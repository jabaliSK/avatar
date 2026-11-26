"""Push-driven wrapper around Ditto's atomic components (audio -> motion -> stitch -> warp -> decode).

Ditto's StreamSDK is a one-shot, file-writing pipeline with six worker threads. A live avatar needs the same arithmetic
driven chunk by chunk, restartable for every reply and without a video writer, so this reproduces StreamSDK.setup() and its
audio2motion worker as plain methods. Differences from upstream are listed in services/renderer/PATCHES.md.

Timing contract: video frame g covers audio samples [g*640, (g+1)*640) at 16 kHz (40 ms), so a caller that emits frame g
together with that audio slice is lip-synchronised by construction.
"""
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.cfg import parse_cfg
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.motion_stitch import MotionStitch, transform_keypoint
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.wav2feat import Wav2Feat

# HuBERT streaming geometry (40 ms frames, 16 kHz): 3 frames of left context, 5 emitted, 2 of lookahead.
CHUNKSIZE = (3, 5, 2)
SAMPLES_PER_FRAME = 640
HUBERT_WINDOW = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80  # 6480 samples per call
HUBERT_STEP = CHUNKSIZE[1] * SAMPLES_PER_FRAME  # 3200 new samples = 5 video frames per call
HUBERT_LEFT_PAD = CHUNKSIZE[0] * SAMPLES_PER_FRAME  # 1920

_COND_KEYS = ("x_s_info_lst", "sc", "eye_open_lst", "eye_ball_lst")

# After the last word the body template keeps moving until it is back at its rest pose (frame 0), so a gesture is finished the way
# a person finishes it instead of being cut. The walk is a speed profile in template frames per output frame: it carries on at
# the speed the template was running at, turns around smoothly if the rest pose is behind it, cruises, and brakes into the rest.
# The speed is capped where the template itself moves fast (the start of a gesture): there the walk plays at the template's own
# pace, never faster, so the hand is not whipped back.
TAIL_CRUISE = 1.75      # template frames per output frame on a calm stretch
TAIL_ACCEL = 0.25       # change of that speed per output frame
TAIL_PACE = 1.6         # the walk may change the picture at most this many times the template's median change per frame
TAIL_REST_SPEED = 0.35  # speed on arrival at the rest pose

# Strength of the live head when the caller does not say (see avatar_store.DEFAULT_SETTINGS).
DEFAULT_EXPRESSION = 0.6
DEFAULT_HEAD_MOTION = 0.5

# How a video template drives the rest face (MotionStitch.setup rest_opts). A loop whose head was placed on the template's real face
# frame by frame (Loop.align, source_info["aligned"]) keeps the rest face's own scale, because the per-frame scale estimate is a few
# percent off in a different way in every frame and shows as the head swelling; the placement restores the real size. It also takes
# every non-lip expression dimension (brows, eyes, cheeks) from the template frame, so the head keeps the template's own expression.
# A loop registered without that placement keeps the estimated scale and takes only the eyes, as before.
REST_ALIGNED = {"scale": "rest", "exp": "nonlip"}
REST_LEGACY = {"scale": "frame", "exp": "eyes"}


def rest_options(source_info, override=None):
    """MotionStitch rest_opts for a video template; `override` (a begin() argument, used by the lab tools) wins over the policy."""
    return {**(REST_ALIGNED if source_info.get("aligned") else REST_LEGACY), **(override or {})}


def tail_caps(n, motion=None):
    """caps[i]: the fastest the walk may go, in template frames per output frame, when it leaves template frame i towards frame 0.

    Built from the rest pose outwards, so a slow stretch is announced early enough to brake for it."""
    caps = np.full(n, TAIL_CRUISE)
    if motion is not None and len(motion) >= n:
        typical = max(float(np.median(motion[:n])), 1e-6)
        for i in range(1, n):
            change = float(motion[i - 1]) / typical  # leaving frame i for i-1 plays the change between those two frames
            caps[i] = min(TAIL_CRUISE, max(1.0, TAIL_PACE / max(change, 1e-6)))
    caps[0] = TAIL_REST_SPEED
    for i in range(1, n):
        caps[i] = min(caps[i], math.sqrt(caps[i - 1] ** 2 + 2.0 * TAIL_ACCEL))
    return caps


def tail_path(last, direction, n, motion=None):
    """Template frame indices of the silent frames that bring the body from frame `last` of an `n`-frame template, moving in
    `direction` (+1 away from the rest pose, -1 towards it), back to frame 0. Empty when the body is already at rest.
    `motion` is the template's per-frame change (avatar.motion_profile); without it the walk cruises everywhere."""
    caps = tail_caps(n, motion)
    pos, v, path = float(last), float(direction), []
    while pos > 0.5 and len(path) < 400:
        at = min(pos, n - 1.0)
        lo = int(at)
        cap = caps[lo] + (caps[min(lo + 1, n - 1)] - caps[lo]) * (at - lo)
        v += max(-TAIL_ACCEL, min(TAIL_ACCEL, -cap - v))
        pos = min(n - 1.0, max(0.0, pos + v))
        path.append(int(round(pos)))
    if path:
        path[-1] = 0
    return path


def has_blinks(source_info):
    """Whether the template's own eyes close at some point. A Stage A idle clip often does not, and a face that never blinks reads
    as a photograph; such a template gets Ditto's synthetic blinks instead (see HeadEngine.begin)."""
    eyes = np.asarray(source_info["eye_open_lst"], dtype=np.float32)
    eyes = eyes.reshape(len(eyes), -1).mean(axis=1)
    return bool(eyes.min() < 0.6 * np.median(eyes))


class StageTimer:
    def __init__(self):
        self.t = defaultdict(list)

    def add(self, name, seconds):
        self.t[name].append(seconds * 1e3)

    def reset(self):
        self.t.clear()

    def summary(self):
        out = {}
        for k, v in self.t.items():
            a = np.asarray(v)
            out[k] = {
                "n": int(a.size),
                "mean_ms": round(float(a.mean()), 3),
                "p50_ms": round(float(np.percentile(a, 50)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "max_ms": round(float(a.max()), 3),
            }
        return out


@dataclass
class _Reply:
    source_info: dict
    n_source: int
    audio_buf: np.ndarray  # float32; absolute index of audio_buf[0] is buf_offset
    buf_offset: int = 0
    audio_pos: int = 0  # absolute start of the next HuBERT window (left-padded timeline)
    real_end: int = HUBERT_LEFT_PAD  # absolute end of real samples (left-padded timeline)
    item_buffer: np.ndarray = None
    res_kp_seq: np.ndarray = None
    valid_start: int = None
    global_idx: int = 0
    local_idx: int = 0
    gen_frame_idx: int = 0
    finished: bool = False
    tail_from: int = None  # first output frame of the walk back to the rest pose; None while the reply is still open
    tail_path: list = field(default_factory=list)  # template frame of each frame of that walk

    @property
    def samples_in(self):
        return self.real_end - HUBERT_LEFT_PAD


class HeadEngine:
    def __init__(self, cfg_pkl, data_root, replace_cfg=None, timer=None, engine_dir=None, lmdm_precision="fp32"):
        (reg_cfg, ch_cfg, lmdm_cfg, stitch_cfg, warp_cfg, dec_cfg, w2f_cfg, default_kwargs) = parse_cfg(
            cfg_pkl, data_root, replace_cfg
        )
        self.registrar_cfg = reg_cfg
        self.kw = dict(default_kwargs)
        self.timer = timer or StageTimer()

        self.condition_handler = ConditionHandler(**ch_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_cfg)
        self.renderer = None
        if engine_dir:
            # Live path: TensorRT LMDM sampler and warp+decode on GPU tensors (see trt_models.py).
            from .trt_models import LMDMTrt, TrtRenderer
            self.audio2motion.lmdm = LMDMTrt(
                os.path.join(engine_dir, f"lmdm_v0.4_hubert_{lmdm_precision}.engine"),
                seq_frames=self.audio2motion.lmdm.seq_frames,
            )
            self.renderer = TrtRenderer(engine_dir)
            self.warp_f3d = self.decode_f3d = None
        else:
            self.warp_f3d = WarpF3D(warp_cfg)
            self.decode_f3d = DecodeF3D(dec_cfg)
        self.wav2feat = Wav2Feat(**w2f_cfg)
        assert self.wav2feat.support_streaming
        self.feat_dim = self.wav2feat.feat_dim
        self.reply = None
        self._silent_feat = {}

    def _silent_features(self, n_frames):
        # Silence is deterministic, so compute it once per length instead of once per reply.
        if n_frames not in self._silent_feat:
            feat = self.wav2feat.wav2feat(np.zeros((n_frames * SAMPLES_PER_FRAME,), np.float32), sr=16000)
            assert len(feat) == n_frames, len(feat)
            self._silent_feat[n_frames] = feat
        return self._silent_feat[n_frames].copy()

    # ------------------------------------------------------------------ reply lifecycle
    def begin(self, source_info, **kwargs):
        """Start a reply on a registered template (image, or a looping video of the body).

        The face is always rendered from the appearance of the template's first frame (its rest pose, a closed mouth): a video
        template's later frames are talking, and a talking frame's own appearance feature carries its open mouth, which the warp
        cannot close again (it is what made the live mouth follow the template's). A video template therefore only brings its head
        pose and eyes; the live head adds the audio-driven mouth on top of the rest face (see MotionStitch rest_info). A single image
        has no pose of its own: its head motion comes from the audio model, scaled by `head_motion`. `expression` scales the mouth /
        jaw amplitude.
        """
        is_image = source_info["is_image_flag"]
        kw = {**self.kw, **({} if is_image else {"overall_ctrl_info": {}}), **kwargs}
        self.emo = kw.get("emo", 4)
        self.overlap_v2 = kw.get("overlap_v2", 10)
        assert kw.get("online_mode", False), "live driving needs the online (streaming) configuration"
        self.ctrl_info = kw.get("ctrl_info", {})

        use_d_keys = kw.get("use_d_keys")
        if use_d_keys is None:
            use_d_keys = {"exp": float(kw.get("expression", DEFAULT_EXPRESSION))}
            if is_image:
                head = float(kw.get("head_motion", DEFAULT_HEAD_MOTION))
                use_d_keys.update({"pitch": head, "yaw": head, "roll": head, "t": head})
        rest_info = None
        rest_opts = None
        if not is_image:
            rest_info = source_info["x_s_info_lst"][0]
            rest_opts = rest_options(source_info, kw.get("rest_opts"))
        drive_eye = kw.get("drive_eye")
        if drive_eye is None and not is_image:
            drive_eye = not has_blinks(source_info)  # None lets MotionStitch decide (a single image always gets synthetic blinks)

        # The shipped configs carry a reference character (ch_info) that conditions LMDM and sets the motion scale ratio,
        # exactly as StreamSDK does. Without one, condition on the template itself. Either way pass a dict:
        # ConditionHandler.setup deep-copies source_info (every frame and f_s) when ch_info is None.
        ch_info = kw.get("ch_info")
        cond_src = ch_info if ch_info is not None else {k: source_info[k] for k in _COND_KEYS}
        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=kw.get("eye_f0_mode", False), ch_info=cond_src)
        self.audio2motion.setup(
            self.condition_handler.x_s_info_0,
            overlap_v2=self.overlap_v2,
            fix_kp_cond=kw.get("fix_kp_cond", 0),
            fix_kp_cond_dim=kw.get("fix_kp_cond_dim"),
            sampling_timesteps=kw.get("sampling_timesteps", 50),
            online_mode=True,
            v_min_max_for_clip=kw.get("v_min_max_for_clip"),
            smo_k_d=kw.get("smo_k_d", 3),
        )
        self.motion_stitch.setup(
            N_d=-1,
            use_d_keys=use_d_keys,
            relative_d=kw.get("relative_d", True),
            drive_eye=drive_eye,
            delta_eye_arr=kw.get("delta_eye_arr"),
            delta_eye_open_n=kw.get("delta_eye_open_n", 0),
            fade_out_keys=kw.get("fade_out_keys", ("exp",)),
            fade_type=kw.get("fade_type", ""),
            flag_stitching=kw.get("flag_stitching", True),
            is_image_flag=is_image,
            x_s_info=source_info["x_s_info_lst"][0],
            d0=None,
            ch_info=ch_info,
            overall_ctrl_info=kw.get("overall_ctrl_info", {}),
            rest_info=rest_info,
            rest_opts=rest_opts,
        )

        # The first LMDM window is seq_frames long: overlap_v2 frames of silence, then real audio.
        self.audio_feat = self._silent_features(self.overlap_v2)
        self.cond_idx_start = -len(self.audio_feat)

        self.reply = _Reply(
            source_info=source_info,
            n_source=len(source_info["x_s_info_lst"]),
            audio_buf=np.zeros((HUBERT_LEFT_PAD,), np.float32),
            item_buffer=np.zeros((0, self.feat_dim), np.float32),
        )

    # ------------------------------------------------------------------ audio in -> driving motion out
    def feed_audio(self, pcm16k, final=False):
        """Append 16 kHz mono float32 audio. Returns [(frame_idx, x_d_info), ...] that became ready.

        final=True flushes the tail (zero-padded HuBERT windows) and ends the reply.
        """
        r = self.reply
        if r is None or r.finished:
            raise RuntimeError("feed_audio() called without an active reply; call begin() first")
        if pcm16k is not None and len(pcm16k):
            r.audio_buf = np.concatenate([r.audio_buf, np.asarray(pcm16k, dtype=np.float32)])
            r.real_end += len(pcm16k)

        feats = []
        while True:
            s = r.audio_pos - r.buf_offset
            if s + HUBERT_WINDOW <= len(r.audio_buf):
                window = r.audio_buf[s:s + HUBERT_WINDOW]
            elif final and r.audio_pos < r.real_end:
                window = np.zeros((HUBERT_WINDOW,), np.float32)
                tail = r.audio_buf[s:]
                window[:len(tail)] = tail
            else:
                break
            t0 = time.perf_counter()
            feats.append(self.wav2feat(window, chunksize=CHUNKSIZE))
            self.timer.add("hubert_chunk", time.perf_counter() - t0)
            r.audio_pos += HUBERT_STEP

        drop = r.audio_pos - r.buf_offset
        if drop > 0:
            r.audio_buf = r.audio_buf[drop:]
            r.buf_offset = r.audio_pos

        out = []
        if feats:
            out += self._feed_features(np.concatenate(feats, 0), is_end=False)
        if final:
            out += self._feed_features(None, is_end=True)
            # The tail window is padded, so the model can emit a few frames past the real audio; drop them.
            n_expected = math.ceil(r.samples_in / SAMPLES_PER_FRAME)
            out = [(i, x) for i, x in out if i < n_expected]
            r.finished = True
        return out

    def _feed_features(self, feat, is_end):
        r = self.reply
        a2m = self.audio2motion
        seq_frames, valid_clip_len = a2m.seq_frames, a2m.valid_clip_len
        out = []

        if feat is not None:
            r.item_buffer = np.concatenate([r.item_buffer, feat], 0)
        if not is_end and r.item_buffer.shape[0] < valid_clip_len:
            return out  # wait for at least valid_clip_len new frames
        self.audio_feat = np.concatenate([self.audio_feat, r.item_buffer], 0)
        r.item_buffer = np.zeros((0, self.feat_dim), np.float32)

        while True:
            aud_feat = self.audio_feat[r.local_idx:r.local_idx + seq_frames]
            real_valid_len = valid_clip_len
            if len(aud_feat) == 0:
                break
            if len(aud_feat) < seq_frames:
                if not is_end:
                    break  # wait for the next chunk
                real_valid_len = len(aud_feat)
                pad = np.stack([aud_feat[-1]] * (seq_frames - len(aud_feat)), 0)
                aud_feat = np.concatenate([aud_feat, pad], 0)

            t0 = time.perf_counter()
            aud_cond = self.condition_handler(aud_feat, r.global_idx + self.cond_idx_start)[None]
            r.res_kp_seq = a2m(aud_cond, r.res_kp_seq)
            self.timer.add("lmdm_call", time.perf_counter() - t0)

            if r.valid_start is None:
                # First window only seeds d0 (the reference expression); it emits nothing.
                r.valid_start = r.res_kp_seq.shape[1] - a2m.fuse_length
                self.motion_stitch.d0 = a2m.cvt_fmt(r.res_kp_seq[0:1])[0]
            else:
                valid = r.res_kp_seq[:, r.valid_start:r.valid_start + real_valid_len]
                for x_d_info in a2m.cvt_fmt(valid):
                    out.append((r.gen_frame_idx, x_d_info))
                    r.gen_frame_idx += 1
                r.valid_start += real_valid_len

            r.local_idx += real_valid_len
            r.global_idx += real_valid_len

            n = r.res_kp_seq.shape[1]
            if n > seq_frames * 2:
                cut = n - seq_frames * 2
                r.res_kp_seq = r.res_kp_seq[:, cut:]
                r.valid_start -= cut
            if r.local_idx >= len(self.audio_feat):
                break

        n = len(self.audio_feat)
        if n > seq_frames * 2:
            cut = n - seq_frames * 2
            self.audio_feat = self.audio_feat[cut:]
            r.local_idx -= cut
        return out

    # ------------------------------------------------------------------ one driving frame -> 512x512 head render
    def set_tail(self, speech_frames):
        """Declare that the speech occupies output frames [0, speech_frames). Returns how many silent frames the caller has to
        append so the body template can walk back to its rest pose (0 for a single image or a body already at rest)."""
        r = self.reply
        r.tail_from = speech_frames
        r.tail_path = []
        if speech_frames > 0 and r.n_source > 1:
            forward = ((speech_frames - 1) // r.n_source) % 2 == 0  # which way _mirror_index was running at the last word
            r.tail_path = tail_path(_mirror_index(speech_frames - 1, r.n_source), 1 if forward else -1, r.n_source,
                                    r.source_info.get("body_motion"))
        return len(r.tail_path)

    def template_index(self, frame_idx):
        """Template frame that output frame `frame_idx` is composited on: ping-pong over the loop, then the walk to the rest pose."""
        r = self.reply
        if r.tail_from is None or frame_idx < r.tail_from:
            return _mirror_index(frame_idx, r.n_source)
        path = r.tail_path
        return path[min(frame_idx - r.tail_from, len(path) - 1)] if path else 0

    def stitch(self, frame_idx, x_d_info):
        """Driving motion -> (template frame index, source keypoints, driving keypoints), each [1,21,3] numpy."""
        r = self.reply
        fidx = self.template_index(frame_idx)
        ctrl = self.ctrl_info.get(frame_idx, {}) if isinstance(self.ctrl_info, dict) else {}
        t0 = time.perf_counter()
        x_s, x_d = self.motion_stitch(r.source_info["x_s_info_lst"][fidx], x_d_info, **ctrl)
        self.timer.add("stitch", time.perf_counter() - t0)
        return fidx, x_s, x_d

    def render(self, frame_idx, x_d_info):
        """PyTorch/numpy path. Returns (render [512, 512, 3] float32 in 0..255, template frame index)."""
        fidx, x_s, x_d = self.stitch(frame_idx, x_d_info)
        t1 = time.perf_counter()
        f3d = self.warp_f3d(self.reply.source_info["f_s_lst"][0], x_s, x_d)  # always the first frame's appearance, see begin()
        t2 = time.perf_counter()
        img = self.decode_f3d(f3d)
        self.timer.add("warp", t2 - t1)
        self.timer.add("decode", time.perf_counter() - t2)
        return img, fidx

    def reconstruct(self, f_s, rest_info, frame_info):
        """Template frame `frame_info` re-rendered from the rest face (appearance `f_s`, keypoints `rest_info`) as float32 [512,512,3]
        in 0..255, on the live backend: the head the live path starts from, with the frame's own pose and expression.

        Compared with the frame's crop it yields the colour residual that makes a neutral live head match the body loop.
        """
        x_s, x_d = transform_keypoint(rest_info), transform_keypoint(frame_info)
        if self.renderer is None:
            return self.decode_f3d(self.warp_f3d(f_s, x_s, x_d))
        f = torch.from_numpy(np.ascontiguousarray(f_s, np.float32)).to(self.renderer.device)
        img = self.renderer(f, x_s, x_d)
        return (img[0].clamp(0, 1).permute(1, 2, 0).float() * 255.0).cpu().numpy()

    def render_gpu(self, frame_idx, x_d_info):
        """TensorRT path. Returns (render [1,3,512,512] GPU tensor, nominally 0..1, template frame index).

        The call only enqueues GPU work; the caller synchronises when it reads the composited result back.
        source_info must carry f_s_gpu_lst (the first frame's appearance feature is used; see begin()).
        """
        fidx, x_s, x_d = self.stitch(frame_idx, x_d_info)
        return self.renderer(self.reply.source_info["f_s_gpu_lst"][0], x_s, x_d), fidx
