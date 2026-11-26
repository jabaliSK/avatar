"""Drive one registered template with a speech clip through HeadEngine and report quality + latency.

Audio is fed in 200 ms slices as fast as the engine accepts them, which models a TTS that runs faster than real time.
Lip-sync is scored objectively: the mouth opening of the rendered faces (MediaPipe landmarks) is cross-correlated with the
audio energy envelope, per 40 ms video frame.

  python tools/run_head_clip.py --image /data/portrait.png --audio /data/speech.wav --out /work/clip1
"""
import argparse
import json
import os
import subprocess
import time

import cv2
import librosa
import numpy as np
import torch

from app.avatar import Loop
from app.compositor import PatchReader, blend_roi, paste
from app.head_engine import HUBERT_STEP, SAMPLES_PER_FRAME, HeadEngine
from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.putback import PutBack

# MediaPipe 478-landmark indices
UPPER_INNER_LIP, LOWER_INNER_LIP, EYE_OUTER_R, EYE_OUTER_L = 13, 14, 33, 263


def mouth_opening(landmarker, rgb512):
    res = landmarker(np.ascontiguousarray(rgb512))
    if res.ndim != 3 or res.shape[0] == 0:
        return np.nan
    p = res[0]
    iod = np.linalg.norm(p[EYE_OUTER_R, :2] - p[EYE_OUTER_L, :2]) + 1e-6
    return float(np.linalg.norm(p[UPPER_INNER_LIP, :2] - p[LOWER_INNER_LIP, :2]) / iod)


def lip_sync_score(mouth, audio16k, max_lag=10):
    """Cross-correlates mouth opening with the audio energy envelope per 40 ms frame (positive lag: mouth follows audio).

    Reported for a linear and a log envelope, plus the mean opening in silent vs speech frames: a mouth that opens with the
    voice and rests in pauses gives a clear ratio even when the correlation is capped by the coarse landmark measurement.
    """
    n = len(mouth)
    rms = np.array([np.sqrt(np.mean(audio16k[i * SAMPLES_PER_FRAME:(i + 1) * SAMPLES_PER_FRAME] ** 2) + 1e-12) for i in range(n)])
    m = np.asarray(mouth, np.float64)
    ok = ~np.isnan(m)
    if ok.sum() < 10:
        return {"valid_frames": int(ok.sum())}
    smooth = lambda v: np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    out = {"valid_frames": int(ok.sum()), "mouth_range": [float(np.nanmin(m)), float(np.nanmax(m))]}
    for kind, env in (("linear", smooth(rms)), ("log", np.log(rms + 1e-4))):
        res, best = {}, (-2.0, 0)
        for lag in range(-max_lag, max_lag + 1):
            a = env[: n - lag] if lag >= 0 else env[-lag:]
            b = m[lag:] if lag >= 0 else m[: n + lag]
            k = ~np.isnan(b)
            if k.sum() < 10:
                continue
            c = np.corrcoef(a[k], b[k])[0, 1]
            res[lag] = float(c)
            if c > best[0]:
                best = (c, lag)
        out[kind] = {"corr_lag0": res.get(0), "best_corr": float(best[0]), "best_lag_ms": int(best[1] * 40)}
    thr = 0.15 * np.median(rms[rms > np.percentile(rms, 50)])
    sil, sp = ok & (rms < thr), ok & (rms >= thr)
    if sil.sum() >= 3 and sp.sum() >= 3:
        out["mouth_mean_silence"] = float(m[sil].mean())
        out["mouth_mean_speech"] = float(m[sp].mean())
        out["speech_to_silence_ratio"] = float(m[sp].mean() / max(m[sil].mean(), 1e-6))
        out["silent_frames"] = int(sil.sum())
    return out


def to_u8(render01):
    return (render01[0].clamp(0, 1).permute(1, 2, 0).float() * 255.0).round().to(torch.uint8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="portrait image or looping video of the body")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="/models/ditto-talkinghead/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--data-root", default="/models/ditto-talkinghead/ditto_pytorch")
    ap.add_argument("--steps", type=int, default=10, help="LMDM sampling steps")
    ap.add_argument("--overlap", type=int, default=70, help="overlap_v2: 70 -> LMDM call per 10 frames, 75 -> per 5")
    ap.add_argument("--drive-eye", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--no-ctrl", action="store_true", help="drop the shipped overall_ctrl_info (+2 deg pitch)")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--warmup", type=int, default=1, help="full dry runs before the measured run")
    ap.add_argument("--engine-dir", default="", help="TensorRT engines (GPU live path + GPU compositor); empty = PyTorch + CPU PutBack")
    ap.add_argument("--lmdm-precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    replace = {"default_kwargs": {"online_mode": True, "overlap_v2": args.overlap, "sampling_timesteps": args.steps}}
    use_trt = bool(args.engine_dir)
    engine = HeadEngine(args.cfg, args.data_root, replace_cfg=replace, engine_dir=args.engine_dir or None,
                        lmdm_precision=args.lmdm_precision)
    registrar = AvatarRegistrar(**engine.registrar_cfg)
    kw = engine.kw
    crop_kw = {k: kw[k] for k in ("crop_scale", "crop_vx_ratio", "crop_vy_ratio", "crop_flag_do_rot") if k in kw}
    t0 = time.perf_counter()
    if use_trt:
        loop = Loop.register("clip", args.image, registrar, engine, kw)
        info, frames_src = loop.info, loop.frames
    else:
        info = registrar(args.image, max_dim=kw.get("max_size", 1920), **crop_kw)
        if len(info["x_s_info_lst"]) > 1 and kw.get("smo_k_s", 1) > 1:
            info["x_s_info_lst"] = smooth_x_s_info_lst(info["x_s_info_lst"], smo_k=kw["smo_k_s"])
        frames_src = info["img_rgb_lst"]
    n_src = len(info["x_s_info_lst"])
    print(f"registered {n_src} template frame(s) in {time.perf_counter() - t0:.2f}s (image={info['is_image_flag']})")

    audio, _ = librosa.load(args.audio, sr=16000)
    if args.max_seconds > 0:
        audio = audio[: int(args.max_seconds * 16000)]
    print(f"audio {len(audio) / 16000:.2f}s")

    begin_kw = {}
    if args.drive_eye != "auto":
        begin_kw["drive_eye"] = args.drive_eye == "true"
    if args.no_ctrl:
        begin_kw["overall_ctrl_info"] = {}
    putback = None if use_trt else PutBack()
    reader = PatchReader()

    def run(measure):
        engine.timer.reset()
        engine.begin(info, **begin_kw)
        renders, frames, t_emit = [], [], []
        first_audio_s, overhead = None, 0.0
        t_start = time.perf_counter()
        for off in range(0, len(audio), HUBERT_STEP):
            final = off + HUBERT_STEP >= len(audio)
            for idx, x_d in engine.feed_audio(audio[off:off + HUBERT_STEP], final=final):
                t1 = time.perf_counter()
                if use_trt:
                    render, fidx = engine.render_gpu(idx, x_d)
                    roi = loop.rois[fidx]
                    out = paste(frames_src[fidx], reader.read(blend_roi(render, roi)), roi)  # the read synchronises the GPU
                    engine.timer.add("render+blend+d2h+paste", time.perf_counter() - t1)
                else:
                    img, fidx = engine.render(idx, x_d)
                    out = putback(frames_src[fidx], img, info["M_c2o_lst"][fidx])
                    engine.timer.add("putback", time.perf_counter() - t1)
                if first_audio_s is None:
                    first_audio_s = (off + HUBERT_STEP) / 16000
                    t_first = time.perf_counter() - t_start
                if measure:
                    t2 = time.perf_counter()
                    renders.append(to_u8(render) if use_trt else np.clip(img, 0, 255).astype(np.uint8))
                    frames.append(out)
                    overhead += time.perf_counter() - t2  # keeping the frames for scoring is not part of the pipeline cost
                t_emit.append(time.perf_counter() - t_start)
        torch.cuda.synchronize()
        return renders, frames, t_emit, first_audio_s, t_first, time.perf_counter() - t_start - overhead

    for _ in range(args.warmup):
        run(False)
    renders, frames, t_emit, first_audio_s, t_first, total = run(True)

    n = len(frames)
    summary = {
        "frames": n,
        "audio_s": round(len(audio) / 16000, 3),
        "expected_frames": int(np.ceil(len(audio) / SAMPLES_PER_FRAME)),
        "wall_s": round(total, 3),
        "rtf": round(total / (len(audio) / 16000), 3),
        "compute_ms_per_frame": round(total / max(n, 1) * 1e3, 2),
        "fps_if_compute_bound": round(n / total, 2),
        "audio_needed_before_first_frame_s": first_audio_s,
        "wall_to_first_frame_ms": round(t_first * 1e3, 1),
        "stages": engine.timer.summary(),
        "cfg": {"steps": args.steps, "overlap": args.overlap, "drive_eye": args.drive_eye, "no_ctrl": args.no_ctrl,
                "backend": f"tensorrt(lmdm {args.lmdm_precision}) + gpu compositor" if use_trt else "pytorch + cpu putback"},
    }

    lm = registrar.source2info.landmark478
    mouth = [mouth_opening(lm, r) for r in renders]
    summary["lip_sync"] = lip_sync_score(mouth, audio)
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "clip_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    idx = np.linspace(0, n - 1, 8).astype(int)
    row = lambda ids: np.concatenate([renders[i] for i in ids], 1)
    cv2.imwrite(os.path.join(args.out, "contact.png"), cv2.cvtColor(np.concatenate([row(idx[:4]), row(idx[4:])], 0), cv2.COLOR_RGB2BGR))
    np.save(os.path.join(args.out, "mouth.npy"), np.asarray(mouth))

    if not args.no_video:
        h, w = frames[0].shape[:2]
        tmp = os.path.join(args.out, "silent.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-i", args.audio, "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        os.path.join(args.out, "clip.mp4")], check=False)
        os.remove(tmp)


if __name__ == "__main__":
    main()
