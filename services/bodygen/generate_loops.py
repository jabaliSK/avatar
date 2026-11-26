"""Stage A of the hybrid avatar stack: the speaking body / gesture loop from one portrait, with EchoMimicV3-Flash.

EchoMimicV3 (Apache-2.0) on the Wan2.1-Fun-V1.1-1.3B-InP base (Apache-2.0). The models are loaded once and every requested
state is rendered for every requested image:

    python generate_loops.py --job aisha=/in/aisha.png --subject "aisha=A woman in a blue jacket" --speech /in/speech.wav --out /out

Output per image: <out>/<name>/<state>.mp4 (25 fps), <state>.json (settings, timings) and <state>_sheet.jpg (contact sheet).

The speaking clip is driven by real speech, so the body gestures with a speaking cadence; its first frame is the portrait and
the last one is free, which gives much larger gestures than pinning both ends (the renderer plays the clip ping-pong and walks
it back to the first frame when a reply ends, see renderer/app/head_engine.py). The live head renderer replaces the face region
at run time, so only the body, hands, shoulders and head pose of the clip are used.

The quiet states (idle, listening, thinking) can be generated too, driven by silent audio (use --end-frame pin so they stay at
rest), but the renderer does not use them: a quiet clip has no live head (blinks, micro-motion), and a silent EchoMimicV3 clip
shimmers.

The inference code follows upstream infer_flash.py (third_party/echomimic_v3), without its CLI, moviepy and decord.
"""
import argparse
import contextlib
import gc
import json
import logging
import math
import os
import sys
import time

import av
import cv2
import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from prompts import STATES, build_prompt
from src.cache_utils import get_teacache_coefficients
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
from src.utils import color_transfer, filter_kwargs, get_image_to_video_latent2
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
from src.wan_vae import AutoencoderKLWan
from src.wav2vec2 import Wav2Vec2Model

log = logging.getLogger("kka.bodygen")

FPS = 25
AUDIO_RATE = 16000
# PyTorch's default flash-attention kernel runs at ~5-11 TFLOPS on the licence-throttled vGPU this was built on, the cuDNN fused
# attention at ~84 TFLOPS (47k tokens): ~10x faster on the dominant cost. Falls back through the others if cuDNN rejects a shape.
ATTENTION_BACKENDS = {
    "cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    "default": None,
}
SCHEDULERS = {"Flow": FlowMatchEulerDiscreteScheduler, "Flow_Unipc": FlowUniPCMultistepScheduler, "Flow_DPM++": FlowDPMSolverMultistepScheduler}


def sample_size_for(image, area):
    """(h, w) multiples of 16 with the image's aspect ratio and at most area x area pixels (as upstream does)."""
    w, h = image.size
    if area * area < w * h:
        r = math.sqrt(w * h / (area * area))
        w, h = w / r // 16 * 16, h / r // 16 * 16
    else:
        w, h = w // 16 * 16, h // 16 * 16
    return int(h), int(w)


def loudness_norm(x, sr=AUDIO_RATE, lufs=-23):
    loudness = pyln.Meter(sr).integrated_loudness(x)
    return x if abs(loudness) > 100 else pyln.normalize.loudness(x, loudness, lufs)  # silence has no defined loudness


class Generator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        t0 = time.perf_counter()
        base = os.path.join(args.model_dir, "Wan2.1-Fun-V1.1-1.3B-InP")
        config = OmegaConf.load(os.path.join(args.code_dir, "config", "config.yaml"))

        self.audio_encoder = Wav2Vec2Model.from_pretrained(os.path.join(args.model_dir, "chinese-wav2vec2-base"), local_files_only=True).to("cpu")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_fx = Wav2Vec2FeatureExtractor.from_pretrained(os.path.join(args.model_dir, "chinese-wav2vec2-base"), local_files_only=True)

        transformer = self._load_transformer(base, config, os.path.join(args.model_dir, "transformer", "diffusion_pytorch_model.safetensors"))
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(base, config["vae_kwargs"].get("vae_subpath", "vae")), additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])
        ).to(self.dtype)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(base, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(base, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]), low_cpu_mem_usage=True, torch_dtype=self.dtype,
        ).to(self.dtype).eval()  # the loader's fallback path ignores torch_dtype, and the pipeline takes its compute dtype from this model
        clip = CLIPModel.from_pretrained(os.path.join(base, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))).to(self.dtype).eval()

        cls = SCHEDULERS[args.sampler]
        if args.sampler in ("Flow_Unipc", "Flow_DPM++"):
            config["scheduler_kwargs"]["shift"] = 1  # the pipeline applies `shift` itself
        scheduler = cls(**filter_kwargs(cls, OmegaConf.to_container(config["scheduler_kwargs"])))

        for name, model in (("transformer", transformer), ("vae", vae), ("text_encoder", text_encoder), ("clip", clip)):
            got = next(model.parameters()).dtype
            if got != self.dtype:  # one fp32 model turns the whole pipeline fp32, which is ~7x slower on tensor-core GPUs
                raise RuntimeError(f"{name} loaded as {got}, expected {self.dtype}")
        self.pipeline = WanFunInpaintAudioPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
                                                   scheduler=scheduler, clip_image_encoder=clip)
        self.pipeline.to(device=self.device)
        self.vae = vae
        coefficients = get_teacache_coefficients(base) if args.teacache > 0 else None
        if coefficients is not None:
            transformer.enable_teacache(coefficients, args.steps, args.teacache, num_skip_start_steps=args.teacache_skip, offload=False)
        log.info("models ready in %.1fs (GPU memory %.1f GB)", time.perf_counter() - t0, torch.cuda.memory_allocated() / 2**30)

    def _load_transformer(self, base, config, flash_ckpt):
        kw = OmegaConf.to_container(config["transformer_additional_kwargs"])
        transformer = WanTransformer.from_pretrained(
            os.path.join(base, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
            transformer_additional_kwargs=dict(kw), low_cpu_mem_usage=True, torch_dtype=self.dtype,
        )
        missing, unexpected = transformer.load_state_dict(load_file(flash_ckpt), strict=False)
        log.info("Flash checkpoint: %d missing, %d unexpected keys", len(missing), len(unexpected))
        if any(p.is_meta for p in transformer.parameters()):  # the low-memory loader left weights unmaterialised
            log.warning("transformer has meta tensors after loading; reloading without low_cpu_mem_usage")
            transformer = WanTransformer.from_pretrained(
                os.path.join(base, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
                transformer_additional_kwargs=dict(kw), low_cpu_mem_usage=False, torch_dtype=self.dtype,
            )
            transformer.load_state_dict(load_file(flash_ckpt), strict=False)
        return transformer.to(self.dtype)

    # ------------------------------------------------------------------ conditioning
    def audio_embeds(self, wave, n_frames):
        """wave: float32 mono 16 kHz. -> [1, n_frames, 5, 12, 768] on the GPU (5-frame window of the 12 wav2vec2 layers)."""
        wave = loudness_norm(wave)[: int(n_frames / FPS * AUDIO_RATE)]
        feats = np.squeeze(self.audio_fx(wave, sampling_rate=AUDIO_RATE).input_values)
        feats = torch.from_numpy(feats).float().unsqueeze(0)
        with torch.no_grad():
            hs = self.audio_encoder(feats, seq_len=int(n_frames), output_hidden_states=True).hidden_states
        emb = rearrange(torch.stack(hs[1:], dim=1).squeeze(0), "b s d -> s b d")  # [frames, layers, 768]
        idx = torch.arange(n_frames).unsqueeze(1) + (torch.arange(5) - 2).unsqueeze(0)
        emb = emb[idx.clamp(0, emb.shape[0] - 1)]
        return emb.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def wave_for(self, state, speech, n_frames):
        n = int(n_frames / FPS * AUDIO_RATE)
        if STATES[state]["audio"] == "silence":
            return np.zeros(n, np.float32)  # exact zeros stay zeros after the feature extractor's normalisation
        if speech is None:
            raise SystemExit(f"state {state!r} needs --speech")
        return speech[:n]

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def clip(self, image, prompt, wave, seed, length):
        a = self.args
        n = (length - 1) // 4 * 4 + 1  # the VAE compresses time 4x
        h, w = sample_size_for(image, a.sample_area)
        end = image if a.end_frame == "pin" else None  # pinned: the clip comes back to the portrait (a closed loop, but smaller gestures)
        video, mask, clip_image = get_image_to_video_latent2(image, end, video_length=n, sample_size=[h, w])
        if self.pipeline.transformer.teacache is not None:
            self.pipeline.transformer.teacache.reset()
        backends = ATTENTION_BACKENDS[a.attention]
        with sdpa_kernel(backends, set_priority=True) if backends else contextlib.nullcontext():
            out = self.pipeline(
                prompt, num_frames=n, negative_prompt=a.negative, audio_embeds=self.audio_embeds(wave, n),
                ip_mask=None, use_un_ip_mask=False, height=h, width=w, generator=torch.Generator(device=self.device).manual_seed(seed),
                neg_scale=1.0, neg_steps=0, use_dynamic_cfg=False, use_dynamic_acfg=False, guidance_scale=a.guidance,
                audio_guidance_scale=a.audio_guidance, num_inference_steps=a.steps, video=video, mask_video=mask, clip_image=clip_image,
                cfg_skip_ratio=0.0, shift=a.shift,
            ).videos
        frames = (torch.as_tensor(out)[0, :, :n].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # [T,H,W,3]
        ref = frames[0].copy()
        for i in range(1, len(frames)):  # per-frame colour statistics follow frame 0, as in upstream's save_videos_grid
            frames[i] = color_transfer(frames[i], ref)
        return frames


# ---------------------------------------------------------------------- output
def write_mp4(frames, path, fps=FPS, crf=12):
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = {"crf": str(crf), "preset": "slow"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def contact_sheet(frames, path, count=9, height=420):
    idx = np.linspace(0, len(frames) - 1, count).round().astype(int)
    tiles = [cv2.resize(frames[i], (int(frames.shape[2] * height / frames.shape[1]), height), interpolation=cv2.INTER_AREA) for i in idx]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def loop_report(frames, source):
    """How well the clip closes (first vs last frame) and how much it moves (mean abs. difference to frame 0, per frame)."""
    f = frames.astype(np.float32)
    src = cv2.resize(source, (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_AREA).astype(np.float32)
    drift = np.abs(f - f[0]).mean(axis=(1, 2, 3))
    return {"first_last_mae": round(float(np.abs(f[0] - f[-1]).mean()), 3), "first_vs_source_mae": round(float(np.abs(f[0] - src).mean()), 3),
            "motion_mae_mean": round(float(drift.mean()), 3), "motion_mae_max": round(float(drift.max()), 3),
            "step_mae_max": round(float(np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3)).max()), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", action="append", default=[], metavar="NAME=IMAGE", required=True)
    ap.add_argument("--subject", action="append", default=[], metavar="NAME=TEXT", help="who is in the image (clothing, setting) for the prompt")
    ap.add_argument("--states", default="speaking", help="comma-separated: " + ",".join(sorted(STATES)) + "; the renderer only uses speaking")
    ap.add_argument("--speech", help="speech wav for the speaking state")
    ap.add_argument("--speech-offset", type=float, default=0.0, help="start of the excerpt within --speech, seconds")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--model-dir", default=os.environ.get("KKA_BODYGEN_MODELS", "/models/echomimic_v3/flash"))
    ap.add_argument("--code-dir", default="/opt/echomimic_v3")
    ap.add_argument("--steps", type=int, default=8, help="upstream: 8 for Flash; its README suggests more steps for body motion")
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--audio-guidance", type=float, default=5.0,
                    help="audio CFG; measured on one portrait (free end frame): 3 -> mean motion 10.4, 4 -> 11.6, 5 -> 13.0, hands stay clean at 5")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--sampler", default="Flow_Unipc", choices=sorted(SCHEDULERS))
    ap.add_argument("--sample-area", type=int, default=768, help="the clip has at most sample-area^2 pixels, aspect ratio kept")
    ap.add_argument("--length", type=int, default=81, help="frames per clip (81 = 3.24 s, the model's maximum)")
    ap.add_argument("--attention", default="cudnn", choices=sorted(ATTENTION_BACKENDS), help="SDPA backend (see ATTENTION_BACKENDS)")
    ap.add_argument("--end-frame", default="free", choices=("pin", "free"),
                    help="free: only the first frame is the portrait (about 70%% more motion; the renderer walks the body back to the first "
                         "frame after a reply); pin: the last frame is the portrait too (closed loop, smaller gestures)")
    ap.add_argument("--teacache", type=float, default=0.0,
                    help="TeaCache threshold; 0 (default) is off. Upstream's pipeline calls the transformer for the audio-conditioned and the "
                         "unconditioned pass without TeaCache's cond_flag, so TeaCache skips the second pass and the audio guidance fades out: "
                         "measured, it halves the body motion")
    ap.add_argument("--teacache-skip", type=int, default=5)
    ap.add_argument("--negative", default="")
    ap.add_argument("--seed", type=int, default=0, help="added to every state's own seed")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, args.code_dir)

    states = [s.strip() for s in args.states.split(",") if s.strip()]
    for s in states:
        if s not in STATES:
            raise SystemExit(f"unknown state {s!r}; choose from {sorted(STATES)}")
    subjects = dict(item.split("=", 1) for item in args.subject)
    speech = None
    if args.speech:
        wave, _ = librosa.load(args.speech, sr=AUDIO_RATE, mono=True)
        speech = wave[int(args.speech_offset * AUDIO_RATE):]

    jobs = []
    for item in args.job:
        name, path = item.split("=", 1)
        todo = [s for s in states if args.overwrite or not os.path.isfile(os.path.join(args.out, name, f"{s}.mp4"))]
        if todo:
            jobs.append((name, path, todo))
    if not jobs:
        log.info("nothing to do (use --overwrite to regenerate)")
        return

    gen = Generator(args)
    for name, path, todo in jobs:
        image = Image.open(path).convert("RGB")
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        for state in todo:
            prompt = build_prompt(state, subjects.get(name))
            seed = STATES[state]["seed"] + args.seed
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            log.info("%s/%s: generating (%d steps, image %dx%d, seed %d)", name, state, args.steps, *image.size, seed)
            try:
                frames = gen.clip(image, prompt, gen.wave_for(state, speech, args.length), seed, args.length)
            except torch.cuda.OutOfMemoryError:
                log.error("%s/%s: out of GPU memory; lower --sample-area", name, state)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            secs = time.perf_counter() - t0
            base = os.path.join(args.out, name, state)
            write_mp4(frames, base + ".mp4", crf=args.crf)
            contact_sheet(frames, base + "_sheet.jpg")
            report = loop_report(frames, np.asarray(image))
            meta = {"avatar": name, "state": state, "image": path, "frames": len(frames), "size": [frames.shape[2], frames.shape[1]], "fps": FPS,
                    "seed": seed, "steps": args.steps, "guidance": args.guidance, "audio_guidance": args.audio_guidance, "teacache": args.teacache,
                    "attention": args.attention, "sample_area": args.sample_area, "end_frame": args.end_frame,
                    "prompt": prompt, "negative": args.negative, "generate_s": round(secs, 1),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), **report}
            with open(base + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            log.info("%s/%s: %.0f s, peak %.1f GB, %s", name, state, secs, meta["peak_gpu_gb"], report)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
