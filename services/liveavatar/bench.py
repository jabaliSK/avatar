"""Resident two-device LiveAvatar benchmark; run only against the patched, pinned checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave

import torch
import torch.distributed as dist

REV = "c3c47d031d8bf3247428333c1fe610579c71a551"
FPS, BLOCK, SEED, AREA = 25, 48, 420, 704 * 384
PROMPT = ("A man in a black bisht addresses the camera naturally. His face, shoulders and torso "
          "move with the spoken phrases. One hand gestures gently while the other rests. The hands "
          "remain separate. Eyes on the viewer, a still camera, unchanged clothing and lighting.")


def atomic(path, text):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(text)
    temp.replace(path)


def playlist(lengths, done):
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:2", "#EXT-X-PLAYLIST-TYPE:EVENT",
             "#EXT-X-MEDIA-SEQUENCE:0"]
    for i, frames in enumerate(lengths):
        if i:
            lines.append("#EXT-X-DISCONTINUITY")  # each independently encoded segment starts its clocks at zero
        lines.extend([f"#EXTINF:{frames / FPS:.3f},", f"{i}.ts"])
    if done:
        lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


class Decoder:
    """One ordered VAE worker; at most one older clip in flight while the DiT computes the next."""
    def __init__(self, pool, out, audio, target, t0):
        self.pool, self.out, self.audio, self.target, self.t0 = pool, out, audio, target, t0
        self.pending = None
        self.rows = []

    def start(self, vae, motion, pixels, overlap, infer_frames):
        self.vae, self.motion, self.pixels, self.overlap, self.block = vae, motion, pixels, overlap, infer_frames
        torch.cuda.synchronize(vae.device)

    def submit(self, index, latent_cpu):
        if self.pending is not None:
            self.pending.result()
        self.pending = self.pool.submit(self.decode, index, latent_cpu)

    def finish(self):
        self.pending.result()
        atomic(self.out / "index.m3u8", playlist([r["frames"] for r in self.rows], True))

    @torch.inference_mode()
    def decode(self, index, latent_cpu):
        dev = self.vae.device
        torch.cuda.set_device(dev)
        with torch.autocast("cuda", dtype=self.vae.dtype):
            latent = latent_cpu.to(device=dev, dtype=self.vae.dtype)
            data = torch.cat([self.motion, latent.unsqueeze(0)], dim=2)
            img = torch.stack(self.vae.decode(data))[:, :, -self.block:]
            if index == 0:
                img = img[:, :, 3:]
            overlap = min(self.overlap, img.shape[2])
            self.pixels = torch.cat([self.pixels[:, :, overlap:], img[:, :, -overlap:]], dim=2)
            self.motion = torch.stack(self.vae.encode(self.pixels.to(self.motion))).type_as(latent)
            seen = sum(r["frames"] for r in self.rows)
            img = img[:, :, :self.target - seen]
            if img.shape[2] == 0:
                return
            if not torch.isfinite(img).all().item():
                raise ValueError("non-finite decoded video")
            frames = ((img[0].clamp(-1, 1) + 1) * 127.5 + .5).byte().permute(1, 2, 3, 0).contiguous().cpu().numpy()
        if frames.max() < 8:
            raise ValueError("black decoded clip")
        decoded = time.monotonic() - self.t0
        n, h, w, _ = frames.shape
        command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", f"{w}x{h}", "-framerate", str(FPS), "-i", "pipe:0",
                   "-ss", str(seen / FPS), "-i", str(self.audio), "-t", str(n / FPS),
                   "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-threads", "3", "-preset", "fast",
                   "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-f", "mpegts", str(self.out / f"{index}.ts")]
        subprocess.run(command, input=frames.tobytes(), check=True)
        lengths = [r["frames"] for r in self.rows] + [n]
        atomic(self.out / "index.m3u8", playlist(lengths, False))
        self.rows.append(dict(frames=n, width=w, height=h, cpu_rgb_s=decoded, available_s=time.monotonic() - self.t0))
        logging.info("chunk %d: %s", index, self.rows[-1])


def load(args):
    from liveavatar.models.wan import causal_model_s2v
    from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
    from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS
    from liveavatar.models.wan.wan_2_2.modules import attention, model
    from liveavatar.utils.args_config import parse_args_for_training_config
    from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE):
        def sdpa(q, k, v, k_lens=None, window_size=(-1, -1), **kw):
            if window_size != (-1, -1) or any(kw.values()):
                raise ValueError("unsupported specialized attention arguments")
            mask = None
            if k_lens is not None:
                mask = torch.arange(k.shape[1], device=k.device)[None, :] < k_lens.to(k.device)[:, None]
                if mask.all().item():
                    mask = None
                else:
                    mask = mask[:, None, None, :]
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                y = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                                   v.transpose(1, 2), attn_mask=mask)
            return y.transpose(1, 2).contiguous().type_as(q)

        model.flash_attention = causal_model_s2v.flash_attention = sdpa
    cfg = WAN_CONFIGS["s2v-14B"]
    pipe = WanS2V(config=cfg, checkpoint_dir=str(args.weights / "Wan2.2-S2V-14B"),
                  convert_model_dtype=True, single_gpu=True, init_on_cpu=False, decoder_device=torch.device("cuda:1"))
    settings = parse_args_for_training_config("liveavatar/configs/s2v_causal_sft.yaml")
    pipe.noise_model = pipe.add_lora_to_model(pipe.noise_model, **{k: settings[k] for k in
        ("lora_rank", "lora_alpha", "lora_target_modules", "init_lora_weights")},
        pretrained_lora_path=str(args.weights / "Live-Avatar/liveavatar.safetensors"))
    replace_linear_with_scaled_fp8(pipe.noise_model, ignore_keys=["text_embedding", "time_embedding", "time_projection",
        "head.head", "casual_audio_encoder.encoder.final_linear"])
    # Keep the KV caches resident, not the 11 GiB T5. Reuse only an exact prompt match.
    encode = pipe.encode_prompt
    @functools.lru_cache(maxsize=1)
    def cached(input_prompt, n_prompt, offload_model):
        return encode(input_prompt, n_prompt, offload_model=True)
    pipe.encode_prompt = cached
    return pipe


def benchmark(args, pipe, out):
    with wave.open(str(args.audio)) as wf:
        seconds = wf.getnframes() / wf.getframerate()
    target = round(seconds * FPS)
    if not 1 <= target <= FPS * 60:
        raise ValueError("benchmark expects at most 60 seconds of audio")
    for i in range(2):
        torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(i)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool, torch.inference_mode():
        decoder = Decoder(pool, out, args.audio, target, t0)
        pipe.generate(input_prompt=PROMPT, ref_image_path=str(args.image), audio_path=str(args.audio),
                      max_area=AREA, infer_frames=BLOCK, sampling_steps=4, shift=pipe.config.sample_shift,
                      sample_solver="euler", guide_scale=0, seed=SEED, num_repeat=None, offload_model=False,
                      split_decoder=decoder)
    rows = decoder.rows
    frames = sum(r["frames"] for r in rows)
    if frames != target:
        raise ValueError(f"expected {target} presentation frames, got {frames}")
    elapsed = rows[-1]["available_s"]
    stats = dict(upstream=REV, layout="one FP8 DiT + one VAE, eager, single request", torch=torch.__version__,
                 audio_s=seconds, frames=frames, seed=SEED, steps=4, fps=frames/elapsed, audio_rtf=elapsed/seconds,
                 renderer_first_chunk_s=rows[0]["available_s"], last_chunk_s=elapsed,
                 after_first_chunk_fps=(frames-rows[0]["frames"])/(elapsed-rows[0]["available_s"]) if len(rows)>1 else None,
                 peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)], chunks=rows)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(out/"index.m3u8"), "-i", str(args.audio),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-t", str(seconds),
                    "-movflags", "+faststart", str(out/"preview.mp4")], check=True)
    atomic(out/"result.json", json.dumps(stats, indent=2))
    return stats


def serve(args, pipe, load_s):
    lock = threading.Lock()
    state = {"state":"ready", "load_s":round(load_s, 2)}

    def run(out):
        try:
            benchmark(args, pipe, out)
            status = {"state":"ready", "error":None}
        except Exception as exc:
            logging.exception("benchmark failed")
            status = {"state":"failed", "error":str(exc)}
        with lock:
            state.update(status)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, mime="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                with lock:
                    body = json.dumps(state).encode()
                return self.send(200, body)
            match = re.fullmatch(r"/runs/([0-9a-f]{32})/(result\.json|index\.m3u8|preview\.mp4|[0-9]+\.ts)", self.path)
            if match:
                path = args.out.joinpath(*match.groups())
                if path.is_file():
                    mime = {".json":"application/json", ".mp4":"video/mp4", ".ts":"video/mp2t",
                            ".m3u8":"application/vnd.apple.mpegurl"}[path.suffix]
                    return self.send(200, path.read_bytes(), mime)
            self.send_error(404)

        def do_POST(self):
            if self.path != "/benchmark":
                return self.send_error(404)
            # Browser form/cross-origin POSTs must not allocate GPU work on the local host.
            if self.headers.get("X-LiveAvatar-Benchmark") != "1" or self.headers.get("Origin"):
                return self.send_error(403)
            with lock:
                if state["state"] == "running":
                    return self.send_error(409)
                run_id = uuid.uuid4().hex
                out = args.out / run_id
                out.mkdir()
                state.update(state="running", run=run_id, error=None)
                threading.Thread(target=run, args=(out,), daemon=True).start()
            self.send(202, json.dumps({"run":run_id}).encode())

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("upstream", "weights", "image", "audio", "out"):
        ap.add_argument("--"+key, type=lambda x: Path(x).resolve(), required=True)
    ap.add_argument("--port", type=int, default=18160)
    args = ap.parse_args()
    if subprocess.check_output(["git","-C",args.upstream,"rev-parse","HEAD"],text=True).strip() != REV:
        raise SystemExit("unexpected upstream revision")
    if torch.cuda.device_count() != 2:
        raise SystemExit("select exactly two GPUs using CUDA_VISIBLE_DEVICES")
    if not args.image.is_file() or not args.audio.is_file():
        raise SystemExit("missing reference image or reply WAV")
    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(args.upstream)
    sys.path.insert(0, str(args.upstream))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        dist.init_process_group("gloo", init_method=Path(tmp, "rendezvous").as_uri(), rank=0, world_size=1)
        try:
            t0 = time.monotonic()
            pipe = load(args)
            serve(args, pipe, time.monotonic()-t0)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
