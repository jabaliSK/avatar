#!/usr/bin/env python3
"""Streaming per-chunk latency benchmark for SoulX-FlashHead (Pro) on the pinned GPU.

Mirrors the upstream `generate_video.py --audio_encode_mode stream` loop. Each chunk yields
`frame_num - motion_frames_num` new frames, i.e. a chunk is real-time only if it is produced in
less than slice_len / tgt_fps seconds. Upstream reads flash_head/configs/infer_params.yaml
relative to the CWD, so this script chdirs into the upstream checkout.

Pin the device from the caller: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id>.
"""
import argparse
import collections
import contextlib
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FH = ROOT / "third_party" / "SoulX-FlashHead"
STAGE_RE = re.compile(r"\[generate\] (.+?): ([0-9.eE+-]+)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", type=int, default=1, choices=[0, 1], help="torch.compile DiT + VAE")
    ap.add_argument("--chunks", type=int, default=6, help="measured chunks (after warmup)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--image", default=str(FH / "examples" / "girl.png"))
    ap.add_argument("--audio", default=str(FH / "examples" / "podcast_sichuan_16k.wav"))
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "SoulX-FlashHead-1_3B"))
    ap.add_argument("--wav2vec", default=str(ROOT / "models" / "wav2vec2-base-960h"))
    ap.add_argument("--cached-audio-s", type=float, default=None, help="override wav2vec context window (s)")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    os.chdir(FH)
    sys.path.insert(0, str(FH))
    import imageio
    import librosa
    import numpy as np
    import torch

    import flash_head.src.pipeline.flash_head_pipeline as fhp

    fhp.COMPILE_MODEL = fhp.COMPILE_VAE = bool(a.compile)
    from flash_head.inference import (get_audio_embedding, get_base_data, get_infer_params,
                                      get_pipeline, run_pipeline)

    print(f"device: {torch.cuda.get_device_name(0)} | visible devices: {torch.cuda.device_count()} | compile={a.compile}", flush=True)

    t0 = time.perf_counter()
    pipe = get_pipeline(world_size=1, ckpt_dir=a.ckpt, model_type="pro", wav2vec_dir=a.wav2vec)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    get_base_data(pipe, a.image, 42, False)
    torch.cuda.synchronize()
    prep_s = time.perf_counter() - t0

    p = get_infer_params()
    sr, fps = p["sample_rate"], p["tgt_fps"]
    frame_num, motion = p["frame_num"], p["motion_frames_num"]
    slice_len = frame_num - motion
    per_slice = slice_len * sr // fps
    window_s = a.cached_audio_s or p["cached_audio_duration"]
    cache = int(window_s * sr)
    end_idx = int(window_s * fps)
    start_idx = end_idx - frame_num
    budget_ms = slice_len / fps * 1000

    audio, _ = librosa.load(a.audio, sr=sr, mono=True)
    total = a.warmup + a.chunks
    audio = np.tile(audio, (total * per_slice) // len(audio) + 1)[: total * per_slice]
    slices = audio.reshape(-1, per_slice)

    ring = collections.deque([0.0] * cache, maxlen=cache)
    rows, last = [], None
    for i, s in enumerate(slices):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ring.extend(s.tolist())
        emb = get_audio_embedding(pipe, np.array(ring, dtype=np.float32), start_idx, end_idx)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            video = run_pipeline(pipe, emb)[motion:]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        last = video.to(torch.uint8).cpu().numpy()
        t3 = time.perf_counter()

        stages = collections.defaultdict(float)
        for k, v in STAGE_RE.findall(buf.getvalue()):
            stages[k] += float(v) * 1000
        rows.append({
            "chunk": i, "warmup": i < a.warmup, "frames": int(last.shape[0]),
            "audio_embed_ms": (t1 - t0) * 1e3, "generate_ms": (t2 - t1) * 1e3, "d2h_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3, **{f"st_{k}": v for k, v in stages.items()},
        })
        r = rows[-1]
        print(f"chunk {i:2d}{' (warmup)' if r['warmup'] else '         '} total={r['total_ms']:8.1f} ms  embed={r['audio_embed_ms']:7.1f}  "
              f"generate={r['generate_ms']:8.1f}  d2h={r['d2h_ms']:6.1f}  frames={r['frames']}", flush=True)

    steady = [r for r in rows if not r["warmup"]]
    med = statistics.median(r["total_ms"] for r in steady)
    p95 = sorted(r["total_ms"] for r in steady)[max(0, int(len(steady) * 0.95) - 1)]
    keys = sorted({k for r in steady for k in r if k.startswith("st_")})
    summary = {
        "tag": a.tag, "compile": a.compile, "device": torch.cuda.get_device_name(0),
        "load_s": load_s, "prepare_s": prep_s, "frames_per_chunk": slice_len, "chunk_audio_ms": budget_ms,
        "median_chunk_ms": med, "p95_chunk_ms": p95, "fps": slice_len / (med / 1000),
        "rtf": med / budget_ms, "realtime": med < budget_ms,
        "median_audio_embed_ms": statistics.median(r["audio_embed_ms"] for r in steady),
        "median_generate_ms": statistics.median(r["generate_ms"] for r in steady),
        "stage_median_ms": {k[3:]: statistics.median(r[k] for r in steady if k in r) for k in keys},
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "window_s": window_s,
    }
    print("\n" + json.dumps(summary, indent=2), flush=True)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"bench_flashhead_{a.tag}.json").write_text(json.dumps({"summary": summary, "chunks": rows}, indent=2))
    for n, idx in (("first", 0), ("mid", last.shape[0] // 2), ("last", last.shape[0] - 1)):
        imageio.imwrite(out / f"bench_flashhead_{a.tag}_{n}.png", last[idx])


if __name__ == "__main__":
    main()
