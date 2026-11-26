"""Push every image in a folder through the live head path (register -> speech -> GPU composite) and report speed + quality.

One HeadEngine is loaded once. Per image: register the portrait, warm up, render a speech clip while timing the pipeline
stages, score lip-sync from the rendered mouth, and write clip.mp4 / contact.jpg / face_strip.jpg under <out>/<image>/.

  python tools/test_images.py --images /images --audio /data/audio.wav --out /work/test_images \
      --engine-dir /engines/sm90_trt10.11.0.33 --max-seconds 6
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import time
import traceback

import cv2
import librosa
import numpy as np
import torch

from run_head_clip import lip_sync_score, mouth_opening, to_u8
from app.avatar import Loop
from app.compositor import PatchReader, blend_roi, paste
from app.head_engine import HUBERT_STEP, HeadEngine
from app.speech_pipeline import SpeechPipeline
from core.atomic_components.avatar_registrar import AvatarRegistrar

FPS = 25


def write_clip(frames, audio_path, out_path):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-i", audio_path, "-map", "0:v", "-map", "1:a", "-t", f"{len(frames) / FPS:.3f}",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    return p.wait()


def square_crop(img, cx, cy, half):
    """Square window centred on (cx, cy); replicated border where it leaves the image."""
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    pad = max(0, -x0, -y0, x0 + side - img.shape[1], y0 + side - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return img[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]


def label(tile, text):
    tile = np.ascontiguousarray(tile)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return tile


def save_sheets(out_dir, loop, frames, mouth, size=300):
    roi = loop.rois[0]
    cx, cy = (roi.x0 + roi.x1) / 2, (roi.y0 + roi.y1) / 2
    half = max(roi.x1 - roi.x0, roi.y1 - roi.y0) / 2
    crop = lambda im: cv2.resize(square_crop(im, cx, cy, half), (size, size), interpolation=cv2.INTER_AREA)

    m = np.asarray(mouth, np.float64)
    ok = np.where(~np.isnan(m))[0]
    if len(ok) >= 6:  # spread across the mouth's range of motion, closed ... widest open
        order = ok[np.argsort(m[ok])]
        pick = [int(order[int(q * (len(order) - 1))]) for q in (0.0, 0.3, 0.55, 0.75, 0.9, 1.0)]
    else:
        pick = np.linspace(0, len(frames) - 1, 6).astype(int).tolist()
    tiles = [label(crop(loop.frames[0]), "original")]
    tiles += [label(crop(frames[i]), f"f{i} mouth {m[i]:.2f}" if not np.isnan(m[i]) else f"f{i}") for i in pick]
    cv2.imwrite(os.path.join(out_dir, "face_strip.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    idx = np.linspace(len(frames) * 0.15, len(frames) - 1, 4).astype(int)
    full = [loop.frames[0]] + [frames[i] for i in idx]
    scale = min(420 / full[0].shape[1], 640 / full[0].shape[0])
    full = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in full]
    cv2.imwrite(os.path.join(out_dir, "contact.jpg"), cv2.cvtColor(np.concatenate(full, 1), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def speak(engine, loop, audio, reader, keep):
    """Stream `audio` through the head engine in 200 ms slices (a TTS faster than real time) and composite every frame."""
    engine.timer.reset()
    engine.begin(loop.info)
    renders, frames = [], []
    first, overhead = None, 0.0
    t_start = time.perf_counter()
    for off in range(0, len(audio), HUBERT_STEP):
        final = off + HUBERT_STEP >= len(audio)
        for idx, x_d in engine.feed_audio(audio[off:off + HUBERT_STEP], final=final):
            t1 = time.perf_counter()
            render, fidx = engine.render_gpu(idx, x_d)
            roi = loop.rois[fidx]
            out = paste(loop.frames[fidx], reader.read(blend_roi(render, roi)), roi)
            engine.timer.add("render+blend+d2h+paste", time.perf_counter() - t1)
            if first is None:
                first = (time.perf_counter() - t_start, (off + HUBERT_STEP) / 16000)
            if keep:
                t2 = time.perf_counter()
                renders.append(to_u8(render))
                frames.append(out)
                overhead += time.perf_counter() - t2  # keeping frames for scoring is not pipeline cost
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start - overhead
    return renders, frames, wall, first


def speak_pipelined(pipe, loop, audio, keep):
    """Same job through SpeechPipeline: motion (HuBERT+LMDM) and render run concurrently on their own CUDA streams."""
    engine = pipe.engine
    engine.timer.reset()
    pipe.keep_render = keep
    t_start = time.perf_counter()
    pipe.start(loop)
    for off in range(0, len(audio), HUBERT_STEP):
        pipe.push(audio[off:off + HUBERT_STEP])
    pipe.finish()
    renders, frames, first, t_last = [], [], None, t_start
    while True:
        f = pipe.frames.get()
        if f is None:
            break
        first = first or f.t_ready - t_start
        t_last = f.t_ready
        if keep:
            renders.append(f.render)
            frames.append(f.rgb)
    if pipe.error:
        raise pipe.error
    pipe.join()
    overhead = sum(engine.timer.t.get("keep_render", [])) / 1e3  # scoring-only work, not pipeline cost
    return renders, frames, t_last - t_start - overhead, (first, None)


def process(i, path, args, engine, registrar, kw, audio, reader, pipe):
    stem = os.path.splitext(os.path.basename(path))[0]
    name = f"{i:02d}_{stem[:28]}"
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    res = {"image": os.path.basename(path), "dir": name}
    try:
        res["src_size"] = list(cv2.imread(path, cv2.IMREAD_UNCHANGED).shape[1::-1])
    except Exception:
        pass
    t0 = time.perf_counter()
    try:
        loop = Loop.register(name, path, registrar, engine, kw)
    except Exception as e:
        res["error"] = f"registration failed: {type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc().strip().splitlines()[-4:]
        return res
    res["register_s"] = round(time.perf_counter() - t0, 2)
    roi = loop.rois[0]
    res["frame_size"] = list(loop.size)
    res["roi_px"] = [roi.x1 - roi.x0, roi.y1 - roi.y0]
    res["render_px_per_frame_px"] = round(512 / max(roi.x1 - roi.x0, 1), 2)  # > 1: the 512 render is downscaled into the frame

    warm, run = audio[: int(2.0 * 16000)], audio
    if pipe is not None:
        speak_pipelined(pipe, loop, warm, keep=False)  # warm-up: new ROI shapes, allocator, TRT contexts
        renders, frames, wall, first = speak_pipelined(pipe, loop, run, keep=True)
    else:
        speak(engine, loop, warm, reader, keep=False)
        renders, frames, wall, first = speak(engine, loop, run, reader, keep=True)

    n = len(frames)
    seconds = len(audio) / 16000
    res.update(frames=n, audio_s=round(seconds, 2), wall_s=round(wall, 2), rtf=round(wall / seconds, 2),
               ms_per_frame=round(wall / max(n, 1) * 1e3, 1), fps_if_compute_bound=round(n / wall, 1),
               mode="pipelined" if pipe is not None else "serial",
               audio_needed_before_first_frame_s=first[1], compute_to_first_frame_ms=round(first[0] * 1e3, 0),
               stages={k: v["mean_ms"] for k, v in engine.timer.summary().items()})

    lm = registrar.source2info.landmark478
    mouth = [mouth_opening(lm, r) for r in renders]
    res["lip_sync"] = lip_sync_score(mouth, audio)

    if not args.no_clips:
        save_sheets(out_dir, loop, frames, mouth)
        write_clip(frames, args.audio, os.path.join(out_dir, "clip.mp4"))
        np.save(os.path.join(out_dir, "mouth.npy"), np.asarray(mouth))
    del loop, frames, renders
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with portraits / body photos")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--cfg", default="/models/ditto-talkinghead/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--data-root", default="/models/ditto-talkinghead/ditto_pytorch")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=70)
    ap.add_argument("--lmdm-precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--only", default="", help="comma separated image indices to run")
    ap.add_argument("--serial", action="store_true", help="single-threaded motion->render loop instead of SpeechPipeline")
    ap.add_argument("--no-clips", action="store_true", help="skip mp4/jpg output (timing runs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(p for ext in ("png", "jpg", "jpeg", "webp") for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
    only = {int(x) for x in args.only.split(",") if x.strip()}
    replace = {"default_kwargs": {"online_mode": True, "overlap_v2": args.overlap, "sampling_timesteps": args.steps}}
    engine = HeadEngine(args.cfg, args.data_root, replace_cfg=replace, engine_dir=args.engine_dir, lmdm_precision=args.lmdm_precision)
    registrar = AvatarRegistrar(**engine.registrar_cfg)
    audio, _ = librosa.load(args.audio, sr=16000)
    audio = audio[: int(args.max_seconds * 16000)]
    reader = PatchReader()
    pipe = None if args.serial else SpeechPipeline(engine, reader)

    results = []
    for i, path in enumerate(paths):
        if only and i not in only:
            continue
        print(f"\n=== [{i}] {os.path.basename(path)}", flush=True)
        res = process(i, path, args, engine, registrar, engine.kw, audio, reader, pipe)
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "stages"}, indent=1), flush=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "image".ljust(34) + "frame".ljust(11) + "roi".ljust(9) + "ms/f".ljust(7) + "rtf".ljust(6) + "corr".ljust(7) + "sp/sil".ljust(8) + "mouth range")
    for r in results:
        if "error" in r:
            print(r["image"][:32].ljust(34) + "ERROR " + r["error"])
            continue
        ls = r["lip_sync"]
        print(r["image"][:32].ljust(34) + f"{r['frame_size'][0]}x{r['frame_size'][1]}".ljust(11) + f"{r['roi_px'][0]}".ljust(9)
              + f"{r['ms_per_frame']}".ljust(7) + f"{r['rtf']}".ljust(6)
              + (f"{ls['linear']['corr_lag0']:.2f}" if "linear" in ls else "n/a").ljust(7)
              + (f"{ls['speech_to_silence_ratio']:.2f}" if "speech_to_silence_ratio" in ls else "n/a").ljust(8)
              + (f"{ls['mouth_range'][0]:.3f}-{ls['mouth_range'][1]:.3f}" if "mouth_range" in ls else f"valid={ls['valid_frames']}"))


if __name__ == "__main__":
    main()
