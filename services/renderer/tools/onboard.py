"""Register avatars once so the live service only has to load them.

    python tools/onboard.py --image portrait.png --name aisha                       # one image
    python tools/onboard.py --images-dir /images                                   # every image in a folder (name = file stem)
    python tools/onboard.py --image portrait.png --name aisha --loop idle=idle.mp4 --loop speaking=gestures.mp4   # Stage A clips
    python tools/onboard.py --name aisha --add-loop idle=idle.mp4                   # one more clip on an onboarded avatar

Needs the TensorRT engines (built on first use) because registration renders every frame once from the neutral face: to derive
the colour residual that makes the live head match the body, and to place the head on each frame's real face (app/calibrate.py).
Output goes to <avatars>/<name>/ (see app/avatar_store.py).
With --loop the clips' own size becomes the avatar's frame size (all clips must share it) and the photo is fitted to it, so
the photo and the Stage A loops line up pixel for pixel.
Pixels of a loop that do not move (the clip stays within a few grey levels of its first frame) are replaced by the photo's
own pixels, which removes the generator's shimmer and colour cast from the background and clothes; --no-pin turns that off.
"""
import argparse
import glob
import logging
import os
import re
import time

import torch  # noqa: F401  (loads CUDA libs before onnxruntime / tensorrt)

from app.avatar_store import LOOP_STATES, add_state_loop, onboard_image, video_size
from app.engine_cache import ensure_engines
from app.head_engine import HeadEngine
from core.atomic_components.avatar_registrar import AvatarRegistrar


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "avatar"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--images-dir")
    ap.add_argument("--name")
    ap.add_argument("--loop", action="append", default=[], metavar="STATE=VIDEO", help="Stage A body loop; STATE in " + ",".join(LOOP_STATES))
    ap.add_argument("--add-loop", action="append", default=[], metavar="STATE=VIDEO",
                    help="register a Stage A clip on an avatar that is already onboarded (--name); the clip must have its frame size")
    ap.add_argument("--no-pin", action="store_true", help="keep the generated pixels everywhere instead of the photo's where the clip does not move")
    ap.add_argument("--avatars", default=os.environ.get("KKA_AVATARS", "/avatars"))
    ap.add_argument("--max-size", type=int, default=int(os.environ.get("KKA_MAX_SIZE", "1280")))
    ap.add_argument("--engine-root", default=os.environ.get("KKA_ENGINE_ROOT", "/engines"))
    ap.add_argument("--onnx-dir", default=os.environ.get("KKA_ONNX_DIR", "/models/ditto-talkinghead/ditto_onnx"))
    ap.add_argument("--cfg", default="/models/ditto-talkinghead/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--data-root", default="/models/ditto-talkinghead/ditto_pytorch")
    # Registration renders the live head (colour match, placement on the real face), so the engine runs as the server runs it.
    ap.add_argument("--overlap", type=int, default=int(os.environ.get("KKA_OVERLAP", "70")), help="LMDM overlap (the server's KKA_OVERLAP)")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("KKA_STEPS", "10")), help="LMDM sampling steps (the server's KKA_STEPS)")
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    jobs = []
    if args.image:
        jobs.append((args.name or slug(os.path.splitext(os.path.basename(args.image))[0]), args.image))
    if args.images_dir:
        for p in sorted(sum((glob.glob(os.path.join(args.images_dir, f"*.{e}")) for e in ("png", "jpg", "jpeg", "webp")), [])):
            jobs.append((slug(os.path.splitext(os.path.basename(p))[0]), p))
    loops = dict(item.split("=", 1) for item in args.loop)
    add_loops = dict(item.split("=", 1) for item in args.add_loop)
    if add_loops and (jobs or loops or not args.name):
        ap.error("--add-loop goes with --name only (no --image, --images-dir or --loop)")
    if not jobs and not add_loops:
        ap.error("give --image, --images-dir or --name with --add-loop")
    if loops and len(jobs) != 1:
        ap.error("--loop applies to exactly one --image/--name")

    engine_dir = ensure_engines(args.engine_root, args.onnx_dir)
    engine = HeadEngine(args.cfg, args.data_root,
                        replace_cfg={"default_kwargs": {"online_mode": True, "overlap_v2": args.overlap, "sampling_timesteps": args.steps}},
                        engine_dir=engine_dir, lmdm_precision="fp16")
    registrar = AvatarRegistrar(**engine.registrar_cfg)
    kw = {**engine.kw, "max_size": args.max_size}
    if add_loops:
        for state, video in add_loops.items():
            t0 = time.perf_counter()
            add_state_loop(args.avatars, args.name, state, video, registrar, engine, kw, pin=not args.no_pin)
            print(f"[ok] {args.name}: {state} loop from {video} ({time.perf_counter() - t0:.1f}s)", flush=True)
        return
    size = None
    if loops:  # the Stage A clips decide the frame geometry; the photo is fitted to it and no state may differ
        sizes = {state: video_size(video) for state, video in loops.items()}
        if len(set(sizes.values())) != 1:
            ap.error(f"all --loop clips must have one size, got {sizes}")
        size = next(iter(sizes.values()))
        kw["max_size"] = max(size)

    failed = []
    for name, path in jobs:
        t0 = time.perf_counter()
        try:
            onboard_image(args.avatars, name, path, registrar, engine, kw, size=size)
            for state, video in loops.items():
                add_state_loop(args.avatars, name, state, video, registrar, engine, kw, pin=not args.no_pin)
            print(f"[ok] {name}: {path} ({time.perf_counter() - t0:.1f}s)", flush=True)
        except Exception as e:  # keep going: one unusable photo must not stop the rest
            failed.append(name)
            print(f"[fail] {name}: {path}: {type(e).__name__}: {e}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(jobs)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
