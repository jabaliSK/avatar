"""Take Live Avatar's gestures back to the photo's pose with a short Wan 2.2 first-and-last-frame bridge.

    unshare --mount --propagation private sh -c 'mount --bind /dev/null /dev/nvidia1 &&
      CUDA_VISIBLE_DEVICES=0 exec .venv-flashhead/bin/python -m photostage.clipbridge <avatar> --clips data/clips/<avatar>'

Live Avatar (photostage.clipgen_liveavatar) is driven by speech alone and cannot be told where to end, so its gestures
settle in a resting pose of its own instead of the photo's, and clippack would leave them out (the player could only cut
from one pose to the other). For each of its talking runs this ends the gesture once the hands are still
(clipgen_liveavatar.calm), asks Wan 2.2 I2V A14B Lightning for a few frames from that frame to the clip's first one, and
appends them, so that the clip ends where it began. A quiet clip (idle, listening, thinking) that drifts away from the
photo's pose is bridged from its last frame when it is named with --gesture. The run keeps the clip as Live Avatar made it
in gesture.mp4, and a later run bridges that again. Wan holds the GPU one 28 GB expert at a time for a few minutes per clip: run it when nobody
is talking to an avatar. --dry-run reports the cuts and needs neither a GPU nor a model.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from photostage import clipgen, clipgen_liveavatar as liveavatar, clips

WAN = "wan-i2v-a14b-lightning"
FRAMES = 33  # Wan's frames, played at the gesture's 25 fps: 1.3 s back to the photo's pose
JOIN = 6  # frames over which Wan's first frame is made to meet the gesture's last
CLOSE = 8  # frames over which its last frame is made the clip's first
MAX_CLOSE = 6.0  # most the last frame may differ from the first (levels) to be closed by fading, not left to the gates
PROMPT = (  # no word of speech: Wan makes a person who "finishes speaking" talk on
    "A calm, silent moment. The person is not talking: the lips stay gently closed in a relaxed smile the whole time. "
    "The hands move slowly and smoothly, straight back to their resting positions{rest}, and then stay still. The head, "
    "neck and shoulders move together naturally. Locked-off camera; the framing, clothing, identity, background and "
    "lighting never change. It ends in exactly the original resting pose."
)


def ramp(count):
    """Weights from 1 down towards 0 over `count` frames, smoothly: how much of a correction each frame takes."""
    t = np.arange(count, dtype=np.float32) / count
    return 1 - t * t * (3 - 2 * t)


def join(gesture, bridge, max_close=MAX_CLOSE):
    """The gesture followed by the bridge, which starts on the gesture's last frame and should end on its first. The few
    levels by which the bridge misses either are spread over its first JOIN and last CLOSE frames; an end that misses by
    more than `max_close` is left as it is. Returns the frames, and how far the bridge missed at each end (levels)."""
    bridge = bridge.astype(np.float32)
    seam = gesture[-1].astype(np.float32) - bridge[0]
    count = min(JOIN, len(bridge))
    bridge[:count] += ramp(count)[:, None, None, None] * seam
    miss = gesture[0].astype(np.float32) - bridge[-1]
    gap = float(np.abs(miss).mean())
    if gap <= max_close:
        count = min(CLOSE, len(bridge) - 1)
        bridge[-count:] += ramp(count)[::-1, None, None, None] * miss
    tail = np.clip(bridge[1:] + 0.5, 0, 255).astype(np.uint8)
    return np.concatenate([gesture, tail]), {"seam": round(float(np.abs(seam).mean()), 3), "end_gap": round(gap, 3),
                                              "closed": gap <= max_close}


def quiet(info):
    return clips.ROLES[info["gesture"]][0] in clips.LOOP_ROLES


def bridged_runs(folder, gestures):
    """The Live Avatar runs in `folder` to bridge, by name: those of `gestures` if any are named, else the talking ones."""
    found = {}
    for info_path in sorted(folder.glob("*/info.json")):
        info = json.loads(info_path.read_text())
        if info.get("model") != liveavatar.MODEL or info.get("gesture") not in clips.ROLES:
            continue
        if (info["gesture"] not in gestures) if gestures else quiet(info):
            continue
        found[info_path.parent.name] = info_path.parent
    return found


def cut(frames, info):
    """Where the clip ends: a gesture once the hands are still after its speech, a quiet clip where it was cut."""
    if quiet(info):
        return len(frames) - 1
    return liveavatar.calm(frames, liveavatar.speech_end(info["speech_seconds"], info.get("lead_in", 0.0)))


def prompt(rest=None):
    """What Wan is asked for; `rest` describes the photo's resting pose."""
    return PROMPT.format(rest=f" ({rest.strip().rstrip('.')})" if rest else "")


def generate(pipe, start, end, text, seed, frames, steps):
    """Wan's frames from `start` to `end`, as an (n, h, w, 3) uint8 RGB array."""
    import torch
    from PIL import Image

    video = pipe(
        image=Image.fromarray(start), last_image=Image.fromarray(end), prompt=text, negative_prompt=clips.NEGATIVE,
        height=start.shape[0], width=start.shape[1], num_frames=frames, num_inference_steps=steps,
        guidance_scale=clipgen.MODELS[WAN].guidance, generator=torch.Generator("cpu").manual_seed(seed), output_type="np",
    ).frames[0]
    return (video.clip(0, 1) * 255 + 0.5).astype(np.uint8)


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("avatar", help="the avatar's key (its folder in data/photostage) or a folder with photo.png and rig.json")
    parser.add_argument("--clips", type=Path, help="the folder of runs (default: data/clips/<avatar>)")
    parser.add_argument("--gesture", nargs="+", choices=list(clips.ROLES),
                        help="bridge these (default: every talking gesture); name a quiet clip to bridge it too")
    parser.add_argument("--rest", help="the photo's resting pose, e.g. 'the right hand held loosely in front of the chest'")
    parser.add_argument("--end", nargs="+", default=[], metavar="GESTURE=FRAME",
                        help="end a gesture at this frame instead (after watching it: Live Avatar can garble a still hand)")
    parser.add_argument("--frames", type=int, default=FRAMES, help="Wan's frames: a multiple of 4, plus 1")
    parser.add_argument("--steps", type=int, default=clipgen.MODELS[WAN].steps)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--tile-vae", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 5 or (args.frames - 1) % 4:
        parser.error("--frames must be a multiple of 4, plus 1")
    try:
        args.end = {name: int(frame) for name, frame in (item.split("=") for item in args.end)}
    except ValueError:
        parser.error("--end takes GESTURE=FRAME")
    return args


def main(argv=None):
    args = parse(argv)
    folder = clipgen.avatar_folder(args.avatar)
    runs = bridged_runs((args.clips or clipgen.OUTPUT / folder.name).resolve(), args.gesture)
    if not runs:
        raise SystemExit("No Live Avatar clips to bridge there.")
    pipe = None
    for name, run in runs.items():
        info = json.loads((run / "info.json").read_text())
        source = run / "gesture.mp4" if (run / "gesture.mp4").is_file() else run / "clip.mp4"
        gesture, fps = clips.read_video(source)
        chosen = info["gesture"] in args.end
        end = min(args.end[info["gesture"]], len(gesture) - 1) if chosen else cut(gesture, info)
        gesture = gesture[:end + 1]
        print(f"{name}: {'chosen end' if chosen else 'it ends' if quiet(info) else 'the hands are still'} at frame {end}; {len(gesture) / fps:.2f} s, "
              f"then {args.frames} bridging frames", flush=True)
        if args.dry_run:
            continue
        if pipe is None:
            path = clipgen.ROOT / clipgen.MODELS[WAN].path
            if not path.is_dir():
                raise SystemExit(f"Wan is not at {path}.")
            pipe = clipgen.load(clipgen.MODELS[WAN], path, "cuda", args.tile_vae)
        started = time.time()
        text = prompt(args.rest)
        bridge = generate(pipe, gesture[-1], gesture[0], text, args.seed, args.frames, args.steps)
        frames, numbers = join(gesture, bridge)
        if source.name == "clip.mp4":
            shutil.copy2(source, run / "gesture.mp4")
        image = cv2.cvtColor(cv2.imread(str(run / "first.png")), cv2.COLOR_BGR2RGB)
        import diffusers

        info.update({
            "frames": len(frames),
            "bridge": {"model": WAN, "frames": args.frames, "steps": args.steps, "seed": args.seed, "prompt": text,
                       "gesture_frames": len(gesture), "end_chosen": chosen, "seconds": round(time.time() - started, 1),
                       "diffusers": diffusers.__version__, **numbers},
            "measure": clips.measure(frames, image, tuple(info["band"]), info["face_h"]),
        })
        clipgen.save(run, image, frames, fps, info)
        print(f"  {info['bridge']['seconds']} s, {len(frames)} frames, {json.dumps(numbers)}\n  {json.dumps(info['measure'])}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
