"""Make body clips of an avatar's photo with an image-to-video model, and measure them.

    .venv-flashhead/bin/python -m photostage.clipgen <avatar key or folder> --model wan-i2v-a14b --gesture explain emphasis

Each clip lands in data/clips/<avatar>/<model>-<gesture>-s<seed>/: clip.mp4, a contact sheet, the picture the model was given
(first.png) and info.json (the settings, the time it took and the numbers of `clips.measure`). `--dry-run` only plans the
region and writes first.png; it needs neither a GPU nor a model. Choose the GPU with CUDA_VISIBLE_DEVICES. A 14B model visits
the GPU one 28 GB expert at a time, which fits beside the PhotoStage server, but a clip keeps the GPU busy for many minutes
and the live renderer needs it to stay real-time: generate when nobody is talking to an avatar. --tile-vae lowers the
decoder's peak memory.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from photostage import clips

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "data" / "photostage"
OUTPUT = ROOT / "data" / "clips"


@dataclass(frozen=True)
class Model:
    path: str  # under the repository root
    stride: int  # the size granularity of its pictures
    area: int  # the pixels it is given
    frames: int
    fps: int
    steps: int
    guidance: float
    last_frame: bool  # whether it can be asked to come back to the first frame, so that the clip loops
    offload: bool  # whether it is too big for the GPU along with its text encoder, and visits the GPU a part at a time


MODELS = {
    "wan-ti2v-5b": Model("models/wan22-ti2v-5b", 32, 870_000, 73, 24, 35, 5.0, False, False),
    "wan-i2v-a14b": Model("models/wan22-i2v-a14b", 16, 900_000, 81, 16, 40, 3.5, True, True),
    "wan-i2v-a14b-lightning": Model("models/wan22-i2v-a14b-lightning", 16, 900_000, 81, 16, 4, 1.0, True, True),
}


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("avatar", help="the avatar's key (its folder in data/photostage) or a folder with photo.png and rig.json")
    parser.add_argument("--model", choices=MODELS, default="wan-ti2v-5b")
    parser.add_argument("--model-dir", type=Path, help="where the model is, if not where the registry says")
    parser.add_argument("--gesture", nargs="+", choices=clips.GESTURES, default=list(clips.GESTURES))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--frames", type=int, help="a multiple of 4, plus 1")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--area", type=int, help="pixels given to the model")
    parser.add_argument("--above-chin", type=float, default=clips.ABOVE_CHIN, help="face heights of the face shown above the chin")
    parser.add_argument("--framing", choices=("body", "avatar"), default="body",
                        help="body crop, or the whole avatar video for tracked facial reenactment")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="ask the model to come back to the first frame, if it can")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, help="CPU threads, for --device cpu")
    parser.add_argument("--tile-vae", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    model = MODELS[args.model]
    args.frames, args.steps = args.frames or model.frames, args.steps or model.steps
    if (args.frames - 1) % 4:
        parser.error("--frames must be a multiple of 4, plus 1")
    return args


def avatar_folder(name):
    folder = Path(name) if Path(name).is_dir() else LIBRARY / name
    if not (folder / "photo.png").is_file() or not (folder / "rig.json").is_file():
        raise SystemExit(f"No avatar with a photo and a rig at {folder}.")
    return folder


def load(model, path, device, tile):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    # The video decoder stays in float32: bfloat16 leaves visible banding in smooth skin and cloth.
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(path, vae=vae, torch_dtype=torch.bfloat16)
    if tile:
        pipe.vae.enable_tiling()
    if device == "cpu":
        pipe.to("cpu")
    elif model.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def generate(pipe, model, image, prompt, seed, frames, steps, loop):
    """The clip as an (n, h, w, 3) uint8 RGB array."""
    import torch
    from PIL import Image

    started = time.time()

    def report(pipeline, index, timestep, tensors):
        print(f"  step {index + 1}/{steps}  {time.time() - started:.0f} s", flush=True)
        return tensors

    first = Image.fromarray(image)
    extra = {"last_image": first} if loop and model.last_frame else {}
    video = pipe(
        image=first, prompt=prompt, negative_prompt=clips.NEGATIVE, height=image.shape[0], width=image.shape[1],
        num_frames=frames, num_inference_steps=steps, guidance_scale=model.guidance,
        generator=torch.Generator("cpu").manual_seed(seed), callback_on_step_end=report, output_type="np", **extra,
    ).frames[0]
    return (video.clip(0, 1) * 255 + 0.5).astype(np.uint8)


def save(folder, image, frames, fps, info):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "first.png"), image[:, :, ::-1])
    clips.write_video(folder / "clip.mp4", frames, fps)
    cv2.imwrite(str(folder / "sheet.jpg"), clips.contact_sheet(frames)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    (folder / "info.json").write_text(json.dumps(info, indent=1))


def main(argv=None):
    args = parse(argv)
    model = MODELS[args.model]
    folder = avatar_folder(args.avatar)
    rig = json.loads((folder / "rig.json").read_text())
    photo = cv2.cvtColor(cv2.imread(str(folder / "photo.png")), cv2.COLOR_BGR2RGB)
    if list(photo.shape[1::-1]) != list(rig["size"]):
        raise SystemExit(f"The photo is {photo.shape[1]}x{photo.shape[0]} but the rig was made for {rig['size'][0]}x{rig['size'][1]}.")
    try:
        plan = (clips.plan_avatar(rig["size"], rig["face"], model.stride, args.area or model.area)
                if args.framing == "avatar" else
                clips.plan_region(rig["size"], rig["face"], model.stride, args.area or model.area, args.above_chin))
    except clips.Unsuitable as error:
        raise SystemExit(str(error))
    image = clips.crop(photo, plan)
    rows = clips.band_rows(plan, rig["face"])
    face_h = (rig["face"][3] - rig["face"][1]) * clips.work_scale(plan)
    geometry = {**plan, "band": list(rows), "face_h": round(face_h, 1), "framing": args.framing}
    if args.framing == "avatar":
        geometry["track_box"] = clips.face_box(plan, rig["face"])
    print(json.dumps(geometry))
    if args.dry_run:
        target = args.out / folder.name / f"{args.model}-plan"
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target / "first.png"), image[:, :, ::-1])
        (target / "plan.json").write_text(json.dumps(geometry, indent=1))
        print(f"The model would be given {target / 'first.png'}")
        return 0

    import diffusers
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    path = args.model_dir or ROOT / model.path
    if not path.is_dir():
        raise SystemExit(f"The model is not at {path}.")
    pipe = load(model, path, args.device, args.tile_vae)
    for gesture in args.gesture:
        print(f"{args.model} / {gesture}", flush=True)
        started = time.time()
        prompt = clips.avatar_prompt(gesture) if args.framing == "avatar" else clips.GESTURES[gesture]
        frames = generate(pipe, model, image, prompt, args.seed, args.frames, args.steps, args.loop)
        seconds = round(time.time() - started, 1)
        numbers = clips.measure(frames, image, rows, face_h)
        info = {
            "avatar": folder.name, "model": args.model, "gesture": gesture, "prompt": prompt,
            "negative": clips.NEGATIVE, "seed": args.seed, "frames": args.frames, "fps": model.fps, "steps": args.steps,
            "guidance": model.guidance, "loop": bool(args.loop and model.last_frame), **geometry, "seconds": seconds,
            "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__}, "measure": numbers,
        }
        target = args.out / folder.name / f"{args.model}-{gesture}-s{args.seed}"
        save(target, image, frames, model.fps, info)
        print(f"  {seconds} s -> {target}\n  {json.dumps(numbers)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
