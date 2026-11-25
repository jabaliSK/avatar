"""Make an avatar's clips with Live Avatar (Alibaba-Quark/LiveAvatar): whole-avatar video in which the head, the hands and
the body move as one, made offline from the photo and a drive track: speech for the gestures, silence for the quiet clips.

    unshare --mount --propagation private sh -c 'mount --bind /dev/null /dev/nvidia1 &&
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 exec .venv-liveavatar/bin/python -m photostage.clipgen_liveavatar <avatar> \
        --gesture idle listening thinking greet explain --describe "A bearded man in a white thobe"'

Each clip lands where photostage.clipgen puts its clips (data/clips/<avatar>/liveavatar-<gesture>-s<seed>/), with the same
files plus the drive track (drive.wav), so `photostage.clippack <avatar> --model liveavatar` judges and installs them and
MuseTalk dubs the face inside them for every reply. A gesture is driven by LEAD_IN seconds of silence, a line of Piper speech
and then TAIL seconds of silence, in which the hands settle; it ends once they are still. Live Avatar cannot be asked to
end in the photo's pose, so run photostage.clipbridge on the gestures before packing them: it takes them back there. A
quiet clip is cut where it looks most like its first frame, so that it loops. Live Avatar generates at the size it was
trained for (about 704x384, 25 fps), and Real-ESRGAN doubles it.

It is far from real time on one GPU (on this host's GPU 0, about 95 s per second of clip) and holds about 70 GB, so
generate when nobody is talking to an avatar. Needs the venv with Live Avatar's requirements, spandrel and piper-tts, the
repository in external/LiveAvatar and the weights in models/liveavatar and models/upscale. --dry-run only plans the crop and
writes first.png and the drive tracks; it needs neither a GPU nor a model.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from photostage import clipgen, clips

ROOT = clipgen.ROOT
MODEL = "liveavatar"
REPO = ROOT / "external" / "LiveAvatar"
CHECKPOINT = ROOT / "models" / "liveavatar" / "Wan2.2-S2V-14B"
LORA = ROOT / "models" / "liveavatar" / "Live-Avatar" / "liveavatar.safetensors"
UPSCALER = ROOT / "models" / "upscale" / "RealESRGAN_x2plus.pth"
VOICE = ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"
AREA, STRIDE = 704 * 384, 64  # the pixels Live Avatar was trained at, and the granularity of its sizes
ASPECT_SLACK = 0.05  # how far (as a log ratio) the photo's shape may be cropped to fit a size Live Avatar takes
UPSCALE = 2  # RealESRGAN_x2plus
FPS, BLOCK, LEAD = 25, 48, 3  # it makes 48-frame pieces at 25 fps, and the first piece loses 3 frames
STEPS = 4
LEAD_IN = 0.5  # silence before a gesture's speech, so that it starts from the photo's pose rather than mid-gesture
TAIL = 1.2  # silence after a gesture's speech, in which the hands settle back
CALM_FRAMES, CALM_SLACK = 5, 0.15  # a gesture ends once its motion, over this many frames, is this near its stillest
QUIET_BLOCKS = 3  # pieces in a quiet clip: enough that it comes back close to where it began
QUIET_KEEP = 2.5  # a quiet clip keeps at least this many seconds; its end is looked for after them
MEMORY = 0.8  # of GPU 0 that Live Avatar may take, leaving room for the live PhotoStage worker

# Live Avatar is prompted with who is in the picture and what they do; the photo shows the rest.
SCENE = (" Static camera: the framing, the background and the lighting never change. Photorealistic, smooth and natural "
         "motion.")
QUIET = " The mouth stays closed and still. Natural eye blinks."
# A gesture has to end in the photo's pose, so {rest} can say what that is (see --rest).
TALK = (" The gestures are calm and unhurried and stay in front of the chest, never reaching toward the camera; then the "
        "hands return to the resting pose{rest} and stay there.")
PROMPTS = {
    "idle": "{who} waits quietly facing the camera with a gentle, relaxed smile and calm breathing. The head and the body "
            "stay almost still and the hands stay in their resting position, the fingers moving only very slightly." + QUIET,
    "listening": "{who} listens attentively to someone in front of the camera with a soft smile and gives one small, slow "
                 "nod. The hands stay in their resting position." + QUIET,
    "thinking": "{who} pauses thoughtfully before answering, glancing briefly aside and back to the camera, the head "
                "upright. The hands stay in their resting position{rest}, the fingers shifting a little." + QUIET,
    "greet": "{who} starts still in the resting pose, then warmly welcomes the viewer while speaking, one hand opening a "
             "little outward with the palm turned up in a friendly, inviting gesture." + TALK,
    "explain": "{who} starts still in the resting pose, then explains an idea warmly to the camera while speaking, both "
               "hands rising to chest height with open palms and moving in rhythm with the words." + TALK,
    "emphasis": "{who} starts still in the resting pose, then makes an important point to the camera while speaking, one "
                "hand lifting with the palm open and giving a few soft beats of emphasis." + TALK,
    "ask": "{who} starts still in the resting pose, then asks the viewer a question, one hand turning palm up and opening "
           "a little outward as if waiting for an answer." + TALK,
    "calm": "{who} starts still in the resting pose, then speaks calmly and kindly to the camera with small, relaxed "
            "movements of the hands." + TALK,
}
# What a gesture is driven with: its rhythm moves the hands and the head; MuseTalk re-voices the mouth for each reply.
LINES = {
    "greet": "Hello, and welcome! It's lovely to meet you.",
    "explain": "Let me explain how this works. First we look at what you need, and then we find the best way forward "
               "together.",
    "emphasis": "This really matters, and it makes a big difference.",
    "ask": "What would you like to know? Is there anything I can help you with today?",
    "calm": "Thank you. I understand, and I'm here to help.",
}


def prompt(gesture, who, rest=None):
    """What Live Avatar is asked for: `who` doing `gesture`, coming back to `rest` (the photo's pose, in words)."""
    return PROMPTS[gesture].format(who=who, rest=f", {rest}" if rest else "") + SCENE


def model_size(size):
    """The (width, height) Live Avatar is given for a photo of `size`: multiples of STRIDE, at most AREA pixels, the largest
    whose shape is within ASPECT_SLACK of the photo's (else the nearest in shape)."""
    aspect = size[0] / size[1]
    options = [(w, h) for w in range(STRIDE, 4096, STRIDE) for h in range(STRIDE, 4096, STRIDE) if w * h <= AREA]
    error = {wh: abs(math.log(wh[0] / wh[1] / aspect)) for wh in options}
    fits = [wh for wh in options if error[wh] <= ASPECT_SLACK]
    if not fits:
        return min(options, key=lambda wh: (error[wh], -wh[0] * wh[1]))
    return max(fits, key=lambda wh: (wh[0] * wh[1], -error[wh]))


def plan(size, face, upscale=UPSCALE):
    """The part of the photo Live Avatar is shown, its size, and the size the clip is kept at: {"region", "given", "work"}.
    The region is the largest one with the given size's proportions, as near the face's centre as the photo allows."""
    width, height = model_size(size)
    if face[2] <= face[0] or face[3] <= face[1]:
        raise clips.Unsuitable("The face must have a positive size.")
    if size[0] / size[1] > width / height:
        crop_w, crop_h = size[1] * width / height, size[1]
    else:
        crop_w, crop_h = size[0], size[0] * height / width
    left = min(max((face[0] + face[2]) / 2 - crop_w / 2, 0.0), size[0] - crop_w)
    top = min(max((face[1] + face[3]) / 2 - crop_h / 2, 0.0), size[1] - crop_h)
    region = [round(left), round(top), round(left) + round(crop_w), round(top) + round(crop_h)]
    return {"region": region, "given": [width, height], "work": [width * upscale, height * upscale]}


def drive(speech, rate, quiet_blocks=QUIET_BLOCKS):
    """The track Live Avatar is driven with, as float samples at `rate`: LEAD_IN seconds of silence, `speech`, then at
    least TAIL seconds of silence, made up to just short of a whole number of its pieces (it makes
    floor(seconds * FPS / BLOCK) + 1 of them); with no speech, `quiet_blocks` pieces of silence. Returns the track and
    the number of pieces."""
    lead = round(LEAD_IN * rate) if len(speech) else 0
    if len(speech):
        # room for the half frame the track stops short
        blocks = max(1, math.ceil((((lead + len(speech)) / rate + TAIL) * FPS + 0.5) / BLOCK - 1e-9))
    else:
        blocks = quiet_blocks
    track = np.zeros(int((blocks * BLOCK - 0.5) / FPS * rate), np.float32)
    track[lead:lead + len(speech)] = speech
    return track, blocks


def settle(frames, earliest):
    """The last frame a clip keeps: of those from `earliest` on, the one that differs least from the first, by the mean
    difference in levels that clippack's gap gate measures."""
    earliest = min(max(1, earliest), len(frames) - 1)
    first = frames[0].astype(np.float32)
    costs = [float(np.abs(frame.astype(np.float32) - first).mean()) for frame in frames[earliest:]]
    return earliest + int(np.argmin(costs))


def speech_end(speech_seconds, lead_in=LEAD_IN):
    """The frame of a gesture in which its speech ends."""
    return round((lead_in + speech_seconds) * FPS) - LEAD


def calm(frames, earliest, slack=CALM_SLACK):
    """The last frame a gesture keeps: the first from `earliest` on whose motion (levels per frame, over CALM_FRAMES) is
    within `slack` of the stillest there. Live Avatar's gestures do not come back to the photo's pose (clipbridge takes
    them there), so they end as soon as the hands are still: from the end of the speech (speech_end), as they may settle
    before it, and in the silence after it they may start a gesture of their own."""
    earliest = min(max(1, earliest), len(frames) - 1)
    low = np.stack([cv2.resize(clips.gray(frame), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                    for frame in frames]).astype(np.float32)
    step = np.abs(np.diff(low, axis=0)).mean((1, 2))
    step = np.concatenate([step[:1], step])  # the motion into each frame
    half = CALM_FRAMES // 2
    padded = np.pad(step, half, mode="edge")
    motion = np.convolve(padded, np.ones(CALM_FRAMES) / CALM_FRAMES, mode="valid")
    tail = motion[earliest:]
    return earliest + int(np.argmax(tail <= tail.min() + slack))


def speak(text):
    """`text` in the avatar's Piper voice, as (float samples, rate)."""
    from piper import PiperVoice

    with tempfile.NamedTemporaryFile(suffix=".wav") as file:
        with wave.open(file.name, "wb") as output:
            PiperVoice.load(str(VOICE)).synthesize_wav(text, output)
        with wave.open(file.name) as source:
            rate = source.getframerate()
            samples = np.frombuffer(source.readframes(source.getnframes()), "<i2")
    # Piper leaves a little silence at the ends; the gesture should start with the voice.
    loud = np.flatnonzero(np.abs(samples) > 300)
    samples = samples[max(0, loud[0] - rate // 50):loud[-1] + rate // 50] if len(loud) else samples
    return samples.astype(np.float32) / 32768, rate


def write_wav(path, samples, rate):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class LiveAvatar:
    """Live Avatar's single-GPU pipeline, made to run on this host's vGPU: see the comments."""

    def __init__(self, prompts, fp8=True, memory=MEMORY):
        import torch
        import torch.distributed as dist

        self.torch = torch
        torch.cuda.set_per_process_memory_fraction(memory, 0)
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))
        spec = importlib.util.spec_from_file_location("s2v_streaming_interact", REPO / "minimal_inference" /
                                                      "s2v_streaming_interact.py")
        s2v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s2v)
        argv, sys.argv = sys.argv, [
            "s2v", "--task", "s2v-14B", "--size", "704*384", "--training_config",
            str(REPO / "liveavatar" / "configs" / "s2v_causal_sft.yaml"), "--offload_model", "False",
            "--convert_model_dtype", "--infer_frames", str(BLOCK), "--load_lora", "--lora_path_dmd", str(LORA),
            "--sample_steps", str(STEPS), "--sample_guide_scale", "0", "--num_clip", "10000", "--num_gpus_dit", "1",
            "--sample_solver", "euler", "--single_gpu", "--ckpt_dir", str(CHECKPOINT)]
        try:
            self.args = s2v._parse_args()
        finally:
            sys.argv = argv
        settings = s2v.training_config_parser(self.args.training_config)

        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("MASTER_PORT", "29641"), RANK="0",
                          WORLD_SIZE="1", LOCAL_RANK="0")
        torch.cuda.set_device(0)
        # NCCL cannot start on this vGPU (no CUDA IPC); one process only needs CPU barriers.
        dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)
        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS
        import liveavatar.models.wan.causal_model_s2v as causal_model
        import liveavatar.models.wan.wan_2_2.modules.attention as attention
        import liveavatar.models.wan.wan_2_2.modules.model as wan_model

        if not (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # Without flash-attn, every flash_attention call left on one GPU (the text and audio cross-attention) has no
            # padding, so PyTorch's own kernels compute the same. Its cuDNN kernel returns NaN for the audio
            # cross-attention here, and the video comes out black.
            def unpadded(q, k, v, k_lens=None, window_size=(-1, -1), **_):
                assert window_size == (-1, -1)
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                return out.transpose(1, 2).contiguous().type_as(q)

            wan_model.flash_attention = causal_model.flash_attention = unpadded

        self.config = WAN_CONFIGS[self.args.task]
        # init_on_cpu=False: with it the VAE is left on the CPU after every piece but the last decode, when the model is
        # not offloaded, expects it on the GPU.
        self.model = WanS2V(config=self.config, checkpoint_dir=str(CHECKPOINT), device_id=0, rank=0, t5_fsdp=False,
                            dit_fsdp=False, use_sp=False, sp_size=1, t5_cpu=False, convert_model_dtype=True,
                            single_gpu=True, offload_kv_cache=False, init_on_cpu=False)
        self.model.noise_model = self.model.add_lora_to_model(
            self.model.noise_model, lora_rank=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
            lora_target_modules=settings["lora_target_modules"], init_lora_weights=settings["init_lora_weights"],
            pretrained_lora_path=str(LORA), load_lora_weight_only=False)
        if fp8:
            from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8

            replace_linear_with_scaled_fp8(self.model.noise_model, ignore_keys=[
                "text_embedding", "time_embedding", "time_projection", "head.head",
                "casual_audio_encoder.encoder.final_linear"])
        self.fp8 = fp8
        # Encode every prompt now and move the 11 GB text encoder off the GPU: the four KV caches of a piece (about
        # 46 GB) must fit beside the model without starving the live worker.
        encode, self.prompts = self.model.encode_prompt, {}

        def cached(input_prompt, n_prompt=None, offload_model=True):
            if (input_prompt, n_prompt) not in self.prompts:
                self.prompts[input_prompt, n_prompt] = encode(input_prompt, n_prompt, offload_model=True)
            return self.prompts[input_prompt, n_prompt]

        self.model.encode_prompt = cached
        for prompt in prompts:
            cached(prompt, "")  # as generate() asks for it
        torch.cuda.empty_cache()

    def generate(self, image_path, audio_path, prompt, seed):
        """The clip as (n, h, w, 3) uint8 RGB at the image's size."""
        torch, args = self.torch, self.args
        video, _ = self.model.generate(
            input_prompt=prompt, ref_image_path=str(image_path), audio_path=str(audio_path), enable_tts=False,
            tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, num_repeat=args.num_clip, pose_video=None,
            generate_size=args.size, max_area=AREA, infer_frames=BLOCK, shift=args.sample_shift,
            sample_solver=args.sample_solver, sampling_steps=STEPS, guide_scale=0.0, seed=seed, offload_model=False,
            init_first_frame=args.start_from_ref, use_dataset=False, dataset_sample_idx=0,
            drop_motion_noisy=args.drop_motion_noisy, num_gpus_dit=1, enable_vae_parallel=False,
            input_video_for_sam2=None, enable_online_decode=args.enable_online_decode)
        frames = ((video.float().clamp(-1, 1) + 1) * 127.5 + 0.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        del video
        torch.cuda.empty_cache()
        if frames.max() < 8:
            raise RuntimeError("Live Avatar made a black clip; see the attention note in LiveAvatar.__init__.")
        return frames

    def close(self):
        import torch.distributed as dist

        dist.destroy_process_group()


class Upscaler:
    def __init__(self, path=UPSCALER):
        import torch
        from spandrel import ModelLoader

        self.torch = torch
        self.model = ModelLoader().load_from_file(str(path)).cuda().eval().half()
        if self.model.scale != UPSCALE:
            raise SystemExit(f"{path} upscales {self.model.scale}x, not {UPSCALE}x.")

    def __call__(self, frames, batch=8):
        torch, out = self.torch, []
        with torch.inference_mode():
            for start in range(0, len(frames), batch):
                x = torch.from_numpy(frames[start:start + batch]).cuda().permute(0, 3, 1, 2).half() / 255
                y = self.model(x).float().clamp(0, 1)
                out.append((y * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("avatar", help="the avatar's key (its folder in data/photostage) or a folder with photo.png and rig.json")
    parser.add_argument("--gesture", nargs="+", choices=PROMPTS, default=["idle", "listening", "thinking", "greet", "explain"])
    parser.add_argument("--describe", default="A person", help="who is in the photo, for the prompts, e.g. 'A woman in a "
                        "blue blazer'; the photo shows the rest")
    parser.add_argument("--rest", help="the pose in the photo, which gestures come back to, e.g. 'the right hand resting "
                        "against the chest and the left hand low in front of the body'")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True, help="FP8 linear layers (faster)")
    parser.add_argument("--upscale", action=argparse.BooleanOptionalAction, default=True,
                        help=f"double the clip with Real-ESRGAN ({UPSCALER.name})")
    parser.add_argument("--out", type=Path, default=clipgen.OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def versions():
    import torch

    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"torch": torch.__version__, "liveavatar": commit}


def main(argv=None):
    args = parse(argv)
    folder = clipgen.avatar_folder(args.avatar)
    out = args.out.resolve()
    rig = json.loads((folder / "rig.json").read_text())
    photo = cv2.cvtColor(cv2.imread(str(folder / "photo.png")), cv2.COLOR_BGR2RGB)
    if list(photo.shape[1::-1]) != list(rig["size"]):
        raise SystemExit(f"The photo is {photo.shape[1]}x{photo.shape[0]} but the rig was made for {rig['size'][0]}x{rig['size'][1]}.")
    try:
        layout = plan(rig["size"], rig["face"], UPSCALE if args.upscale else 1)
    except clips.Unsuitable as error:
        raise SystemExit(str(error))
    work_plan = {"region": layout["region"], "work": layout["work"]}
    image = clips.crop(photo, work_plan)
    given = clips.crop(photo, {"region": layout["region"], "work": layout["given"]})
    rows = clips.band_rows(work_plan, rig["face"])
    face_h = (rig["face"][3] - rig["face"][1]) * clips.work_scale(work_plan)
    geometry = {**work_plan, "band": list(rows), "face_h": round(face_h, 1), "framing": "avatar",
                "track_box": clips.face_box(work_plan, rig["face"]), "given": layout["given"]}
    print(json.dumps(geometry), flush=True)

    jobs = []
    for gesture in args.gesture:
        line = LINES.get(gesture)
        speech, rate = speak(line) if line else (np.zeros(0, np.float32), 22050)
        track, blocks = drive(speech, rate)
        jobs.append({"gesture": gesture, "line": line, "speech_seconds": round(len(speech) / rate, 3), "rate": rate,
                     "track": track, "blocks": blocks, "prompt": prompt(gesture, args.describe, args.rest),
                     "target": out / folder.name / f"{MODEL}-{gesture}-s{args.seed}"})
        print(f"{gesture}: {len(track) / rate:.2f} s drive, {blocks} pieces, {BLOCK * blocks - LEAD} frames", flush=True)
    if args.dry_run:
        target = out / folder.name / f"{MODEL}-plan"
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target / "first.png"), image[:, :, ::-1])
        cv2.imwrite(str(target / "given.png"), given[:, :, ::-1])
        for job in jobs:
            write_wav(target / f"{job['gesture']}.wav", job["track"], job["rate"])
        (target / "plan.json").write_text(json.dumps(geometry, indent=1))
        print(f"Live Avatar would be given {target / 'given.png'}")
        return 0

    began = time.time()
    model = LiveAvatar([job["prompt"] for job in jobs], fp8=args.fp8)
    upscaler = Upscaler() if args.upscale else None
    print(f"loaded in {time.time() - began:.0f} s", flush=True)
    with tempfile.TemporaryDirectory(prefix="liveavatar-") as scratch:
        given_path = Path(scratch) / "given.png"
        cv2.imwrite(str(given_path), given[:, :, ::-1])
        for job in jobs:
            gesture, target = job["gesture"], job["target"]
            print(f"{MODEL} / {gesture}", flush=True)
            started = time.time()
            audio = Path(scratch) / f"{gesture}.wav"
            write_wav(audio, job["track"], job["rate"])
            frames = model.generate(given_path, audio, job["prompt"], args.seed)
            made = len(frames)
            quiet = clips.ROLES[gesture][0] in clips.LOOP_ROLES
            if quiet:
                frames = frames[:settle(frames, round(QUIET_KEEP * FPS)) + 1]
            else:
                frames = frames[:calm(frames, speech_end(job["speech_seconds"])) + 1]
            if upscaler:
                frames = upscaler(frames)
            seconds = round(time.time() - started, 1)
            numbers = clips.measure(frames, image, rows, face_h)
            info = {
                "avatar": folder.name, "model": MODEL, "gesture": gesture, "prompt": job["prompt"], "negative": None,
                "line": job["line"], "speech_seconds": job["speech_seconds"], "lead_in": LEAD_IN if job["line"] else 0.0,
                "drive_seconds": round(len(job["track"]) / job["rate"], 3), "made_frames": made, "seed": args.seed,
                "frames": len(frames), "fps": FPS, "steps": STEPS, "guidance": 0.0, "loop": False, "fp8": args.fp8,
                "upscaler": UPSCALER.name if upscaler else None, **geometry, "seconds": seconds,
                "versions": versions(), "measure": numbers,
            }
            clipgen.save(target, image, frames, FPS, info)
            write_wav(target / "drive.wav", job["track"], job["rate"])
            print(f"  {seconds} s, kept {len(frames)} of {made} frames -> {target}\n  {json.dumps(numbers)}", flush=True)
    model.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
