"""MuseTalk lip-sync inside the avatar's own gesture clips (PHOTOSTAGE_RENDERER=musetalk).

The head is never pasted onto the body: every picture is a frame of one of the avatar's gesture clips, and MuseTalk 1.5
regenerates only its lower face (jaw, lips, cheeks) for the sound of that picture, at the clip's own resolution. Head and
body are one image, so they cannot come apart. The generated face is 256 px and softer than the clip, so the clip's own
skin detail is carried back onto it along optical flow, as far as the two agree (they differ where the mouth opens).

A clip is prepared once per avatar (face boxes from FAN landmarks, smoothed over time; VAE latents; face-parsing masks) and
kept next to it (dub-<clip>.npz) until the clip changes. Speech is then a whisper pass over the reply and, per picture, one
UNet step and a VAE decode: about 10 ms on the GPU and 3 ms of CPU, several times faster than real time.

Used by scripts/live_worker.py under the main environment (.venv), which has MuseTalk's dependencies.
"""

import base64
import json
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUSETALK = ROOT / "external" / "MuseTalk"
CACHE_VERSION = 1
FPS = 25
BATCH = 16  # one batch shape, autotuned once at warm-up
SEGMENT = 25  # pictures per streamed segment
QUALITY = 82
STILL = "photo"  # the photo itself, one frame, for an avatar without clips
SMOOTH = (3, 1.2)  # face boxes are smoothed over this many frames each way, with this sigma
DETAIL_TAU = 20.0  # how different (0-255) generated and original may be before the original's detail is left out
DETAIL_SCALE = 70  # the detail is what is finer than the face's width over this
SHARPEN = (1.0, 0.7)  # unsharp mask (sigma, amount) where the original's detail cannot be used


class Material:
    """One clip as MuseTalk needs it: its frames (BGR), face boxes, latents (on the GPU), blend masks and their boxes."""

    def __init__(self, frames, boxes, latents, masks, crops):
        self.frames, self.boxes, self.latents, self.masks, self.crops = frames, boxes, latents, masks, crops
        self.quiet = None


def quiet_frames(folder, spec, count):
    """The exact JPEGs used by the browser between turns, not another lossy encoding of the face."""
    info = json.loads((folder / spec["index"]).read_text())
    data = (folder / spec["frames"]).read_bytes()
    rows = info["frames"]
    if len(rows) != count:
        raise ValueError("The silent frame bank does not match the motion loop.")
    found = []
    for offset, length in rows:
        if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length < 4 or offset + length > len(data):
            raise ValueError("Invalid silent frame bank offsets.")
        jpeg = data[offset:offset + length]
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("Invalid silent frame bank JPEG.")
        found.append(base64.b64encode(jpeg).decode("ascii"))
    return found


def smooth_boxes(boxes):
    import numpy as np

    radius, sigma = SMOOTH
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(boxes, np.float64), ((radius, radius), (0, 0)), mode="edge")
    return np.round(np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(4)], 1)).astype(int)


def face_box(points):
    """MuseTalk's face box from 68 landmarks: from as far above the nose as the chin is below it, down past the chin."""
    nose, chin = points[29, 1], points[:, 1].max()
    return [points[:, 0].min(), nose - (chin - nose), points[:, 0].max(), chin + 10]


def fill_missing(boxes):
    """Frames where no face was found take the box of the nearest frame where one was."""
    found = [i for i, box in enumerate(boxes) if box is not None]
    if not found:
        raise RuntimeError("No face was found in the clip.")
    return [boxes[i] if boxes[i] is not None else boxes[min(found, key=lambda j: abs(j - i))]
            for i in range(len(boxes))]


def blend(frame, patch, box, mask, crop, weight):
    """MuseTalk's get_image_blending in numpy, limited to the mask's box and clipped to the frame."""
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop
    large = np.zeros((cy2 - cy1, cx2 - cx1, 3), np.uint8)
    sx1, sy1, sx2, sy2 = max(cx1, 0), max(cy1, 0), min(cx2, width), min(cy2, height)
    large[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1] = frame[sy1:sy2, sx1:sx2]
    large[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = patch
    alpha = mask[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1, None].astype(np.float32) * (weight / 255.0)
    out = frame.copy()
    region = out[sy1:sy2, sx1:sx2].astype(np.float32)
    region += (large[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1].astype(np.float32) - region) * alpha
    out[sy1:sy2, sx1:sx2] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return out


class Engine:
    def __init__(self):
        os.environ.update(HF_HUB_OFFLINE="1", TORCH_HOME=str(ROOT / "models" / "torch"), TOKENIZERS_PARALLELISM="false")
        sys.path.insert(0, str(MUSETALK))
        os.chdir(MUSETALK)  # MuseTalk finds its models relative to its folder
        import cv2
        import torch
        from transformers import WhisperModel
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing as MuseFaceParsing
        from musetalk.utils.utils import load_all_model

        if not torch.cuda.is_available():
            raise RuntimeError("The dub worker requires a CUDA GPU.")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        cv2.setNumThreads(2)
        self.device = torch.device("cuda:0")
        with torch.inference_mode():
            self.vae, self.unet, self.position = load_all_model(device=self.device)
            self.vae.vae = self.vae.vae.to(device=self.device, dtype=torch.float16)
            self.unet.model = self.unet.model.to(device=self.device, dtype=torch.float16).eval()
            self.position = self.position.to(device=self.device, dtype=torch.float16)
            self.whisper = WhisperModel.from_pretrained("models/whisper", local_files_only=True)
            self.whisper = self.whisper.to(device=self.device, dtype=torch.float16).eval()
        self.processor = AudioProcessor("models/whisper")

        class FaceParsing(MuseFaceParsing):
            def model_init(self):
                return super().model_init(resnet_path="models/face-parse-bisent/resnet18-f37072fd.pth")

        self.parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        self.pool = ThreadPoolExecutor(6, thread_name_prefix="dub")
        self.local = threading.local()
        self.avatar = None
        self.stamp = None
        self.materials = {}
        self.warmed = False
        self.cancelled = threading.Event()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError("Avatar generation cancelled.")

    # Preparing an avatar -------------------------------------------------------------------------------------------------

    def prepare(self, avatar, image, idle, emit):
        folder = Path(image).parent
        pack = folder / "clips.json"
        stamp = (avatar, pack.stat().st_mtime_ns if pack.is_file() else None)
        if self.stamp != stamp:
            started = time.monotonic()
            self.materials = {}
            manifest = json.loads(pack.read_text()) if pack.is_file() else {}
            # Only clips of the whole avatar have a face to dub; body clips (from the chin down) do not.
            clips = (manifest.get("clips") or {}) if manifest.get("mode") == "avatar" else {}
            quiet = manifest.get("continuous")
            # Only an avatar's own folder keeps a cache (not the warm-up portrait's).
            keep = (folder / "rig.json").is_file()
            size = None
            for name, spec in sorted(clips.items()):
                self.check_cancelled()
                path = folder / (spec["source"] if quiet else (spec.get("files") or {}).get("mp4", ""))
                if not path.is_file():
                    continue
                frames = self.decode(path)
                size = size or frames[0].shape[1::-1]
                self.materials[name] = self.material(frames, folder / f"dub-{name}.npz" if keep else None, path)
                if quiet:
                    self.materials[name].quiet = quiet_frames(folder, quiet, len(frames))
            self.materials[STILL] = self.material([self.still(Path(image), size)],
                                                  folder / f"dub-{STILL}.npz" if keep else None, Path(image))
            self.avatar, self.stamp = avatar, stamp
            print(f"prepared {avatar}: {', '.join(f'{n} ({len(m.frames)})' for n, m in self.materials.items())} "
                  f"in {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)
        if not self.warmed:
            # Warm Whisper as well as the face batch before negotiating a time-critical playback phase.
            import numpy as np
            import soundfile as sf

            material = self.materials[STILL]
            rows = [[0, 0, -1, 0, 0.0, 1.0]] * BATCH
            with tempfile.TemporaryDirectory(prefix="dub-warm-") as directory:
                audio = Path(directory) / "silence.wav"
                sf.write(audio, np.zeros(16000, np.float32), 16000)
                self.render({"fps": FPS, "clips": [STILL], "frames": rows}, str(audio),
                            lambda event: None, material=material)
            self.warmed = True
        emit({"event": "prepared", "avatar": avatar})

    @staticmethod
    def decode(path):
        if path.suffix == ".json":
            # A bank of JPEGs (its index and, beside it, .bin): the pictures exactly as the browser shows them.
            import cv2
            import numpy as np

            info = json.loads(path.read_text())
            data = path.with_suffix(".bin").read_bytes()
            frames = [cv2.imdecode(np.frombuffer(data[at:at + size], np.uint8), cv2.IMREAD_COLOR)
                      for at, size in info["frames"]]
            if not frames or any(frame is None for frame in frames):
                raise RuntimeError(f"{path.name} is not a usable frame bank.")
            return frames
        import av

        # PyAV converts with the colour matrix the clip is tagged with (BT.709), as browsers do; OpenCV would use BT.601.
        with av.open(str(path)) as container:
            frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
        if len(frames) < 1:
            raise RuntimeError(f"{path.name} has no frames.")
        return frames

    @staticmethod
    def still(path, size):
        """The photo at the clips' size (or 1264 px high without clips), as the clips show it."""
        import cv2
        import numpy as np
        from PIL import Image

        with Image.open(path) as photo:
            rgb = np.asarray(photo.convert("RGB"))
        if size is None:
            size = (round(rgb.shape[1] * 1264 / rgb.shape[0]), 1264)
        return cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()

    def material(self, frames, cache, source):
        import numpy as np
        import torch

        key = f"{CACHE_VERSION}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{len(frames)}:{frames[0].shape}"
        if cache and cache.is_file():
            try:
                with np.load(cache) as data:
                    if str(data["key"]) == key:
                        sizes = data["sizes"]
                        flat = np.split(data["masks"], np.cumsum([h * w for h, w in sizes])[:-1])
                        masks = [m.reshape(h, w) for m, (h, w) in zip(flat, sizes)]
                        latents = torch.from_numpy(data["latents"]).to(self.device)
                        return Material(frames, data["boxes"], latents, masks, data["crops"])
            except (OSError, KeyError, ValueError) as error:
                print(f"ignoring {cache.name}: {error}", file=sys.stderr, flush=True)
        boxes = smooth_boxes(fill_missing(self.landmark_boxes(frames)))
        latents, masks, crops = self.encode(frames, boxes)
        if cache:
            scratch = cache.with_suffix(".tmp.npz")
            np.savez(scratch, key=np.array(key), boxes=boxes, crops=np.array(crops),
                     latents=latents.cpu().numpy(), sizes=np.array([m.shape for m in masks]),
                     masks=np.concatenate([m.ravel() for m in masks]))
            scratch.replace(cache)
        return Material(frames, boxes, latents, masks, np.array(crops))

    def landmark_boxes(self, frames):
        import cv2
        import face_alignment
        import numpy as np
        import torch

        detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda", flip_input=False)
        boxes = []
        try:
            for frame in frames:
                self.check_cancelled()
                found = detector.get_landmarks_from_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not found:
                    boxes.append(None)
                    continue
                largest = max(found, key=lambda p: np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
                boxes.append(face_box(largest))
        finally:
            del detector
            torch.cuda.empty_cache()
        return boxes

    def encode(self, frames, boxes):
        import cv2
        import torch
        from musetalk.utils.blending import get_image_prepare_material

        latents, masks, crops = [], [], []
        with torch.inference_mode():
            for frame, (x1, y1, x2, y2) in zip(frames, boxes):
                self.check_cancelled()
                crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents.append(self.vae.get_latents_for_unet(crop).to(dtype=torch.float16))
                mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.parser, mode="jaw")
                masks.append(mask)
                crops.append(crop_box)
        return torch.cat(latents), masks, crops

    # Speaking ------------------------------------------------------------------------------------------------------------

    def stream(self, audio, output, emit, plan=None):
        if self.avatar is None:
            raise ValueError("Prepare an avatar before streaming.")
        if not isinstance(plan, dict):
            raise ValueError("A dubbed reply needs its plan.")
        self.render(plan, audio, emit)

    def check_plan(self, plan):
        if plan.get("fps") != FPS or not isinstance(plan.get("frames"), list) or not plan["frames"]:
            raise ValueError(f"The plan must have frames at {FPS} fps.")
        names = plan.get("clips")
        if not isinstance(names, list) or any(name not in self.materials for name in names):
            raise ValueError("The plan names a clip this avatar does not have.")
        if len(plan["frames"]) > FPS * 50:
            raise ValueError("The plan is too long.")
        clips = [self.materials[name] for name in names]
        quiet_from = plan.get("quiet_from")
        if quiet_from is not None and (isinstance(quiet_from, bool) or not isinstance(quiet_from, int) or
                                       not 0 <= quiet_from < len(plan["frames"]) or len(clips) != 1 or
                                       clips[0].quiet is None):
            raise ValueError("The plan needs a matching silent frame bank.")
        for row in plan["frames"]:
            a, ai, b, bi, mix, dub = row
            if not (0 <= a < len(clips) and 0 <= ai < len(clips[a].frames) and 0 <= mix <= 1 and 0 <= dub <= 1):
                raise ValueError("The plan has a frame that does not exist.")
            if b >= 0 and not (b < len(clips) and 0 <= bi < len(clips[b].frames)):
                raise ValueError("The plan has a frame that does not exist.")
            if quiet_from is not None and b != -1:
                raise ValueError("Continuous motion cannot crossfade clips.")
        return clips

    def render(self, plan, audio, emit, material=None):
        import numpy as np
        import soundfile as sf
        import torch

        clips = [material] if material else self.check_plan(plan)
        rows = plan["frames"]
        quiet_from = plan.get("quiet_from", len(rows))
        started = time.monotonic()
        if audio:
            samples, rate = sf.read(audio, dtype="float32", always_2d=True)
            samples = samples.mean(1)
            if not 0 < len(samples) <= rate * 45:
                raise ValueError("Speech must be between 0 and 45 seconds.")
            # Include real encoded silence beyond the final frame's two-frame Whisper lookahead.
            heard, padded = str(audio), None
            if len(samples) < (len(rows) + 3) * rate / FPS:
                padded = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                padded.close()
                silence = np.zeros(math.ceil((len(rows) + 3) * rate / FPS) - len(samples), np.float32)
                sf.write(padded.name, np.concatenate([samples, silence]), rate)
                heard = padded.name
            try:
                with torch.inference_mode():
                    features, length = self.processor.get_audio_feature(heard)
                    prompts = self.processor.get_whisper_chunk(features, self.device, torch.float16, self.whisper,
                                                               length, fps=FPS)
            finally:
                if padded:
                    os.unlink(padded.name)
        else:
            rate, samples = 16000, np.zeros(0, np.float32)
            prompts = torch.zeros((1, 50, 384), dtype=torch.float16, device=self.device)
        whisper_seconds = time.monotonic() - started
        first, gpu, cpu = None, 0.0, 0.0
        for index, begin in enumerate(range(0, len(rows), SEGMENT)):
            self.check_cancelled()
            span = range(begin, min(len(rows), begin + SEGMENT))
            tick = time.monotonic()
            faces = self.faces(clips, rows, range(span.start, min(span.stop, quiet_from)), prompts)
            gpu += time.monotonic() - tick
            tick = time.monotonic()
            pictures = list(self.pool.map(lambda i: clips[0].quiet[rows[i][1]] if i >= quiet_from else
                                          self.picture(clips, rows[i], faces.get((i, 0)), faces.get((i, 1))),
                                          span))
            cpu += time.monotonic() - tick
            if audio is None:
                continue
            at, to = round(span.start * rate / FPS), round(span.stop * rate / FPS)
            pcm = np.zeros(to - at, np.float32)
            heard = samples[at:to]
            pcm[:len(heard)] = heard
            pcm = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
            first = first or time.monotonic() - started
            emit({"event": "segment", "index": index, "fps": FPS, "rate": int(rate),
                  "audio": base64.b64encode(pcm).decode("ascii"), "frames": pictures})
        if audio is None:
            return
        emit({
            "event": "rendered", "first_segment_seconds": round(first, 3),
            "render_seconds": round(time.monotonic() - started, 3), "whisper_seconds": round(whisper_seconds, 3),
            "gpu_ms_per_frame": round(gpu / len(rows) * 1000, 2), "cpu_ms_per_frame": round(cpu / len(rows) * 1000, 2),
            "audio_seconds": round(len(samples) / rate, 3), "frames": len(rows),
        })

    def faces(self, clips, rows, span, prompts):
        """MuseTalk's lower faces for the pictures in `span`: {(picture, 0 for its clip or 1 for the clip faded in): face}."""
        import torch

        jobs = []
        for i in span:
            a, ai, b, bi, mix, dub = rows[i]
            if dub <= 0 or not len(prompts):
                continue
            jobs.append(((i, 0), clips[a].latents[ai], min(i, len(prompts) - 1)))
            if b >= 0 and mix > 0:
                jobs.append(((i, 1), clips[b].latents[bi], min(i, len(prompts) - 1)))
        found = {}
        with torch.inference_mode():
            for begin in range(0, len(jobs), BATCH):
                batch = jobs[begin:begin + BATCH]
                padded = batch + [batch[-1]] * (BATCH - len(batch))
                latent = torch.stack([job[1] for job in padded])
                sound = self.position(prompts[[job[2] for job in padded]])
                predicted = self.unet.model(latent, torch.tensor([0], device=self.device),
                                            encoder_hidden_states=sound).sample
                image = self.vae.vae.decode(predicted / self.vae.scaling_factor).sample
                image = ((image / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
                image = image.permute(0, 2, 3, 1).flip(-1).cpu().numpy()  # RGB to BGR
                for job, face in zip(batch, image):
                    found[job[0]] = face
        return found

    def picture(self, clips, row, face, faded):
        import cv2

        a, ai, b, bi, mix, dub = row
        image = self.dubbed(clips[a], ai, face, dub)
        if b >= 0 and mix > 0:
            image = cv2.addWeighted(image, 1 - mix, self.dubbed(clips[b], bi, faded, dub), mix, 0)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed.")
        return base64.b64encode(jpeg).decode("ascii")

    def dubbed(self, material, index, face, weight):
        import cv2

        frame = material.frames[index]
        if face is None or weight <= 0:
            return frame
        x1, y1, x2, y2 = material.boxes[index]
        patch = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
        patch = self.restore(patch, frame[y1:y2, x1:x2])
        return blend(frame, patch, (x1, y1, x2, y2), material.masks[index], material.crops[index], weight)

    def restore(self, generated, original):
        """The original's fine detail, moved along optical flow onto the generated face where the two agree, over a lightly
        sharpened generated face elsewhere (the open mouth and the lips, which the original does not have)."""
        import cv2
        import numpy as np

        if not hasattr(self.local, "flow"):
            # One per thread: OpenCV's DIS optical flow is not safe to share between threads.
            self.local.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        g = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        flow = self.local.flow.calc(g, cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), None)
        height, width = g.shape
        grid = np.mgrid[0:height, 0:width].astype(np.float32)
        warped = cv2.remap(original, grid[1] + flow[..., 0], grid[0] + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        made = generated.astype(np.float32)
        sigma = max(1.0, width / DETAIL_SCALE)
        low_warped = cv2.GaussianBlur(warped, (0, 0), sigma)
        error = np.abs(low_warped - cv2.GaussianBlur(made, (0, 0), sigma)).mean(2)
        confidence = cv2.GaussianBlur(np.exp(-(error / DETAIL_TAU) ** 2), (0, 0), sigma)[..., None]
        sharpened = made + SHARPEN[1] * (made - cv2.GaussianBlur(made, (0, 0), SHARPEN[0]))
        result = sharpened + confidence * (warped - low_warped) - confidence * (sharpened - made)
        return np.clip(result + 0.5, 0, 255).astype(np.uint8)
