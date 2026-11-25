"""Full-frame, block-streaming adapter for the pinned LiveTalk release."""

import base64
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def frame_layout(width, height, area=384):
    scale = math.sqrt(area * area / (width * height))
    canvas = (max(32, int(width * scale) // 32 * 32),
              max(32, int(height * scale) // 32 * 32))
    ratio = min(canvas[0] / width, canvas[1] / height)
    fitted = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    left, top = (canvas[0] - fitted[0]) // 2, (canvas[1] - fitted[1]) // 2
    return canvas, (left, top, left + fitted[0], top + fitted[1])


class Engine:
    head_only = False  # the whole portrait unless told otherwise (also for an engine made without __init__)

    prompts = None  # (idle, speaking) prompts in place of the built-in ones
    area = 384  # the full-frame canvas has the area of a square this wide (the model was trained at 512)
    keep = None  # a list to which every frame is also appended, as RGB at the model's size

    def __init__(self, *, head_only=False, prompts=None, area=384):
        self.head_only = head_only
        self.prompts = prompts
        self.area = area
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        os.environ.update(HF_HUB_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                          WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false")
        source = ROOT / "external/LiveTalk"
        sys.path[:0] = [str(source), str(source / "OmniAvatar")]
        os.chdir(source)
        sys.argv = [sys.argv[0], "--config", str(source / "configs/causal_inference.yaml")]
        import torch
        from scripts.inference_example import CausalInferencePipeline, args

        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("LiveTalk generation requires the selected CUDA GPU.")
        args.text_encoder_path = str(ROOT / "models/livetalk/base/models_t5_umt5-xxl-enc-bf16.pth")
        args.dit_path = str(ROOT / "models/livetalk/model/model.safetensors")
        args.vae_path = str(ROOT / "models/livetalk/base/Wan2.1_VAE.pth")
        args.wav2vec_path = str(ROOT / "models/wav2vec2")
        engine = self

        class StreamingPipeline(CausalInferencePipeline):
            def _setup_caches(self, batch_size, dtype, device):
                super()._setup_caches(batch_size, dtype, device)
                for cache in self.kv_cache1:
                    # The upstream attention loop reads these with .item(); CUDA scalars stall it.
                    cache["global_end_index"] = torch.zeros(1, dtype=torch.long, device="cpu")
                    cache["local_end_index"] = torch.zeros(1, dtype=torch.long, device="cpu")

            def _encode_text_prompts(self, text_prompts, positive=True):
                key = (text_prompts, positive)
                if key not in self.prompt_cache:
                    self.prompt_cache[key] = super()._encode_text_prompts(text_prompts, positive)
                return self.prompt_cache[key]

            def inference(self, noise, text_prompts, img_lat, audio_embed,
                          initial_latent=None, return_latents=False):
                if initial_latent is not None or return_latents:
                    raise ValueError("This streaming adapter accepts image-conditioned output only.")
                batch = noise.shape[0]
                condition = dict(self._encode_text_prompts(text_prompts))
                self._setup_caches(batch, noise.dtype, noise.device)
                self.vae.model.clear_cache()
                count = 0
                for start in range(0, noise.shape[1], self.num_frame_per_block):
                    engine.check_cancelled()
                    end = start + self.num_frame_per_block
                    current = noise[:, start:end]
                    condition.update(image=img_lat[:, :, start:end], audio=audio_embed[:, start:end])
                    for index, step in enumerate(self.denoising_step_list):
                        if start == 0:
                            current[:, :1] = img_lat[:, :16, :1].permute(0, 2, 1, 3, 4)
                        # Upstream preserves this dtype in its sinusoidal embeddings.
                        timestep = torch.full((batch, end - start), step, device=noise.device,
                                              dtype=noise.dtype)
                        _, clean = self._generator_forward(
                            current, condition, timestep, kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start * self.frame_seq_length,
                        )
                        if index + 1 < len(self.denoising_step_list):
                            next_step = self.denoising_step_list[index + 1]
                            current = self.scheduler.add_noise(
                                clean.flatten(0, 1), torch.randn_like(clean.flatten(0, 1)),
                                next_step * torch.ones(batch * (end - start), device=noise.device,
                                                       dtype=torch.long),
                            ).unflatten(0, clean.shape[:2])
                    if start == 0:
                        clean[:, :1] = img_lat[:, :16, :1].permute(0, 2, 1, 3, 4)
                    self._generator_forward(
                        clean, condition, torch.zeros_like(timestep), kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=start * self.frame_seq_length,
                    )
                    video = (self.vae.decode_to_pixel(clean, use_cache=True) * 0.5 + 0.5).clamp(0, 1)
                    count += engine.on_frames(video[0])
                return {"frames": count}

        self.pipeline = StreamingPipeline(args, torch.device("cuda:0"))
        self.pipeline.prompt_cache = {}
        self.directory = tempfile.TemporaryDirectory(prefix="avatar-fullbody-")
        self.image = Path(self.directory.name) / "reference.png"
        self.avatar = None
        self.warmed = False
        self.cancelled = threading.Event()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError("Avatar generation cancelled.")

    def prepare(self, avatar, image, idle, emit):
        import numpy as np
        import soundfile as sf
        from PIL import Image, ImageOps

        if avatar != self.avatar:
            with Image.open(image) as reference:
                if self.head_only:
                    if reference.size != (512, 512):
                        raise ValueError("LiveTalk head input must be the prepared 512x512 head crop.")
                    canvas, self.crop = (512, 512), (0, 0, 512, 512)
                else:
                    canvas, self.crop = frame_layout(*reference.size, self.area)
                self.original_size = reference.size
                fitted = ImageOps.contain(reference.convert("RGB"), (
                    self.crop[2] - self.crop[0], self.crop[3] - self.crop[1],
                ))
                padded = Image.new("RGB", canvas)
                padded.paste(fitted, self.crop[:2])
                padded.save(self.image)
            self.canvas = canvas
            self.pipeline.args.image_sizes_720 = [[canvas[1], canvas[0]]]
            self.pipeline.frame_seq_length = canvas[0] * canvas[1] // 256
            self.pipeline.kv_cache1 = None
            self.avatar = avatar
        self.check_cancelled()
        if not Path(idle).is_file() or not self.warmed:
            with tempfile.TemporaryDirectory(prefix="fullbody-idle-", dir=Path(idle).parent) as directory:
                silence, rendered = Path(directory) / "silence.wav", Path(directory) / "idle.mp4"
                sf.write(silence, np.zeros(16000 * 3, dtype=np.float32), 16000)
                self.stream(str(silence), str(rendered), lambda event: None, idle=True)
                rendered.replace(idle)
            self.warmed = True
        emit({"event": "prepared", "avatar": avatar})

    def stream(self, audio, output, emit, idle=False):
        import cv2
        import imageio_ffmpeg
        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        if self.avatar is None:
            raise ValueError("Prepare a LiveTalk reference image first.")
        samples, _ = librosa.load(audio, sr=16000, mono=True)
        if not 0 < len(samples) <= 16000 * 45:
            raise ValueError("Speech must be between 0 and 45 seconds.")
        fps, rate = 16, 16000
        target = math.ceil(len(samples) * fps / rate)
        latent_frames = math.ceil((target + 4) / 12) * 3
        padded_length = (latent_frames * 4 - 4) * rate // fps
        prompt = (
            ("A close-up of the same person's head, natural blinking, subtle breathing, "
             "lips closed at rest. " if idle else
             "A close-up of the same person speaking naturally, synchronized lips, expressive "
             "eyes and eyebrows, small natural head movements. ")
            + "Keep the head centered and the neck and shoulders steady. No hands in the frame. "
            "The camera is stationary. "
        ) if self.head_only else (
            "A full-body shot of the same person, standing quietly with subtle breathing and "
            "natural blinking, hands relaxed, mouth at rest. The camera is stationary. "
            if idle else
            "A full-body shot of the same person speaking naturally, with synchronized lips, "
            "expressive face, gentle hand gestures and subtle body movement. Keep the entire "
            "person, head to feet, inside the frame. The camera is stationary. "
        )
        if self.prompts:
            prompt = self.prompts[0] if idle else self.prompts[1]
        prompt += "Preserve the original clothing, identity, scene, background and composition."
        count, index, first = 0, 0, None
        started = time.monotonic()
        left, top, right, bottom = self.crop
        width, height = right - left, bottom - top
        with tempfile.TemporaryDirectory(prefix="avatar-fullbody-stream-") as directory:
            silent, padded_audio = Path(directory) / "silent.mp4", Path(directory) / "padded.wav"
            sf.write(padded_audio, np.pad(samples, (0, max(0, padded_length - len(samples)))), rate)
            writer = imageio_ffmpeg.write_frames(
                str(silent), (width, height), fps=fps, codec="libx264",
                pix_fmt_in="rgb24", pix_fmt_out="yuv444p", macro_block_size=1,
                output_params=["-preset", "ultrafast", "-crf", "18", "-threads", "4"],
            )
            writer.send(None)

            def on_frames(video):
                nonlocal count, index, first
                self.check_cancelled()
                frames = video[:max(0, target - count)].permute(0, 2, 3, 1)
                frames = (frames.float().cpu().numpy() * 255).astype(np.uint8)
                if not len(frames):
                    return 0
                encoded = []
                for frame in frames:
                    rgb = np.ascontiguousarray(frame[top:bottom, left:right])
                    writer.send(rgb)
                    if self.keep is not None:
                        self.keep.append(rgb)
                    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError("Full-frame JPEG encoding failed.")
                    encoded.append(base64.b64encode(jpeg).decode("ascii"))
                chunk = samples[count * rate // fps:(count + len(frames)) * rate // fps]
                if first is None:
                    first = time.monotonic() - started
                emit({
                    "event": "segment", "index": index, "fps": fps, "rate": rate,
                    "width": width, "height": height,
                    "audio": base64.b64encode((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()).decode("ascii"),
                    "frames": encoded,
                })
                count += len(frames)
                index += 1
                return len(frames)

            self.on_frames = on_frames
            try:
                with torch.inference_mode():
                    noise = torch.randn(
                        (1, latent_frames, 16, self.canvas[1] // 8, self.canvas[0] // 8),
                        device=self.pipeline.device, dtype=self.pipeline.dtype,
                    )
                    self.pipeline(noise, prompt, str(self.image), str(padded_audio))
            finally:
                writer.close()
                self.pipeline.vae.model.clear_cache()
            self.check_cancelled()
            if count != target:
                raise RuntimeError(f"Expected {target} LiveTalk frames; generated {count}.")
            output_width, output_height = self.original_size
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(silent), "-i", audio,
                "-map", "0:v:0", "-map", "1:a:0", "-vf", f"scale={output_width}:{output_height}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "20",
                "-threads", "4", "-c:a", "aac", "-movflags", "+faststart", "-shortest", output,
            ], check=True, timeout=60, capture_output=True)
        emit({"event": "rendered", "first_segment_seconds": first,
              "render_seconds": time.monotonic() - started, "audio_seconds": len(samples) / rate,
              "frames": count, "native_size": [width, height], "output_size": list(self.original_size)})
