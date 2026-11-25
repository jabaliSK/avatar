"""Private stdio worker: keep Lite on one GPU and emit audio/JPEG chunks."""

import base64
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Engine:
    def __init__(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        os.environ.update(HF_HUB_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                          OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        sys.path.insert(0, str(ROOT / "external" / "FlashHead"))
        os.chdir(ROOT / "external" / "FlashHead")
        import torch
        import flash_head.src.pipeline.flash_head_pipeline as module
        from flash_head.inference import get_pipeline, get_infer_params

        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("The live worker requires a CUDA GPU.")
        # Compiling takes minutes once (the kernels are then cached) and renders about 1.7x faster.
        module.COMPILE_MODEL = module.COMPILE_VAE = os.getenv("AVATAR_COMPILE") == "1"
        self.pipeline = get_pipeline(
            world_size=1, ckpt_dir=str(ROOT / "models/flashhead"),
            wav2vec_dir=str(ROOT / "models/wav2vec2"), model_type="lite",
        )
        self.params = get_infer_params()
        self.avatar = None
        self.warmed = False
        self.cancelled = threading.Event()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError("Avatar generation cancelled.")

    def prepare(self, avatar, image, idle, emit):
        from flash_head.inference import get_base_data
        from render_flashhead import prepare_portrait
        import numpy as np
        import soundfile as sf
        import torch

        if self.avatar != avatar:
            with tempfile.TemporaryDirectory(prefix="avatar-live-prepare-") as directory:
                portrait = Path(directory) / "portrait.png"
                prepare_portrait(Path(image), portrait)
                with torch.inference_mode():
                    get_base_data(
                        self.pipeline, cond_image_path_or_dir=str(portrait),
                        base_seed=42, use_face_crop=False,
                    )
            self.avatar = avatar
        self.check_cancelled()
        if not Path(idle).is_file() or not self.warmed:
            with tempfile.TemporaryDirectory(prefix="avatar-idle-", dir=Path(idle).parent) as directory:
                silence = Path(directory) / "silence.wav"
                rendered = Path(directory) / "idle.mp4"
                sf.write(silence, np.zeros(16000 * 4, dtype=np.float32), 16000)
                self.stream(str(silence), str(rendered), lambda event: None, emit_segments=False)
                rendered.replace(idle)
            self.warmed = True
        emit({"event": "prepared", "avatar": avatar})

    def stream(self, audio, output, emit, *, emit_segments=True):
        import cv2
        import imageio_ffmpeg
        import librosa
        import numpy as np
        import torch
        from flash_head.inference import get_audio_embedding, run_pipeline
        from render_flashhead import audio_chunks

        if self.avatar is None:
            raise ValueError("Prepare an avatar before streaming.")
        p = self.params
        rate, fps, overlap = p["sample_rate"], p["tgt_fps"], p["motion_frames_num"]
        hop = (p["frame_num"] - overlap) * rate // fps
        samples, _ = librosa.load(audio, sr=rate, mono=True)
        if not 0 < len(samples) <= rate * 45:
            raise ValueError("Speech must be between 0 and 45 seconds.")
        history = np.zeros(p["cached_audio_duration"] * rate, dtype=np.float32)
        end = p["cached_audio_duration"] * fps
        count = 0
        target = math.ceil(len(samples) * fps / rate)
        self.pipeline.reset_person_name()
        started = time.monotonic()
        first = None
        with tempfile.TemporaryDirectory(prefix="avatar-live-") as directory:
            silent = Path(directory) / "silent.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(silent), (512, 512), fps=fps, codec="libx264",
                pix_fmt_in="rgb24", macro_block_size=1,
                output_params=["-preset", "ultrafast", "-crf", "20", "-threads", "4"],
            )
            writer.send(None)
            try:
                with torch.inference_mode():
                    for index, chunk in enumerate(audio_chunks(samples, hop)):
                        self.check_cancelled()
                        history = np.concatenate((history[hop:], chunk))
                        embedding = get_audio_embedding(
                            self.pipeline, history, end - p["frame_num"], end
                        )
                        frames = run_pipeline(self.pipeline, embedding)[overlap:].cpu().numpy()
                        frames = frames[:min(len(frames), target - count)]
                        encoded = []
                        for frame in frames:
                            rgb = np.ascontiguousarray(frame.astype(np.uint8))
                            writer.send(rgb)
                            if not emit_segments:
                                continue
                            ok, jpeg = cv2.imencode(
                                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 88],
                            )
                            if not ok:
                                raise RuntimeError("JPEG encoding failed.")
                            encoded.append(base64.b64encode(jpeg).decode("ascii"))
                        count += len(frames)
                        if not emit_segments:
                            continue
                        pcm = np.clip(samples[index * hop:(index + 1) * hop], -1, 1)
                        pcm = (pcm * 32767).astype("<i2").tobytes()
                        if first is None:
                            first = time.monotonic() - started
                        emit({
                            "event": "segment", "index": index, "fps": fps, "rate": rate,
                            "audio": base64.b64encode(pcm).decode("ascii"),
                            "frames": encoded,
                        })
            finally:
                writer.close()
            if count != target:
                raise RuntimeError(f"Expected {target} frames, generated {count}.")
            self.check_cancelled()
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                 "-i", str(silent), "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", "-shortest", output],
                check=True, timeout=30, capture_output=True,
            )
        emit({
            "event": "rendered", "first_segment_seconds": first,
            "render_seconds": time.monotonic() - started,
            "audio_seconds": len(samples) / rate, "frames": count,
        })


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(event):
        protocol.write(json.dumps(event, allow_nan=False) + "\n")
        protocol.flush()

    renderer = os.getenv("AVATAR_RENDERER", "flashhead")
    if renderer == "livetalk":
        from render_livetalk import Engine as LiveTalkEngine
        engine = LiveTalkEngine(head_only=os.getenv("AVATAR_HEAD_ONLY") == "1")
    elif renderer == "flashhead":
        engine = Engine()
    elif renderer == "musetalk":
        from dub_worker import Engine as DubEngine
        engine = DubEngine()
    else:
        raise ValueError(f"Unsupported streaming renderer: {renderer}")
    incoming = queue.Queue()

    def read_commands():
        cancelled = None
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                incoming.put((line, threading.Event()))
                continue
            if isinstance(command, dict) and command.get("command") == "cancel":
                if cancelled:
                    cancelled.set()
                continue
            if isinstance(command, dict) and command.get("command") == "quit" and cancelled:
                cancelled.set()
            cancelled = threading.Event()
            incoming.put((line, cancelled))
        incoming.put((None, None))

    threading.Thread(target=read_commands, daemon=True).start()
    emit({"event": "ready"})
    while True:
        line, engine.cancelled = incoming.get()
        if line is None:
            break
        try:
            command = json.loads(line)
            if command["command"] == "quit":
                break
            if command["command"] == "prepare":
                engine.prepare(command["avatar"], command["image"], command["idle"], emit)
            elif command["command"] == "stream":
                # A dubbed reply (musetalk) comes with the plan of the body under it (photostage/dubplan.py).
                extra = {"plan": command["plan"]} if "plan" in command else {}
                engine.stream(command["audio"], command["output"], emit, **extra)
            else:
                raise ValueError("Unknown worker command.")
        except InterruptedError:
            emit({"event": "cancelled"})
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            traceback.print_exc(file=sys.stderr)
            emit({"event": "error", "message": str(error)})
        finally:
            emit({"event": "done"})


if __name__ == "__main__":
    main()
