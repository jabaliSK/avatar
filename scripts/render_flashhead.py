"""Audio-driven full-portrait generation with FlashHead Pro, not mouth compositing."""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "FlashHead"


def audio_chunks(samples, chunk_size):
    import numpy as np

    if chunk_size <= 0 or len(samples) == 0:
        raise ValueError("Audio and chunk size must be nonempty and positive.")
    for offset in range(0, len(samples), chunk_size):
        chunk = samples[offset:offset + chunk_size]
        yield np.pad(chunk, (0, chunk_size - len(chunk)))


def prepare_portrait(image: Path, target: Path):
    import numpy as np
    from PIL import Image

    from flash_head.utils.cpu_face_handler import CPUFaceHandler
    from flash_head.utils.facecrop import get_scaled_bbox

    with Image.open(image) as source:
        portrait = source.convert("RGB")
    detector = CPUFaceHandler(min_detection_confidence=0.6)
    try:
        boxes, _ = detector(np.asarray(portrait))
    finally:
        detector.detector.close()
    if len(boxes) != 1:
        raise ValueError("The portrait must contain exactly one clear, front-facing face.")
    width, height = portrait.size
    x1, y1, x2, y2 = boxes[0]
    crop = get_scaled_bbox(
        [x1 * width, y1 * height, x2 * width, y2 * height],
        width, height, ratio=2.0, face_image=portrait,
    )
    crop.resize((512, 512), Image.Resampling.LANCZOS).save(target)


def render(image: Path, audio: Path, output: Path, seed: int):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)

    import imageio_ffmpeg
    import librosa
    import numpy as np
    import torch

    import flash_head.src.pipeline.flash_head_pipeline as pipeline_module
    from flash_head.inference import (
        get_audio_embedding, get_base_data, get_infer_params, get_pipeline, run_pipeline,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("FlashHead requires an available CUDA GPU.")
    torch.set_num_threads(4)
    # These are upstream configuration switches. Compilation adds large cold-start
    # overhead to a renderer that intentionally releases its GPU after every turn.
    pipeline_module.COMPILE_MODEL = False
    pipeline_module.COMPILE_VAE = False
    with tempfile.TemporaryDirectory(prefix="avatar-motion-") as directory:
        temporary = Path(directory)
        portrait = temporary / "portrait.png"
        prepare_portrait(image, portrait)
        with torch.inference_mode():
            pipeline = get_pipeline(
                world_size=1, ckpt_dir=str(ROOT / "models" / "flashhead"),
                wav2vec_dir=str(ROOT / "models" / "wav2vec2"), model_type="pro",
            )
            get_base_data(
                pipeline, cond_image_path_or_dir=str(portrait), base_seed=seed, use_face_crop=False,
            )
            params = get_infer_params()
            fps, rate = params["tgt_fps"], params["sample_rate"]
            samples, _ = librosa.load(str(audio), sr=rate, mono=True)
            if not 0 < len(samples) <= 45 * rate:
                raise ValueError("Speech must be between 0 and 45 seconds.")
            overlap = params["motion_frames_num"]
            hop_frames = params["frame_num"] - overlap
            hop_samples = hop_frames * rate // fps
            history = np.zeros(params["cached_audio_duration"] * rate, dtype=np.float32)
            end_frame = params["cached_audio_duration"] * fps
            target_frames = math.ceil(len(samples) * fps / rate)
            frame_count = 0
            silent = temporary / "silent.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(silent), (512, 512), fps=fps, codec="libx264",
                pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
                output_params=["-crf", "18", "-threads", "4"],
            )
            writer.send(None)
            try:
                for chunk in audio_chunks(samples, hop_samples):
                    history = np.concatenate((history[hop_samples:], chunk))
                    embedding = get_audio_embedding(
                        pipeline, history, end_frame - params["frame_num"], end_frame
                    )
                    generated = run_pipeline(pipeline, embedding)[overlap:].cpu().numpy()
                    for frame in generated:
                        if frame_count >= target_frames:
                            break
                        writer.send(np.ascontiguousarray(frame.astype(np.uint8)))
                        frame_count += 1
            finally:
                writer.close()
            if frame_count != target_frames:
                raise RuntimeError(f"Expected {target_frames} frames, generated {frame_count}.")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                "-i", str(silent), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-movflags", "+faststart", "-shortest", str(output),
            ],
            check=True, timeout=60, capture_output=True,
        )
    print(f"FlashHead Pro rendered {frame_count} full-portrait frames at {fps} fps.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--image", type=Path, required=True)
    cli.add_argument("--audio", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    render(args.image.resolve(), args.audio.resolve(), args.output.resolve(), args.seed)
