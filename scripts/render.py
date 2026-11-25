"""Single-image MuseTalk 1.5 inference, isolated to one GPU subprocess per reply."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["TORCH_HOME"] = str(ROOT / "models" / "torch")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, str(ROOT / "external" / "MuseTalk"))


def render(image: Path, audio: Path, output: Path):
    import cv2
    import face_alignment
    import imageio_ffmpeg
    import numpy as np
    import torch
    from transformers import WhisperModel

    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.blending import get_image_blending, get_image_prepare_material
    from musetalk.utils.face_parsing import FaceParsing as MuseFaceParsing
    from musetalk.utils.utils import datagen, load_all_model

    if not torch.cuda.is_available():
        raise RuntimeError("MuseTalk requires an available CUDA GPU for this application.")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError("Cannot decode portrait.")
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device="cuda", flip_input=False
    )
    landmarks = detector.get_landmarks_from_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if landmarks is None or len(landmarks) != 1:
        raise ValueError("The portrait must contain exactly one clear, front-facing face.")
    points = landmarks[0].astype(int)
    # Match MuseTalk's nose/chin-based crop, using FAN's equivalent 68 landmarks.
    nose_y = points[29, 1]
    top = max(0, nose_y - (points[:, 1].max() - nose_y))
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(points[:, 0].min())), int(top)
    x2, y2 = min(width, int(points[:, 0].max())), min(height, int(points[:, 1].max()) + 10)
    if x2 - x1 < 32 or y2 - y1 < 32:
        raise ValueError("Detected face is too small to animate.")
    box = (x1, y1, x2, y2)
    del detector
    torch.cuda.empty_cache()
    with torch.inference_mode():
        vae, unet, position = load_all_model(device=device)
        vae.vae = vae.vae.to(device=device, dtype=torch.float16)
        unet.model = unet.model.to(device=device, dtype=torch.float16).eval()
        position = position.to(device=device, dtype=torch.float16)
        whisper = WhisperModel.from_pretrained("models/whisper", local_files_only=True)
        whisper = whisper.to(device=device, dtype=torch.float16).eval()
        processor = AudioProcessor("models/whisper")
        features, length = processor.get_audio_feature(str(audio))
        prompts = processor.get_whisper_chunk(
            features, device, torch.float16, whisper, length, fps=25
        )
        if len(prompts) == 0:
            raise ValueError("Audio is too short to animate.")
        del whisper
        torch.cuda.empty_cache()
        latent = vae.get_latents_for_unet(
            cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        )
        class FaceParsing(MuseFaceParsing):
            def model_init(self):
                # The full face-parser checkpoint overwrites this backbone. Use the
                # modern ResNet serialization so PyTorch 2.6+ can keep weights_only=True.
                return super().model_init(
                    resnet_path="models/face-parse-bisent/resnet18-f37072fd.pth"
                )

        parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        mask, crop_box = get_image_prepare_material(frame, box, fp=parser, mode="jaw")
        del parser
        torch.cuda.empty_cache()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory(prefix="avatar-render-") as directory:
            silent = Path(directory) / "silent.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(silent), (width, height), fps=25, codec="libx264",
                pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
                output_params=["-crf", "18", "-threads", "4"],
            )
            writer.send(None)
            try:
                for prompt, batch in datagen(prompts, [latent], batch_size=8, device=device):
                    embeddings = position(prompt.to(device=device, dtype=torch.float16))
                    predicted = unet.model(
                        batch.to(device=device, dtype=torch.float16),
                        torch.tensor([0], device=device),
                        encoder_hidden_states=embeddings,
                    ).sample
                    for face in vae.decode_latents(predicted):
                        patch = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
                        blended = get_image_blending(frame, patch, box, mask, crop_box)
                        writer.send(np.ascontiguousarray(blended[:, :, ::-1]))
            finally:
                writer.close()
            subprocess.run(
                [
                    ffmpeg, "-y", "-v", "error", "-i", str(silent), "-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-movflags", "+faststart", "-shortest", str(output),
                ],
                check=True, timeout=60, capture_output=True,
            )
    print(f"Rendered {len(prompts)} frames at 25 fps: {output}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--image", required=True, type=Path)
    cli.add_argument("--audio", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    image, audio, output = args.image.resolve(), args.audio.resolve(), args.output.resolve()
    os.chdir(ROOT / "external" / "MuseTalk")
    render(image, audio, output)
