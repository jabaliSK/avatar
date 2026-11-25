"""Download public model artifacts into this workspace, without changing system packages."""

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatar.config import Settings

MUSE_COMMIT = "2536eaf06b85f775d8e04b7fd81bf6699155a58d"


def download(url: str, target: Path, sha256: str | None = None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix(target.suffix + ".download")
        print(f"Downloading {target.name}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as output:
                import shutil

                shutil.copyfileobj(source, output)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256:
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest() if sys.version_info >= (3, 11) else None
        if digest is None:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != sha256:
            raise RuntimeError(f"Checksum mismatch: {target}. Remove this file and retry setup.")


def main():
    from huggingface_hub import hf_hub_download, snapshot_download

    s = Settings.from_env()
    muse = ROOT / "external" / "MuseTalk"
    if not muse.exists():
        muse.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/TMElyralab/MuseTalk.git", str(muse)], check=True)
        subprocess.run(["git", "-C", str(muse), "checkout", "--detach", MUSE_COMMIT], check=True)
    revision = subprocess.check_output(["git", "-C", str(muse), "rev-parse", "HEAD"], text=True).strip()
    if revision != MUSE_COMMIT:
        raise RuntimeError(f"Expected MuseTalk {MUSE_COMMIT}, found {revision}; refusing to alter existing checkout.")
    artifacts = muse / "models"
    for repo, destination, names in [
        ("TMElyralab/MuseTalk", artifacts, [
            "musetalkV15/musetalk.json", "musetalkV15/unet.pth", "LICENSE.txt", "NOTICE",
        ]),
        ("stabilityai/sd-vae-ft-mse", artifacts / "sd-vae", ["config.json", "diffusion_pytorch_model.bin"]),
        ("openai/whisper-tiny", artifacts / "whisper", ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
    ]:
        for name in names:
            print(f"Downloading {repo}/{name}", flush=True)
            hf_hub_download(repo, name, local_dir=destination)
    # This mirror supplies the same parsing checkpoint linked by upstream.
    parser = artifacts / "face-parse-bisent"
    checkpoint = Path(hf_hub_download(
        "camenduru/MuseTalk", "face-parse-bisent/79999_iter.pth", local_dir=artifacts,
    ))
    download(
        checkpoint.as_uri(), checkpoint,
        "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567",
    )
    download(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        parser / "resnet18-f37072fd.pth",
    )
    face = ROOT / "models" / "torch" / "hub" / "checkpoints"
    download(
        "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip",
        face / "2DFAN4-cd938726ad.zip",
    )
    download(
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        face / "s3fd-619a316812.pth",
    )
    if not s.stt_base:
        snapshot_download(
            s.stt_model, local_dir=ROOT / "models/stt",
            allow_patterns=["*.json", "*.bin", "vocabulary.*", "README.md"],
        )
    if not s.llm_base:
        snapshot_download(
            s.llm_model, local_dir=ROOT / "models/llm",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "LICENSE", "README.md"],
        )
    if not s.tts_base:
        for name in ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json", "MODEL_CARD"]:
            hf_hub_download(
                "rhasspy/piper-voices", f"en/en_US/lessac/medium/{name}",
                local_dir=ROOT / "models" / "piper-download",
            )
            source = ROOT / "models/piper-download/en/en_US/lessac/medium" / name
            destination = ROOT / "models" / "tts" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copyfile(source, destination)
    print("Model setup complete. Start: .venv/bin/python run.py", flush=True)


if __name__ == "__main__":
    main()
