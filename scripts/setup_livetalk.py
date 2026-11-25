"""Install the pinned full-frame renderer and its inference checkpoints."""

import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "e0d555fc5edf8516fbed2c955f1c0f4b32476abb"
OMNI_REVISION = "1536bf31abaec74364fb7d5883470d5b23ffa7f8"
MODEL_REVISION = "be14ffdf356e73e328db88595ccf62e7e1374b74"
BASE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"


def main():
    source = ROOT / "external/LiveTalk"
    for path, url, revision in [
        (source, "https://github.com/GAIR-NLP/LiveTalk.git", SOURCE_REVISION),
        (source / "OmniAvatar", "https://github.com/Omni-Avatar/OmniAvatar.git", OMNI_REVISION),
    ]:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Unexpected source revision for {path.name}: {actual}")
    subprocess.run(["bash", "scripts/add_patch.sh"], cwd=source, check=True)
    for repository, revision, directory, patterns in [
        ("GAIR/LiveTalk-1.3B-V0.1", MODEL_REVISION, "model", ["model.safetensors", "README.md", "*LICENSE*"]),
        ("Wan-AI/Wan2.1-T2V-1.3B", BASE_REVISION, "base", [
            "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/*", "README.md", "*LICENSE*",
        ]),
        ("facebook/wav2vec2-base-960h", "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
         "wav2vec2", ["config.json", "preprocessor_config.json", "model.safetensors", "README.md"]),
    ]:
        destination = ROOT / "models/livetalk" / directory if directory != "wav2vec2" else ROOT / "models/wav2vec2"
        snapshot_download(repository, revision=revision, local_dir=destination, allow_patterns=patterns)
    print("LiveTalk full-frame model setup complete.", flush=True)


if __name__ == "__main__":
    main()
