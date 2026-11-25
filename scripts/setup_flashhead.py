"""Install pinned FlashHead source and download Pro or Lite inference weights."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "9bc03de06bb0de82cd6bc477804512ae06144bf2"
MODEL_REVISION = "74d81686a7bd750b276ec1ade295bc4077c5868a"
WAV2VEC_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def main(model_type="pro"):
    from huggingface_hub import snapshot_download

    source = ROOT / "external" / "FlashHead"
    if not source.exists():
        source.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Soul-AILab/SoulX-FlashHead.git", str(source)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", SOURCE_REVISION], check=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"Unexpected FlashHead source revision: {revision}")
    components = ["Model_Pro/*", "VAE_Wan/*"] if model_type == "pro" else ["Model_Lite/*", "VAE_LTX/*"]
    snapshot_download(
        "Soul-AILab/SoulX-FlashHead-1_3B",
        revision=MODEL_REVISION,
        local_dir=ROOT / "models" / "flashhead",
        allow_patterns=[*components, "README.md", "*LICENSE*", "*NOTICE*"],
    )
    snapshot_download(
        "facebook/wav2vec2-base-960h",
        revision=WAV2VEC_REVISION,
        local_dir=ROOT / "models" / "wav2vec2",
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"],
    )
    print(f"FlashHead {model_type} model setup complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["pro", "lite"], default="pro")
    main(parser.parse_args().model_type)
