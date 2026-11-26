#!/usr/bin/env python3
"""Fetch model weights into ./models with explicit allow-lists.

Only components whose licence is Apache-2.0 or MIT are fetched. Deliberately absent:
  * FunAudioLLM/SenseVoiceSmall      - weights are under the FunASR "model-license" (HF: license=other)
  * SoulX-FlashHead Model_Lite/VAE_LTX - LTX-0.9 VAE is Open-RAIL-M
Revisions are pinned to the commit SHAs verified on 2026-11-25.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# name -> (repo, revision, local dir, allow_patterns, licence)
ITEMS = {
    "qwen3": (
        "Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3", "qwen3-8b-fp8", ["*"], "Apache-2.0"),
    "flashhead-pro": (  # Pro only; the card's licence tag is "other" because Lite/VAE_LTX (Open-RAIL-M) share the repo
        "Soul-AILab/SoulX-FlashHead-1_3B", "74d81686a7bd750b276ec1ade295bc4077c5868a", "SoulX-FlashHead-1_3B",
        ["README.md", "Model_Pro/*", "VAE_Wan/*"], "Apache-2.0"),
    "wav2vec2-960h": (
        "facebook/wav2vec2-base-960h", "22aad52d435eb6dbaf354bdad9b0da84ce7d6156", "wav2vec2-base-960h",
        ["config.json", "model.safetensors", "preprocessor_config.json", "README.md"], "Apache-2.0"),
    "whisper": (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        "faster-whisper-large-v3-turbo", ["*"], "MIT"),
    "kokoro": (
        "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", "Kokoro-82M",
        ["config.json", "kokoro-v1_0.pth", "voices/*", "README.md", "LICENSE*"], "Apache-2.0"),
    "echomimic-flash": (
        "BadToBest/EchoMimicV3", "311e176905a8c4c24b240b530488fe636ce4d249", "echomimic_v3/flash/transformer",
        ["echomimicv3-flash-pro/*"], "Apache-2.0"),
    "wan-base": (
        "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "fc913c34361f4ec879e2f9c78b4f11ae50a937d1",
        "echomimic_v3/flash/Wan2.1-Fun-V1.1-1.3B-InP", ["*"], "Apache-2.0"),
    "cn-wav2vec2": (
        "TencentGameMate/chinese-wav2vec2-base", "3991242c806928916fff4a8c0e4f76acf661b743",
        "echomimic_v3/flash/chinese-wav2vec2-base",
        ["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"], "MIT"),
    # Ditto's repo is Apache-2.0 but its HF bundle also ships InsightFace-derived detectors/landmarkers
    # (non-commercial). They are never fetched: see DENY below. Prebuilt TRT engines and the prebuilt
    # grid-sample plugin are skipped too (built for TRT 8.6 and unusable on this driver); we rebuild them.
    "ditto": (
        "digital-avatar/ditto-talkinghead", "e4a2f60328ee7c32af585ac4b3cce299e4c8e254", "ditto-talkinghead",
        ["LICENSE", "README.md", "ditto_cfg/*", "ditto_pytorch/models/*",
         "ditto_pytorch/aux_models/landmark203.onnx", "ditto_pytorch/aux_models/face_landmarker.task",
         "ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx",
         "ditto_onnx/appearance_extractor.onnx", "ditto_onnx/decoder.onnx", "ditto_onnx/hubert.onnx",
         "ditto_onnx/landmark203.onnx", "ditto_onnx/lmdm_v0.4_hubert.onnx", "ditto_onnx/motion_extractor.onnx",
         "ditto_onnx/stitch_network.onnx", "ditto_onnx/warp_network.onnx"], "Apache-2.0"),
}

# Never download these, whatever the allow-list says (non-commercial / non-Apache-MIT assets).
DENY = ["*insightface*", "*landmark106*", "*2d106det*", "*det_10g*", "*VAE_LTX*", "*Model_Lite*", "*SenseVoice*"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", default=list(ITEMS), help=f"subset of {list(ITEMS)}")
    ap.add_argument("--force", action="store_true", help="re-download even if the target directory is populated")
    args = ap.parse_args()
    for name in args.items:
        repo, rev, sub, patterns, lic = ITEMS[name]
        dest = MODELS / sub
        if dest.is_dir() and any(dest.iterdir()) and not args.force:
            print(f"[{name}] present -> skip ({dest})", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] {repo}@{rev or 'main'} ({lic}) -> {dest}", flush=True)
        snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=patterns, ignore_patterns=DENY)
    # EchoMimicV3 expects flash/transformer/diffusion_pytorch_model.safetensors directly.
    nested = MODELS / "echomimic_v3/flash/transformer/echomimicv3-flash-pro"
    if nested.is_dir():
        for f in nested.iterdir():
            f.replace(nested.parent / f.name)
        nested.rmdir()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
