"""Locate the TensorRT engines for the GPU in use, building them on first start.

Engines are specific to the GPU architecture and TensorRT version, so they live in <root>/sm<CC>_trt<VERSION>/ and a new
GPU model (for example an L40S, sm89) triggers a one-off rebuild. On the GPU this was developed on, the three engines the
live path needs take roughly 20-25 minutes to build.
"""
import logging
import os
import subprocess
import sys

import tensorrt as trt
import torch

log = logging.getLogger("kka.engines")

# name in tools/build_engines.py SPECS -> file it writes
REQUIRED = {
    "warp_network": "warp_network_fp16.engine",
    "decoder": "decoder_fp16.engine",
    "lmdm_fp16": "lmdm_v0.4_hubert_fp16.engine",
}
BUILDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "build_engines.py")


def engine_tag():
    cc = torch.cuda.get_device_capability()
    return f"sm{cc[0]}{cc[1]}_trt{trt.__version__}"


def ensure_engines(root, onnx_dir, build=True):
    """Return the engine directory for this GPU, building any missing engine first."""
    directory = os.path.join(root, engine_tag())
    missing = [name for name, fname in REQUIRED.items() if not os.path.isfile(os.path.join(directory, fname))]
    if not missing:
        return directory
    if not build:
        raise FileNotFoundError(f"TensorRT engines {missing} are missing in {directory}")
    log.warning("TensorRT engines %s are missing for %s; building them now. This runs once per GPU model and takes "
                "about 20-25 minutes; the service reports 'loading' until it is done.", missing, engine_tag())
    subprocess.run([sys.executable, BUILDER, "--onnx-dir", onnx_dir, "--out-dir", root, "--only", ",".join(missing)], check=True)
    return directory
