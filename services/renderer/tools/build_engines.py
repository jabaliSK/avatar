"""Build the per-frame TensorRT engines for the live head renderer from the Ditto ONNX files.

Engines are built natively for the GPU in use (no hardware-compatibility mode) and cached under
<out-dir>/sm<CC>_trt<VERSION>/. The GridSample3D plugin is loaded at build time but NOT serialised into the engine; the
runtime preloads the same .so (see app/trt_runtime.py).

  python tools/build_engines.py --onnx-dir /models/ditto-talkinghead/ditto_onnx --out-dir /engines [--only decoder,warp_network]
"""
import argparse
import ctypes
import json
import os
import time

import torch  # noqa: F401  (loads CUDA libs first)
import tensorrt as trt

SPECS = {
    "warp_network": dict(onnx="warp_network.onnx", out="warp_network_fp16.engine", fp16=True, plugin=True),
    "decoder": dict(onnx="decoder.onnx", out="decoder_fp16.engine", fp16=True),
    "stitch_network": dict(onnx="stitch_network.onnx", out="stitch_network_fp16.engine", fp16=True),
    "lmdm_fp32": dict(onnx="lmdm_v0.4_hubert.onnx", out="lmdm_v0.4_hubert_fp32.engine", fp16=False),
    "lmdm_fp16": dict(onnx="lmdm_v0.4_hubert.onnx", out="lmdm_v0.4_hubert_fp16.engine", fp16=True),
}

# Shape/bookkeeping layers that must not be pinned to a precision when forcing GridSample to fp32.
_UNCONSTRAINED = {"SHAPE", "ASSERTION", "SHUFFLE", "IDENTITY", "CONSTANT", "CONCATENATION", "GATHER", "SLICE",
                  "CONDITION", "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT", "FILL", "NON_ZERO", "ONE_HOT"}


def build(onnx_path, engine_path, fp16, opt_level, workspace_gb, cache_path, plugin=None):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    if plugin:
        ctypes.CDLL(plugin)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise RuntimeError(f"cannot parse {onnx_path}")

    config = builder.create_builder_config()
    config.builder_optimization_level = opt_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if plugin:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if str(layer.type)[10:] in _UNCONSTRAINED:
                continue
            if "GridSample" in layer.name:
                layer.precision = trt.float32
    cache_bytes = open(cache_path, "rb").read() if os.path.isfile(cache_path) else b""
    cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(cache, False)

    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    with open(engine_path + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(engine_path + ".tmp", engine_path)
    with open(cache_path, "wb") as f:
        f.write(memoryview(cache.serialize()))
    return os.path.getsize(engine_path)


def record(manifest_path, header, name, entry, overwrite=True):
    """Merge one engine entry into the manifest. It is re-read on every call so concurrent builders keep each other's entries."""
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest.update(header)
    engines = manifest.setdefault("engines", {})
    if overwrite or name not in engines:
        engines[name] = entry
    tmp = f"{manifest_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="/models/ditto-talkinghead/ditto_onnx")
    ap.add_argument("--out-dir", default="/engines")
    ap.add_argument("--plugin", default=os.environ.get("GRID_SAMPLE_3D_PLUGIN", "/opt/trt-plugins/libgrid_sample_3d_plugin.so"))
    ap.add_argument("--only", default="", help="comma separated subset of: " + ",".join(SPECS))
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--workspace-gb", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cc = torch.cuda.get_device_capability()
    tag = f"sm{cc[0]}{cc[1]}_trt{trt.__version__}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, f"timing_{tag}.cache")
    manifest_path = os.path.join(out_dir, "manifest.json")
    header = {"tensorrt": trt.__version__, "compute_capability": f"{cc[0]}.{cc[1]}",
              "device": torch.cuda.get_device_name(0), "opt_level": args.opt_level}

    names = [n for n in args.only.split(",") if n] or list(SPECS)
    for name in names:
        spec = SPECS[name]
        onnx_path = os.path.join(args.onnx_dir, spec["onnx"])
        engine_path = os.path.join(out_dir, spec["out"])
        entry = {"file": spec["out"], "fp16": spec["fp16"], "onnx_bytes": os.path.getsize(onnx_path)}
        if os.path.isfile(engine_path) and not args.force:
            print(f"[skip] {name}: {engine_path} exists", flush=True)
            record(manifest_path, header, name, {**entry, "bytes": os.path.getsize(engine_path)}, overwrite=False)
            continue
        print(f"[build] {name}  fp16={spec['fp16']}  opt={args.opt_level}", flush=True)
        t0 = time.perf_counter()
        size = build(onnx_path, engine_path, spec["fp16"], args.opt_level, args.workspace_gb, cache_path,
                     plugin=args.plugin if spec.get("plugin") else None)
        dt = time.perf_counter() - t0
        record(manifest_path, header, name, {**entry, "bytes": size, "build_s": round(dt, 1)})
        print(f"[done] {name}: {size / 1e6:.0f} MB in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
