"""Minimal GPU-native TensorRT runner (torch tensors in, torch tensors out, no host round-trips).

Ditto's own TRTWrapper copies every input from a numpy array and every output back to host memory on the default stream.
This runner binds torch device pointers directly, preallocates outputs per input-shape signature and enqueues on the
caller's current CUDA stream, so a chain of engines and torch ops needs no synchronisation until the final host copy.
"""
import ctypes
import os

import tensorrt as trt
import torch

_LOGGER = trt.Logger(trt.Logger.WARNING)
_plugins_ready = False

_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def init_plugins(plugin_so=None):
    """Register TensorRT's bundled plugins and preload the GridSample3D plugin (it is not serialised into engines)."""
    global _plugins_ready
    if _plugins_ready:
        return
    trt.init_libnvinfer_plugins(_LOGGER, "")
    so = plugin_so or os.environ.get("GRID_SAMPLE_3D_PLUGIN")
    if so and os.path.isfile(so):
        ctypes.CDLL(so)
    _plugins_ready = True


class TRTModule:
    def __init__(self, engine_path, plugin_so=None, device="cuda"):
        init_plugins(plugin_so)
        with open(engine_path, "rb") as f, trt.Runtime(_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path}; rebuild with tools/build_engines.py on this GPU")
        self.context = self.engine.create_execution_context()
        self.device = device
        self.path = engine_path
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dtype = {n: _TO_TORCH[self.engine.get_tensor_dtype(n)] for n in names}
        self._dynamic = {n: -1 in tuple(self.engine.get_tensor_shape(n)) for n in self.inputs}
        self._outs = {}

    def __call__(self, **tensors):
        key = tuple(tuple(t.shape) for t in tensors.values())
        keep = []  # hold references until the enqueue has been issued
        for name in self.inputs:
            t = tensors[name]
            if t.dtype != self.dtype[name]:
                t = t.to(self.dtype[name])
            if not t.is_contiguous():
                t = t.contiguous()
            keep.append(t)
            if self._dynamic[name]:
                self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
        outs = self._outs.get(key)
        if outs is None:
            outs = {n: torch.empty(tuple(self.context.get_tensor_shape(n)), dtype=self.dtype[n], device=self.device)
                    for n in self.outputs}
            self._outs[key] = outs
        for n, t in outs.items():
            self.context.set_tensor_address(n, t.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"TensorRT enqueue failed for {self.path}")
        return outs
