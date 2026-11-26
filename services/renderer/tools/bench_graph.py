"""Micro-benchmark: warp+decoder TensorRT engines, eager enqueue vs CUDA-graph replay.

  python tools/bench_graph.py --engine-dir /engines/sm90_trt10.11.0.33
"""
import argparse
import time

import numpy as np
import torch

from app.trt_models import TrtRenderer


def timed(fn, n, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    r = TrtRenderer(args.engine_dir)
    dev = r.device
    rng = np.random.default_rng(0)
    f_s = torch.from_numpy(rng.standard_normal((1, 32, 16, 64, 64)).astype(np.float16)).to(dev)
    x_s = (rng.standard_normal((1, 21, 3)) * 0.1).astype(np.float32)
    x_d = x_s + (rng.standard_normal((1, 21, 3)) * 0.01).astype(np.float32)

    eager = lambda: r(f_s, x_s, x_d)
    ref = eager().clone()
    ms_eager = timed(eager, args.n)

    f_buf = f_s.clone()
    kp_buf = torch.from_numpy(np.concatenate([x_s, x_d], 0)).to(dev)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            feat = r.warp(feature_3d=f_buf, kp_source=kp_buf[0:1], kp_driving=kp_buf[1:2])["out"]
            r.decoder(feature=feat)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        feat = r.warp(feature_3d=f_buf, kp_source=kp_buf[0:1], kp_driving=kp_buf[1:2])["out"]
        out = r.decoder(feature=feat)["output"]

    def replay():
        f_buf.copy_(f_s)
        kp_buf.copy_(torch.from_numpy(np.concatenate([x_s, x_d], 0)), non_blocking=False)
        graph.replay()
        return out

    got = replay().clone()
    torch.cuda.synchronize()
    err = (got - ref).abs()
    ms_graph = timed(replay, args.n)

    # Split: engine time alone, no input copies.
    ms_replay_only = timed(graph.replay, args.n)
    print(f"eager  warp+decoder : {ms_eager:7.2f} ms")
    print(f"graph  warp+decoder : {ms_graph:7.2f} ms (with input copies), {ms_replay_only:7.2f} ms replay only")
    print(f"graph vs eager output: max abs {err.max().item():.5f}, mean abs {err.mean().item():.6f}")


if __name__ == "__main__":
    main()
