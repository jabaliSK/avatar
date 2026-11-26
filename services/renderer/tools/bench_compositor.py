"""Micro-benchmark of the compositor on registered avatars: GPU blend, device-to-host read-back and CPU paste, per frame.

The head render is replaced by random pixels (the cost does not depend on their values), so no engine is needed, only the GPU and
each avatar's cached speaking loop. Run it while the renderer is idle: two processes on one GPU time-slice each other.

  docker run --rm --gpus '"device=0"' --entrypoint python -v $PWD/avatars:/avatars:ro -v $PWD/services/renderer/tools:/tools:ro \
      kka-renderer:latest /tools/bench_compositor.py gulf-man-white seated-man
"""
import argparse
import os
import time

import numpy as np
import torch

from app.avatar import Loop
from app.compositor import PatchReader, blend_roi, paste

WARMUP = 20  # the first frames pay for allocator and pinned-buffer set-up


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("avatars", nargs="+")
    ap.add_argument("--root", default="/avatars")
    ap.add_argument("--frames", type=int, default=400, help="frames to time per avatar (the loop is cycled)")
    args = ap.parse_args()

    render = torch.rand(1, 3, 512, 512, device="cuda")
    reader = PatchReader()
    print(f"{'':<24}{'':>11}{'':>11}  {'mean ms per frame':^30}  {'total ms per frame':^30}")
    print(f"{'avatar':<24}{'frame':>11}{'mean ROI':>11}  {'GPU blend':>10}{'read-back':>10}{'CPU paste':>10}  {'median':>7}{'mean':>7}{'p95':>7}{'> 5 ms':>8}")
    for name in args.avatars:
        directory = os.path.join(args.root, name)
        state = "speaking" if os.path.isdir(os.path.join(directory, "speaking")) else "still"
        loop = Loop.load(os.path.join(directory, state))
        blend, read, cpu = [], [], []
        for i in range(args.frames + WARMUP):
            roi, frame = loop.rois[i % len(loop)], loop.frames[i % len(loop)]
            t0 = time.perf_counter()
            patch = blend_roi(render, roi)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            host = reader.read(patch)
            t2 = time.perf_counter()
            paste(frame, host, roi)
            t3 = time.perf_counter()
            if i >= WARMUP:
                blend.append((t1 - t0) * 1e3)
                read.append((t2 - t1) * 1e3)
                cpu.append((t3 - t2) * 1e3)
        total = np.array(blend) + np.array(read) + np.array(cpu)
        h, w = np.mean([r.shape for r in loop.rois], axis=0).round().astype(int)
        fw, fh = loop.size
        print(f"{name:<24}{f'{fw}x{fh}':>11}{f'{w}x{h}':>11}  {np.mean(blend):>10.2f}{np.mean(read):>10.2f}{np.mean(cpu):>10.2f}  "
              f"{np.median(total):>7.2f}{np.mean(total):>7.2f}{np.percentile(total, 95):>7.2f}{100 * np.mean(total > 5):>7.1f}%")
        del loop


if __name__ == "__main__":
    main()
