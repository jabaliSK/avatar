"""Time the released MP4 pipeline without changing the live avatar service."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
UPSTREAM = ROOT / "external" / "AvatarForever"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=153)
    parser.add_argument("--no-fast-infer", dest="fast", action="store_false")
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(UPSTREAM))
    import inference as upstream
    import torch

    torch.cuda.set_per_process_memory_fraction(0.79, 0)
    make_pipeline = upstream.ARA2VidDistilledPipeline
    encode = upstream.encode_video
    pipeline = None
    start = 0.0
    timings = {}

    def cached_pipeline(**kwargs):
        nonlocal pipeline
        if pipeline is None:
            pipeline = make_pipeline(loras=[], **kwargs)
        return pipeline

    def timed_encode(*, video, **kwargs):
        torch.cuda.synchronize()
        timings["decode_iterator_ready_seconds"] = round(time.perf_counter() - start, 3)

        def timed_video():
            for frames in video:
                torch.cuda.synchronize()
                elapsed = round(time.perf_counter() - start, 3)
                timings.setdefault("first_decoded_batch_seconds", elapsed)
                timings.setdefault("batches", []).append([int(frames.shape[0]), elapsed])
                yield frames

        return encode(video=timed_video(), **kwargs)

    upstream.ARA2VidDistilledPipeline = cached_pipeline
    upstream.encode_video = timed_encode
    source = ROOT / "models" / "avatarforever"
    prompt = (
        "Fixed camera, single continuous full-body shot, head and both shoes always in frame. "
        "Keep the exact reference composition, clothing, and background. "
        "The person speaks naturally with occasional small hand gestures and shifts of posture. "
        + upstream.DEFAULT_PROMPT
    )
    for turn in range(args.repeat):
        output = args.output if turn == 0 else args.output.with_stem(args.output.stem + "-warm")
        sys.argv = [
            "inference.py",
            "--distilled-checkpoint-path", str(source / "avatarforever-ltx-2.3-22b.safetensors"),
            "--gemma-root", str(source / "gemma-3-12b-it-qat-q4_0-unquantized"),
            "--audio-path", str(args.audio.resolve()),
            "--first-frame-condition-image-path", str(args.image.resolve()),
            "--height", str(args.height), "--width", str(args.width),
            "--num-frames", str(args.frames), "--output-path", str(output.resolve()),
            "--temporal-tile-size", "48", "--ar-history-feature-cache", "--prompt", prompt,
            "--fast-infer" if args.fast else "--no-fast-infer",
        ]
        torch.cuda.reset_peak_memory_stats()
        timings = {"turn": turn + 1, "output": str(output), "frames": args.frames}
        start = time.perf_counter()
        upstream.main()
        timings["wall_seconds"] = round(time.perf_counter() - start, 3)
        timings["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / (2**30), 2)
        timings["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (2**30), 2)
        print("BENCHMARK " + json.dumps(timings), flush=True)


if __name__ == "__main__":
    main()
