"""Contact sheet of a recorded clip, for checking a reply (pose, gestures, mouth, seams) without a video player.

  docker run --rm --entrypoint python -v $PWD/work:/work -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest \
      /tools/contact_sheet.py /work/final/live/gulf-man-white.mp4 /work/final/live/gulf-man-white_sheet.jpg --frames 12 --start 2 --end 12

--crop takes x0,y0,x1,y1 as fractions of the frame, for example 0.15,0.02,0.85,0.30 for the head of a portrait, so mouth shapes can be
compared at full size. Every tile is labelled with its time in seconds.
"""
import argparse
import math

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=12, help="number of tiles, evenly spaced between --start and --end")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=0.0, help="seconds, 0 = end of the clip")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile-height", type=int, default=420)
    ap.add_argument("--crop", default="", help="x0,y0,x1,y1 as fractions of the frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = max(0, int(args.start * fps))
    last = min(total - 1, int(args.end * fps)) if args.end else total - 1
    if last < first:
        raise SystemExit(f"{args.video}: {total} frames at {fps:g} fps, nothing between {args.start}s and {args.end}s")
    wanted = set(np.linspace(first, last, max(1, args.frames)).round().astype(int).tolist())
    crop = [float(v) for v in args.crop.split(",")] if args.crop else None
    if crop and (len(crop) != 4 or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1)):
        raise SystemExit("--crop needs x0,y0,x1,y1 as fractions with x0<x1 and y0<y1")

    tiles = []
    for index in range(last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if index not in wanted:
            continue
        if crop:
            h, w = frame.shape[:2]
            frame = frame[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
        scale = args.tile_height / frame.shape[0]
        tile = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), args.tile_height), interpolation=cv2.INTER_AREA)
        label = f"{index / fps:.2f}s"
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise SystemExit(f"{args.video}: no frames could be read")

    cols = max(1, min(args.cols, len(tiles)))
    width = max(t.shape[1] for t in tiles)
    rows = []
    for r in range(math.ceil(len(tiles) / cols)):
        row = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in tiles[r * cols:(r + 1) * cols]]
        row += [np.zeros_like(row[0])] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(args.out, np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {args.out} ({len(tiles)} tiles from {total} frames at {fps:g} fps)")


if __name__ == "__main__":
    main()
