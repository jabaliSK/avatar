"""Measure how a recorded session moves: mouth opening, head against its surroundings, body motion and state changes.

(lipsync_probe.py measures sync; this measures the look.) Record with ws_client.py, which writes the state of every frame with --meta:

  python tools/ws_client.py --url ws://127.0.0.1:8290/ws --avatar NAME --audio reply.wav --out reply.mp4 --meta reply.json --lead-in 4 --tail 4
  docker run --rm --entrypoint python -v $PWD/work:/work -v $PWD/models/ditto-talkinghead:/models/ditto-talkinghead:ro \
      -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest /tools/motion_probe.py /work/reply.mp4 --meta /work/reply.json

Frames are grouped as `speech` (frames that carry audio), `tail` (the silent frames the player appends so a gesture can finish) and
`quiet` (idle / listening / thinking). Reported per group:

  lip_gap, jaw_open   inner-lip gap / face height and the jawOpen blendshape (0..1): p50, p95, max. How wide the mouth opens.
  face_vs_surround    speed of the face relative to the hair, neck and clothes around it, px per frame at a 1280 px tall frame: p50, p95,
                      and `range_px`, the peak-to-peak of that speed summed over the group. A rigid face that slides over a picture that
                      does not move with it scores high; a head that moves together with its body scores about zero.
                      (Assumes the surroundings of the face belong to the person, not to a static backdrop.)
  body_motion         mean |difference| of consecutive frames (0..255) in the part of the picture below the chin. 0 = a frozen photo.
  head_size_pulse_pct how far the distance between the outer eye corners strays from its median over +-0.3 s, in percent: p95 and max.
                      A head that swells for a few frames (a template whose crop followed its jaw) scores several percent; under 2 is steady.
  blinks_per_min      eye closures (blendshape > 0.5) in the quiet frames.
  transitions         every change of state: the largest change of a two-frame average within +-0.4 s as a multiple of the median during
                      speech. About 1 is invisible; several times that is a jump.
"""
import argparse
import json
import os
import sys

import cv2
import mediapipe as mp  # noqa: F401  (imported by lipsync_probe too; fail early with a clear error)
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lipsync_probe import CHIN, DEFAULT_TASK, FACE_TOP, LIP_BOTTOM, LIP_TOP, detect, locate_face  # noqa: E402

SCALE = 0.5  # optical flow runs at half resolution
QUIET = ("idle", "listening", "thinking")
EYE_CORNERS = (33, 263)  # FaceMesh outer eye corners: they do not move with the jaw


def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def track(frames, task, min_height):
    """Per frame: face box (px), inner-lip gap / face height, jawOpen, blink score, outer-eye-corner distance (px). NaN where no face was found."""
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task), output_face_blendshapes=True, num_faces=1)
    n = len(frames)
    h, w = frames[0].shape[:2]
    box, lips, jaw, blink = np.full((n, 4), np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    eyes = np.full(n, np.nan)
    with vision.FaceLandmarker.create_from_options(options) as lm:
        crop = locate_face(lm, frames[n // 3:n // 3 + 40]) or locate_face(lm, frames)
        if crop is None:
            raise SystemExit("no face found in the recording")
        cw, ch = (crop[2] - crop[0]) * w, (crop[3] - crop[1]) * h
        for i, frame in enumerate(frames):
            found = detect(lm, frame, crop, min_height)
            if found is None:
                continue
            marks, blend, _ = found
            xs = np.array([crop[0] * w + p.x * cw for p in marks])
            ys = np.array([crop[1] * h + p.y * ch for p in marks])
            box[i] = (xs.min(), ys.min(), xs.max(), ys.max())
            face_h = abs(marks[CHIN].y - marks[FACE_TOP].y)
            lips[i] = abs(marks[LIP_BOTTOM].y - marks[LIP_TOP].y) / max(face_h, 1e-6)
            jaw[i] = blend.get("jawOpen", np.nan)
            blink[i] = max(blend.get("eyeBlinkLeft", 0.0), blend.get("eyeBlinkRight", 0.0)) if blend else np.nan
            a, b = (marks[k] for k in EYE_CORNERS)
            eyes[i] = float(np.hypot((a.x - b.x) * cw, (a.y - b.y) * ch))
    return box, lips, jaw, blink, eyes


def size_pulse(eyes, half=7):
    """Percent each frame's eye-corner distance differs from the median of the frames around it (NaN where unknown)."""
    out = np.full(len(eyes), np.nan)
    for i in range(len(eyes)):
        around = eyes[max(0, i - half):i + half + 1]
        around = around[~np.isnan(around)]
        if len(around) and not np.isnan(eyes[i]):
            out[i] = abs(eyes[i] / np.median(around) - 1.0) * 100.0
    return out


def fill_box(box):
    good = ~np.isnan(box[:, 0])
    if not good.any():
        raise SystemExit("no face found in the recording")
    idx = np.arange(len(box))
    return np.stack([np.interp(idx, idx[good], box[good, c]) for c in range(4)], 1)


def region(shape, box, lo, hi=None):
    """Mask of `box` scaled by `lo` about its centre (half-resolution coordinates); with `hi`, the ring between scale `lo` and `hi`."""
    h, w = shape
    cx, cy = (box[0] + box[2]) / 2 * SCALE, (box[1] + box[3]) / 2 * SCALE
    bw, bh = (box[2] - box[0]) * SCALE, (box[3] - box[1]) * SCALE

    def rect(s):
        m = np.zeros(shape, bool)
        m[int(max(cy - bh * s / 2, 0)):int(min(cy + bh * s / 2, h)), int(max(cx - bw * s / 2, 0)):int(min(cx + bw * s / 2, w))] = True
        return m

    return rect(lo) if hi is None else rect(hi) & ~rect(lo)


def motion(frames, box):
    """Per consecutive pair: whole-frame change, body change (below the chin), face-minus-surround flow (px at full resolution)."""
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) for f in frames]
    shape = gray[0].shape
    total, body, rel = [], [], []
    for i in range(len(gray) - 1):
        b = box[i]
        diff = np.abs(gray[i + 1].astype(np.float32) - gray[i].astype(np.float32))
        total.append(float(diff.mean()))
        below = np.zeros(shape, bool)
        below[int((b[3] + 0.15 * (b[3] - b[1])) * SCALE):, :] = True
        body.append(float(diff[below].mean()))
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        face, ring = region(shape, b, 0.6), region(shape, b, 1.15, 1.45)
        if face.any() and ring.any():
            rel.append((np.median(flow[face], 0) - np.median(flow[ring], 0)) / SCALE)
        else:
            rel.append(np.zeros(2))
    return np.array(total), np.array(body), np.array(rel)


def stats(x, qs=(50, 95)):
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    out = {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs}
    out["max"] = round(float(x.max()), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--meta", required=True, help="the JSON ws_client.py wrote with --meta")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--min-height", type=int, default=512)
    ap.add_argument("--json", default="", help="also write the report here")
    args = ap.parse_args()

    frames = read_frames(args.video)
    meta = json.load(open(args.meta))
    states = meta["states"][:len(frames)]
    n = len(states)
    done = [e for e in meta["events"] if e.get("type") == "speak_done"]
    speech_n = round(done[-1]["audio_s"] * meta.get("fps", 25)) if done else None
    group = []
    seen_speaking = 0
    for s in states:
        if s in QUIET:
            group.append("quiet")
        else:
            group.append("speech" if speech_n is None or seen_speaking < speech_n else "tail")
            seen_speaking += 1
    group = np.array(group)

    box, lips, jaw, blink, eyes = track(frames, args.task, args.min_height)
    pulse = size_pulse(eyes)
    box = fill_box(box)
    face_h_px = float(np.median(box[:, 3] - box[:, 1]))
    unit = 1280.0 / frames[0].shape[0]  # report flow in px of a 1280 px tall frame
    total, body, rel = motion(frames, box)
    rel = rel * unit

    report = {"video": args.video, "frames": n, "frames_per_group": {g: int((group == g).sum()) for g in ("quiet", "speech", "tail")}}
    for g in ("speech", "tail", "quiet"):
        sel = np.flatnonzero(group[:-1] == g)  # pair i covers frames i -> i+1
        selv = np.flatnonzero(group == g)
        if not len(selv):
            continue
        entry = {"lip_gap": stats(lips[selv]), "jaw_open": stats(jaw[selv])}
        entry["head_size_pulse_pct"] = stats(pulse[selv], qs=(95,))
        if len(sel) > 2:
            speed = np.linalg.norm(rel[sel], axis=1)
            walk = np.cumsum(rel[sel], 0)
            entry["face_vs_surround"] = {"p50": round(float(np.percentile(speed, 50)), 3), "p95": round(float(np.percentile(speed, 95)), 3),
                                         "range_px": round(float(np.ptp(walk, 0).max()), 1)}
            entry["body_motion"] = round(float(body[sel].mean()), 3)
        if g == "quiet":
            closed = np.nan_to_num(blink[selv]) > 0.5
            starts = int(np.sum(closed[1:] & ~closed[:-1]))
            entry["blinks_per_min"] = round(starts * 60.0 * meta.get("fps", 25) / len(selv), 1)
        report[g] = entry

    # A frame that is late is shown twice and the next one jumps ahead, so the change of a single frame says little about the motion:
    # judge two-frame averages (the baseline numbers are re-scored the same way).
    per_tick = np.convolve(total, [0.5, 0.5], "same")
    speech_sel = np.flatnonzero(group[:-1] == "speech")
    scale = float(np.median(per_tick[speech_sel])) if len(speech_sel) else float(np.median(per_tick))
    transitions = []
    for i in range(1, n):
        if states[i] != states[i - 1]:
            lo, hi = max(0, i - 10), min(len(per_tick), i + 10)
            peak = lo + int(np.argmax(per_tick[lo:hi]))
            transitions.append({"at_s": round(i / meta.get("fps", 25), 2), "from": states[i - 1], "to": states[i],
                                "spike": round(float(per_tick[peak] / max(scale, 1e-6)), 2), "peak_at_s": round((peak + 1) / meta.get("fps", 25), 2)})
    report["transitions"] = transitions
    report["median_frame_change_during_speech"] = round(scale, 3)
    report["face_height_px"] = round(face_h_px, 1)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
