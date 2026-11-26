"""Measure how far the live face strays from the template's own face at the same template frame: the head wobbling over its body.

Every speech frame is the template frame `t` (found by matching the body, which is copied from the template untouched) with the
live head pasted over the face. The hair, headdress and neck are therefore the template's own, so any difference between the live
face's landmarks and the template face's landmarks at frame `t` is the face sliding, swelling or turning against its surroundings.

Record with ws_client.py (`--meta`, `--lead-in 4 --tail 4`; see motion_probe.py), then run in the renderer image (MediaPipe):

  docker run --rm --entrypoint python -v $PWD/work/run1:/work -v $PWD/avatars:/avatars:ro -v $PWD/services/renderer/tools:/tools:ro \
      -v $PWD/models/ditto-talkinghead:/models/ditto-talkinghead:ro kka-renderer:latest \
      /tools/wobble_probe.py /work/reply.mp4 --meta /work/reply.json --template /avatars/gulf-man-black/speaking/frames.npy --json /work/wobble.json

Reported under `deviation_live_minus_template`, live minus template, over the speech frames:

  eyes, nose, brows, forehead, cheeks   groups of FaceMesh points the mouth does not move. `hf_p50` / `hf_p95`: how far the group's
                      position strays from its own median over +-7 frames, in px of a 1280 px tall frame (the sliding that the eye
                      sees). `mean_dx_dy` / `std_dx_dy` / `range_dx_dy` are the offset itself. Steady is about 1.5 px or less at p95.
  jaw, lips           the same for the mouth, which is audio-driven: expect more.
  roll_deg            the line between the eyes, live minus template, in degrees (mean, std, hf_p95).
  width_pct           the face width (FaceMesh 234-454, cheek to cheek) as a percentage of the template's. `std` and `hf_p95` are the
                      swelling and shrinking of the head. The jaw contour is part of it, so it grows a little with the mouth opening.
  scale_pct           the same for the outer eye corners (33-263). Blinks move those points, so read `width_pct` first.

`--dump` also writes the landmark arrays (npz: `out`, `ref`, `dev`, `idx`, `sp`) for further analysis. Frames that repeat or jump
because the renderer was late (`index_steps`) raise every figure equally, so compare recordings of one GPU in one state.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lipsync_probe import DEFAULT_TASK, detect, locate_face  # noqa: E402

# FaceMesh indices that the mouth does not move: outer / inner eye corners, nose bridge and tip, brow, forehead, cheek bones
GROUPS = {
    "eyes": [33, 133, 362, 263],
    "nose": [6, 168, 1, 4],
    "brows": [70, 105, 300, 334],
    "forehead": [10, 109, 338],
    "cheeks": [50, 280, 205, 425],
    "jaw": [152, 148, 377, 172, 397],
    "lips": [13, 14, 61, 291],
}


def landmarks(frames_bgr, task, crop, min_height=512):
    n = len(frames_bgr)
    out = np.full((n, 478, 2), np.nan, np.float32)
    h, w = frames_bgr[0].shape[:2]
    cw, ch = (crop[2] - crop[0]) * w, (crop[3] - crop[1]) * h
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task), output_face_blendshapes=False, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as lm:
        for i, f in enumerate(frames_bgr):
            found = detect(lm, f, crop, min_height)
            if found is None:
                continue
            marks = found[0]
            out[i, :, 0] = [crop[0] * w + p.x * cw for p in marks]
            out[i, :, 1] = [crop[1] * h + p.y * ch for p in marks]
    return out


def match_template(frame_bgr, tmpl_gray, y0):
    """Index of the template frame whose body (rows from y0 down, quarter size) is closest to this frame."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g[y0:], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
    d = np.abs(tmpl_gray - g[None]).mean(axis=(1, 2))
    return int(np.argmin(d)), float(d.min())


def highpass(x, half=7):
    """x minus its moving median over +-half frames, per column (NaN-safe)."""
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - half):i + half + 1]
        out[i] = x[i] - np.nanmedian(seg, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--template", required=True, help="frames.npy of the speaking loop")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--json", default="")
    ap.add_argument("--dump", default="", help="write the per-frame deviation arrays here (npz)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    meta = json.load(open(args.meta))
    states = meta["states"][:len(frames)]
    done = [e for e in meta["events"] if e.get("type") == "speak_done"]
    speech_n = round(done[-1]["audio_s"] * meta.get("fps", 25)) if done else 10 ** 9
    sp = [i for i, s in enumerate(states) if s == "speaking"][:speech_n]
    qu = [i for i, s in enumerate(states) if s in ("idle", "listening", "thinking")]
    tmpl = np.load(args.template, mmap_mode="r")
    n_t = len(tmpl)
    H = frames[0].shape[0]
    tg = np.stack([cv2.resize(cv2.cvtColor(np.ascontiguousarray(t), cv2.COLOR_RGB2GRAY)[int(0.45 * H):], None, fx=0.25, fy=0.25,
                              interpolation=cv2.INTER_AREA) for t in tmpl]).astype(np.float32)
    idx, err = zip(*[match_template(frames[i], tg, int(0.45 * H)) for i in sp])
    idx, err = np.array(idx), np.array(err)
    print(f"speech frames {len(sp)}; template {n_t} frames; body match error median {np.median(err):.2f} max {err.max():.2f}")
    print("template index of the first 60 speech frames:", idx[:60].tolist())
    d = np.diff(idx)
    print("index steps: repeat %d, +1 %d, -1 %d, other %d (max |step| %d)" % ((d == 0).sum(), (d == 1).sum(), (d == -1).sum(),
                                                                          ((np.abs(d) > 1)).sum(), np.abs(d).max()))

    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=args.task), output_face_blendshapes=False, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as lm:
        crop = locate_face(lm, [frames[sp[len(sp) // 2]]]) or locate_face(lm, frames)
    out_lm = landmarks([frames[i] for i in sp], args.task, crop)
    t_bgr = [cv2.cvtColor(np.ascontiguousarray(t), cv2.COLOR_RGB2BGR) for t in tmpl]
    t_lm = landmarks(t_bgr, args.task, crop)
    ref = t_lm[idx]  # template landmarks at the matched index, [K, 478, 2]
    dev = out_lm - ref

    report = {"video": args.video, "speech_frames": len(sp), "index_steps": {"repeat": int((d == 0).sum()), "fwd": int((d == 1).sum()),
                                                                         "back": int((d == -1).sum()), "jump": int((np.abs(d) > 1).sum())}}
    unit = 1280.0 / H
    rows = {}
    for g, ids in GROUPS.items():
        v = np.nanmean(dev[:, ids, :], axis=1) * unit  # [K, 2] mean displacement of the group, px at 1280 tall
        hp = highpass(v)
        mag = np.linalg.norm(hp, axis=1)
        rows[g] = {"mean_dx_dy": [round(float(x), 2) for x in np.nanmean(v, 0)], "std_dx_dy": [round(float(x), 2) for x in np.nanstd(v, 0)],
                   "hf_p50": round(float(np.nanpercentile(mag, 50)), 2), "hf_p95": round(float(np.nanpercentile(mag, 95)), 2),
                   "range_dx_dy": [round(float(np.nanmax(v[:, c]) - np.nanmin(v[:, c])), 1) for c in range(2)]}
    # roll and scale of the eyes line, live minus template
    def eye_line(lm):
        a = lm[:, [33, 133], :].mean(1)
        b = lm[:, [263, 362], :].mean(1)
        v = b - a
        return np.degrees(np.arctan2(v[:, 1], v[:, 0])), np.linalg.norm(v, axis=1)
    def width(lm):
        return np.linalg.norm(lm[:, 234, :] - lm[:, 454, :], axis=1)

    r_o, s_o = eye_line(out_lm)
    r_t, s_t = eye_line(ref)
    roll, scale = r_o - r_t, (s_o / s_t - 1.0) * 100.0
    wide = (width(out_lm) / width(ref) - 1.0) * 100.0

    def series(x):
        return {"mean": round(float(np.nanmean(x)), 2), "std": round(float(np.nanstd(x)), 2),
                "hf_p95": round(float(np.nanpercentile(np.abs(highpass(x[:, None])[:, 0]), 95)), 2)}

    rows["roll_deg"] = series(roll)
    rows["width_pct"] = series(wide)
    rows["scale_pct"] = series(scale)
    report["deviation_live_minus_template"] = rows
    # the template's own motion of the same landmarks over the played indices, for scale
    own = np.nanmean(ref[:, GROUPS["eyes"] + GROUPS["nose"], :], axis=1) * unit
    report["template_own_motion_px_range"] = [round(float(np.nanmax(own[:, c]) - np.nanmin(own[:, c])), 1) for c in range(2)]
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    if args.dump:
        np.savez(args.dump, idx=idx, dev=dev, out=out_lm, ref=ref, sp=np.array(sp))


if __name__ == "__main__":
    main()
