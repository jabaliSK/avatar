"""Measure how well the mouth follows the sound in a recorded reply.

MediaPipe FaceLandmarker (Apache-2.0) tracks the mouth opening in every video frame (inner-lip gap relative to face height, and the
`jawOpen` blendshape). Both are cross-correlated with the loudness of the audio track. The lag of the correlation peak is how far the
mouth is behind the sound (positive) or ahead of it (negative); `--max-lag` frames are searched in each direction.

  docker run --rm --entrypoint python -v $PWD/work:/work -v $PWD/models/ditto-talkinghead:/models/ditto-talkinghead:ro \
      -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest /tools/lipsync_probe.py /work/final/live/seated-man.mp4

The face is located automatically. When it is too small for MediaPipe's face detector (128 px input), pass a tight --crop around the head
(fractions x0,y0,x1,y1 of the frame, for example 0.50,0,0.68,0.30).

How to read the result (one entry each for `lip_gap` and `jaw_open`):
  lag_ms               where the correlation peaks, in ms. Positive: the mouth is behind the sound. Negative: it is ahead of it.
  lag_ms_90pct         90% interval of that peak from resampling the clip in one-second blocks, searched only around the peak.
  corr_at_peak         correlation at the peak. A mouth follows phonemes, not loudness, so 0.2-0.4 is normal: read the peak as a lag, not a score.
  corr_best_elsewhere  highest correlation more than 3 frames from the peak. It should be clearly lower than corr_at_peak; if it is close
                       the peak is ambiguous (a clip with few syllables or a face that is not tracked well) and the lag should not be trusted.
The resolution is one video frame (40 ms at 25 fps) and the clip needs a few seconds of continuous speech. Check against ground truth:
delaying the audio of a recording by exactly 200 ms with ffmpeg (`-af adelay=200`) moved the reading by -191 and -195 ms (lip gap and
jaw). A mouth up to 45 ms behind the sound or up to 125 ms ahead of it is not noticed by viewers (ITU-R BT.1359). The recording's audio is
AAC, whose encoder delay ffmpeg removes when decoding.
"""
import argparse
import json
import subprocess

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_TASK = "/models/ditto-talkinghead/ditto_pytorch/aux_models/face_landmarker.task"
# FaceMesh indices: inner lip top/bottom, forehead top, chin.
LIP_TOP, LIP_BOTTOM, FACE_TOP, CHIN = 13, 14, 10, 152


def read_audio(path, rate=24000):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def detect(landmarker, frame_bgr, crop, min_height):
    """Landmarks and blendshapes of the face in (a crop of) a BGR frame, or None."""
    if crop:
        h, w = frame_bgr.shape[:2]
        frame_bgr = frame_bgr[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
    if frame_bgr.shape[0] < min_height:
        scale = min_height / frame_bgr.shape[0]
        frame_bgr = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    blend = {b.category_name: b.score for b in result.face_blendshapes[0]} if result.face_blendshapes else {}
    return result.face_landmarks[0], blend, frame_bgr.shape[0]


def locate_face(landmarker, frames, margin=0.5):
    """Crop box (fractions of the frame) around the first face found in the frames; the box has `margin` of the face size on each side."""
    for frame in frames:
        found = detect(landmarker, frame, None, 0)
        if found:
            xs, ys = [p.x for p in found[0]], [p.y for p in found[0]]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            return [max(0.0, min(xs) - margin * w), max(0.0, min(ys) - margin * h), min(1.0, max(xs) + margin * w), min(1.0, max(ys) + margin * h)]
    return None


def track_mouth(path, task, crop, min_height):
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task), output_face_blendshapes=True, num_faces=1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    lips, jaw = [], []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        if crop is None:
            # Frames from the middle of the clip are speaking frames; the first ones may show a different (idle) pose.
            crop = locate_face(landmarker, frames[len(frames) // 3:len(frames) // 3 + 40])
            if crop is None:
                raise SystemExit("no face found in the full frame (the face is probably small): pass --crop x0,y0,x1,y1")
        for frame in frames:
            found = detect(landmarker, frame, crop, min_height)
            if found is None:
                lips.append(np.nan)
                jaw.append(np.nan)
                continue
            lm, blend, height = found
            face_h = abs(lm[CHIN].y - lm[FACE_TOP].y) * height
            lips.append(abs(lm[LIP_BOTTOM].y - lm[LIP_TOP].y) * height / max(face_h, 1e-6))
            jaw.append(blend.get("jawOpen", np.nan))
    return np.array(lips), np.array(jaw), fps, crop


def fill_gaps(x):
    bad = np.isnan(x)
    if bad.all():
        return x
    x = x.copy()
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def smooth(x):
    return np.convolve(x, np.array([0.25, 0.5, 0.25]), mode="same")


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


REACH = 3  # frames around the correlation peak that count as its main lobe (3 frames = 120 ms at 25 fps)


def parabolic_peak(corr, lo, hi):
    """Lag in frames of the highest correlation within lo..hi, refined to a fraction of a frame by a parabola through its neighbours."""
    k = max(range(lo, hi + 1), key=lambda i: corr[i])
    frac = 0.0
    if lo < k < hi:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    return k, frac


def best_lag(audio, mouth, max_lag):
    """corr[k] = correlation of audio[t] with mouth[t + k]; k > 0 means the mouth is late."""
    corr = {}
    for k in range(-max_lag, max_lag + 1):
        a, m = (audio[:len(audio) - k], mouth[k:]) if k >= 0 else (audio[-k:], mouth[:len(mouth) + k])
        corr[k] = float(np.mean(zscore(a) * zscore(m))) if len(a) > 8 else float("nan")
    k, frac = parabolic_peak(corr, -max_lag, max_lag)
    return k, frac, corr


def lag_interval(audio, mouth, k0, fps, block=25, rounds=300):
    """90% interval in ms of the main-lobe peak (k0 +- REACH frames), from resampling whole one-second blocks of the clip.

    Blocks are resampled whole because the traces are correlated over time. The search stays in the main lobe: a ten-second clip has too
    few syllables for a resample to tell the lobe from its neighbours, which is why `corr_best_elsewhere` is reported separately.
    """
    a, m = zscore(audio), zscore(mouth)
    blocks = len(a) // block
    lags = list(range(k0 - REACH, k0 + REACH + 1))
    products, counts = np.zeros((blocks, len(lags))), np.zeros((blocks, len(lags)))
    for b in range(blocks):
        ab, mb = a[b * block:(b + 1) * block], m[b * block:(b + 1) * block]
        for i, k in enumerate(lags):
            x, y = (ab[:block - k], mb[k:]) if k >= 0 else (ab[-k:], mb[:block + k])
            products[b, i], counts[b, i] = float(np.dot(x, y)), len(x)
    rng = np.random.default_rng(0)
    found = []
    for _ in range(rounds):
        pick = rng.integers(0, blocks, blocks)
        corr = dict(zip(lags, products[pick].sum(0) / counts[pick].sum(0)))
        k, frac = parabolic_peak(corr, lags[0], lags[-1])
        found.append((k + frac) * 1000 / fps)
    return [round(float(v)) for v in np.percentile(found, [5, 95])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--crop", default="", help="x0,y0,x1,y1 as fractions of the frame")
    ap.add_argument("--min-height", type=int, default=512, help="upscale the (cropped) frame to at least this height before detection")
    ap.add_argument("--max-lag", type=int, default=10, help="frames searched in each direction")
    ap.add_argument("--json", default="", help="also write the report here")
    args = ap.parse_args()
    crop = [float(v) for v in args.crop.split(",")] if args.crop else None

    lips, jaw, fps, crop = track_mouth(args.video, args.task, crop, args.min_height)
    pcm, rate = read_audio(args.video)
    n = len(lips)
    per = int(rate / fps)
    loud = np.array([np.sqrt(np.mean(pcm[i * per:(i + 1) * per] ** 2)) if (i + 1) * per <= len(pcm) else 0.0 for i in range(n)])
    speech = np.flatnonzero(loud > 0.05 * loud.max())
    if len(speech) < 20:
        raise SystemExit("not enough speech in the recording to measure sync")
    lo, hi = max(0, speech[0] - 5), min(n, speech[-1] + 6)

    report = {"video": args.video, "frames": n, "fps": fps, "face_crop": [round(v, 3) for v in crop],
              "speech_window_s": [round(lo / fps, 2), round(hi / fps, 2)], "face_found_pct": round(100 * float(np.mean(~np.isnan(lips[lo:hi]))), 1)}
    for name, signal in (("lip_gap", lips), ("jaw_open", jaw)):
        if np.isnan(signal[lo:hi]).mean() > 0.3:
            report[name] = "face not found in most frames (try a tighter --crop)"
            continue
        audio, mouth = smooth(loud[lo:hi]), smooth(fill_gaps(signal[lo:hi]))
        k, frac, corr = best_lag(audio, mouth, args.max_lag)
        elsewhere = [v for i, v in corr.items() if abs(i - k) > REACH]
        report[name] = {"lag_ms": round((k + frac) * 1000 / fps), "lag_ms_90pct": lag_interval(audio, mouth, k, fps),
                        "corr_at_peak": round(corr[k], 3), "corr_best_elsewhere": round(max(elsewhere), 3) if elsewhere else None,
                        "corr_at_zero": round(corr[0], 3),
                        "mouth_p5_p95": [round(float(np.nanpercentile(signal[lo:hi], 5)), 3), round(float(np.nanpercentile(signal[lo:hi], 95)), 3)]}
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
