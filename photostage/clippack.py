"""Make an avatar's clip pack from the body clips a video model made of it, and install it where the player finds it.

    .venv/bin/python -m photostage.clippack <avatar key> --model wan-i2v-a14b-lightning

Reads the runs `photostage.clipgen` left in data/clips/<avatar>/ (clip.mp4 and info.json), judges every clip against `Gates`
and prints the numbers, puts the light of the clips that pass right (a video model lets the exposure drift), widens a
full-avatar clip that shows only part of the photo to the whole of it with the photo's own edge, closes the
loops of the thinking clips, finds the poses close to the photo's at which a clip can end, follows the neck through each
clip (the live head is moved with it, see clips.js), and writes clips.json and clip-<name>.mp4/.webm into the avatar's
folder. A clip that fails a gate is left out, unless it is named with --force. It needs no GPU and can run while the
server is up: a player page that is reloaded plays the new pack. --dry-run only prints the numbers; --remove takes the
pack away.
"""

import argparse
import filecmp
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np

from photostage import clipgen, clips, guide

DEFAULT_MODEL = "wan-i2v-a14b-lightning"
TYPES = {"mp4": 'video/mp4; codecs="avc1.640029"', "webm": 'video/webm; codecs="vp9"'}
ENCODERS = {
    "mp4": ("libx264", ["-crf", "18", "-preset", "slow", "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart"]),
    "webm": ("libvpx-vp9", ["-crf", "28", "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "3"]),
}
NAME = re.compile(r"[a-z0-9-]+")
STALE = re.compile(r"clip-[a-z0-9-]+\.(mp4|webm)")
RUN_FIELDS = {"model", "gesture", "seed", "frames", "fps", "region", "work", "band", "face_h"}

# The light of a frame is put right by a gain and a bias per colour channel; a fit that asks for more than this is a clip
# that is wrong, not one that has drifted, and is left alone.
MAX_GAIN, MAX_BIAS = 0.25, 40.0
# The first frame of the clip may differ from the picture the model was given by this much (levels) before the clip is
# taken not to be of this photo.
SAME_PHOTO = 2.0
# A loop's last frame is laid over its first one over this many seconds.
CLOSE_SECONDS = 1.0
# Exits: a clip can end where its pose is within the larger of EXIT_LEVEL and EXIT_SHARE of its peak distance from rest,
# at a local minimum, no sooner than EXIT_SKIP after it begins, and no closer together than EXIT_SPACING.
EXIT_LEVEL, EXIT_SHARE, EXIT_SKIP, EXIT_SPACING = 2.0, 0.2, 0.4, 0.8
TRACK_SIGMA = 1.2  # frames
# A full-avatar clip widened to the whole photo fades into the photo's own edge over this many work pixels.
SEAT_FEATHER = 8


@dataclass(frozen=True)
class Gates:
    """What a clip has to be, in the units of `clips.measure` (levels are 0-255)."""

    fidelity: float = 28.0  # least dB of agreement of the first frame with the picture the model was given
    band_lost: float = 0.15  # most of the frames in which the neck cannot be followed
    tone_drift: float = 10.0  # most the mean colour may wander once the light has been put right
    softening: float = 0.5  # least sharpness of the softest frame against the first
    flicker: float = 3.0  # most shimmer
    gap: float = 4.0  # most the last frame may differ from the first: a clip has to come back to the pose it began in
    motion: float = 2.5  # least a gesture must move at its peak
    calm: float = 10.0  # most a thinking clip may move at its peak: it plays on a loop
    exit_gap: float = 2.5  # longest a thinking clip goes without a pose it can be let go in, in seconds


def fit_levels(source, target, step=4):
    """The gain and bias per channel that bring `source` onto `target` (both (h, w, 3) arrays), a robust fit that takes
    no notice of the pixels that really differ between them, as where a hand has moved."""
    x = source[::step, ::step].reshape(-1, 3).astype(np.float64)
    y = target[::step, ::step].reshape(-1, 3).astype(np.float64)
    gain, bias = np.ones(3), np.zeros(3)
    for c in range(3):
        a, b = 1.0, float(np.median(y[:, c] - x[:, c]))
        for _ in range(8):
            r = y[:, c] - (a * x[:, c] + b)
            spread = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1.0)
            w = np.clip(1 - (r / (4.685 * spread)) ** 2, 0, None) ** 2
            total = float(w.sum())
            if total < 16:
                break
            mx, my = float((w * x[:, c]).sum()) / total, float((w * y[:, c]).sum()) / total
            var = float((w * (x[:, c] - mx) ** 2).sum())
            a = float((w * (x[:, c] - mx) * (y[:, c] - my)).sum()) / var if var > 1e-3 * total else 1.0
            b = my - a * mx
        gain[c], bias[c] = a, b
    return gain, bias


def sensible(gain, bias):
    return bool(np.all(np.abs(gain - 1) <= MAX_GAIN) and np.all(np.abs(bias) <= MAX_BIAS))


def smoothed(values, sigma, wrap=False):
    """`values` (a 1-d series) blurred by a Gaussian of `sigma` samples."""
    radius = max(1, int(round(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    padded = np.pad(values, radius, mode="wrap" if wrap else "edge")
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")


def correct_tone(frames, reference):
    """The frames with their light put right: held to the first frame's through the clip (what drifts is the light, so
    the fit looks past a hand that moves), and the first frame brought onto `reference`, the picture the model was
    given. Returns the frames and {"drift": largest change of gain and bias through the clip, "first": the same for the
    first frame, "refused": whether a fit was too wild to use}."""
    count = len(frames)
    gains, biases = np.ones((count, 3)), np.zeros((count, 3))
    first = frames[0].astype(np.float32)
    refused = False
    for i in range(1, count):
        gain, bias = fit_levels(frames[i].astype(np.float32), first)
        if sensible(gain, bias):
            gains[i], biases[i] = gain, bias
        else:
            refused = True
    # What drifts, drifts slowly: a fit that jumps from frame to frame would be shimmer put in.
    for c in range(3):
        gains[1:, c], biases[1:, c] = smoothed(gains[1:, c], 1.5), smoothed(biases[1:, c], 1.5)
    base_gain, base_bias = fit_levels(first, reference.astype(np.float32))
    if not sensible(base_gain, base_bias):
        base_gain, base_bias, refused = np.ones(3), np.zeros(3), True
    out = np.empty_like(frames)
    for i in range(count):
        gain, bias = base_gain * gains[i], base_gain * biases[i] + base_bias
        out[i] = np.clip(frames[i] * gain + bias + 0.5, 0, 255).astype(np.uint8)
    drift = {"gain": float(np.abs(gains - 1).max()), "bias": float(np.abs(biases).max())}
    return out, {"drift": drift, "first": {"gain": float(np.abs(base_gain - 1).max()), "bias": float(np.abs(base_bias).max())},
                 "refused": refused}


def close_loop(frames, fps):
    """The frames with the difference between the last and the first spread over the last second, so that the last
    is the first: the clip is then a loop once that repeated frame is dropped."""
    count = len(frames)
    span = max(2, min(count // 2, round(CLOSE_SECONDS * fps)))
    delta = frames[0].astype(np.float32) - frames[-1]
    out = frames.copy()
    for k in range(1, span + 1):
        i = count - 1 - span + k
        t = k / span
        out[i] = np.clip(frames[i] + t * t * (3 - 2 * t) * delta + 0.5, 0, 255).astype(np.uint8)
    return out


def rest_curve(frames, rows):
    """How far each frame is from the first, in levels, below the head's seat: the pose's distance from the photo's."""
    low = np.stack([cv2.resize(clips.gray(frame)[rows[0]:], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) for frame in frames])
    low = low.astype(np.float32)
    return np.abs(low - low[0]).mean((1, 2))


def find_exits(rest, fps):
    """The times, in seconds, at which the pose is close to the photo's: the dips of the curve, calm enough, far enough
    from the start and from each other, the calmest first."""
    smooth = np.convolve(np.pad(rest, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    limit = max(EXIT_LEVEL, EXIT_SHARE * float(smooth.max()))
    dips = [(float(smooth[i]), i) for i in range(1, len(smooth) - 1)
            if i / fps >= EXIT_SKIP and smooth[i] <= limit and smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]]
    chosen = []
    for _, i in sorted(dips):
        if all(abs(i - j) / fps >= EXIT_SPACING for j in chosen):
            chosen.append(i)
    return [round(i / fps, 4) for i in sorted(chosen)]


def neck_track(frames, rows, scale, loop, columns=None):
    """Per frame [zoom, turn in radians, dx, dy in photo pixels] of the head's seat against the first frame, the holes
    filled and the jitter smoothed (a loop's track is a loop too)."""
    table = clips.band_motion(frames, rows, columns)
    count = len(table)
    out = np.empty((count, 4))
    for c, rest in enumerate((1.0, 0.0, 0.0, 0.0)):
        seen = ~np.isnan(table[:, c])
        series = np.interp(np.arange(count), np.flatnonzero(seen), table[seen, c]) if seen.any() else np.full(count, rest)
        out[:, c] = smoothed(series, TRACK_SIGMA, wrap=loop)
    out[:, 1] = np.radians(out[:, 1])
    out[:, 2:] /= scale
    return out


FACE_POINTS = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 1, 234, 454]


def face_pose(reference, points):
    """Fit eye corners, nose and cheeks; exclude lips, eyelids and the moving jaw."""
    if points is None:
        return None
    source = np.asarray(reference[FACE_POINTS], dtype=np.float64)
    target = np.asarray(points[FACE_POINTS], dtype=np.float64)
    design = np.column_stack([source, np.ones(len(source))])
    weights = np.ones(len(source))
    for _ in range(6):
        scale = np.sqrt(weights)
        fitted = np.linalg.lstsq(design * scale[:, None], target * scale[:, None], rcond=None)[0]
        residual = np.linalg.norm(design @ fitted - target, axis=1)
        median = np.median(residual)
        spread = max(1.4826 * np.median(np.abs(residual - median)), 0.5 / guide.SIZE)
        cutoff = 1.345 * spread
        weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
    matrix = fitted.T
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0.1:
        return None
    return matrix.reshape(-1)


class FaceRegistration:
    """Match both video and photo to the renderer's actual neutral face, not the video's first frame."""

    def __init__(self, reference, rig):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""
        try:
            import mediapipe as mp
        except ImportError as error:
            raise SystemExit("Full-avatar packs need MediaPipe; run clippack with .venv-flashhead/bin/python.") from error
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        self.reference = self.landmarks(reference)
        if self.reference is None:
            self.mesh.close()
            raise SystemExit("No clear face in the renderer's neutral reference; no pack was installed.")
        self.crop = np.array(rig["crop"], dtype=float)

    def landmarks(self, image):
        found = self.mesh.process(image)
        if not found.multi_face_landmarks:
            return None
        return np.array([(p.x, p.y) for p in found.multi_face_landmarks[0].landmark], dtype=np.float32)

    def mouth_region(self):
        points = self.reference
        lips = points[[61, 291, 0, 17]]
        low, high = lips.min(axis=0), lips.max(axis=0)
        width = high[0] - low[0]
        center = (low + high) / 2
        box = [low[0] - 0.2 * width, max(points[2, 1], low[1] - 0.15 * width),
               high[0] + 0.2 * width, min(points[152, 1] - 0.1 * width, high[1] + 0.35 * width)]
        if not (box[0] < center[0] < box[2] and box[1] < center[1] < box[3]):
            raise SystemExit("The mouth could not be separated from the nose and chin; no pack was installed.")
        return {"box": [round(float(v), 7) for v in box], "center": center.round(7).tolist()}

    def pose(self, frame, region):
        scale = np.array(frame.shape[1::-1]) / (np.array(region[2:]) - region[:2])
        extent = (self.crop[2:] - self.crop[:2]) * scale / guide.SIZE
        offset = (self.crop[:2] - region[:2]) * scale
        mapping = np.array([[extent[0], 0, offset[0]], [0, extent[1], offset[1]]], dtype=np.float32)
        head = cv2.warpAffine(frame, mapping, (guide.SIZE, guide.SIZE),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return face_pose(self.reference, self.landmarks(head))

    def track(self, frames, region):
        poses = [self.pose(frame, region) for frame in frames]
        seen = np.array([pose is not None for pose in poses])
        if not seen.any():
            return None, 1.0
        valid = np.stack([pose for pose in poses if pose is not None])
        track = np.column_stack([np.interp(np.arange(len(poses)), np.flatnonzero(seen), valid[:, c]) for c in range(6)])
        return track.round(7).tolist(), float((~seen).mean())


def judge(numbers, role, gap, exits, duration, gates):
    """Why a clip should not be used, as a list of sentences; empty where it passes."""
    problems = []
    if numbers["fidelity"] < gates.fidelity:
        problems.append(f"it does not start as the photo does (fidelity {numbers['fidelity']} dB < {gates.fidelity})")
    if numbers.get("band_lost", 1.0) > gates.band_lost:
        problems.append(f"the neck cannot be followed ({numbers.get('band_lost', 1.0)} of the frames lost > {gates.band_lost})")
    if numbers["tone_drift"] > gates.tone_drift:
        problems.append(f"the colours wander ({numbers['tone_drift']} levels > {gates.tone_drift})")
    if numbers["softening"] < gates.softening:
        problems.append(f"it blurs ({numbers['softening']} of the first frame's sharpness < {gates.softening})")
    if numbers["flicker"] > gates.flicker:
        problems.append(f"it shimmers (flicker {numbers['flicker']} > {gates.flicker})")
    if gap > gates.gap:
        problems.append(f"it does not come back to the pose it began in ({gap:.1f} levels > {gates.gap})")
    if role == "talk" and numbers["motion_peak"] < gates.motion:
        problems.append(f"it hardly moves (peak {numbers['motion_peak']} levels < {gates.motion})")
    if role in clips.LOOP_ROLES:
        if numbers["motion_peak"] > gates.calm:
            problems.append(f"it moves too much to be played on a loop (peak {numbers['motion_peak']} levels > {gates.calm})")
        stops = [0.0, *exits, duration]
        longest = max(b - a for a, b in zip(stops, stops[1:]))
        if longest > gates.exit_gap:
            problems.append(f"it has no calm moment to be let go at for {longest:.1f} s > {gates.exit_gap}")
    return problems


@dataclass
class Prepared:
    name: str
    role: str
    mood: str
    fps: int
    frames: np.ndarray
    entry: dict  # the clip's manifest entry, but for its files
    numbers: dict
    gap: float
    problems: list


def tracking_box(info, face_mode="full"):
    if info.get("framing") != "avatar":
        return None
    box = info.get("track_box")
    work = info["work"]
    if (not isinstance(box, list) or len(box) != 4 or
            not all(isinstance(v, int) for v in box) or
            not (0 <= box[0] < box[2] <= work[0] and 0 <= box[1] < box[3] <= work[1])):
        raise SystemExit("A full-avatar clip needs a valid track_box inside its video frame; regenerate it.")
    if face_mode == "head":
        # Anchor the replacement head to the collar, not the video's independently moving face.
        top = box[3]
        bottom = min(work[1], top + round(0.6 * (box[3] - box[1])))
        if bottom - top < 8:
            raise SystemExit("The full-head overlay needs a visible collar below the face.")
        return [box[0], top, box[2], bottom]
    return box


def full_seat(info, size):
    """A full-avatar run's plan moved onto the whole photo of `size`, and the work pixels (left, top, right, bottom) its
    frames need on each side for that; no pixels where it shows the whole photo already. The player and the dubbing take
    a full-avatar clip to be the whole photo, but a model may only make some frame shapes (Live Avatar's sides are
    multiples of 16, so it is shown a slightly narrower part of this photo)."""
    region, work = info["region"], info["work"]
    if info.get("framing") != "avatar":
        return info, None
    across, down = work[0] / (region[2] - region[0]), work[1] / (region[3] - region[1])
    pad = [round(region[0] * across), round(region[1] * down),
           round((size[0] - region[2]) * across), round((size[1] - region[3]) * down)]
    if not any(pad):
        return info, None
    # H.264 needs even sides.
    if (work[0] + pad[0] + pad[2]) % 2:
        pad[0 if pad[0] >= pad[2] else 2] += 1
    if (work[1] + pad[1] + pad[3]) % 2:
        pad[1 if pad[1] >= pad[3] else 3] += 1
    seated = {**info, "region": [0, 0, *size], "work": [work[0] + pad[0] + pad[2], work[1] + pad[1] + pad[3]],
              "band": [info["band"][0] + pad[1], info["band"][1] + pad[1]]}
    box = info.get("track_box")
    if isinstance(box, list) and len(box) == 4:
        seated["track_box"] = [v + pad[i % 2] if isinstance(v, int) else v for i, v in enumerate(box)]
    return seated, pad


def widen(frames, base, pad):
    """Frames (n, h, w, 3) set into `base`, the photo at the widened size, fading into it near each side it shows."""
    left, top, right, bottom = pad
    count, height, width = frames.shape[:3]

    def ramp(length, low, high):
        weight = np.ones(length, np.float32)
        fade = (np.arange(SEAT_FEATHER, dtype=np.float32) + 0.5) / SEAT_FEATHER
        if low:
            weight[:SEAT_FEATHER] = np.minimum(weight[:SEAT_FEATHER], fade[:length])
        if high:
            weight[-SEAT_FEATHER:] = np.minimum(weight[-SEAT_FEATHER:], fade[::-1][-length:])
        return weight

    alpha = (ramp(height, top, bottom)[:, None] * ramp(width, left, right)[None, :])[..., None]
    behind = base[top:top + height, left:left + width].astype(np.float32) * (1 - alpha)
    out = np.empty((count, *base.shape), np.uint8)
    for i, frame in enumerate(frames):
        out[i] = base
        out[i, top:top + height, left:left + width] = np.clip(frame * alpha + behind + 0.5, 0, 255)
    return out


def prepare(name, info, frames, reference, gates, registration=None, face_mode="full", photo=None):
    """A clip made ready: its light put right, widened to the whole photo if it is of the full avatar (with `photo`), a
    loop closed, its exits and its neck found, and judged."""
    role, mood = clips.ROLES[info["gesture"]]
    loop = role in clips.LOOP_ROLES
    frames, tone = correct_tone(frames, reference)
    gap = float(np.abs(frames[-1].astype(np.float32) - frames[0]).mean())
    if photo is not None:
        info, pad = full_seat(info, photo.shape[1::-1])
        if pad:
            # After the light is put right, so that the photo's own edge is not given the clip's correction.
            base = clips.crop(photo, info)
            frames, reference = widen(frames, base, pad), widen(reference[None], base, pad)[0]
    fps, rows = info["fps"], tuple(info["band"])
    if loop:
        frames = close_loop(frames, fps)[:-1]
    numbers = clips.measure(frames, reference, rows, info["face_h"])
    duration = len(frames) / fps
    exits = find_exits(rest_curve(frames, rows), fps)
    box = tracking_box(info, face_mode)
    track_rows, columns = ((box[1], box[3]), (box[0], box[2])) if box else (rows, None)
    if box:
        tracked = clips.band_motion(frames, track_rows, columns)
        numbers["band_lost"] = round(float(np.isnan(tracked[:, 0]).mean()), 4)
    region, work = info["region"], info["work"]
    scale = (work[0] / (region[2] - region[0]), work[1] / (region[3] - region[1]))
    track = neck_track(frames, track_rows, scale, loop, columns)
    face_track = None
    if registration is not None:
        face_track, lost = registration.track(frames, region)
        numbers["band_lost"] = max(numbers["band_lost"], lost)
    stats = {key: numbers[key] for key in ("fidelity", "softening", "flicker", "motion_peak", "tone_drift", "band_lost",
                                           "band_shift", "band_turn", "band_zoom") if key in numbers}
    stats.update({"model": info["model"], "gesture": info["gesture"], "seed": info["seed"], "gap": round(gap, 2),
                  "tone": {"refused": tone["refused"], "drift": {k: round(v, 3) for k, v in tone["drift"].items()}}})
    entry = {
        "role": role, "mood": mood, "duration": round(duration, 6), "fps": fps, "loop": loop, "exits": exits,
        "track": [[round(float(v), 5) for v in row] for row in track], "stats": stats,
    }
    if face_track is not None:
        entry["faceTrack"] = face_track
    return Prepared(name, role, mood, fps, frames, entry, numbers, gap, judge(numbers, role, gap, exits, duration, gates))


def geometry(rig, info, face_mode="full"):
    """What clips.json says about where the clips sit on the photo; the same for every clip made from one plan."""
    region, work, band = info["region"], info["work"], info["band"]
    scale = work[1] / (region[3] - region[1])
    face = rig["face"]
    result = {
        "size": list(rig["size"]), "box": list(region),
        "pivot": [round(region[0] + work[0] / 2 / scale, 2), round(region[1] + (band[0] + band[1]) / 2 / scale, 2)],
        "follow": [round(face[1], 2), round(face[3], 2)],
    }
    if info.get("framing") == "avatar":
        box = tracking_box(info, face_mode)
        result.update(mode="avatar", face=list(face),
                      pivot=[round(region[0] + (box[0] + box[2]) / 2 * (region[2] - region[0]) / work[0], 2),
                             round(region[1] + (box[1] + box[3]) / 2 / scale, 2)])
    return result


def find_runs(root, model):
    """The runs of `model` in `root` as [(folder, info)], in order of name; with no model named, those of the one there is."""
    runs = []
    for path in sorted(root.glob("*/info.json")) if root.is_dir() else []:
        try:
            info = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(info, dict) and RUN_FIELDS <= set(info):
            runs.append((path.parent, info))
    models = sorted({info["model"] for _, info in runs})
    if model is None and len(models) > 1:
        raise SystemExit(f"{root} has clips of several models ({', '.join(models)}); name one with --model.")
    chosen = model or (models[0] if models else DEFAULT_MODEL)
    return [(folder, info) for folder, info in runs if info["model"] == chosen]


def clip_name(folder, info):
    """A run's name in the pack: its folder's, without the model's, which every clip of a pack shares."""
    name = folder.name
    return name[len(info["model"]) + 1:] if name.startswith(info["model"] + "-") else name


def load_run(run, info, photo):
    """The run's frames and the picture the model was given, or a sentence on why it is not a clip of this photo."""
    plan = {"region": info["region"], "work": info["work"]}
    try:
        reference = cv2.imread(str(run / "first.png"))
        frames, _ = clips.read_video(run / "clip.mp4")
        given = clips.crop(photo, plan)
    except (OSError, ValueError, cv2.error) as error:
        return f"it cannot be read ({error})"
    if reference is None:
        return "it has no first.png"
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    if reference.shape != given.shape or np.abs(reference.astype(np.float32) - given).mean() > SAME_PHOTO:
        return "it was not made from this avatar's photo"
    if len(frames) != info["frames"] or frames.shape[1:3] != reference.shape[:2]:
        return "its video is not the clip that was made"
    return frames, reference


def table(rows):
    header = ("clip", "role", "mood", "frames", "fidelity", "lost", "drift", "soft", "flicker", "peak", "gap", "exits", "verdict")
    cells = [header] + rows
    widths = [max(len(str(row[i])) for row in cells) for i in range(len(header) - 1)]
    return "\n".join("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) + "  " + str(row[-1]) for row in cells)


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("avatar", help="the avatar's key (its folder in data/photostage) or a folder with photo.png and rig.json")
    parser.add_argument("--model", help=f"whose clips to use (default: the only model in the clips folder, else {DEFAULT_MODEL})")
    parser.add_argument("--clips", type=Path, help="the folder of runs (default: data/clips/<avatar>)")
    parser.add_argument("--only", nargs="+", metavar="CLIP", default=[], help="use only these clips")
    parser.add_argument("--force", nargs="+", metavar="CLIP", default=[], help="use these clips even if they fail a gate")
    parser.add_argument("--gate", action="append", default=[], metavar="NAME=VALUE", help="change a gate, e.g. --gate softening=0.4")
    parser.add_argument("--dry-run", action="store_true", help="judge the clips and print the numbers, install nothing")
    parser.add_argument("--face-mode", choices=["full", "mouth", "head"], default="full",
                        help="full-avatar composition: facial reenactment (full), mouth only, or a replacement head over a masked video")
    parser.add_argument("--remove", action="store_true", help="take the avatar's clip pack away")
    args = parser.parse_args(argv)
    gates, names = Gates(), {field.name for field in fields(Gates)}
    for item in args.gate:
        key, _, value = item.partition("=")
        if key not in names:
            parser.error(f"no gate {key!r}; the gates are {', '.join(sorted(names))}")
        try:
            gates = replace(gates, **{key: float(value)})
        except ValueError:
            parser.error(f"{item!r} is not NAME=NUMBER")
    args.gates = gates
    return args


def remove(folder):
    """Takes the pack away, its clips.json first, so that no page that loads meanwhile is shown half of it."""
    gone = [path for path in folder.iterdir() if path.name == "clips.json" or STALE.fullmatch(path.name)]
    for path in sorted(gone, key=lambda p: p.name != "clips.json"):
        path.unlink()
    print(f"Removed {len(gone)} files of the clip pack from {folder}." if gone else f"{folder} has no clip pack.")


def install(folder, manifest, encoded):
    """Puts the clips and then clips.json in place, each file whole or not at all, and removes the files of an older pack."""
    for name, path in encoded.items():
        target = folder / name
        if target.is_file() and filecmp.cmp(path, target, shallow=False):
            # MuseTalk's prepared materials are keyed by the source file's modification time.
            path.unlink()
        else:
            os.replace(path, target)
    pending = folder / "clips.json.new"
    pending.write_text(json.dumps(manifest, separators=(",", ":")))
    os.replace(pending, folder / "clips.json")
    for path in folder.iterdir():
        if STALE.fullmatch(path.name) and path.name not in encoded:
            path.unlink()


def main(argv=None):
    args = parse(argv)
    folder = clipgen.avatar_folder(args.avatar)
    if args.remove:
        remove(folder)
        return 0
    rig = json.loads((folder / "rig.json").read_text())
    photo = cv2.cvtColor(cv2.imread(str(folder / "photo.png")), cv2.COLOR_BGR2RGB)
    if list(photo.shape[1::-1]) != list(rig["size"]):
        raise SystemExit(f"The photo is {photo.shape[1]}x{photo.shape[0]} but the rig was made for {rig['size'][0]}x{rig['size'][1]}.")
    root = args.clips or clipgen.OUTPUT / folder.name
    runs = find_runs(root, args.model)
    if not runs:
        raise SystemExit(f"There are no clips to use in {root}; make some with photostage.clipgen.")

    wanted = [(run, info, clip_name(run, info)) for run, info in runs
              if (not args.only or clip_name(run, info) in args.only) and info["gesture"] in clips.ROLES]
    # One pack sits on the photo in one way; if the runs were made for several, those of the most common are used.
    seats = [(geometry(rig, seated, args.face_mode), seated["work"])
             for seated in (full_seat(info, rig["size"])[0] for _, info, _ in wanted)]
    seat = max(seats, key=seats.count, default=None)
    registration, home_pose, mouth = None, None, None
    if args.face_mode == "mouth" and (not seat or seat[0].get("mode") != "avatar"):
        raise SystemExit("Mouth-only reenactment requires full-avatar clips made with --framing avatar.")
    if args.face_mode == "head" and (not seat or seat[0].get("mode") != "avatar"):
        raise SystemExit("The full-head overlay requires full-avatar clips made with --framing avatar.")
    if seat and seat[0].get("mode") == "avatar" and args.face_mode != "head":
        reference = cv2.imread(str(folder / "ref.png"))
        if reference is None or reference.shape != (guide.SIZE, guide.SIZE, 3):
            raise SystemExit("Full-avatar face registration needs the avatar's prepared 512px ref.png.")
        registration = FaceRegistration(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), rig)
        home_pose = registration.pose(photo, [0, 0, *rig["size"]])
        if home_pose is None:
            registration.mesh.close()
            raise SystemExit("The renderer's neutral face could not be registered to the photo; no pack was installed.")
        if args.face_mode == "mouth":
            mouth = registration.mouth_region()
    made, used, rows = [], [], []
    for (run, info, name), here in zip(wanted, seats):
        if here != seat:
            print(f"{name}: left out, it was made for another part of the photo than most of the clips.")
            continue
        loaded = load_run(run, info, photo) if NAME.fullmatch(name) else "its name must be lowercase letters, digits and dashes"
        if isinstance(loaded, str):
            print(f"{name}: left out, {loaded}.")
            continue
        clip = prepare(name, info, *loaded, args.gates, registration, args.face_mode, photo)
        n = clip.numbers
        verdict = "ok" if not clip.problems else ("FORCED: " if name in args.force else "left out: ") + "; ".join(clip.problems)
        rows.append((name, clip.role, clip.mood, len(clip.frames), n["fidelity"], n.get("band_lost", "-"), n["tone_drift"], n["softening"],
                     n["flicker"], n["motion_peak"], round(clip.gap, 1), len(clip.entry["exits"]), verdict))
        fit = not clip.problems or name in args.force
        if fit:
            used.append(clip)
        if args.dry_run or not fit:
            clip.frames = None  # a clip is a few hundred megabytes of frames
        made.append(clip)
    if registration is not None:
        registration.mesh.close()
    print(table(rows))
    if not used:
        print("No clip is fit to be used; nothing was installed.")
        return 1
    if not any(clip.role == "think" for clip in used):
        print("Warning: there is no thinking clip; another quiet role is used if available.")
    if not any(clip.role == "talk" for clip in used):
        print("Warning: there is no gesture clip, so nothing will play while a reply is spoken.")
    if args.dry_run:
        print(f"{len(used)} of {len(made)} clips would be installed.")
        return 0

    manifest = {"version": 1, **seat[0], "types": TYPES, "clips": {}}
    if seat[0].get("mode") == "avatar":
        manifest["faceMode"] = args.face_mode
    if home_pose is not None:
        manifest["facePose"] = home_pose.round(7).tolist()
        manifest["faceAnchors"] = registration.reference[FACE_POINTS].round(7).tolist()
    if mouth is not None:
        manifest["mouth"] = mouth
    with tempfile.TemporaryDirectory(prefix=".clips-", dir=folder) as scratch:
        encoded = {}
        for clip in used:
            files = {}
            for kind, (codec, params) in ENCODERS.items():
                files[kind] = f"clip-{clip.name}.{kind}"
                encoded[files[kind]] = Path(scratch) / files[kind]
                clips.encode_video(encoded[files[kind]], clip.frames, clip.fps, codec, params)
            manifest["clips"][clip.name] = {**clip.entry, "files": files}
        install(folder, manifest, encoded)
    size = sum((folder / name).stat().st_size for name in encoded) / 1e6
    print(f"Installed {len(used)} clips ({size:.1f} MB) in {folder}; reload the player to see them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
