"""Body clips: what an image-to-video model is shown of a photo, and how the video it makes is judged.

A body clip is a few seconds of the person's body gesturing, made from the photo by an image-to-video model; the live head
is meant to be drawn over the clip's head. Nothing here needs a GPU or a model. `plan_region` says which part of the photo
the model is shown and at what size, `GESTURES` are the prompts, and `measure` puts numbers on a finished clip (geometry in
the photo's face heights, so photos of any size compare) so that clips of different models can be ranked without anyone
having to watch them all.
"""

import math

import cv2
import numpy as np

# Face heights of photo that must show below the face, or there is no body to gesture with.
MIN_BODY = 4.0
# How much of the face, from the chin up, the model is shown: the live head covers all of it, but the picture should still be
# of a person and not of a headless torso.
ABOVE_CHIN = 0.5
# A clip is at most this much taller than wide and at most this many face heights wide: a body, not a landscape.
MAX_ASPECT, MAX_WIDTH = 1.6, 8.0
# The rows the head rides on, from the chin down, in face heights: the neck, the shoulders and the top of the chest.
BAND = (0.0, 1.0)

# The prompts name no clothes, scene or gender: the photo is the first frame and tells the model all of that.
STYLE = (
    "Realistic, smooth motion at a natural pace. Locked-off tripod camera: the framing, the background and the lighting "
    "never change. The head stays still, and the clothes and the body stay as in the photo."
)
NEGATIVE = (
    "blurry, low quality, jpeg artifacts, deformed hands, extra fingers, missing fingers, fused fingers, extra arms, "
    "extra limbs, bad anatomy, distorted body, changing identity, changing clothes, morphing, melting, flickering, "
    "camera movement, zoom, pan, scene change, cartoon, painting, text, watermark, static, motionless, stiff, robotic"
)
GESTURES = {
    "idle": (
        "A person stands quietly with a gentle relaxed smile. Only a tiny adjustment of the fingers of the lower hand, "
        "then the fingers relax into the original resting pose. Keep the arms, torso, shoulders and head almost still. "
        "No nodding, swaying, arm raising or broad gestures. " + STYLE
    ),
    "listening": (
        "A person listens attentively to someone in front of the camera, giving one small slow nod of acknowledgement "
        "while the hands stay in their original resting positions. Return to the original pose. " + STYLE
    ),
    "explain": (
        "A person stands facing a static camera and gestures as if explaining an idea warmly. Both hands rise from rest "
        "to chest height with the palms open, move outward and back in time with an imagined rhythm and turn a little to "
        "make the point; then both arms relax and settle back into the resting pose they started in. " + STYLE
    ),
    "emphasis": (
        "A person stands facing a static camera and makes a point. One hand lifts to chest height with the palm open and "
        "gives two small, soft beats of emphasis while the other arm stays relaxed; then the hand lowers back into the "
        "resting pose it started in. " + STYLE
    ),
    "calm": (
        "A person stands facing a static camera, relaxed and attentive. The shoulders rise and fall gently with each "
        "breath, the hands shift a little and the fingers loosen, then everything settles back into the resting pose. "
        "Very small, smooth, natural movements. " + STYLE
    ),
    "thinking": (
        "A person stands facing a static camera, listening and thinking before answering. The hands rest low in front of "
        "the body, well below the chest and away from the face, and shift slowly: the fingers loosen and one hand "
        "adjusts its hold on the other; the shoulders rise and fall with slow breathing; then everything settles back "
        "into the resting pose it started in. Very small, smooth, natural movements. " + STYLE
    ),
    "greet": (
        "A person stands facing a static camera and welcomes the viewer warmly. One hand opens outward from the waist "
        "with the palm turned up, in a friendly, inviting gesture that rises to chest height at most, and holds there a "
        "moment; then the hand lowers and everything settles back into the resting pose it started in. " + STYLE
    ),
    "ask": (
        "A person stands facing a static camera and asks a question. One hand turns palm up and opens a little outward "
        "at waist height, held for a moment as if waiting for an answer, then lowers back into the resting pose it "
        "started in, while the other arm stays relaxed. " + STYLE
    ),
}
LOOP_ROLES = frozenset({"idle", "listen", "think"})
ROLES = {
    "idle": ("idle", "idle"),
    "listening": ("listen", "listen"),
    "thinking": ("think", "think"),
    "calm": ("talk", "calm"),
    "explain": ("talk", "explain"),
    "emphasis": ("talk", "emphasis"),
    "greet": ("talk", "greet"),
    "ask": ("talk", "ask"),
}
# Video that browsers show with the colours it was made with: BT.709, limited range, which is what they assume of HD video.
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
FROM_YUV = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc"


class Unsuitable(ValueError):
    """The photo has no body for a clip to move."""


def plan_region(size, face, stride, area, above_chin=ABOVE_CHIN):
    """The part of the photo the model is shown and the size it is generated at, as {"region": [x1, y1, x2, y2], "work": [w, h]}.

    From `above_chin` face heights over the chin down to the photo's lower edge, as wide as the person can reach (the
    photo's width, but no more than MAX_WIDTH face heights around the face) and no taller than MAX_ASPECT times that.
    About `area` pixels are spent on it, in multiples of `stride`, never more than the photo has. The region has exactly
    the work size's proportions, so the picture is not stretched.
    """
    width, height = size
    x1, y1, x2, y2 = face
    face_h = y2 - y1
    if face_h <= 0 or height - y2 < MIN_BODY * face_h:
        raise Unsuitable("The photo shows too little of the body below the face for a body clip.")
    reach = min(width, MAX_WIDTH * face_h)
    left = min(max((x1 + x2) / 2 - reach / 2, 0.0), width - reach)
    top = max(0.0, y2 - above_chin * face_h)
    tall = min(height - top, MAX_ASPECT * reach)
    area = min(area, reach * tall)
    work_w = max(stride, round(math.sqrt(area * reach / tall) / stride) * stride)
    work_h = max(stride, int(work_w * tall / reach // stride) * stride)
    drawn = reach * work_h / work_w
    return {"region": [round(left), round(top), round(left + reach), round(top + drawn)], "work": [work_w, work_h]}


def plan_avatar(size, face, stride, area):
    """Keep the entire image, including the head and feet, at the model's size granularity."""
    width, height = size
    if min(width, height) < stride or face[3] <= face[1]:
        raise Unsuitable("The image and face must have positive dimensions.")
    scale = min(1.0, math.sqrt(area / (width * height)))
    work = [max(stride, round(side * scale / stride) * stride) for side in size]
    return {"region": [0, 0, width, height], "work": work}


def face_box(plan, face):
    """A tracking box around the face, excluding most of the static background."""
    width, height = plan["work"]
    left, top, right, bottom = plan["region"]
    sx, sy = width / (right - left), height / (bottom - top)
    return [max(0, round((face[0] - left) * sx)), max(0, round((face[1] - top) * sy)),
            min(width, round((face[2] - left) * sx)), min(height, round((face[3] - top) * sy))]


def avatar_prompt(gesture):
    quiet = {
        "idle": (
            "A person waits quietly facing the camera with a gentle relaxed smile. Keep that smile steady. "
            "The head, neck, shoulders and torso stay almost still. Only the fingers of the lower hand make one "
            "tiny relaxed adjustment, then settle back. Both hands and arms stay in their original resting positions. "
            "No nodding, head tilting, swaying, shoulder lifting, arm raising or broad gestures. "
        ),
        "listening": (
            "A person listens attentively to someone directly in front of the camera. Give one small, slow nod "
            "of acknowledgement, then return upright. The hands remain in their original resting positions. "
        ),
        "thinking": (
            "A person pauses thoughtfully before answering, keeping the head upright and facing the viewer. "
            "The fingers of the lower hand slowly curl a little, then relax again as the shoulders settle with "
            "a soft exhale. Keep both arms and the upper hand in their original positions. "
        ),
    }
    if gesture in quiet:
        return quiet[gesture] + (
            "Keep the eyes open at the start and end. Blink just once briefly near the middle, with both eyelids "
            "immediately reopening together. No repeated eye closures. The mouth stays still and silent. "
            "Small, smooth natural movements with the head, neck and shoulders connected. "
            "Locked-off camera, unchanged framing, clothing, identity, background and lighting. "
            "Finish in exactly the original resting pose."
        )
    return GESTURES[gesture] + " The person is silent, with no lip movements. Natural occasional eye blinks."


def crop(photo, plan):
    """The photo's region at the work size: the picture the model is given."""
    left, top, right, bottom = plan["region"]
    width, height = plan["work"]
    shrinking = width < right - left
    return cv2.resize(photo[top:bottom, left:right], (width, height), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)


def work_scale(plan):
    """Work pixels per photo pixel."""
    left, top, right, bottom = plan["region"]
    return plan["work"][1] / (bottom - top)


def band_rows(plan, face):
    """The rows of the work picture (first, last) that the head rides on: BAND, measured from the chin."""
    top, scale = plan["region"][1], work_scale(plan)
    face_h = face[3] - face[1]
    first, last = (int(round((face[3] + k * face_h - top) * scale)) for k in BAND)
    return max(0, first), min(plan["work"][1], last)


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def band_motion(frames, rows, columns=None):
    """How the band of rows moves against the first frame: (zoom, turn in degrees, dx, dy) per frame.

    The shift is that of the band's centre, in pixels. Features of the first frame are followed through the clip and a
    similarity is fitted to them, so a hand that passes in front of the band does not drag the fit along with it.
    A frame in which the band cannot be followed is NaN.
    """
    top, bottom = rows
    left, right = columns if columns is not None else (0, frames[0].shape[1])
    first = gray(frames[0])[top:bottom, left:right]
    table = np.full((len(frames), 4), np.nan)
    points = cv2.goodFeaturesToTrack(first, 300, 0.01, 8)
    if points is None or len(points) < 8:
        return table
    centre = np.array([first.shape[1] / 2, first.shape[0] / 2])
    for i, frame in enumerate(frames):
        moved, found, _ = cv2.calcOpticalFlowPyrLK(first, gray(frame)[top:bottom, left:right], points, None, winSize=(21, 21), maxLevel=3)
        kept = found.ravel() == 1
        if kept.sum() < 8:
            continue
        matrix, _ = cv2.estimateAffinePartial2D(points[kept], moved[kept], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if matrix is None:
            continue
        shift = matrix[:, :2] @ centre + matrix[:, 2] - centre
        table[i] = [math.hypot(matrix[0, 0], matrix[1, 0]), math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), *shift]
    return table


def measure(frames, reference=None, rows=None, face_h=None):
    """Numbers for a finished clip: `frames` is (n, h, w, 3) uint8 RGB, `reference` the picture the model was given.

    fidelity     dB between the first frame and `reference`: a model that re-draws the person scores low
    loop_gap     mean difference in levels (0-255) between the last frame and the first: how far the clip is from a loop
    motion_*     mean and peak difference in levels of a frame from the first: a clip that hardly moves is not worth playing
    tone_drift   the largest change of the picture's mean colour, in levels: the light drifting, the colours washing out
    softening    the sharpness of the softest frame in the first frame's: how far the clip blurs
    flicker      mean second difference from frame to frame, in levels: shimmer that smooth motion does not make
    band_*       how far the head's seat (see band_rows) moves, in face heights, degrees and zoom, and the share of frames
                 in which it cannot be followed; needs `rows` and `face_h` (the face's height in the work picture)
    """
    count = len(frames)
    low = np.stack([cv2.resize(gray(frame), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) for frame in frames]).astype(np.float32)
    moved = np.abs(low - low[0]).mean((1, 2))
    colour = np.stack([frame.reshape(-1, 3).mean(0) for frame in frames])
    sharp = [cv2.Laplacian(gray(frame), cv2.CV_32F).var() for frame in frames]
    result = {
        "frames": count,
        "loop_gap": float(np.abs(frames[-1].astype(np.float32) - frames[0]).mean()),
        "motion_mean": float(moved.mean()),
        "motion_peak": float(moved.max()),
        "tone_drift": float(np.abs(colour - colour[0]).max()),
        "softening": float(min(sharp) / max(sharp[0], 1e-6)),
        "flicker": float(np.abs(low[:-2] - 2 * low[1:-1] + low[2:]).mean()) if count >= 3 else 0.0,
    }
    if reference is not None:
        error = float(np.mean((frames[0].astype(np.float32) - reference.astype(np.float32)) ** 2))
        result["fidelity"] = 99.0 if error < 1e-6 else min(99.0, 10 * math.log10(255.0 ** 2 / error))
    if rows is not None and face_h:
        table = band_motion(frames, rows)
        seen = ~np.isnan(table[:, 0])
        result["band_lost"] = float(1 - seen.mean())
        if seen.any():
            result["band_shift"] = float(np.hypot(table[seen, 2], table[seen, 3]).max() / face_h)
            result["band_turn"] = float(np.abs(table[seen, 1]).max())
            result["band_zoom"] = float(np.abs(table[seen, 0] - 1).max())
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()}


def contact_sheet(frames, count=8, height=400):
    """A strip of `count` evenly spaced frames, each numbered, to be judged at a glance."""
    picks = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    tiles = []
    for index in picks:
        frame = frames[index]
        tile = cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(tile, str(index), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def encode_video(path, frames, fps, codec, params):
    """`frames` (n, h, w, 3 uint8 RGB) as a video file: `codec` with ffmpeg's `params`, tagged as BT.709 limited range."""
    import imageio_ffmpeg

    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec=codec, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        macro_block_size=1, quality=None, output_params=["-vf", TO_YUV, *params],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def write_video(path, frames, fps):
    """The clip as H.264, close to lossless: it is for judging, and the clip pack is made from it."""
    encode_video(path, frames, fps, "libx264", ["-crf", "14", "-preset", "slow"])


def read_video(path):
    """A video written by `write_video` as (frames (n, h, w, 3) uint8 RGB, fps)."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=["-vf", FROM_YUV])
    meta = next(reader)
    width, height = meta["size"]
    frames = [np.frombuffer(data, np.uint8).reshape(height, width, 3) for data in reader]
    if not frames:
        raise ValueError(f"{path} has no frames.")
    return np.stack(frames), meta["fps"]
