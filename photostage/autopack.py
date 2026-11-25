"""The motion pack of a newly made avatar, built without anyone choosing takes: what was curated by hand for the first
avatar (a speaking loop, its silent frames, a subtle rest with random blinks and lip-synced filler phrases), made from
any photo.

The server (photostage.server) runs the stages one after the other; this module holds the stages that need a model
of their own, each run as a process in the Python environment the model lives in:

    python -m photostage.autopack wan <avatar folder>          # .venv-flashhead: a wide photo's loop and rest take
    python -m photostage.autopack livetalk <avatar folder>     # .venv-livetalk: a close-up's speaking takes
    python -m photostage.autopack liveavatar <avatar folder>   # .venv-liveavatar: the same with Live Avatar (by hand)
    python -m photostage.autopack bridge <avatar folder>       # .venv-flashhead: a close-up's return and rest take
    python -m photostage.autopack rest <avatar folder>         # .venv-flashhead (CPU): the rest bank and its blinks

A close-up (a portrait: head and shoulders filling the photo) moves with LiveTalk, driven by a spoken line
(build/drive.wav, which the server records in the avatar's voice); it keeps the whole photo, its framing and the
person's likeness better there than Live Avatar did, and comes back near the photo's pose by itself, so its end is
eased into its start. A wider photo moves with Wan alone; Wan also makes both kinds' near-still idle take. MuseTalk
lip-syncs both live. Every stage writes into <avatar>/build/, and reports how far it has got as lines
`PROGRESS {"part": 0-1, "label": ...}`.
"""

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from photostage import clipbridge, clipgen, clips, clippack

ROOT = clipgen.ROOT
CLOSEUP = 0.25  # face height / photo height from which a photo is a close-up
FPS = 16  # every bank plays at Wan's rate
LOOP_FRAMES = 97  # Wan frames of a wide photo's loop (4k + 1); the last one is the first again
REST_FRAMES = 65  # Wan frames of the rest take: 64 rest frames and the anchor again
SEEDS = (9, 19, 5)
BRIDGE_SEED = 19
REST_SEED = 5
EYE_CLOSED = 0.55  # eye openness, as a share of the photo's, below which an eye counts as shut
LONGEST_CLOSURE = 0.25  # seconds an eye may stay shut in the speaking loop
MOST_CLOSURES = 2
BLINK_LENGTH = 4  # frames of a blink in the rest bank
BLINK_EVERY, BLINK_OFFSET = 16, 8  # the rest bank's blink slots: frames i with i % 16 in 8..11
MAX_BANK = 200  # the player's largest frame bank
# Points that move with the skull rather than the lips or the jaw: eye corners, brows, bridge, forehead, cheekbones.
RIGID = [33, 133, 263, 362, 70, 105, 300, 334, 6, 168, 197, 195, 5, 10, 151, 9, 116, 345, 123, 352, 50, 280]

LOOP_PROMPT = (
    "A person waits quietly, keeping exactly the gentle smile in the reference photo. "
    "The lips and jaw remain still in that same smile throughout. "
    "The head, neck, shoulders and torso stay almost still, with calm breathing. "
    "One hand makes one small, relaxed movement, pauses, then eases back to its original resting position. "
    "Keep both eyes open; just once near the middle, blink very briefly and immediately reopen both eyes. "
    "No lingering eye closure or repeated blinks. "
    "Small, unhurried, natural motion. Locked-off camera, unchanged framing, facial proportions, "
    "clothing, background and lighting. Finish in the original resting pose."
)
REST_PROMPT = (
    "A still portrait with a constant warm slight smile. Both eyes stay comfortably open. The hands and arms are "
    "motionless. Only a barely perceptible breath in the clothing. Fixed camera, unchanged lighting."
)
LIVE_AVATAR_GESTURE = "calm"
LIVE_AVATAR_WHO = "A person"
LIVETALK_PROMPTS = (
    "A portrait of the same person listening quietly with a calm, relaxed face, natural blinking, subtle breathing and "
    "tiny head movements, lips closed. The camera is stationary. ",
    "A portrait of the same friendly person speaking warmly to the camera with a light smile, bright fully open eyes "
    "looking straight into the camera, clearly synchronized lips, lively eyebrows, small natural nods and head "
    "movements and a gentle sway of the shoulders. The camera is stationary. ",
)
LIVETALK_AREA = 512  # its training size: the canvas has the area of a 512 x 512 square
LIVETALK_LEAD, LIVETALK_TAIL = 0.5, 1.2  # seconds of silence around the line: it starts still and settles after


def progress(part, label):
    print("PROGRESS " + json.dumps({"part": round(min(1.0, max(0.0, part)), 3), "label": label}), flush=True)


def kind(rig):
    """"closeup" for a photo the face fills a good part of (head and shoulders), else "wide"."""
    face, size = rig["face"], rig["size"]
    return "closeup" if (face[3] - face[1]) / size[1] >= CLOSEUP else "wide"


def even(value):
    return max(2, int(round(value / 2)) * 2)


def canvas_size(size, scale):
    """The pack's frame size for a photo of `size` at `scale` frame pixels per photo pixel (even, for H.264)."""
    return [even(size[0] * scale), even(size[1] * scale)]


def resample(frames, source_fps, fps=FPS):
    """The frames at `fps`, by picking the nearest one: no blending, which would show as ghosting."""
    count = max(1, int(math.floor((len(frames) - 1) * fps / source_fps)) + 1)
    picks = [min(len(frames) - 1, round(i * source_fps / fps)) for i in range(count)]
    return frames[picks]


def close_loop(frames, count=clipbridge.CLOSE):
    """A loop that Wan brought back to its first frame: its last frame (the first again) dropped, and the little by which
    it missed spread over the frames before it."""
    frames = frames.astype(np.float32)
    miss = frames[0] - frames[-1]
    gap = float(np.abs(miss).mean())
    count = min(count, len(frames) - 2)
    frames[-count - 1:-1] += (1 - clipbridge.ramp(count))[:, None, None, None] * miss
    return np.clip(frames[:-1] + 0.5, 0, 255).astype(np.uint8), round(gap, 3)


def paste(canvas, frames, box, feather=8):
    """`frames` (of one size) placed into `box` [x1, y1, x2, y2] of `canvas` (an RGB picture), fading in over `feather`
    pixels on the sides that are not the canvas's own edges."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    mask = np.ones((height, width), np.float32)
    ramp = np.linspace(0, 1, feather + 2, dtype=np.float32)[1:-1]
    if x1 > 0:
        mask[:, :feather] *= ramp[None]
    if x2 < canvas.shape[1]:
        mask[:, -feather:] *= ramp[::-1][None]
    if y1 > 0:
        mask[:feather] *= ramp[:, None]
    if y2 < canvas.shape[0]:
        mask[-feather:] *= ramp[::-1][:, None]
    mask = mask[..., None]
    out = np.empty((len(frames), *canvas.shape), np.uint8)
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        picture = canvas.copy()
        region = picture[y1:y2, x1:x2].astype(np.float32)
        picture[y1:y2, x1:x2] = np.clip(region * (1 - mask) + frame * mask + 0.5, 0, 255).astype(np.uint8)
        out[i] = picture
    return out


# Faces ----------------------------------------------------------------------------------------------------------------

def landmarks(mesh, image, box, margin=0.35):
    """MediaPipe's face mesh of the face in `box` [x1, y1, x2, y2] of `image` (RGB), in image pixels, or None."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    left, top = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
    right, bottom = min(image.shape[1], int(x2 + margin * w)), min(image.shape[0], int(y2 + margin * h))
    found = mesh.process(np.ascontiguousarray(image[top:bottom, left:right]))
    if not found.multi_face_landmarks:
        return None
    return np.array([(p.x * (right - left) + left, p.y * (bottom - top) + top)
                     for p in found.multi_face_landmarks[0].landmark], np.float32)


def face_mesh():
    import mediapipe as mp

    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


def openness(points):
    """How open the eyes are: eyelid gap over eye width, both eyes' average; None without a face."""
    if points is None:
        return None
    return float((np.linalg.norm(points[159] - points[145]) / np.linalg.norm(points[33] - points[133]) +
                  np.linalg.norm(points[386] - points[374]) / np.linalg.norm(points[362] - points[263])) / 2)


def closures(open_by_frame, reference, fps=FPS):
    """The runs of frames with the eyes shut, as [first, last, seconds]; a frame without a face counts as open."""
    shut = [value is not None and value < EYE_CLOSED * reference for value in open_by_frame]
    found, start = [], None
    for i, closed in enumerate(shut + [False]):
        if closed and start is None:
            start = i
        elif not closed and start is not None:
            found.append([start, i - 1, (i - start) / fps])
            start = None
    return found


def eye_report(frames, box):
    with face_mesh() as mesh:
        values = [openness(landmarks(mesh, frame, box)) for frame in frames]
    reference = values[0] if values[0] else max((v for v in values if v), default=0.3)
    return values, closures(values, reference)


def eyes_pass(found):
    return len(found) <= MOST_CLOSURES and all(seconds <= LONGEST_CLOSURE for _, _, seconds in found)


def eye_mask(shape, points, corners):
    width = np.linalg.norm(points[corners[0]] - points[corners[1]])
    center = points[list(corners)].mean(axis=0)
    mask = np.zeros(shape[:2], np.float32)
    cv2.ellipse(mask, tuple(np.round(center).astype(int)), (max(2, round(width * .86)), max(2, round(width * .52))),
                0, 0, 360, 1, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 3.0)


def face_mask(shape, points):
    """The central face, brows to chin and inside the cheeks: the part of the still face that rides on the head."""
    top, chin, left, right = points[9], points[152], points[234], points[454]
    center = (top + chin) / 2 + [0, 4]
    axes = (round(np.linalg.norm(right - left) * .40), round(np.linalg.norm(chin - top) * .58))
    mask = np.zeros(shape[:2], np.float32)
    cv2.ellipse(mask, tuple(np.rint(center).astype(int)), axes, 0, 0, 360, 1, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 6.0)


def similarity(src, dst):
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None
    scale = np.hypot(matrix[0, 0], matrix[1, 0])
    return np.array([np.log(scale), np.arctan2(matrix[1, 0], matrix[0, 0]), matrix[0, 2], matrix[1, 2]])


def to_matrix(p):
    s, a = np.exp(p[0]), p[1]
    return np.float32([[s * np.cos(a), -s * np.sin(a), p[2]], [s * np.sin(a), s * np.cos(a), p[3]]])


def circular_smooth(values, sigma):
    offsets = np.arange(-3 * int(sigma) - 1, 3 * int(sigma) + 2)
    weights = np.exp(-.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return sum(w * np.roll(values, o, 0) for o, w in zip(offsets, weights))


def paint(image, reference, mask, matrix):
    size = image.shape[1], image.shape[0]
    overlay = cv2.warpAffine(reference, matrix, size, flags=cv2.INTER_CUBIC)
    blend = cv2.warpAffine(mask, matrix, size)[..., None]
    return np.clip(image * (1 - blend) + overlay * blend + .5, 0, 255).astype(np.uint8)


def closed_eyes(mesh, anchor, target, sources, box):
    """Each of `sources` (frames of a blink) with its eyes aligned onto the anchor's, and the rest of it the anchor."""
    result = []
    for image in sources:
        points = landmarks(mesh, image, box)
        if points is None:
            return None
        composite = anchor.astype(np.float32)
        for corners, brow in (((33, 133), 105), ((362, 263), 334)):
            ids = [*corners, brow]
            matrix, _ = cv2.estimateAffinePartial2D(points[ids], target[ids], method=cv2.LMEDS)
            if matrix is None:
                return None
            aligned = cv2.warpAffine(image, matrix, (anchor.shape[1], anchor.shape[0]), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REFLECT)
            mask = eye_mask(anchor.shape, target, corners)[..., None]
            composite = composite * (1 - mask) + aligned.astype(np.float32) * mask
        result.append(np.clip(composite + .5, 0, 255).astype(np.uint8))
    return result


def blink_frames(frames, box):
    """BLINK_LENGTH frames around the shortest full eye closure in `frames` (closing, shut, opening), or None."""
    values, found = eye_report(frames, box)
    found = [c for c in found if c[0] >= 1 and c[1] + 2 < len(frames)]
    if not found:
        return None
    first, last, _ = min(found, key=lambda c: (c[2], -min(values[c[0]:c[1] + 1])))
    deepest = first + int(np.argmin(values[first:last + 1]))
    start = min(max(0, deepest - 1), len(frames) - BLINK_LENGTH)
    return frames[start:start + BLINK_LENGTH]


def rest_layout(count, slots):
    """Which unique picture each entry of a rest bank shows: the `count` body frames, then the same again with blink
    pictures (numbered from `count`, BLINK_LENGTH per slot) in the given slots."""
    ids = list(range(count)) * 2
    for j, start in enumerate(slots):
        ids[count + start:count + start + BLINK_LENGTH] = range(count + j * BLINK_LENGTH, count + (j + 1) * BLINK_LENGTH)
    return ids


def write_bank(folder, stem, jpegs, ids, fps=FPS):
    """jpegs (bytes) as <stem>.bin, and <stem>.json listing [offset, length] of the picture each frame shows."""
    offsets, position = [], 0
    with (folder / f"{stem}.bin").open("wb") as output:
        for data in jpegs:
            offsets.append([position, len(data)])
            output.write(data)
            position += len(data)
    (folder / f"{stem}.json").write_text(json.dumps({"fps": fps, "frames": [offsets[i] for i in ids]}))


def read_bank(folder, stem):
    info = json.loads((folder / f"{stem}.json").read_text())
    data = (folder / f"{stem}.bin").read_bytes()
    return [data[a:a + n] for a, n in info["frames"]]


def decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def encode(image, quality=90):
    ok, data = cv2.imencode(".jpg", np.ascontiguousarray(image[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode a frame.")
    return data.tobytes()


# Stages ---------------------------------------------------------------------------------------------------------------

def load_avatar(folder):
    rig = json.loads((folder / "rig.json").read_text())
    photo = cv2.cvtColor(cv2.imread(str(folder / "photo.png")), cv2.COLOR_BGR2RGB)
    if list(photo.shape[1::-1]) != list(rig["size"]):
        raise SystemExit("The photo does not match its rig.")
    return rig, photo


def wan_pipe():
    import torch

    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    spec = clipgen.MODELS[clipbridge.WAN]
    return spec, clipgen.load(spec, ROOT / spec.path, "cuda", True)


def save_loop(build, frames, info):
    np.save(build / "loop.npy", frames)
    clips.encode_video(build / "source.mp4", frames, FPS, "libx264", ["-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
    cv2.imwrite(str(build / "loop-sheet.jpg"), clips.contact_sheet(frames)[:, :, ::-1])
    (build / "loop.json").write_text(json.dumps(info, indent=1))


def make_rest_take(pipe, spec, anchor, work):
    """Wan's near-still take from the loop's first frame back to it, at the pack's frame size."""
    given = cv2.resize(anchor, tuple(work), interpolation=cv2.INTER_AREA) if list(anchor.shape[1::-1]) != list(work) else anchor
    frames = clipbridge.generate(pipe, given, given, REST_PROMPT, REST_SEED, REST_FRAMES, spec.steps)
    if frames.shape[1:3] != anchor.shape[:2]:
        frames = np.stack([cv2.resize(f, (anchor.shape[1], anchor.shape[0]), interpolation=cv2.INTER_CUBIC) for f in frames])
    return frames


def stage_wan(folder, build):
    rig, photo = load_avatar(folder)
    progress(0.0, "Loading the motion model")
    spec, pipe = wan_pipe()
    geometry = clips.plan_avatar(rig["size"], rig["face"], spec.stride, spec.area)
    image = clips.crop(photo, geometry)
    scale = geometry["work"][0] / rig["size"][0]
    box = [v * scale for v in rig["face"]]
    best = None
    for attempt, seed in enumerate(SEEDS):
        progress(0.08 + 0.62 * attempt / len(SEEDS), f"Animating the speaking motion (take {attempt + 1})")
        raw = clipgen.generate(pipe, spec, image, LOOP_PROMPT, seed, LOOP_FRAMES, spec.steps, True)
        corrected, tone = clippack.correct_tone(raw, image)
        frames, gap = close_loop(raw if tone["refused"] else corrected)
        _, found = eye_report(frames, box)
        score = (not eyes_pass(found), gap)
        print(json.dumps({"seed": seed, "closures": found, "end_gap": gap, "tone": tone}), flush=True)
        if best is None or score < best[0]:
            best = (score, frames, {"seed": seed, "closures": found, "end_gap": gap, "tone": tone})
        if eyes_pass(found):
            break
    _, frames, info = best
    save_loop(build, frames, {**info, "model": clipbridge.WAN, "kind": "wide", "fps": FPS, "frames": len(frames),
                              "work": geometry["work"], "prompt": LOOP_PROMPT})
    progress(0.72, "Animating the idle breathing")
    rest = make_rest_take(pipe, spec, frames[0], geometry["work"])
    np.save(build / "rest-take.npy", rest)
    progress(1.0, "Motion ready")


def stage_liveavatar(folder, build):
    from photostage import clipgen_liveavatar as la

    rig, photo = load_avatar(folder)
    layout = la.plan(rig["size"], rig["face"])
    work_plan = {"region": layout["region"], "work": layout["work"]}
    image = clips.crop(photo, work_plan)
    given = clips.crop(photo, {"region": layout["region"], "work": layout["given"]})
    text = la.prompt(LIVE_AVATAR_GESTURE, LIVE_AVATAR_WHO)
    progress(0.0, "Loading Live Avatar")
    model = la.LiveAvatar([text])
    upscaler = la.Upscaler()
    speech, rate = la.speak(la.LINES[LIVE_AVATAR_GESTURE])
    track, _ = la.drive(speech, rate)
    progress(0.15, "Animating the speaking motion (Live Avatar)")
    with tempfile.TemporaryDirectory(prefix="autopack-la-") as scratch:
        given_path, audio = Path(scratch) / "given.png", Path(scratch) / "drive.wav"
        cv2.imwrite(str(given_path), given[:, :, ::-1])
        la.write_wav(audio, track, rate)
        frames = model.generate(given_path, audio, text, SEEDS[0])
    model.close()
    frames = frames[:la.calm(frames, la.speech_end(len(speech) / rate)) + 1]
    progress(0.85, "Sharpening the frames")
    frames = upscaler(frames)
    corrected, tone = clippack.correct_tone(frames, image)
    frames = frames if tone["refused"] else corrected
    # Leave room in the bank (at FPS, once resampled) for the bridge back to the first frame.
    frames = frames[:int(MAX_BANK * la.FPS / FPS) - clipbridge.FRAMES]
    np.save(build / "liveavatar.npy", frames)
    (build / "liveavatar.json").write_text(json.dumps({**layout, "fps": la.FPS, "tone": tone, "frames": len(frames), "prompt": text}))
    progress(1.0, "Speaking motion ready")


def stage_livetalk(folder, build):
    import librosa
    import soundfile as sf
    import torch

    from photostage import clipgen_liveavatar as la

    sys.path.insert(0, str(ROOT / "scripts"))
    import render_livetalk

    rig, photo = load_avatar(folder)
    drive = build / "drive.wav"
    if not drive.is_file():
        raise SystemExit("The spoken line that drives the motion (build/drive.wav) is missing.")
    speech, rate = librosa.load(str(drive), sr=16000, mono=True)
    loud = np.flatnonzero(np.abs(speech) > 0.01)
    speech = speech[max(0, loud[0] - rate // 50):loud[-1] + rate // 50] if len(loud) else speech
    track = np.concatenate([np.zeros(round(LIVETALK_LEAD * rate), np.float32), speech,
                            np.zeros(round(LIVETALK_TAIL * rate), np.float32)])
    progress(0.0, "Loading LiveTalk")
    engine = render_livetalk.Engine(prompts=LIVETALK_PROMPTS, area=LIVETALK_AREA)
    upscaler = la.Upscaler()
    takes = []
    with tempfile.TemporaryDirectory(prefix="autopack-lt-") as scratch:
        scratch = Path(scratch)
        audio, idle = scratch / "drive.wav", scratch / "idle.mp4"
        sf.write(audio, track, rate)
        idle.touch()
        engine.warmed = True  # no idle clip: the rest is made from Wan's near-still take
        engine.prepare("build", str((folder / "photo.png").resolve()), str(idle), lambda event: None)
        for attempt, seed in enumerate(SEEDS):
            progress(0.05 + 0.9 * attempt / len(SEEDS), f"Animating the speaking motion (take {attempt + 1})")
            torch.manual_seed(seed)
            engine.keep = []
            engine.stream(str(audio), str(scratch / "take.mp4"), lambda event: None)
            frames = np.stack(engine.keep)
            engine.keep = None
            frames = frames[:la.calm(frames, round((LIVETALK_LEAD + len(speech) / rate) * FPS)) + 1]
            frames = upscaler(frames[:MAX_BANK - clipbridge.FRAMES])
            work = frames.shape[2], frames.shape[1]
            given = cv2.resize(photo, work, interpolation=cv2.INTER_AREA)
            corrected, tone = clippack.correct_tone(frames, given)
            np.save(build / f"talk-{attempt}.npy", frames if tone["refused"] else corrected)
            takes.append({"seed": seed, "frames": len(frames), "tone": tone})
    (build / "talk.json").write_text(json.dumps({"region": [0, 0, *rig["size"]], "work": list(work), "fps": FPS,
                                                 "takes": takes, "prompt": LIVETALK_PROMPTS[1]}, indent=1))
    progress(1.0, "Speaking motion ready")


def best_talk(build, rig):
    """The LiveTalk take whose eyes stay open best (stage_livetalk made a few), and its layout."""
    layout = json.loads((build / "talk.json").read_text())
    best = None
    for n, info in enumerate(layout["takes"]):
        take = np.load(build / f"talk-{n}.npy")
        sx, sy = take.shape[2] / rig["size"][0], take.shape[1] / rig["size"][1]
        _, found = eye_report(take, [rig["face"][0] * sx, rig["face"][1] * sy, rig["face"][2] * sx, rig["face"][3] * sy])
        score = (not eyes_pass(found), round(sum(c[2] for c in found), 3))
        print(json.dumps({"take": n, "seed": info["seed"], "closures": found}), flush=True)
        if best is None or score < best[0]:
            best = score, take, {**layout, "take": n, "closures": found}
    return best[1], best[2]


def livetalk_loop(take):
    """A LiveTalk take as a loop: it comes back close to the photo's pose by itself, so its end is eased into its start
    (no Wan bridge, which would add a gesture of its own). The photo itself (frame 0) is left out: its texture is not the
    generated frames', and would flicker at every wrap."""
    frames, gap = close_loop(take[1:], count=12)
    return frames, {"seam": None, "end_gap": gap, "closed": True}


def stage_bridge(folder, build):
    rig, photo = load_avatar(folder)
    progress(0.0, "Loading the motion model")
    spec, pipe = wan_pipe()
    if (build / "talk.json").is_file():
        take, layout = best_talk(build, rig)
        looped, seam = livetalk_loop(take)
        model = "livetalk"
    else:
        take = np.load(build / "liveavatar.npy")
        layout = json.loads((build / "liveavatar.json").read_text())
        progress(0.1, "Bringing the motion back to the photo's pose")
        bridge = clipbridge.generate(pipe, take[-1], take[0], clipbridge.PROMPT.format(rest=""), BRIDGE_SEED,
                                     clipbridge.FRAMES, spec.steps)
        joined, seam = clipbridge.join(take, bridge)
        looped = resample(joined[:-1] if seam["closed"] else joined, layout["fps"])[:MAX_BANK]
        model = "liveavatar"
    region, work = layout["region"], layout["work"]
    scale = work[1] / (region[3] - region[1])
    size = canvas_size(rig["size"], scale)
    canvas = cv2.resize(photo, tuple(size), interpolation=cv2.INTER_AREA)
    sx, sy = size[0] / rig["size"][0], size[1] / rig["size"][1]
    box = [round(region[0] * sx), round(region[1] * sy), min(size[0], round(region[2] * sx)),
           min(size[1], round(region[3] * sy))]
    frames = paste(canvas, looped, box)
    face = [rig["face"][0] * sx, rig["face"][1] * sy, rig["face"][2] * sx, rig["face"][3] * sy]
    _, found = eye_report(frames, face)
    save_loop(build, frames, {"model": model, "kind": "closeup", "fps": FPS, "frames": len(frames),
                              "seam": seam, "closures": found, "box": box, "size": size})
    progress(0.45, "Animating the idle breathing")
    wan_work = clips.plan_avatar(rig["size"], rig["face"], spec.stride, spec.area)["work"]
    rest = make_rest_take(pipe, spec, frames[0], wan_work)
    np.save(build / "rest-take.npy", rest)
    progress(1.0, "Motion ready")


def build_rest(anchor_jpeg, take, loop, face):
    """The rest bank: the still face riding a smoothed head path through Wan's near-still take, twice over, the second
    time with a short blink (from the speaking loop's own) in every slot. Returns (jpegs, ids, report)."""
    anchor = decode(anchor_jpeg)
    corrected, tone = clippack.correct_tone(take, anchor)
    joined, seam = clipbridge.join(anchor[None], take if tone["refused"] else corrected)
    body = joined[:-1] if seam["closed"] else joined[:REST_FRAMES - 1]
    with face_mesh() as mesh:
        ref = landmarks(mesh, anchor, face)
        if ref is None:
            raise SystemExit("No face was found in the avatar's still frame.")
        params, last = [], np.zeros(4)
        for frame in body:
            points = landmarks(mesh, frame, face)
            found = similarity(ref[RIGID], points[RIGID]) if points is not None else None
            last = found if found is not None else last
            params.append(last)
        params = np.stack(params)
        params[0] = 0
        smooth = circular_smooth(params, 2.5)
        matrices = [to_matrix(p) for p in smooth]
        mask = face_mask(anchor.shape, ref)
        frames = [anchor] + [paint(body[i], anchor, mask, matrices[i]) for i in range(1, len(body))]
        count = len(frames)
        slots = list(range(BLINK_OFFSET, count - BLINK_LENGTH + 1, BLINK_EVERY))
        source = blink_frames(loop, face)
        pictures = closed_eyes(mesh, anchor, ref, source, face) if source is not None else None
    jpegs = [anchor_jpeg] + [encode(f) for f in frames[1:]]
    if pictures is None or count % BLINK_EVERY:
        return jpegs, list(range(count)), {"blink": False, "frames": count, "seam": seam, "tone": tone}
    eyes = np.maximum(eye_mask(anchor.shape, ref, (33, 133)), eye_mask(anchor.shape, ref, (362, 263)))
    for start in slots:
        jpegs += [encode(paint(frames[start + i], picture, eyes, matrices[start + i])) for i, picture in enumerate(pictures)]
    return jpegs, rest_layout(count, slots), {"blink": True, "frames": count, "slots": slots, "seam": seam, "tone": tone}


def stage_rest(folder, build):
    rig, _ = load_avatar(folder)
    progress(0.0, "Building the idle animation")
    quiet = read_bank(build, "quiet")
    loop = np.load(build / "loop.npy")
    take = np.load(build / "rest-take.npy")
    size = loop.shape[2], loop.shape[1]
    sx, sy = size[0] / rig["size"][0], size[1] / rig["size"][1]
    face = [rig["face"][0] * sx, rig["face"][1] * sy, rig["face"][2] * sx, rig["face"][3] * sy]
    jpegs, ids, report = build_rest(quiet[0], take, loop, face)
    write_bank(build, "rest", jpegs, ids)
    clips.encode_video(build / "quiet.mp4", np.stack([decode(j) for j in quiet]), FPS, "libx264",
                       ["-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
    (build / "rest-report.json").write_text(json.dumps(report, indent=1))
    progress(1.0, "Idle animation ready")


STAGES = {"wan": stage_wan, "livetalk": stage_livetalk, "liveavatar": stage_liveavatar, "bridge": stage_bridge, "rest": stage_rest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    folder = args.folder.resolve()
    build = folder / "build"
    build.mkdir(exist_ok=True)
    started = time.monotonic()
    STAGES[args.stage](folder, build)
    print(f"{args.stage} finished in {time.monotonic() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
