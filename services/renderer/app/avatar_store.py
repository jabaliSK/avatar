"""Avatar assets on disk and in memory.

    avatars/<name>/
        meta.json                       name, max_size, source image, which states have a Stage A body loop
        settings.json                   optional: expression / head_motion strength (see DEFAULT_SETTINGS)
        source.<ext>                    the onboarding image
        still/                          Loop cache of the single image (head template when there is no speaking loop)
        idle/ listening/ thinking/ speaking/   optional body loops from Stage A (each a Loop cache of the generated clip)

A state without a body loop falls back (thinking -> listening -> idle). A quiet state that has a Stage A body loop is rendered
once at load time through the live head on silence over one full cycle of that clip, so it looks like the speaking state (same
face rendering) and the whole body moves. Without any idle loop the idle frames are rendered the same way from the still image,
which gives blinks and micro head motion on a frozen photo: usable, but not natural; generate the idle loop.
"""
import json
import logging
import os
import shutil
import tempfile
import time

import cv2
import imageio
import numpy as np

from core.atomic_components.condition_handler import _mirror_index

from .avatar import Loop
from .head_engine import HUBERT_STEP
from .pinning import pin_static

log = logging.getLogger("kka.avatars")

BASE_STATES = ("idle", "listening", "thinking")
LOOP_STATES = BASE_STATES + ("speaking",)

# Per-avatar strength of the live head. 1.0 is what Ditto produces on its own, which is too busy for a calm presenter.
#   expression   mouth / jaw amplitude added on top of the template (0.3 barely moves, 1.0 = Ditto's native amplitude)
#   head_motion  head rotation / shift of a single-image avatar (video templates bring their own head motion)
DEFAULT_SETTINGS = {"expression": 0.6, "head_motion": 0.5}
SETTING_RANGES = {"expression": (0.2, 1.2), "head_motion": (0.0, 1.0)}


class BaseLoop:
    """Frames played while the avatar is not speaking: one full cycle of a Stage A clip (cyclic) or a clip played ping-pong."""

    def __init__(self, frames, cyclic):
        self.frames = frames  # uint8 [N,H,W,3]
        self.cyclic = cyclic

    def __len__(self):
        return len(self.frames)

    def at(self, pos):
        n = len(self.frames)
        return self.frames[pos % n if self.cyclic else _mirror_index(pos, n)]


class Avatar:
    def __init__(self, name, speak_loop, base, meta, stamp=None, settings=None):
        self.name = name
        self.speak_loop = speak_loop  # template the live head is composited onto while speaking
        self.base = base              # state -> BaseLoop, played while not speaking
        self.meta = meta
        self.stamp = stamp            # avatar_stamp() when it was loaded
        self.settings = settings or dict(DEFAULT_SETTINGS)

    @property
    def size(self):
        return self.speak_loop.size  # (width, height)


def read_settings(directory):
    """settings.json merged over the defaults; unknown keys are dropped and values are clamped to SETTING_RANGES."""
    out = dict(DEFAULT_SETTINGS)
    try:
        with open(os.path.join(directory, "settings.json")) as f:
            stored = json.load(f)
    except (FileNotFoundError, ValueError):
        stored = {}
    for key, (lo, hi) in SETTING_RANGES.items():
        try:
            out[key] = min(hi, max(lo, float(stored.get(key, out[key]))))
        except (TypeError, ValueError):
            pass
    return out


def write_settings(directory, settings):
    """Validate, store and return the settings (only the known keys; values outside the range are clamped)."""
    merged = read_settings(directory)
    for key, (lo, hi) in SETTING_RANGES.items():
        if key in settings:
            merged[key] = round(min(hi, max(lo, float(settings[key]))), 3)
    tmp = os.path.join(directory, "settings.json.tmp")
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, os.path.join(directory, "settings.json"))
    return merged


def list_avatars(root):
    if not os.path.isdir(root):
        return []
    return sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, "meta.json")))


def preload_names(requested, root):
    """The avatars to load at boot: `requested` is a list of names or ["all"]. A name that is not in `root` (an avatar deleted in
    the Studio, say) is skipped with a warning, so that it cannot keep the renderer from starting."""
    present = list_avatars(root)
    if requested == ["all"]:
        return present
    for name in requested:
        if name not in present:
            log.warning("KKA_PRELOAD names %r, which is not in %s: not loaded", name, root)
    return [name for name in requested if name in present]


def avatar_stamp(root, name):
    """Changes whenever the avatar is onboarded again, gets a loop (meta.json is always written last) or gets new settings;
    None if it is not there."""
    directory = os.path.join(root, name)
    try:
        stamp = os.stat(os.path.join(directory, "meta.json")).st_mtime_ns
    except FileNotFoundError:
        return None
    try:
        stamp = max(stamp, os.stat(os.path.join(directory, "settings.json")).st_mtime_ns)
    except FileNotFoundError:
        pass
    return stamp


def _read_meta(directory):
    with open(os.path.join(directory, "meta.json")) as f:
        return json.load(f)


def _write_meta(directory, meta):
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def fit_image(src, dst, size):
    """Write `src` resized to exactly `size` (width, height)."""
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    interp = cv2.INTER_AREA if img.shape[1] * img.shape[0] > size[0] * size[1] else cv2.INTER_CUBIC
    cv2.imwrite(dst, cv2.resize(img, tuple(size), interpolation=interp))


def video_size(path):
    """(width, height) of a video file, read the way the registrar reads it."""
    reader = imageio.get_reader(path, "ffmpeg")
    try:
        return tuple(reader.get_meta_data()["size"])
    finally:
        reader.close()


def onboard_image(root, name, image_path, registrar, engine, kw, target_roi=560, size=None):
    """Register a portrait / body photo once and cache everything the live path needs.

    The head renderer works at 512 px, so a face covering far more than that in the output frame is upscaled and looks soft.
    When the face ROI exceeds `target_roi` the frame is registered again at a smaller size that brings it back to that.

    `size` (width, height) pins the frame geometry instead: Stage A clips come with a fixed size and every state of an avatar
    has to share one frame size, so the photo is resized to it and the ROI-based re-registration is skipped.
    """
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    t0 = time.perf_counter()
    if size is not None:
        kw = {**kw, "max_size": max(size)}
        with tempfile.TemporaryDirectory() as tmp:
            fitted = os.path.join(tmp, "still.png")
            fit_image(image_path, fitted, size)
            loop = Loop.register("still", fitted, registrar, engine, kw)
        if tuple(loop.size) != tuple(size):
            raise ValueError(f"{name}: still registered as {loop.size}, expected {tuple(size)}")
    else:
        loop = Loop.register("still", image_path, registrar, engine, kw)
        roi = max(loop.rois[0].shape)
        if roi > target_roi * 1.1:
            kw = {**kw, "max_size": int(kw["max_size"] * target_roi / roi) // 2 * 2}
            log.info("%s: face ROI %d px is larger than the 512 px render; re-registering at max_size %d", name, roi, kw["max_size"])
            loop = Loop.register("still", image_path, registrar, engine, kw)
    for state in LOOP_STATES:  # loops made for an earlier still may not match this frame size
        shutil.rmtree(os.path.join(directory, state), ignore_errors=True)
    loop.save(os.path.join(directory, "still"))
    ext = os.path.splitext(image_path)[1].lower() or ".png"
    shutil.copy(image_path, os.path.join(directory, "source" + ext))
    meta = {"name": name, "max_size": kw.get("max_size"), "source": "source" + ext, "frame_size": list(loop.size),
            "states": [], "registered_s": round(time.perf_counter() - t0, 1)}
    _write_meta(directory, meta)
    log.info("onboarded %s: frame %s, face ROI %s px", name, loop.size, loop.rois[0].shape[::-1])
    return loop


def add_state_loop(root, name, state, video_path, registrar, engine, kw, pin=True):
    """Register a Stage A clip (25 fps body video of the same person) as the body loop of `state`.

    pin: keep the photo's own pixels wherever the clip does not move (see pinning.pin_static), so the cross-fades between the
    photo-based quiet states and this loop do not change tone or texture.
    """
    if state not in LOOP_STATES:
        raise ValueError(f"state must be one of {LOOP_STATES}")
    directory = os.path.join(root, name)
    meta = _read_meta(directory)
    frame_filter = None
    if pin:
        photo = np.load(os.path.join(directory, "still", "frames.npy"), mmap_mode="r")[0]
        frame_filter = lambda frames: pin_static(frames, np.asarray(photo))  # noqa: E731
    loop = Loop.register(state, video_path, registrar, engine, kw, frame_filter=frame_filter)
    if list(loop.size) != list(meta["frame_size"]):  # the player crossfades between states, which needs one frame size
        raise ValueError(f"{state} loop is {loop.size[0]}x{loop.size[1]} but the avatar frame is {meta['frame_size'][0]}x{meta['frame_size'][1]}")
    loop.save(os.path.join(directory, state))
    meta["states"] = sorted(set(meta.get("states", [])) | {state})
    _write_meta(directory, meta)
    return loop


def render_silence(pipe, loop, seconds, **begin_kw):
    """Run the live head on silence and return the composited frames as uint8 [N,H,W,3] (no real-time gate).

    At least `seconds` of frames come back, rounded up to whole HuBERT steps (5 frames).
    """
    gate, pipe.gate = pipe.gate, None
    try:
        pipe.start(loop, **begin_kw)
        zeros = np.zeros(HUBERT_STEP, np.float32)
        for _ in range(0, int(seconds * 16000), HUBERT_STEP):
            pipe.push(zeros)
        pipe.finish()
        frames = []
        while (f := pipe.frames.get()) is not None:
            frames.append(f.rgb)
        if pipe.error:
            raise pipe.error
        pipe.join()
        return np.stack(frames)
    finally:
        pipe.gate = gate


def drive_kwargs(settings):
    """HeadEngine.begin() arguments for an avatar's settings."""
    return {"expression": settings["expression"], "head_motion": settings["head_motion"]}


BLINK_FRAMES = 15  # length of one of Ditto's synthetic blinks (delta_eye_arr)


def blink_gaps(cycle, mean_gap=85):
    """`delta_eye_open_n` for a loop of `cycle` frames: whole blinks about `mean_gap` frames (3.4 s) apart and unevenly spread, none
    cut by the wrap. Only used while the loop's own eyes never close (HeadEngine.begin)."""
    k = max(1, round(cycle / mean_gap))
    gap = cycle / k
    starts = [int(round((i + 0.5 + (0.12 if i % 2 else -0.12)) * gap)) for i in range(k)]
    # [frames before the first blink, open frames between blinks..., one long gap so no blink follows inside the loop]
    return [starts[0]] + [starts[i] - starts[i - 1] - BLINK_FRAMES for i in range(1, k)] + [cycle]


def render_cycle(pipe, loop, drive):
    """One full ping-pong period of a body loop (0..n-1, n-1..0) through the live head on silence, for cyclic playback."""
    n = len(loop)
    frames = render_silence(pipe, loop, 2 * n / 25.0, **{"delta_eye_open_n": blink_gaps(2 * n), **drive})
    if len(frames) < 2 * n:
        raise RuntimeError(f"rendered {len(frames)} frames for a {2 * n}-frame cycle")
    return frames[:2 * n]


def load_avatar(root, name, pipe, idle_seconds=6.0):
    directory = os.path.join(root, name)
    stamp = avatar_stamp(root, name)  # taken before reading: a change while loading makes the next session load it again
    meta = _read_meta(directory)
    settings = read_settings(directory)
    drive = drive_kwargs(settings)
    t0 = time.perf_counter()
    loops = {s: Loop.load(os.path.join(directory, s)) for s in LOOP_STATES if os.path.isdir(os.path.join(directory, s))}
    still = Loop.load(os.path.join(directory, "still")) if os.path.isdir(os.path.join(directory, "still")) else None
    # Without a speaking clip the idle clip is the template: the head speaks on the calm body instead of a frozen photo.
    speak_loop = loops.get("speaking") or loops.get("idle") or still
    if speak_loop is None:
        raise FileNotFoundError(f"avatar {name!r} has neither a still image nor a body loop")

    base = {s: BaseLoop(render_cycle(pipe, loops[s], drive), cyclic=True) for s in BASE_STATES if s in loops}
    if "idle" not in base:
        if still is None:
            raise FileNotFoundError(f"avatar {name!r} has no idle loop and no still image to render one from")
        base["idle"] = BaseLoop(render_silence(pipe, still, idle_seconds, **drive), cyclic=False)  # doubles as the warm-up
    elif "speaking" in loops:
        render_silence(pipe, speak_loop, 1.5, **drive)  # warm-up for the speaking template's ROI size
    base.setdefault("listening", base["idle"])
    base.setdefault("thinking", base["listening"])
    log.info("loaded avatar %s in %.1fs: speaking template %s frames, base %s, settings %s", name, time.perf_counter() - t0,
             len(speak_loop), {k: len(v) for k, v in base.items()}, settings)
    return Avatar(name, speak_loop, base, meta, stamp, settings)
