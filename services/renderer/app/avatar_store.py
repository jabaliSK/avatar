"""Avatar assets on disk and in memory.

    avatars/<name>/
        meta.json                       name, max_size, source image, which states have a Stage A body loop
        source.<ext>                    the onboarding image
        still/                          Loop cache of the single image (head template when there is no speaking loop)
        idle/ listening/ thinking/ speaking/   optional body loops from Stage A (each a Loop cache of the generated clip)

A state without a body loop falls back (thinking -> listening -> idle). Without any idle loop the idle frames are rendered
once at load time by running the live head on silence, which gives natural blinks and micro head motion on the still image.
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

from .avatar import Loop
from .head_engine import HUBERT_STEP
from .pinning import pin_static

log = logging.getLogger("kka.avatars")

BASE_STATES = ("idle", "listening", "thinking")
LOOP_STATES = BASE_STATES + ("speaking",)


class Avatar:
    def __init__(self, name, speak_loop, base, meta, stamp=None):
        self.name = name
        self.speak_loop = speak_loop  # template the live head is composited onto while speaking
        self.base = base              # state -> uint8 [N,H,W,3], played ping-pong while not speaking
        self.meta = meta
        self.stamp = stamp            # avatar_stamp() when it was loaded

    @property
    def size(self):
        return self.speak_loop.size  # (width, height)


def list_avatars(root):
    if not os.path.isdir(root):
        return []
    return sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, "meta.json")))


def avatar_stamp(root, name):
    """Changes whenever the avatar is onboarded again or gets a loop (meta.json is always written last); None if it is not there."""
    try:
        return os.stat(os.path.join(root, name, "meta.json")).st_mtime_ns
    except FileNotFoundError:
        return None


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


def render_silence(pipe, loop, seconds):
    """Run the live head on silence and return the composited frames as uint8 [N,H,W,3] (no real-time gate)."""
    gate, pipe.gate = pipe.gate, None
    try:
        pipe.start(loop)
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


def load_avatar(root, name, pipe, idle_seconds=6.0):
    directory = os.path.join(root, name)
    stamp = avatar_stamp(root, name)  # taken before reading: a change while loading makes the next session load it again
    meta = _read_meta(directory)
    t0 = time.perf_counter()
    loops = {s: Loop.load(os.path.join(directory, s)) for s in LOOP_STATES if os.path.isdir(os.path.join(directory, s))}
    still = Loop.load(os.path.join(directory, "still")) if os.path.isdir(os.path.join(directory, "still")) else None
    speak_loop = loops.get("speaking") or still or loops.get("idle")
    if speak_loop is None:
        raise FileNotFoundError(f"avatar {name!r} has neither a still image nor a speaking/idle loop")

    base = {s: loops[s].frames for s in BASE_STATES if s in loops}
    if "idle" not in base:
        if still is None:
            raise FileNotFoundError(f"avatar {name!r} has no idle loop and no still image to render one from")
        base["idle"] = render_silence(pipe, still, idle_seconds)  # doubles as the engine warm-up
    else:
        render_silence(pipe, speak_loop, 1.5)  # warm-up: TensorRT contexts, allocator, pinned buffers for this ROI size
    base.setdefault("listening", base["idle"])
    base.setdefault("thinking", base["listening"])
    log.info("loaded avatar %s in %.1fs: speaking template %s frames, base %s", name, time.perf_counter() - t0,
             len(speak_loop), {k: len(v) for k, v in base.items()})
    return Avatar(name, speak_loop, base, meta, stamp)
