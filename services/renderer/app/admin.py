"""Avatar administration API (/v1), used by the Studio in the gateway and by the generation worker.

    GET    /v1/avatars                     every avatar: title, description, frame size, loops, settings, size on disk, preview
    GET    /v1/avatars/{id}                one of them
    GET    /v1/avatars/{id}/thumbnail      JPEG preview, made once from the source image
    PATCH  /v1/avatars/{id}                rename / change the description
    PUT    /v1/avatars/{id}/settings       expression / head_motion strength
    DELETE /v1/avatars/{id}
    GET    /v1/avatars/{id}/status         loaded in memory / loading / busy
    POST   /v1/avatars/{id}/warm           load the avatar in the background so that the next conversation starts at once
    POST   /v1/registrations               register an image (and Stage A clips) as an avatar -> 202 and a registration id
    GET    /v1/registrations/{rid}         progress of that registration

Listing, previews, rename and settings only touch files, so they work while the models are still loading. Registration and
deletion are accepted once the renderer is ready: they take the conversation lock (the TensorRT engines are shared with the live
path, so nothing else may render meanwhile) and answer 409 while a conversation or another registration is running.

Display data (title, description, dates) lives in profile.json, apart from meta.json and settings.json: changing those two makes
the avatar load again on the next conversation, a rename must not.

The API has no authentication of its own; the renderer's port is bound to the loopback interface and the Studio in the gateway
checks the access key. Registration reads files, so it only accepts paths below KKA_STUDIO_ROOTS.
"""
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .avatar_store import LOOP_STATES, add_state_loop, avatar_stamp, list_avatars, onboard_image, read_settings, video_size, write_settings

log = logging.getLogger("kka.admin")

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")  # ids the Studio creates
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # avatars made with tools/onboard.py may use any such name
STAGING = ".staging"  # registrations are built here and moved into place when complete
THUMBNAIL_SIDE = 640
MAX_SIZE = int(os.environ.get("KKA_MAX_SIZE", "1280"))
STUDIO_ROOTS = tuple(p for p in os.environ.get("KKA_STUDIO_ROOTS", "/studio:/loops").split(":") if p)
KEEP_REGISTRATIONS = 20


# ----------------------------------------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------------------------------------
def _read_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path, data):
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tree_size(directory):
    total = 0
    for base, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


# control characters become spaces; bidirectional overrides (which can make a name read backwards) are dropped
_CLEAN = {**dict.fromkeys([*range(0x20), *range(0x7F, 0xA0)], " "), **dict.fromkeys([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])}


def _text(value):
    """One line of text: control characters and runs of white space collapse to single spaces."""
    return " ".join(value.translate(_CLEAN).split())


def pretty_title(name):
    return name.replace("-", " ").title()


def mode_of(states):
    """full: idle and speaking body loops; partial: some other loop; quick: one photo and the live face only."""
    if "idle" in states and "speaking" in states:
        return "full"
    return "partial" if states else "quick"


def summary(root, name):
    """What the library shows for one avatar; None if it vanished meanwhile."""
    stamp = avatar_stamp(root, name)
    if stamp is None:
        return None
    directory = os.path.join(root, name)
    meta = _read_json(os.path.join(directory, "meta.json"))
    profile = _read_json(os.path.join(directory, "profile.json"))
    states = [s for s in LOOP_STATES if s in (meta.get("states") or [])]
    source = os.path.join(directory, meta.get("source") or "-")
    created = profile.get("created") or _iso(os.path.getmtime(source) if os.path.isfile(source) else stamp / 1e9)
    return {
        "id": name,
        "title": profile.get("title") or pretty_title(name),
        "subject": profile.get("subject", ""),
        "frame_size": meta.get("frame_size"),
        "states": states,
        "mode": mode_of(states),
        "settings": read_settings(directory),
        "size_bytes": _tree_size(directory),
        "created": created,
        "updated": _iso(stamp / 1e9),
        "stamp": stamp,
        "thumbnail": f"/v1/avatars/{name}/thumbnail?v={stamp}",
    }


def make_thumbnail(directory):
    """Path of the avatar's JPEG preview, (re)made from the source image when it is missing or older than meta.json."""
    meta_path = os.path.join(directory, "meta.json")
    thumb = os.path.join(directory, "thumb.jpg")
    if os.path.isfile(thumb) and os.path.getmtime(thumb) >= os.path.getmtime(meta_path):
        return thumb
    source = os.path.join(directory, _read_json(meta_path).get("source") or "-")
    image = cv2.imread(source, cv2.IMREAD_COLOR) if os.path.isfile(source) else None
    if image is None:  # no usable source file: the first frame of the registered photo
        still = os.path.join(directory, "still", "frames.npy")
        if not os.path.isfile(still):
            return None
        image = cv2.cvtColor(np.asarray(np.load(still, mmap_mode="r")[0]), cv2.COLOR_RGB2BGR)
    scale = THUMBNAIL_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    tmp = f"{thumb}.{threading.get_ident()}.tmp.jpg"
    cv2.imwrite(tmp, image, [cv2.IMWRITE_JPEG_QUALITY, 86])
    os.replace(tmp, thumb)
    return thumb


def studio_file(path):
    """The real path of an existing file below one of the studio roots; None for anything else."""
    real = os.path.realpath(path)
    for root in STUDIO_ROOTS:
        base = os.path.realpath(root)
        if os.path.commonpath([real, base]) == base and os.path.isfile(real):
            return real
    return None


# ----------------------------------------------------------------------------------------------------------------------------
# Request bodies
# ----------------------------------------------------------------------------------------------------------------------------
class ProfilePatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=80)
    subject: Optional[str] = Field(default=None, max_length=600)


class SettingsBody(BaseModel):
    expression: Optional[float] = Field(default=None, allow_inf_nan=False)
    head_motion: Optional[float] = Field(default=None, allow_inf_nan=False)


class Registration(BaseModel):
    id: str
    image: str
    loops: Dict[str, str] = Field(default_factory=dict)  # state -> Stage A clip
    title: Optional[str] = Field(default=None, max_length=80)
    subject: Optional[str] = Field(default=None, max_length=600)


# ----------------------------------------------------------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------------------------------------------------------
def build_router(rt, root):
    """`rt` is the server's Runtime (status, engine, registrar, session, load_lock, loaded, get_avatar); `root` the avatars folder."""
    router = APIRouter(prefix="/v1")
    registrations = {}
    reg_lock = threading.Lock()
    warming = {}  # avatar -> thread loading it
    warm_errors = {}

    def existing(name):
        if not NAME_RE.match(name) or avatar_stamp(root, name) is None:
            raise HTTPException(404, f"unknown avatar {name!r}")
        return os.path.join(root, name)

    def registering():
        with reg_lock:
            return next((r["avatar"] for r in registrations.values() if r["state"] == "running"), None)

    # ------------------------------------------------------------------------------------------------------------ library
    def state():
        return {"renderer": {"status": rt.status, "detail": rt.detail}, "busy": rt.session.locked(), "registering": registering(),
                "loaded": list(rt.loaded)}

    @router.get("/status")
    def renderer_state():
        return state()

    @router.get("/avatars")
    def list_all():
        avatars = [s for s in (summary(root, n) for n in list_avatars(root) if NAME_RE.match(n)) if s]
        avatars.sort(key=lambda a: a["updated"], reverse=True)
        return {"avatars": avatars, **state()}

    @router.get("/avatars/{name}")
    def one(name: str):
        existing(name)
        return summary(root, name)

    @router.get("/avatars/{name}/thumbnail")
    def thumbnail(name: str):
        path = make_thumbnail(existing(name))
        if path is None:
            raise HTTPException(404, "this avatar has no preview")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})

    @router.patch("/avatars/{name}")
    def patch(name: str, body: ProfilePatch):
        directory = existing(name)
        path = os.path.join(directory, "profile.json")
        profile = _read_json(path)
        if body.title is not None:
            title = _text(body.title)
            if not title:
                raise HTTPException(422, "the name cannot be empty")
            profile["title"] = title
        if body.subject is not None:
            profile["subject"] = _text(body.subject)
        profile["updated"] = _iso(time.time())
        _write_json(path, profile)
        return summary(root, name)

    @router.put("/avatars/{name}/settings")
    def put_settings(name: str, body: SettingsBody):
        directory = existing(name)
        return write_settings(directory, body.model_dump(exclude_none=True))

    @router.delete("/avatars/{name}")
    def delete(name: str):
        directory = existing(name)
        if not rt.session.acquire(blocking=False):
            raise HTTPException(409, "A conversation or a registration is in progress; try again when it has finished.")
        try:
            with rt.load_lock:
                rt.loaded.pop(name, None)
                shutil.rmtree(directory)
        finally:
            rt.session.release()
        log.info("deleted avatar %s", name)
        return {"deleted": name}

    # ------------------------------------------------------------------------------------------------------------ warm-up
    def status_of(name):
        stamp = avatar_stamp(root, name)
        cached = rt.loaded.get(name)
        thread = warming.get(name)
        return {"id": name, "renderer": rt.status, "loaded": cached is not None and cached.stamp == stamp,
                "loading": thread is not None and thread.is_alive(), "busy": rt.session.locked(), "error": warm_errors.get(name)}

    def load(name):
        try:
            rt.get_avatar(name)
        except Exception as exc:
            log.exception("loading %s failed", name)
            warm_errors[name] = f"{type(exc).__name__}: {exc}"

    @router.get("/avatars/{name}/status")
    def status(name: str):
        existing(name)
        return status_of(name)

    @router.post("/avatars/{name}/warm")
    def warm(name: str):
        existing(name)
        state = status_of(name)
        # While a conversation or a registration runs the GPU is not disturbed; the caller sees busy=true and asks again later.
        if state["renderer"] == "ready" and not (state["loaded"] or state["loading"] or state["busy"]):
            warm_errors.pop(name, None)
            warming[name] = thread = threading.Thread(target=load, args=(name,), name=f"warm-{name}", daemon=True)
            thread.start()
            state = status_of(name)
        return state

    # ------------------------------------------------------------------------------------------------------------ registration
    def step(reg, name):
        reg["step"] = name
        log.info("registration %s (%s): %s", reg["id"], reg["avatar"], name)

    def install(reg, work, req):
        """Move the finished avatar into place, keeping what the user set on an avatar that is replaced."""
        final = os.path.join(root, req.id)
        old = _read_json(os.path.join(final, "profile.json"))
        old_settings = os.path.join(final, "settings.json")
        if os.path.isfile(old_settings):
            shutil.copy(old_settings, os.path.join(work, "settings.json"))
        now = _iso(time.time())
        _write_json(os.path.join(work, "profile.json"), {
            "title": _text(req.title or "") or old.get("title") or pretty_title(req.id),
            "subject": _text(req.subject) if req.subject is not None else old.get("subject", ""),
            "created": old.get("created", now), "updated": now})
        rt.loaded.pop(req.id, None)
        previous = os.path.join(root, STAGING, req.id + ".previous")
        shutil.rmtree(previous, ignore_errors=True)
        if os.path.isdir(final):
            os.replace(final, previous)
        try:
            os.replace(work, final)
        except BaseException:
            if os.path.isdir(previous):
                os.replace(previous, final)  # the avatar that was there stays usable
            raise
        shutil.rmtree(previous, ignore_errors=True)

    def register(reg, req):
        staging = os.path.join(root, STAGING)
        work = os.path.join(staging, req.id)
        os.makedirs(staging, exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        kw = {**rt.engine.kw, "max_size": MAX_SIZE}
        size = None
        if req.loops:  # the clips decide the frame geometry; the photo is fitted to it
            sizes = {state: video_size(path) for state, path in req.loops.items()}
            if len(set(sizes.values())) != 1:
                raise ValueError(f"the clips must all have one size, got {sizes}")
            size = next(iter(sizes.values()))
            kw["max_size"] = max(size)
        step(reg, "still")
        onboard_image(staging, req.id, req.image, rt.registrar, rt.engine, kw, size=size)
        reg["done"] = 1
        for state in (s for s in LOOP_STATES if s in req.loops):
            step(reg, state)
            add_state_loop(staging, req.id, state, req.loops[state], rt.registrar, rt.engine, kw, pin=True)
            reg["done"] += 1
        install(reg, work, req)

    def run(reg, req):
        t0 = time.perf_counter()
        error = None
        try:
            with rt.load_lock:  # an avatar that is being loaded finishes first
                register(reg, req)
            log.info("registered %s in %.0f s", req.id, time.perf_counter() - t0)
        except Exception as exc:
            log.exception("registering %s failed", req.id)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                shutil.rmtree(os.path.join(root, STAGING, req.id), ignore_errors=True)
                torch.cuda.empty_cache()
            finally:
                rt.session.release()
        # Last, after the lock is free: a client that sees the end may open a conversation at once.
        reg["error"], reg["finished"] = error, time.time()
        reg["state"] = "failed" if error else "done"

    @router.post("/registrations", status_code=202)
    def start_registration(req: Registration):
        if not ID_RE.match(req.id):
            raise HTTPException(422, "the id must be 1-40 characters: lower-case letters, digits and hyphens")
        image = studio_file(req.image)
        if image is None:
            raise HTTPException(422, "the image is not a file below the studio folders")
        clips = {}
        for state, path in req.loops.items():
            if state not in LOOP_STATES:
                raise HTTPException(422, f"unknown loop state {state!r}; choose from {list(LOOP_STATES)}")
            clips[state] = studio_file(path)
            if clips[state] is None:
                raise HTTPException(422, f"the {state} clip is not a file below the studio folders")
        if rt.status != "ready":
            raise HTTPException(503, f"The renderer is {rt.status}: {rt.detail}")
        if not rt.session.acquire(blocking=False):
            raise HTTPException(409, "A conversation or another registration is in progress; try again when it has finished.")
        req = req.model_copy(update={"image": image, "loops": clips})
        reg = {"id": os.urandom(4).hex(), "avatar": req.id, "state": "running", "step": "starting", "done": 0,
               "steps": 1 + len(clips), "started": time.time(), "finished": None, "error": None}
        try:
            with reg_lock:
                registrations[reg["id"]] = reg
                for old in sorted(registrations.values(), key=lambda r: r["started"])[:-KEEP_REGISTRATIONS]:
                    registrations.pop(old["id"], None)
            threading.Thread(target=run, args=(reg, req), name=f"register-{req.id}", daemon=True).start()
        except BaseException:
            rt.session.release()
            raise
        return reg

    @router.get("/registrations/{rid}")
    def registration(rid: str):
        with reg_lock:
            reg = registrations.get(rid)
        if reg is None:
            raise HTTPException(404, "unknown registration")
        return reg

    return router
