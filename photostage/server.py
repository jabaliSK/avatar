"""Independent LAN preview. Launch with python -m photostage.server.

LiveTalk or FlashHead animates a prepared head crop. The browser composites the
head over the photo or a pre-animated body clip, with per-frame detail guides.
MuseTalk (PHOTOSTAGE_RENDERER=musetalk) instead lip-syncs the avatar's own gesture
clips, or its photo, and streams whole pictures: see photostage/dubplan.py.

Two pages: the console at / (make, rename, set up and delete avatars) and the player
at /a/<key>, which shows one avatar with a chat bar below it. What the player shows is
a set of options saved per avatar (see photostage.library) that its address can override.
"""

import asyncio
import dataclasses
import functools
import io
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from contextlib import aclosing, asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from avatar.config import ROOT, Settings
from avatar.live import LiveWorker
from avatar.providers import PipelineError, Providers, run_in_thread
from photostage import builder as autobuild, dubplan, guide, library

BASE = Path(__file__).parent
DATA = ROOT / "data/photostage"
ASSETS = {"photo.png", "detail.png", "ref.png", "bias.png", "rig.json", "idle.json", "idle.bin"}
NO_STORE = {"Cache-Control": "no-store"}
# The body rig is optional: an avatar made before it existed, or whose rig could not be built, keeps its body still.
BODY_ASSETS = {"body.json", "body-field.png"}
BODY_LAYER = re.compile(r"(hand|plate)-\d+\.png")
# So are the body clips (see clippack.py): an avatar without them plays on with the procedural body.
CLIP_ASSETS = {"clips.json"}
CLIP_FILE = re.compile(r"clip-[a-z0-9-]+\.(mp4|webm|json|bin)")
WARMUP_PORTRAIT = ROOT / "external/FlashHead/examples/girl.png"
PYTHON = ROOT / ".venv-flashhead/bin/python"
BODY = [PYTHON, "-m", "photostage.body"]
BODY_SECONDS = 120
# Making, listing, renaming and deleting avatars ask for this token when one is set (the player and its assets never do).
ADMIN_TOKEN = os.environ.get("PHOTOSTAGE_ADMIN_TOKEN", "")
log = logging.getLogger("uvicorn.error")
gate = asyncio.Lock()
settings = Settings.from_env()
settings = dataclasses.replace(settings, renderer=os.environ.get("PHOTOSTAGE_RENDERER", settings.renderer))
providers = Providers(settings)
live = LiveWorker(dataclasses.replace(settings, data=DATA), head_only=True)
warmup = {"done": False, "error": None}
speaking = Counter()  # avatar key -> turns being spoken for it
# New avatars' animations are made in the background (photostage.builder), between conversations.
builds = autobuild.Builder(DATA, ROOT, live, gate, providers, speaking, ready=lambda: warmup["done"])


class Interface(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


async def start_renderer():
    await live.start()
    # Compiling the models takes minutes the first time (later starts reuse the cached
    # kernels). Pay it here, not during somebody's upload.
    patient = live.settings
    live.settings = dataclasses.replace(patient, render_timeout=1800)
    try:
        if WARMUP_PORTRAIT.is_file():
            with tempfile.TemporaryDirectory(prefix="photostage-warmup-") as scratch:
                folder = Path(scratch)
                image = WARMUP_PORTRAIT
                if settings.renderer == "livetalk":
                    await run_in_thread(prepare, image.read_bytes(), folder)
                    image = await run_in_thread(renderer_image, folder)
                await live.prepare("warmup", image, folder / "idle.mp4")
        else:
            log.warning("No warm-up portrait at %s; the first avatar will warm the renderer.", WARMUP_PORTRAIT)
        for folder in sorted(DATA.iterdir()):
            if library.folder(DATA, folder.name):
                rig = json.loads((folder / "rig.json").read_text())
                if rig.get("head_renderer", "flashhead") != settings.renderer:
                    await prepare_head(folder.name, folder)
    finally:
        live.settings = patient


async def warm():
    try:
        os.environ.setdefault("AVATAR_COMPILE", "1")
        await asyncio.gather(start_renderer(), providers.warmup())
        warmup["done"] = True
    except Exception as exc:  # reported through /status instead of lost in the task
        log.exception("PhotoStage warmup failed")
        warmup["error"] = str(exc) or type(exc).__name__


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        raise RuntimeError("PhotoStage requires one approved GPU: set CUDA_VISIBLE_DEVICES to 0 or 1.")
    if settings.renderer not in {"flashhead", "livetalk", "musetalk"}:
        raise RuntimeError("PhotoStage requires PHOTOSTAGE_RENDERER=flashhead, livetalk or musetalk.")
    tasks = [asyncio.create_task(warm()), asyncio.create_task(rig_existing())]
    if settings.renderer == "musetalk":
        tasks.append(builds.start())
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await live.close()


app = FastAPI(title="PhotoStage", lifespan=lifespan)
app.mount("/ui", Interface(directory=BASE / "web"), name="ui")


def unavailable():
    if warmup["error"] or live.error:
        return warmup["error"] or live.error
    if not warmup["done"] or not live.ready:
        return "The avatar renderer is still warming up. This takes about a minute after a restart (several minutes after the cache is cleared)."
    return None


def busy():
    """Whether a conversation (not a background build, which gives way to one) has the renderer."""
    return gate.locked() and not builds.holds


def directory(key):
    if (folder := library.folder(DATA, key)) is None:
        raise HTTPException(404, "Avatar not found")
    return folder


def admin(request: Request):
    given = request.headers.get("x-admin-token", "")
    if ADMIN_TOKEN and not secrets.compare_digest(given.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(401, "Enter the admin token.")


def local_origin(headers):
    origin = headers.get("origin")
    return not origin or urlsplit(origin).netloc == headers.get("host")


def prepare(content, folder):
    with Image.open(io.BytesIO(content), formats=["PNG", "JPEG", "WEBP"]) as img:
        if img.width * img.height > 16_000_000 or max(img.size) > 8192:
            raise ValueError("Use an image up to 16 megapixels and 8192 pixels per side.")
        image = ImageOps.exif_transpose(img).convert("RGBA")
        # Composite transparency on black, appropriate for the Holobox.
        background = Image.new("RGB", image.size)
        background.paste(image, mask=image.getchannel("A"))
        background.save(folder / "photo.png")
    proc = subprocess.run(
        [ROOT / ".venv-flashhead/bin/python", BASE / "enroll.py", folder],
        capture_output=True, text=True, timeout=60, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if proc.returncode == 2:
        raise ValueError(proc.stderr.strip().splitlines()[-1])
    proc.check_returncode()


def finalize_idle(folder):
    """Derive the neutral reference, colour bias and idle pack from the renderer's idle clip."""
    import av

    with av.open(str(folder / "idle.mp4")) as container:
        rate = container.streams.video[0].average_rate
        if rate is None or rate <= 0:
            raise PipelineError("The renderer produced an idle animation without a valid frame rate.")
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(frames) < 8:
        raise PipelineError("The renderer produced too short an idle animation.")
    guide.build_idle(folder, frames, fps=float(rate))


def renderer_image(folder):
    if settings.renderer != "livetalk":
        return folder / "photo.png"
    target = folder / "head.png"
    if not target.is_file():
        rig = json.loads((folder / "rig.json").read_text())
        with Image.open(folder / "photo.png") as source:
            source.convert("RGB").crop(tuple(rig["crop"])).resize(
                (guide.SIZE, guide.SIZE), Image.Resampling.LANCZOS,
            ).save(target)
    return target


def still_idle(folder):
    """The player's idle pack for an avatar that no head renderer has animated (one made under MuseTalk): the photo's head,
    still. An avatar that has one keeps it."""
    if all((folder / name).is_file() for name in ("ref.png", "bias.png", "idle.json", "idle.bin")):
        return
    rig = json.loads((folder / "rig.json").read_text())
    with Image.open(folder / "photo.png") as source:
        head = np.asarray(source.convert("RGB").crop(tuple(rig["crop"])).resize(
            (guide.SIZE, guide.SIZE), Image.Resampling.LANCZOS))
    guide.build_idle(folder, [head] * 8)


def dub_clips(folder):
    """The clips a MuseTalk reply can be pictured in: those of a pack of the whole avatar (mode avatar) that are on disk."""
    pack = folder / "clips.json"
    if not pack.is_file():
        return {}
    manifest = json.loads(pack.read_text())
    if manifest.get("mode") != "avatar":
        return {}
    files = {name: (spec.get("files") or {}).get("mp4") for name, spec in (manifest.get("clips") or {}).items()}
    return {name: clip for name, clip in dubplan.clips_of(manifest).items()
            if files.get(name) and (folder / files[name]).is_file()}


@functools.lru_cache(maxsize=32)
def video_poses(path, stamp):
    """dubplan.pose_of every frame of a clip; `stamp` (size, mtime) tells a clip's versions apart."""
    import av

    with av.open(str(path)) as container:
        return np.stack([dubplan.pose_of(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)])


@functools.lru_cache(maxsize=16)
def photo_pose(path, stamp):
    with Image.open(path) as photo:
        return dubplan.pose_of(np.asarray(photo.convert("RGB")))


def dub_poses(folder, clips):
    """The poses of `clips` (dub_clips) and of the photo, which a reply's rests are made to meet: dubplan.begin_rests.
    Decoding the clips takes seconds, so it is done once per clip version, when the avatar is prepared."""
    if not clips:
        return {}, None
    stamp = lambda path: (path.stat().st_size, path.stat().st_mtime_ns)
    specs = json.loads((folder / "clips.json").read_text())["clips"]
    poses, anchor = {}, None
    for name, clip in clips.items():
        path = folder / specs[name]["files"]["mp4"]
        try:
            found = video_poses(path, stamp(path))
        except Exception as error:  # without its poses a clip's rests only begin where they would anyway
            log.warning("No poses of %s: %s", path.name, error)
            continue
        if len(found) >= clip.frames:
            poses[name] = found
    photo = folder / "photo.png"
    try:
        anchor = photo_pose(photo, stamp(photo))
    except (OSError, UnidentifiedImageError) as error:
        log.warning("No pose of %s: %s", photo, error)
    return poses, anchor


async def prepare_head(key, folder):
    if settings.renderer == "musetalk":
        # MuseTalk works on the photo and the clips themselves: nothing of another renderer's is replaced.
        await run_in_thread(still_idle, folder)
        await run_in_thread(lambda: dub_poses(folder, dub_clips(folder)))
        await live.prepare(key, folder / "photo.png", folder / "idle.mp4")
        return
    pack = folder / "clips.json"
    if settings.renderer == "livetalk" and pack.is_file():
        manifest = json.loads(pack.read_text())
        if manifest.get("mode") == "avatar" and manifest.get("faceMode") != "head":
            raise PipelineError("LiveTalk head replacement requires a clip pack built with --face-mode head.")
    image = await run_in_thread(renderer_image, folder)
    rig = json.loads((folder / "rig.json").read_text())
    if rig.get("head_renderer", "flashhead") == settings.renderer and (folder / "idle.mp4").is_file():
        await live.prepare(key, image, folder / "idle.mp4")
        return
    # A different renderer must never reuse the old idle/reference/guide coordinate system.
    with tempfile.TemporaryDirectory(prefix="head-", dir=folder) as scratch:
        stage = Path(scratch)
        (stage / "photo.png").symlink_to((folder / "photo.png").resolve())
        shutil.copy2(folder / "rig.json", stage / "rig.json")
        await live.prepare(key, image, stage / "idle.mp4")
        await run_in_thread(finalize_idle, stage)
        rig["head_renderer"] = settings.renderer
        (stage / "rig.json").write_text(json.dumps(rig))
        for name in ("idle.mp4", "ref.png", "bias.png", "idle.bin", "idle.json", "rig.json"):
            (stage / name).replace(folder / name)
    reference.cache_clear()


@functools.lru_cache(maxsize=16)
def reference(folder):
    return guide.Reference(cv2.imread(str(folder / "ref.png")))


def remove_body(folder):
    """Whatever a rig that did not finish left behind: the rig is only valid with its body.json, which is written last."""
    for path in (*folder.glob("hand-*.png"), *folder.glob("plate-*.png"),
                 *(folder / name for name in (*BODY_ASSETS, "body.json.tmp"))):
        path.unlink(missing_ok=True)


async def build_body(folder):
    """Rig the photo's body (photostage.body) in a process of its own, on the CPU: the GPU belongs to the renderer. The
    avatar talks without a rig, so nothing that happens here fails the caller. An avatar without one keeps its body still."""
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *BODY, str(folder), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
        output, _ = await asyncio.wait_for(process.communicate(), BODY_SECONDS)
        lines = output.decode(errors="replace").strip().splitlines()
        if process.returncode == 0:
            log.info("%s", next((line for line in lines if line.startswith("body rig")), "The body rig was built."))
        else:
            log.warning("The body rig failed with exit code %s: %s", process.returncode, " | ".join(lines[-3:]))
    except asyncio.TimeoutError:
        log.warning("The body rig took more than %s s and was skipped.", BODY_SECONDS)
    except OSError:
        log.exception("The body rig could not be started.")
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        if not (folder / "body.json").is_file():
            remove_body(folder)


async def rig_existing():
    """Give the avatars made before the body rig existed theirs, one after the other: each takes seconds of CPU. A rig
    that could not be built is still written (empty), so no avatar is tried twice."""
    for folder in sorted(DATA.iterdir()):
        if (re.fullmatch(r"[a-f0-9]{32}", folder.name) and (folder / "photo.png").is_file()
                and (folder / "rig.json").is_file() and not (folder / "body.json").is_file()):
            await build_body(folder)


@app.get("/")
async def console():
    return FileResponse(BASE / "web/console.html", headers=NO_STORE)


@app.get("/a/{key}")
async def player(key: str):
    # The page itself explains an avatar that is not there (any more), which a bare error would not.
    return FileResponse(BASE / "web/player.html", status_code=200 if library.folder(DATA, key) else 404, headers=NO_STORE)


@app.get("/api/options")
async def option_schema():
    return {"options": library.schema()}


@app.get("/api/avatars", dependencies=[Depends(admin)])
async def avatars():
    return {"avatars": await run_in_thread(library.listing, DATA)}


@app.get("/api/avatars/{key}")
async def avatar(key: str):
    return library.describe(directory(key))


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    options: dict[str, Any] | None = None


@app.patch("/api/avatars/{key}", dependencies=[Depends(admin)])
async def change(key: str, request: Request, body: Change):
    if not local_origin(request.headers):
        raise HTTPException(403)
    folder = directory(key)
    try:
        library.update(folder, body.name, body.options)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return library.describe(folder)


@app.delete("/api/avatars/{key}", status_code=204, dependencies=[Depends(admin)])
async def delete(key: str, request: Request):
    if not local_origin(request.headers):
        raise HTTPException(403)
    folder = directory(key)
    if speaking[key]:
        raise HTTPException(409, "This avatar is speaking right now. Try again when it has finished.")
    if key in builds.queued:
        raise HTTPException(409, "This avatar's animations are being made. Delete it once they are done or have failed.")
    await run_in_thread(library.remove, folder)
    return Response(status_code=204)


@app.get("/assets/{key}/{asset}")
async def asset(key: str, asset: str):
    folder = directory(key)
    if asset == "thumb.jpg":
        try:
            return FileResponse(await run_in_thread(library.thumbnail, folder))
        except (OSError, UnidentifiedImageError) as exc:
            raise HTTPException(404, "Asset not found") from exc
    path = folder / asset
    # The body rig and the clips are rebuilt in place, so a page that has the old ones must ask again.
    rebuilt = (asset in library.PUBLISHED or asset in BODY_ASSETS or asset in CLIP_ASSETS
               or BODY_LAYER.fullmatch(asset) or CLIP_FILE.fullmatch(asset))
    if not (asset in ASSETS or rebuilt) or not path.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(path, headers={"Cache-Control": "no-cache"} if rebuilt else None)


@app.post("/avatars", status_code=201, dependencies=[Depends(admin)])
async def create(request: Request, image: UploadFile = File(...), consent: bool = Form(...), name: str = Form("")):
    if not local_origin(request.headers) or not consent:
        raise HTTPException(403, "Permission to animate this photo is required.")
    content = await image.read(15 * 1024**2 + 1)
    if len(content) > 15 * 1024**2:
        raise HTTPException(413, "Maximum upload is 15 MB.")
    if message := unavailable():
        raise HTTPException(503, message)
    if busy():
        raise HTTPException(409, "Preview is busy; try again shortly.")
    async with gate:
        with tempfile.TemporaryDirectory(dir=DATA) as stage:
            folder, key = Path(stage), uuid.uuid4().hex
            try:
                await run_in_thread(prepare, content, folder)
                # The rig is CPU work, so it is made while the renderer prepares the head.
                rigging = asyncio.create_task(build_body(folder))
                try:
                    await prepare_head(key, folder)
                    await rigging
                finally:
                    rigging.cancel()
                    await asyncio.gather(rigging, return_exceptions=True)
            except (ValueError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
                raise HTTPException(422, str(exc)) from exc
            except PipelineError as exc:
                raise HTTPException(502, str(exc)) from exc
            library.write_meta(folder, {"name": library.clean_name(name) or library.name_from_upload(image.filename)
                                        or library.default_name(key), "created": library.stamp(), "options": {}})
            with suppress(OSError, UnidentifiedImageError):  # made when the console first asks for it, if not now
                library.thumbnail(folder)
            animate = settings.renderer == "musetalk"
            if animate:  # published as being built, so that nobody talks to it before its animations are there
                builds.begin(folder, "avatar")
            folder.rename(DATA / key)
    if animate:
        builds.enqueue(key)
    return {"url": f"/a/{key}", "key": key, "build": library.build_state(DATA / key)}


@app.post("/api/avatars/{key}/rebuild", dependencies=[Depends(admin)])
async def rebuild(key: str, request: Request):
    """Make a failed avatar's animations again, from the stage that failed."""
    if not local_origin(request.headers):
        raise HTTPException(403)
    folder = directory(key)
    if settings.renderer != "musetalk":
        raise HTTPException(409, "Animations are made with the MuseTalk renderer only.")
    if library.build_state(folder)["state"] != "failed":
        raise HTTPException(409, "Only an avatar whose animations failed can be rebuilt.")
    builds.begin(folder, "avatar")
    builds.enqueue(key)
    return library.describe(folder)


@app.get("/status")
def status():
    message = unavailable()
    state = "error" if warmup["error"] or live.error else "warming" if message else "busy" if busy() else "ready"
    return {"ready": message is None, "busy": busy(), "state": state, "message": message,
            "renderer": settings.renderer}


@app.post("/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    if not local_origin(request.headers):
        raise HTTPException(403)
    raw = await audio.read(10 * 1024**2 + 1)
    if len(raw) > 10 * 1024**2:
        raise HTTPException(413, "Maximum recording is 10 MB.")
    if busy():
        raise HTTPException(409, "Preview is busy.")
    async with gate:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording"
            path.write_bytes(raw)
            try:
                return {"text": await providers.transcribe(path)}
            except PipelineError as exc:
                raise HTTPException(422, str(exc)) from exc


@app.websocket("/turn/{key}")
async def turn(ws: WebSocket, key: str):
    if not local_origin(ws.headers):
        await ws.close(code=1008)
        return
    await ws.accept()
    credit = asyncio.Event()
    expected = None
    phase = asyncio.Queue(maxsize=1)
    awaiting_phase = False
    tasks, counted = [], False

    async def listen():
        nonlocal awaiting_phase
        # Segment acknowledgements are the player's backpressure signal.
        while True:
            message = await ws.receive_json()
            if isinstance(message, dict) and message.get("type") == "ack" and message.get("index") == expected:
                credit.set()
            elif isinstance(message, dict) and message.get("type") == "phase" and awaiting_phase:
                awaiting_phase = False
                phase.put_nowait(message.get("frame"))
            else:
                raise ValueError("Unexpected message from the player.")

    async def speak(folder, text, mode, history, body):
        nonlocal expected, awaiting_phase
        started = time.monotonic()
        if busy():
            raise ValueError("Another avatar is busy. Please retry.")
        async with gate:
            await ws.send_json({"kind": "stage", "stage": "thinking"})
            reply = text if mode == "speak" else await providers.chat([*history, {"role": "user", "content": text}])
            dub = settings.renderer == "musetalk"
            # A dubbed reply is whole pictures: of the clips the player plays around it, or of the photo where it plays none.
            clips = await run_in_thread(dub_clips, folder) if dub and body["clips"] else {}
            continuous = bool(clips and json.loads((folder / "clips.json").read_text()).get("continuous"))
            if continuous and (not body["continuous"] or len(clips) != 1):
                raise ValueError("Please reload the player to use this continuous-motion pack.")
            await ws.send_json({"kind": "speech", "reply": reply,
                                **({"picture": "clips" if clips else "photo"} if dub else {})})
            with tempfile.TemporaryDirectory() as tmp:
                wav, video = Path(tmp) / "reply.wav", Path(tmp) / "reply.mp4"
                await providers.speak(reply, wav)
                # Lets the player hold back enough speech to ride out a renderer slower than real time.
                total = round(sf.info(str(wav)).duration, 3)
                await prepare_head(key, folder)
                plan = None
                if dub:
                    if continuous:
                        clip = next(iter(clips.values()))
                        awaiting_phase = True
                        await ws.send_json({"kind": "motion", "clip": clip.name, "fps": clip.fps, "frames": clip.frames,
                                            "seconds": dubplan.continuous(clip, total, 0).seconds})
                        start_frame = await asyncio.wait_for(phase.get(), 15)
                        plan = dubplan.continuous(clip, total, start_frame)
                    else:
                        poses, pose = await run_in_thread(dub_poses, folder, clips)
                        plan = dubplan.plan(clips, reply, total, pace=body["pace"], poses=poses, anchor=pose,
                                            still=dubplan.Clip("photo", "idle", "idle", 1, dubplan.FPS, True))
                    total = round(plan.seconds, 3)
                    log.info("Dubbed reply of %.2f s in %s", plan.seconds,
                             ", ".join(f"{name} {start:.1f}-{end:.1f} s" for name, start, end in plan.gestures) or "rests")
                first, metrics, guide_seconds = None, {}, 0.0
                guided = not dub and json.loads((folder / "rig.json").read_text())["detail"] is not None
                anchor = reference(folder) if guided else None
                async with aclosing(live.stream(wav, video, plan.message() if plan else None)) as events:
                    async for event in events:
                        if event["event"] == "segment":
                            segment = {k: v for k, v in event.items() if k != "event"}
                            segment["total"] = total
                            if guided:
                                tick = time.monotonic()
                                segment["guides"] = await run_in_thread(guide.guide_segment, anchor, segment["frames"])
                                guide_seconds += time.monotonic() - tick
                            first = first or round(time.monotonic() - started, 3)
                            credit.clear()
                            expected = event["index"]
                            await ws.send_json({"kind": "segment", **segment})
                            await asyncio.wait_for(credit.wait(), 30)
                        elif event["event"] == "rendered":
                            metrics = event
                if first is None:
                    raise PipelineError("The renderer produced no speech video.")
                await ws.send_json({"kind": "done", "first_segment_seconds": first,
                                    "guide_seconds": round(guide_seconds, 3), "metrics": metrics})

    try:
        folder = directory(key)
        speaking[key] += 1
        counted = True
        if library.build_state(folder)["state"] == "building":
            raise ValueError("This avatar is still being created. It can talk once its animations are ready.")
        req = await asyncio.wait_for(ws.receive_json(), 15)
        if not isinstance(req, dict) or not isinstance(req.get("text"), str) or not 1 <= len(req["text"].strip()) <= 500:
            raise ValueError("Enter 1 to 500 characters.")
        history = req.get("history", [])
        if not isinstance(history, list) or len(history) > 8 or any(
            not isinstance(m, dict) or m.get("role") not in {"user", "assistant"} or
            not isinstance(m.get("content"), str) or len(m["content"]) > 1600 for m in history
        ):
            raise ValueError("Invalid conversation history.")
        # What the player plays around the reply: its clips (or not), at its amount of motion.
        motion = req.get("motion", 1)
        if not isinstance(req.get("clips", True), bool) or not isinstance(req.get("continuous", False), bool) \
                or isinstance(motion, bool) or not isinstance(motion, (int, float)) \
                or not 0 <= motion <= 2:
            raise ValueError("Invalid player options.")
        body = {"clips": req.get("clips", True) and motion > 0, "pace": min(2.0, max(0.25, float(motion))),
                "continuous": req.get("continuous", False)}
        if message := unavailable():
            raise PipelineError(message)
        tasks = [
            asyncio.create_task(speak(folder, req["text"].strip(), req.get("mode"), history, body)),
            asyncio.create_task(listen()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # The speech task normally finishes first; a listener result means the player left or misbehaved.
        for task in tasks:
            if task.done():
                task.result()
    except WebSocketDisconnect:
        pass
    except HTTPException as exc:
        with suppress(WebSocketDisconnect, RuntimeError):
            await ws.send_json({"kind": "error", "message": str(exc.detail)})
    except (ValueError, PipelineError, TimeoutError) as exc:
        message = str(exc)
        if isinstance(exc, TimeoutError):
            message = ("The player did not synchronize motion. Please reload." if awaiting_phase else
                       "Playback stopped acknowledging media. Please retry.")
        with suppress(WebSocketDisconnect, RuntimeError):
            await ws.send_json({"kind": "error", "message": message})
    except (OSError, RuntimeError):
        log.exception("PhotoStage turn failed")
        with suppress(WebSocketDisconnect, RuntimeError):
            await ws.send_json({"kind": "error", "message": "Rendering failed. See the PhotoStage server log."})
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if counted:
            speaking[key] -= 1
            if not speaking[key]:
                del speaking[key]
        with suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("PHOTOSTAGE_HOST", "127.0.0.1"), port=8001,
                ws_max_size=48 * 1024)
