import asyncio
import io
import logging
import sqlite3
import time
import uuid
from contextlib import aclosing, asynccontextmanager, closing, contextmanager, suppress
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from avatar.config import ROOT, Settings
from avatar.providers import PipelineError, Providers
from avatar.live import LiveWorker

logger = logging.getLogger(__name__)
MAX_IMAGE = 15 * 1024 * 1024
MAX_AUDIO = 10 * 1024 * 1024


def create_app(settings: Settings | None = None, providers: Providers | None = None, live_worker=None):
    settings = settings or Settings.from_env()
    real_providers = providers is None
    providers = providers or Providers(settings)
    settings.data.mkdir(parents=True, exist_ok=True)
    media = settings.data / "media"
    media.mkdir(exist_ok=True)
    lock = asyncio.Lock()
    live = live_worker
    if live is None and real_providers and settings.live_streaming:
        if settings.renderer not in {"flashhead", "livetalk"}:
            raise ValueError("Live streaming requires a FlashHead or LiveTalk renderer.")
        live = LiveWorker(settings)
    warming = None
    warm_error = None
    ai_ready = False

    async def warm():
        nonlocal ai_ready, warm_error
        try:
            await asyncio.gather(live.start(), providers.warmup())
            ai_ready = True
        except (OSError, ImportError, RuntimeError, ValueError, PipelineError, TimeoutError) as error:
            warm_error = str(error)
            logger.exception("Live pipeline warmup failed")

    @asynccontextmanager
    async def lifespan(app):
        nonlocal warming
        if live:
            warming = asyncio.create_task(warm())
        try:
            yield
        finally:
            if warming and not warming.done():
                warming.cancel()
                with suppress(asyncio.CancelledError):
                    await warming
            if live:
                await live.close()

    @contextmanager
    def database():
        with closing(sqlite3.connect(settings.data / "avatar.sqlite3")) as connection:
            with connection:
                connection.row_factory = sqlite3.Row
                yield connection

    with database() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS avatars (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, avatar_id TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                user_text TEXT NOT NULL, assistant_text TEXT NOT NULL,
                video TEXT NOT NULL, elapsed REAL NOT NULL, created REAL NOT NULL
            );
        """)

    app = FastAPI(title="Avatar Studio", version="0.1.0", lifespan=lifespan)
    app.state.providers = providers

    def lookup(table, identifier):
        # Table names come only from the fixed call sites below.
        with database() as db:
            row = db.execute(f"SELECT * FROM {table} WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise HTTPException(404, "Not found")
        return dict(row)

    @app.get("/api/status")
    async def status():
        return {
            "pipeline": providers.status(), "busy": lock.locked(),
            "mode": "live" if live else "turn-based",
            "live": {
                "enabled": live is not None, "ready": bool(live and live.ready and ai_ready),
                "error": warm_error or (live.error if live else None),
            },
        }

    def get_history(session_id, text):
        with database() as db:
            rows = db.execute(
                "SELECT user_text, assistant_text FROM turns WHERE session_id = ? "
                "ORDER BY created DESC LIMIT 8", (session_id,),
            ).fetchall()
        history = []
        for previous in reversed(rows):
            history.extend([
                {"role": "user", "content": previous["user_text"]},
                {"role": "assistant", "content": previous["assistant_text"]},
            ])
        return [*history, {"role": "user", "content": text}]

    def save_turn(result):
        with database() as db:
            db.execute("INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(result.values()))

    def idle_path(avatar):
        suffix = "fullbody-idle" if settings.renderer == "livetalk" else "lite-listening-idle"
        return media / f"{avatar}.{suffix}.mp4"

    def avatar_urls(identifier):
        display = f"{identifier}.display.png"
        image = display if (media / display).is_file() else f"{identifier}.png"
        return {"image": f"/media/{image}", "url": f"/avatar/{identifier}"}

    @app.get("/api/avatars")
    async def avatars():
        with database() as db:
            rows = db.execute("SELECT * FROM avatars ORDER BY created DESC").fetchall()
        return [
            {**dict(row), **avatar_urls(row["id"])}
            for row in rows
        ]

    async def read_limited(upload, limit):
        content = await upload.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(413, f"Upload exceeds the {limit // (1024 * 1024)} MB limit.")
        if not content:
            raise HTTPException(422, "The uploaded file is empty.")
        return content

    @app.post("/api/avatars", status_code=201)
    async def create_avatar(
        name: str = Form(..., min_length=1, max_length=80),
        consent: bool = Form(...),
        image: UploadFile = File(...),
    ):
        if not consent:
            raise HTTPException(422, "Confirm that you have permission to animate this image.")
        if not name.strip():
            raise HTTPException(422, "Avatar name cannot be blank.")
        raw = await read_limited(image, MAX_IMAGE)
        try:
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(422, "Use a JPEG, PNG, or WebP portrait.")
                if source.width * source.height > 25_000_000:
                    raise HTTPException(422, "Image exceeds 25 megapixels.")
                if min(source.size) < 256:
                    raise HTTPException(422, "Image must be at least 256 pixels on each side.")
                oriented = ImageOps.exif_transpose(source)
                has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
                display = oriented.convert("RGBA" if has_alpha else "RGB")
                # Preserve compatible color profiles, not EXIF or other upload metadata.
                profile = oriented.info.get("icc_profile") if oriented.mode in {"RGB", "RGBA", "P"} else None
                display.info.clear()
                portrait = display.convert("RGB")
                portrait.thumbnail((1920, 1920))
                # H.264 requires even dimensions.
                portrait = portrait.crop((0, 0, portrait.width // 2 * 2, portrait.height // 2 * 2))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(422, "The image could not be decoded.") from error
        identifier = uuid.uuid4().hex
        display_path = media / f"{identifier}.display.png"
        render_path = media / f"{identifier}.png"
        try:
            display.save(display_path, icc_profile=profile)
            portrait.save(render_path)
            with database() as db:
                db.execute("INSERT INTO avatars VALUES (?, ?, ?)", (identifier, name.strip(), time.time()))
        except (OSError, sqlite3.Error):
            display_path.unlink(missing_ok=True)
            render_path.unlink(missing_ok=True)
            raise
        return {"id": identifier, **avatar_urls(identifier)}

    class NewSession(BaseModel):
        avatar_id: str = Field(min_length=32, max_length=32)

    @app.post("/api/sessions", status_code=201)
    async def new_session(body: NewSession):
        lookup("avatars", body.avatar_id)
        identifier = uuid.uuid4().hex
        with database() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (identifier, body.avatar_id, time.time()),
            )
        return {"id": identifier}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = lookup("sessions", session_id)
        with database() as db:
            turns = db.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY created", (session_id,)
            ).fetchall()
        return {**session, "turns": [dict(row) for row in turns]}

    @app.post("/api/sessions/{session_id}/turn")
    async def turn(
        session_id: str,
        text: str = Form("", max_length=2000),
        audio: UploadFile | None = File(None),
    ):
        session = lookup("sessions", session_id)
        if bool(text.strip()) == bool(audio):
            raise HTTPException(422, "Provide either a text message or an audio recording.")
        if live and (not ai_ready or not live.ready):
            raise HTTPException(503, warm_error or live.error or "Live renderer is warming up.")
        if lock.locked():
            raise HTTPException(409, "The local inference engine is busy. Please try again shortly.")
        missing = [key for key, value in providers.status().items() if not value["configured"]]
        if missing:
            setup = "scripts/setup_models.py"
            if settings.renderer == "flashhead":
                setup += " and scripts/setup_flashhead.py"
            raise HTTPException(503, f"Models not installed: {', '.join(missing)}. Run {setup}.")
        async with lock:
            identifier = uuid.uuid4().hex
            audio_input = media / f"{identifier}.input.webm"
            speech = media / f"{identifier}.wav"
            video = media / f"{identifier}.mp4"
            started = time.monotonic()
            committed = False
            try:
                if audio:
                    allowed = {
                        "audio/webm": ".webm", "audio/ogg": ".ogg",
                        "audio/wav": ".wav", "audio/x-wav": ".wav",
                        "audio/mp4": ".mp4", "audio/mpeg": ".mp3",
                    }
                    mime = (audio.content_type or "").split(";")[0]
                    if mime not in allowed:
                        raise HTTPException(422, "Unsupported audio format.")
                    audio_input = media / f"{identifier}.input{allowed[mime]}"
                    audio_input.write_bytes(await read_limited(audio, MAX_AUDIO))
                    text = await providers.transcribe(audio_input)
                text = text.strip()
                if len(text) > 2000:
                    raise HTTPException(422, "Transcribed message exceeds 2000 characters.")
                history = get_history(session_id, text)
                reply = await providers.chat(history)
                await providers.speak(reply, speech)
                if live:
                    if not ai_ready or not live.ready:
                        raise PipelineError(warm_error or live.error or "Live renderer is warming up.")
                    await live.prepare(
                        session["avatar_id"], media / f"{session['avatar_id']}.png",
                        idle_path(session["avatar_id"]),
                    )
                    async with aclosing(live.stream(speech, video)) as events:
                        async for _ in events:
                            pass
                else:
                    await providers.render(media / f"{session['avatar_id']}.png", speech, video)
                result = {
                    "id": identifier, "session_id": session_id, "user_text": text,
                    "assistant_text": reply, "video": f"/media/{identifier}.mp4",
                    "elapsed": round(time.monotonic() - started, 2), "created": time.time(),
                }
                save_turn(result)
                committed = True
                return result
            except PipelineError as error:
                logger.exception("Conversation turn failed")
                raise HTTPException(502, str(error)) from error
            except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
                logger.exception("Local pipeline failed")
                raise HTTPException(
                    502, "Inference failed. Check model setup and the server log; no turn was saved."
                ) from error
            finally:
                audio_input.unlink(missing_ok=True)
                speech.unlink(missing_ok=True)
                if not committed:
                    video.unlink(missing_ok=True)

    @app.websocket("/api/sessions/{session_id}/live")
    async def live_conversation(websocket: WebSocket, session_id: str):
        origin = websocket.headers.get("origin")
        if origin and urlsplit(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if live is None:
            await websocket.send_json({"event": "error", "message": "Live streaming is not enabled."})
            await websocket.close(code=1008)
            return
        try:
            session = lookup("sessions", session_id)
        except HTTPException:
            await websocket.send_json({"event": "error", "message": "Conversation not found."})
            await websocket.close(code=1008)
            return
        avatar = session["avatar_id"]
        active = None
        connected = True
        credit = asyncio.Event()
        expected_ack = None

        async def respond(text):
            nonlocal expected_ack, connected
            identifier = uuid.uuid4().hex
            speech, video = media / f"{identifier}.wav", media / f"{identifier}.mp4"
            committed = False
            started = time.monotonic()
            try:
                if lock.locked():
                    raise PipelineError("The GPU is busy with another conversation. Please retry.")
                async with lock:
                    await websocket.send_json({"event": "stage", "stage": "thinking"})
                    reply = await providers.chat(get_history(session_id, text))
                    await websocket.send_json({"event": "stage", "stage": "speaking"})
                    await providers.speak(reply, speech)
                    await live.prepare(avatar, media / f"{avatar}.png", idle_path(avatar))
                    metrics = {}
                    first = None
                    async with aclosing(live.stream(speech, video)) as events:
                        async for event in events:
                            if event["event"] == "segment":
                                if first is None:
                                    first = round(time.monotonic() - started, 3)
                                credit.clear()
                                expected_ack = event["index"]
                                await websocket.send_json(event)
                                await asyncio.wait_for(credit.wait(), 30)
                            elif event["event"] == "rendered":
                                metrics = event
                    if first is None or not video.is_file() or not video.stat().st_size:
                        raise PipelineError("Live renderer did not produce speech video.")
                    result = {
                        "id": identifier, "session_id": session_id, "user_text": text,
                        "assistant_text": reply, "video": f"/media/{identifier}.mp4",
                        "elapsed": round(time.monotonic() - started, 2), "created": time.time(),
                    }
                    save_turn(result)
                    committed = True
                    await websocket.send_json({
                        "event": "complete", "turn": result,
                        "first_segment_seconds": first, "metrics": metrics,
                    })
            except asyncio.CancelledError:
                if connected:
                    await websocket.send_json({"event": "cancelled"})
                raise
            except WebSocketDisconnect:
                connected = False
            except TimeoutError:
                logger.exception("Live playback acknowledgement timed out")
                if connected:
                    await websocket.send_json({
                        "event": "error",
                        "message": "Playback stopped acknowledging media. The reply was not saved; reconnect and retry.",
                    })
            except (PipelineError, OSError, RuntimeError, ValueError, ImportError) as error:
                logger.exception("Live conversation failed")
                if connected:
                    await websocket.send_json({"event": "error", "message": str(error)})
            finally:
                expected_ack = None
                speech.unlink(missing_ok=True)
                if not committed:
                    video.unlink(missing_ok=True)

        try:
            await websocket.send_json({"event": "stage", "stage": "warming"})
            if warming:
                await asyncio.shield(warming)
            if warm_error or not live.ready:
                raise PipelineError(warm_error or live.error or "Live renderer is unavailable.")
            async with lock:
                await websocket.send_json({"event": "stage", "stage": "preparing"})
                await live.prepare(avatar, media / f"{avatar}.png", idle_path(avatar))
            authored_idle = media / f"{avatar}.lite-head-idle.mp4"
            idle = (
                authored_idle if settings.renderer == "flashhead" and authored_idle.is_file()
                else idle_path(avatar)
            )
            await websocket.send_json({"event": "prepared", "idle": f"/media/{idle.name}"})
            while True:
                message = await websocket.receive_json()
                if not isinstance(message, dict):
                    await websocket.send_json({"event": "error", "message": "Expected a message object."})
                    continue
                if message.get("type") == "cancel":
                    if active and not active.done():
                        active.cancel()
                        with suppress(asyncio.CancelledError):
                            await active
                    else:
                        await websocket.send_json({"event": "cancelled"})
                elif message.get("type") == "ack":
                    if expected_ack is not None and message.get("index") == expected_ack:
                        credit.set()
                    else:
                        await websocket.send_json({"event": "error", "message": "Out-of-order segment acknowledgment."})
                elif message.get("type") == "message":
                    text = message.get("text")
                    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                        await websocket.send_json({"event": "error", "message": "Enter 1-2000 characters."})
                    elif active and not active.done():
                        await websocket.send_json({"event": "error", "message": "A reply is already running."})
                    else:
                        active = asyncio.create_task(respond(text.strip()))
                else:
                    await websocket.send_json({"event": "error", "message": "Unknown message type."})
        except WebSocketDisconnect:
            connected = False
        except (PipelineError, OSError, RuntimeError, ValueError, TimeoutError) as error:
            logger.exception("Live connection failed")
            if connected:
                with suppress(WebSocketDisconnect, RuntimeError):
                    await websocket.send_json({"event": "error", "message": str(error)})
        finally:
            was_connected = connected
            connected = False
            if active and not active.done():
                active.cancel()
            if active:
                with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await active
            if was_connected:
                with suppress(WebSocketDisconnect, RuntimeError):
                    await websocket.close()

    @app.get("/media/{filename}")
    async def get_media(filename: str):
        # Only sanitized, generated filenames are public; never expose the database.
        stem, separator, extension = filename.partition(".")
        if (
            not separator or extension not in {
                "png", "display.png", "mp4", "lite-idle.mp4", "lite-listening-idle.mp4",
                "lite-head-idle.mp4", "fullbody-idle.mp4",
            } or len(stem) != 32
            or any(character not in "0123456789abcdef" for character in stem)
        ):
            raise HTTPException(404, "Not found")
        path = media / filename
        if not path.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(path)

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/avatar/{avatar_id}")
    async def avatar_page(avatar_id: str):
        lookup("avatars", avatar_id)
        return FileResponse(ROOT / "web" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app
