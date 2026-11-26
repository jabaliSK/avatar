"""Renderer service: WebSocket API around the live head engine, the compositor and the real-time player.

Client -> server
    text    {"type":"hello","avatar":"<name>","format":"raw"|"jpeg","quality":85}      first message
    text    {"type":"state","state":"idle"|"listening"|"thinking"}
    text    {"type":"speak_start"}  then binary PCM (s16le mono 24 kHz) chunks  then  {"type":"speak_end"}
    text    {"type":"interrupt"}                                                       barge-in
Server -> client
    text    {"type":"ready","avatar","width","height","fps":25,"audio_rate":24000,"format"}
    text    events: speak_started | speak_done | interrupted | error
    binary  one message per 40 ms tick, little endian:
            header "<BBBBIIHHII" = magic 0xA7, version 1, flags (bit0 JPEG, bit1 speaking), state (0 idle, 1 listening,
            2 thinking, 3 speaking), seq, pts_ms, width, height, audio_bytes, video_bytes;
            then video_bytes (RGB24 or JPEG) and audio_bytes (s16le mono 24 kHz, the audio that belongs to this frame).

One session at a time: the GPU is shared with the other services and the pipeline is built for a single live conversation.
Run uvicorn with --ws-per-message-deflate false: its default zlib-compresses every frame inside the event loop, which
caps raw video at ~18 MB/s (6 fps) and starves the audio receive path. Use format=jpeg across a real network.
"""
import asyncio
import json
import logging
import os
import struct
import threading
from contextlib import asynccontextmanager

import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.atomic_components.avatar_registrar import AvatarRegistrar

from .admin import build_router
from .avatar_store import avatar_stamp, list_avatars, load_avatar, preload_names
from .engine_cache import ensure_engines
from .head_engine import HeadEngine
from .player import AUDIO_RATE, FPS, Output, Player
from .speech_pipeline import SpeechPipeline

logging.basicConfig(level=os.environ.get("KKA_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kka.server")

MAGIC, VERSION = 0xA7, 1
HEADER = struct.Struct("<BBBBIIHHII")
STATE_CODE = {"idle": 0, "listening": 1, "thinking": 2, "speaking": 3}
MAX_BACKLOG = 50  # outputs (2 s) a slow client may fall behind before the oldest are discarded


def _env(name, default):
    return os.environ.get(name, default)


class Config:
    engine_root = _env("KKA_ENGINE_ROOT", "/engines")
    onnx_dir = _env("KKA_ONNX_DIR", "/models/ditto-talkinghead/ditto_onnx")
    cfg = _env("KKA_DITTO_CFG", "/models/ditto-talkinghead/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    data_root = _env("KKA_DITTO_DATA", "/models/ditto-talkinghead/ditto_pytorch")
    avatars = _env("KKA_AVATARS", "/avatars")
    overlap = int(_env("KKA_OVERLAP", "70"))            # LMDM overlap: 70 -> ~1.0 s of audio before the first frame, 75 -> ~0.4 s at 2x LMDM cost
    steps = int(_env("KKA_STEPS", "10"))                # LMDM sampling steps
    lmdm_precision = _env("KKA_LMDM_PRECISION", "fp16")
    preload = [n for n in _env("KKA_PRELOAD", "").split(",") if n]
    build_engines = _env("KKA_BUILD_ENGINES", "1") == "1"


class Runtime:
    def __init__(self):
        self.status, self.detail = "loading", "starting"
        self.engine = self.registrar = self.pipe = self.engine_dir = None
        self.loaded = {}
        self.load_lock = threading.Lock()  # one thread at a time on the engines: avatar loading, registration, deletion
        self.session = threading.Lock()  # held by a live conversation and by a registration

    def start(self):
        threading.Thread(target=self._boot, name="boot", daemon=True).start()

    def _boot(self):
        try:
            self.detail = "locating TensorRT engines"
            self.engine_dir = ensure_engines(Config.engine_root, Config.onnx_dir, build=Config.build_engines)
            self.detail = "loading models"
            replace = {"default_kwargs": {"online_mode": True, "overlap_v2": Config.overlap, "sampling_timesteps": Config.steps}}
            self.engine = HeadEngine(Config.cfg, Config.data_root, replace_cfg=replace, engine_dir=self.engine_dir,
                                     lmdm_precision=Config.lmdm_precision)
            self.registrar = AvatarRegistrar(**self.engine.registrar_cfg)
            self.pipe = SpeechPipeline(self.engine)
            for name in preload_names(Config.preload, Config.avatars):
                self.detail = f"loading avatar {name}"
                self.get_avatar(name)
            self.status, self.detail = "ready", "ok"
            log.info("renderer ready (engines %s)", self.engine_dir)
        except Exception as e:
            log.exception("renderer failed to start")
            self.status, self.detail = "error", repr(e)

    def get_avatar(self, name):
        with self.load_lock:
            stamp = avatar_stamp(Config.avatars, name)
            if stamp is None:
                raise KeyError(f"unknown avatar {name!r}")
            cached = self.loaded.get(name)
            if cached is None or cached.stamp != stamp:  # not loaded yet, or onboarded again since it was
                self.loaded.clear()  # one avatar in memory at a time; frees its GPU paste assets
                torch.cuda.empty_cache()
                self.loaded[name] = load_avatar(Config.avatars, name, self.pipe)
            return self.loaded[name]


rt = Runtime()


@asynccontextmanager
async def lifespan(app):
    rt.start()
    yield


app = FastAPI(title="kka-renderer", lifespan=lifespan)
app.include_router(build_router(rt, Config.avatars))  # /v1: avatar library and registration, used by the Studio


@app.get("/healthz")
def healthz():
    body = {"status": rt.status, "detail": rt.detail, "avatars": list_avatars(Config.avatars), "loaded": list(rt.loaded),
            "engine_dir": rt.engine_dir, "session_active": rt.session.locked(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "overlap": Config.overlap, "steps": Config.steps}
    return JSONResponse(body, status_code=200 if rt.status == "ready" else 503)


def _pack(out: Output, video: bytes, jpeg: bool):
    h, w = out.rgb.shape[:2]
    flags = (1 if jpeg else 0) | (2 if out.state == "speaking" else 0)
    head = HEADER.pack(MAGIC, VERSION, flags, STATE_CODE[out.state], out.seq & 0xFFFFFFFF, int(out.pts * 1000) & 0xFFFFFFFF,
                       w, h, len(out.pcm), len(video))
    return head + video + out.pcm


def _encode_jpeg(out: Output, quality: int):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return _pack(out, buf.tobytes(), True)


async def _send_loop(ws: WebSocket, q: asyncio.Queue, fmt: str, quality: int):
    loop = asyncio.get_running_loop()
    while True:
        item = await q.get()
        if isinstance(item, dict):
            await ws.send_text(json.dumps(item))
        elif fmt == "jpeg":
            await ws.send_bytes(await loop.run_in_executor(None, _encode_jpeg, item, quality))
        else:
            await ws.send_bytes(_pack(item, item.rgb.tobytes(), False))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    if rt.status != "ready":
        await ws.send_json({"type": "error", "reason": f"renderer {rt.status}: {rt.detail}"})
        await ws.close(code=1013)
        return
    if not rt.session.acquire(blocking=False):
        await ws.send_json({"type": "error", "reason": "busy: another session is active"})
        await ws.close(code=1013)
        return
    player = sender = None
    try:
        hello = await ws.receive_json()
        if hello.get("type") != "hello":
            raise ValueError("first message must be hello")
        name = hello.get("avatar") or (list_avatars(Config.avatars) or [None])[0]
        fmt = hello.get("format", "raw")
        if fmt not in ("raw", "jpeg"):
            raise ValueError("format must be raw or jpeg")
        quality = int(hello.get("quality", 85))
        try:
            avatar = await loop.run_in_executor(None, rt.get_avatar, name)
        except KeyError as e:
            await ws.send_json({"type": "error", "reason": str(e), "avatars": list_avatars(Config.avatars)})
            await ws.close(code=1008)
            return

        q = asyncio.Queue()
        dropped = 0

        def enqueue(item):
            nonlocal dropped
            if isinstance(item, Output) and q.qsize() > MAX_BACKLOG:
                dropped += 1
                if dropped == 1 or dropped % 100 == 0:
                    log.warning("client is %d outputs behind; %d frames dropped so far (use format=jpeg on slow links)",
                                q.qsize(), dropped)
                return
            q.put_nowait(item)

        player = Player(rt.pipe, avatar, lambda o: loop.call_soon_threadsafe(enqueue, o),
                        lambda e: loop.call_soon_threadsafe(enqueue, e))
        width, height = avatar.size
        await ws.send_json({"type": "ready", "avatar": avatar.name, "width": width, "height": height, "fps": FPS,
                            "audio_rate": AUDIO_RATE, "format": fmt})
        player.start()
        sender = asyncio.create_task(_send_loop(ws, q, fmt, quality))

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                player.push_audio(msg["bytes"])
                continue
            data = json.loads(msg["text"])
            kind = data.get("type")
            if kind == "state":
                player.set_state(data.get("state"))
            elif kind == "speak_start":
                await loop.run_in_executor(None, player.speak_start, data.get("id"))
            elif kind == "speak_end":
                player.speak_end()
            elif kind == "interrupt":
                await loop.run_in_executor(None, player.interrupt)
            elif kind == "stats":
                q.put_nowait({"type": "stats", **player.stats()})
            else:
                q.put_nowait({"type": "error", "reason": f"unknown message type {kind!r}"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("session failed")
        try:
            await ws.send_json({"type": "error", "reason": repr(e)})
        except Exception:
            pass
    finally:
        if sender is not None:
            sender.cancel()
        if player is not None:
            await loop.run_in_executor(None, player.close)
        rt.session.release()
