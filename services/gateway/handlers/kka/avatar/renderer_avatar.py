"""Avatar handler: drives the Ditto renderer service and publishes its 25 fps video and audio to the RTC client.

    TTS audio (AVATAR_AUDIO) --> this handler --> renderer WebSocket (s16le PCM, 24 kHz)
    renderer ticks (RGB frame + the PCM that belongs to those 40 ms) --> AVATAR_VIDEO + AVATAR_AUDIO --> RTC client

The renderer owns the timeline. It produces one 40 ms tick at a time, in real time, whether or not anyone is speaking (idle,
listening and thinking loops included), and the audio the viewer hears travels inside the tick it belongs to. This handler only
forwards ticks, so audio and lip movement cannot drift apart inside the gateway. Between replies the audio track carries silence,
which keeps the browser's audio clock running.

Every TTS stream opens a CLIENT_PLAYBACK lifecycle stream and always closes it again (renderer finished, interrupted, or the reply
was empty), because the RTC client and the VAD's half-duplex gate key off its begin/end.
"""
import json
import queue
import re
import struct
import threading
import time
from math import gcd
from typing import Dict, Optional, Set, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from chat_engine.common.handler_base import ChatDataConsumeMode, HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.chat_stream_config import ChatStreamConfig
from chat_engine.data_models.chat_stream_status import ChatStreamStatus
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry, VariableSize

from kka.common.trace import trace

HEADER = struct.Struct("<BBBBIIHHII")  # magic, version, flags, state, seq, pts_ms, width, height, audio bytes, video bytes
MAGIC, VERSION = 0xA7, 1
FLAG_JPEG, FLAG_SPEAKING = 1, 2
SILENCE_TICK_SAMPLES = 960  # 40 ms at 24 kHz
AVATAR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class RendererAvatarConfig(HandlerBaseConfigModel, BaseModel):
    renderer_url: str = Field(default="ws://kka-renderer:8290/ws")
    avatar: str = Field(default="", description="Avatar folder name in the renderer; empty lets the renderer pick the first one")
    avatar_control_file: Optional[str] = Field(
        default=None, description="If this file exists and holds an avatar name it wins over `avatar`; read at the start of every session")
    frame_format: str = Field(default="raw", description="raw (RGB24) on the compose network, jpeg for slow links")
    jpeg_quality: int = Field(default=90)
    fps: int = Field(default=25)
    audio_sample_rate: int = Field(default=24000)
    connect_timeout_s: float = Field(default=45.0, description="Covers a cold avatar load (~9 s) and a renderer that is still starting")
    asr_timeout_s: float = Field(default=3.0, description="Back to listening when no transcript follows the end of speech")
    thinking_timeout_s: float = Field(default=15.0, description="Back to listening when no reply starts after the end of speech")
    min_reply_s: float = Field(default=0.1, description="Shorter replies (an empty TTS result) are not worth rendering")
    idle_audio: str = Field(default="silence", description="silence: the audio track runs continuously; none: audio only while speaking")


class AvatarContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = RendererAvatarConfig()
        self.definitions: Dict[ChatDataType, DataBundleDefinition] = {}
        self.lock = threading.RLock()
        self.outbox: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self.ws = None
        self.closed = False
        self.last_error = ""
        self.reader: Optional[threading.Thread] = None
        self.writer: Optional[threading.Thread] = None
        self.playback = None
        self.avatar = ""
        self.min_reply_bytes = 0
        # Reply bookkeeping, guarded by `lock`.
        self.tts_key = None
        self.playback_key = None
        self.closed_keys: Set = set()
        self.epoch = 0
        self.reply_id = 0
        self.reply_open = False
        self.speak_sent = False
        self.pending = bytearray()
        self.reply_bytes = 0
        self.first_frame_traced = True
        self.state = "listening"
        self.thinking_since = 0.0
        self.asr_seen = False


def _to_s16le(audio: np.ndarray, rate: int, target: int) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate != target:
        from scipy.signal import resample_poly
        g = gcd(rate, target)
        samples = resample_poly(samples, target // g, rate // g).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class HandlerRendererAvatar(HandlerBase):
    def __init__(self):
        super().__init__()
        self.config = RendererAvatarConfig()
        self.definitions: Dict[ChatDataType, DataBundleDefinition] = {}

    # ------------------------------------------------------------------------------------------------------------------------
    # Handler interface
    # ------------------------------------------------------------------------------------------------------------------------
    def get_handler_info(self) -> HandlerBaseInfo:
        # Prepared before the RTC client so that, as ONCE consumer, this handler gets the TTS audio exclusively.
        return HandlerBaseInfo(config_model=RendererAvatarConfig, load_priority=-999)

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        cfg = handler_config if isinstance(handler_config, RendererAvatarConfig) else RendererAvatarConfig()
        self.config = cfg
        audio = DataBundleDefinition()
        audio.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, cfg.audio_sample_rate))
        audio.lockdown()
        video = DataBundleDefinition()
        video.add_entry(DataBundleEntry.create_framed_entry(
            "avatar_video", [VariableSize(), VariableSize(), VariableSize(), 3], 0, cfg.fps))
        video.lockdown()
        self.definitions = {ChatDataType.AVATAR_AUDIO: audio, ChatDataType.AVATAR_VIDEO: video}

    def create_context(self, session_context, handler_config: Optional[BaseModel] = None) -> HandlerContext:
        cfg = handler_config if isinstance(handler_config, RendererAvatarConfig) else self.config
        context = AvatarContext(session_context.session_info.session_id)
        context.config = cfg
        context.definitions = self.definitions
        context.min_reply_bytes = int(cfg.min_reply_s * cfg.audio_sample_rate) * 2
        return context

    def get_handler_detail(self, session_context, context: HandlerContext) -> HandlerDetail:
        ctx = cast(AvatarContext, context)
        # The renderer's ticks form one continuous timeline that is independent of any single reply, so neither output is linked
        # to the TTS stream and neither can be cancelled by an interrupt.
        continuous = ChatStreamConfig(cancelable=False, auto_link_input=False)
        return HandlerDetail(
            inputs={ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO, input_consume_mode=ChatDataConsumeMode.ONCE)},
            outputs={
                ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                    type=ChatDataType.AVATAR_AUDIO, definition=ctx.definitions[ChatDataType.AVATAR_AUDIO],
                    output_stream_config=continuous),
                ChatDataType.AVATAR_VIDEO: HandlerDataInfo(
                    type=ChatDataType.AVATAR_VIDEO, definition=ctx.definitions[ChatDataType.AVATAR_VIDEO],
                    output_stream_config=continuous),
            },
            signal_filters=[
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, ChatDataType.CLIENT_PLAYBACK),
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, ChatDataType.AVATAR_AUDIO),
                SignalFilterRule(ChatSignalType.STREAM_BEGIN, None, ChatDataType.HUMAN_AUDIO),
                SignalFilterRule(ChatSignalType.STREAM_END, None, ChatDataType.HUMAN_AUDIO),
                SignalFilterRule(ChatSignalType.STREAM_BEGIN, None, ChatDataType.HUMAN_TEXT),
            ],
        )

    def start_context(self, session_context, handler_context: HandlerContext):
        ctx = cast(AvatarContext, handler_context)
        ctx.playback = ctx.stream_manager.create_lifecycle_streamer(
            data_type=ChatDataType.CLIENT_PLAYBACK, producer_name=ctx.owner or "RendererAvatar",
            config=ChatStreamConfig(cancelable=False))
        if not self._connect(ctx):
            raise RuntimeError(f"Renderer at {ctx.config.renderer_url} is not usable: {ctx.last_error}")
        ctx.writer = threading.Thread(target=self._writer_loop, args=(ctx,), name="avatar-writer", daemon=True)
        ctx.reader = threading.Thread(target=self._reader_loop, args=(ctx,), name="avatar-reader", daemon=True)
        ctx.writer.start()
        ctx.reader.start()

    def destroy_context(self, context: HandlerContext):
        ctx = cast(AvatarContext, context)
        ctx.closed = True
        with ctx.lock:
            self._close_playback(ctx)
        ctx.outbox.put(None)
        ws, ctx.ws = ctx.ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        for thread in (ctx.reader, ctx.writer):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        logger.info(f"Avatar: context destroyed for session {ctx.session_id}")

    # ------------------------------------------------------------------------------------------------------------------------
    # TTS audio -> renderer
    # ------------------------------------------------------------------------------------------------------------------------
    def handle(self, context: HandlerContext, inputs: ChatData, output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        ctx = cast(AvatarContext, context)
        if inputs.type != ChatDataType.AVATAR_AUDIO or inputs.stream_id is None:
            return
        key = inputs.stream_id.key
        audio = inputs.data.get_main_data()
        rate = inputs.data.get_main_definition_entry().sample_rate
        pcm = _to_s16le(audio, rate, ctx.config.audio_sample_rate) if audio is not None and np.size(audio) else b""
        with ctx.lock:
            if key in ctx.closed_keys:
                return
            if key != ctx.tts_key:
                stream = ctx.stream_manager.find_stream(inputs.stream_id)
                if (stream is not None and stream.status == ChatStreamStatus.CANCELLED) or ctx.ws is None:
                    ctx.closed_keys.add(key)
                    return
                self._begin_reply(ctx, inputs.stream_id)
            if pcm:
                self._feed(ctx, pcm)
            if inputs.is_last_data:
                self._end_reply(ctx)

    def _begin_reply(self, ctx: AvatarContext, stream_id) -> None:
        if ctx.tts_key is not None and ctx.speak_sent:
            # The previous answer is still being rendered; the new one replaces it.
            self._send_json(ctx, {"type": "interrupt"})
        self._close_playback(ctx)
        ctx.epoch += 1
        ctx.reply_id += 1
        ctx.tts_key = stream_id.key
        ctx.speak_sent = False
        ctx.pending = bytearray()
        ctx.reply_bytes = 0
        ctx.first_frame_traced = False
        if ctx.playback is not None:
            opened = ctx.playback.open_stream(sources=[stream_id], name=f"playback:{stream_id.key}")
            ctx.playback_key = opened.key if opened is not None else None
            ctx.reply_open = True
        trace("avatar_reply_begin", reply=ctx.reply_id)

    def _feed(self, ctx: AvatarContext, pcm: bytes) -> None:
        ctx.reply_bytes += len(pcm)
        if not ctx.speak_sent:
            ctx.pending += pcm
            if ctx.reply_bytes < ctx.min_reply_bytes:
                return
            ctx.speak_sent = True
            self._send_json(ctx, {"type": "speak_start", "id": ctx.reply_id})
            pcm, ctx.pending = bytes(ctx.pending), bytearray()
            trace("avatar_first_pcm", reply=ctx.reply_id)
        ctx.outbox.put(("pcm", ctx.epoch, pcm))

    def _end_reply(self, ctx: AvatarContext) -> None:
        if ctx.speak_sent:
            self._send_json(ctx, {"type": "speak_end"})
            return
        # Nothing worth rendering (an empty reply): close the playback stream here and leave the renderer alone.
        trace("avatar_reply_empty", reply=ctx.reply_id)
        ctx.closed_keys.add(ctx.tts_key)
        ctx.tts_key = None
        ctx.pending = bytearray()
        self._close_playback(ctx)
        self._set_state(ctx, "listening")

    def _close_playback(self, ctx: AvatarContext) -> None:
        if not ctx.reply_open:
            return
        ctx.reply_open = False
        ctx.playback_key = None
        try:
            ctx.playback.finish_current()
        except Exception:
            logger.exception("Avatar: closing the playback stream failed")

    def _finish_reply(self, ctx: AvatarContext) -> None:
        """The renderer is done with the reply (or failed); lock held."""
        if ctx.tts_key is not None:
            ctx.closed_keys.add(ctx.tts_key)
        ctx.tts_key = None
        ctx.state = "listening"
        self._close_playback(ctx)
        if len(ctx.closed_keys) > 512:
            ctx.closed_keys.clear()

    def _interrupt(self, ctx: AvatarContext, reason: str) -> None:
        with ctx.lock:
            if ctx.tts_key is None and not ctx.reply_open:
                return
            was_rendering = ctx.speak_sent
            if ctx.tts_key is not None:
                ctx.closed_keys.add(ctx.tts_key)
            ctx.tts_key = None
            ctx.epoch += 1  # audio still waiting in the outbox belongs to the interrupted reply
            ctx.pending = bytearray()
            ctx.speak_sent = False
            ctx.state = "listening"
            self._close_playback(ctx)
            if was_rendering:
                self._send_json(ctx, {"type": "interrupt"})
        trace("avatar_interrupt", reason=reason)

    # ------------------------------------------------------------------------------------------------------------------------
    # Signals (signal thread: keep it short)
    # ------------------------------------------------------------------------------------------------------------------------
    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        ctx = cast(AvatarContext, context)
        stream = signal.related_stream
        if stream is None:
            return
        dtype = stream.data_type
        if signal.type == ChatSignalType.STREAM_CANCEL:
            if dtype == ChatDataType.CLIENT_PLAYBACK:
                with ctx.lock:
                    current = ctx.playback_key is not None and stream.key == ctx.playback_key
                if current:
                    self._interrupt(ctx, "playback_cancelled")
            elif dtype == ChatDataType.AVATAR_AUDIO and stream.producer_name != ctx.owner:
                with ctx.lock:
                    current = stream.key == ctx.tts_key
                    if not current:
                        ctx.closed_keys.add(stream.key)
                if current:
                    self._interrupt(ctx, "tts_cancelled")
        elif signal.type == ChatSignalType.STREAM_BEGIN and dtype == ChatDataType.HUMAN_AUDIO:
            self._set_state(ctx, "listening")
        elif signal.type == ChatSignalType.STREAM_END and dtype == ChatDataType.HUMAN_AUDIO:
            with ctx.lock:
                ctx.thinking_since = time.monotonic()
                ctx.asr_seen = False
            self._set_state(ctx, "thinking")
        elif signal.type == ChatSignalType.STREAM_BEGIN and dtype == ChatDataType.HUMAN_TEXT:
            ctx.asr_seen = True

    def _set_state(self, ctx: AvatarContext, state: str) -> None:
        with ctx.lock:
            if ctx.state == state:
                return
            ctx.state = state
            self._send_json(ctx, {"type": "state", "state": state})

    @staticmethod
    def _send_json(ctx: AvatarContext, message: dict) -> None:
        ctx.outbox.put(("json", 0, json.dumps(message)))

    # ------------------------------------------------------------------------------------------------------------------------
    # Renderer connection
    # ------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _select_avatar(cfg: RendererAvatarConfig) -> str:
        if cfg.avatar_control_file:
            try:
                with open(cfg.avatar_control_file, "r", encoding="utf-8") as f:
                    name = f.read().strip()
                if AVATAR_NAME.match(name):
                    return name
                if name:
                    logger.warning(f"Avatar: ignoring invalid avatar name in {cfg.avatar_control_file}: {name!r}")
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(f"Avatar: cannot read {cfg.avatar_control_file}: {e}")
        return cfg.avatar

    def _connect(self, ctx: AvatarContext) -> bool:
        cfg = ctx.config
        ctx.avatar = avatar = self._select_avatar(cfg)
        deadline = time.monotonic() + cfg.connect_timeout_s
        while not ctx.closed and time.monotonic() < deadline:
            ws = None
            try:
                ws = connect(cfg.renderer_url, compression=None, max_size=None, open_timeout=5, close_timeout=2)
                ws.send(json.dumps({"type": "hello", "avatar": avatar, "format": cfg.frame_format, "quality": cfg.jpeg_quality}))
                reply = ws.recv(timeout=max(1.0, deadline - time.monotonic()))
                ready = json.loads(reply) if isinstance(reply, str) else {}
                if ready.get("type") == "ready":
                    ctx.ws = ws
                    ctx.state = "listening"
                    logger.info(f"Avatar: renderer ready, avatar={ready.get('avatar')} {ready.get('width')}x{ready.get('height')} "
                                f"{ready.get('format')} ({cfg.renderer_url})")
                    return True
                ctx.last_error = str(ready.get("reason", reply))
                if "avatars" in ready:
                    # An unknown avatar will not appear by waiting.
                    ctx.last_error = f"unknown avatar {avatar!r}; available: {ready['avatars']}"
                    ws.close()
                    return False
            except Exception as e:
                ctx.last_error = repr(e)
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            time.sleep(1.0)
        return False

    def _writer_loop(self, ctx: AvatarContext) -> None:
        while True:
            item = ctx.outbox.get()
            if item is None:
                return
            kind, epoch, payload = item
            if kind == "pcm" and epoch != ctx.epoch:
                continue  # belongs to a reply that has been interrupted or replaced
            ws = ctx.ws
            if ws is None:
                continue
            try:
                ws.send(payload)
            except Exception as e:
                logger.warning(f"Avatar: send to renderer failed: {e!r}")

    def _reader_loop(self, ctx: AvatarContext) -> None:
        while not ctx.closed:
            ws = ctx.ws
            if ws is None:
                if not self._connect(ctx):
                    logger.error(f"Avatar: renderer unavailable: {ctx.last_error}")
                continue
            try:
                message = ws.recv(timeout=1.0)
            except TimeoutError:
                self._watchdog(ctx)
                continue
            except Exception as e:
                if ctx.closed:
                    break
                logger.warning(f"Avatar: renderer connection lost ({e!r}); reconnecting")
                self._lost(ctx, ws)
                continue
            try:
                if isinstance(message, bytes):
                    self._on_frame(ctx, message)
                else:
                    self._on_event(ctx, json.loads(message))
            except Exception:
                logger.exception("Avatar: could not process a renderer message")
            self._watchdog(ctx)

    def _lost(self, ctx: AvatarContext, ws) -> None:
        with ctx.lock:
            if ctx.ws is ws:
                ctx.ws = None
            if ctx.tts_key is not None:
                ctx.closed_keys.add(ctx.tts_key)
            ctx.tts_key = None
            ctx.epoch += 1
            ctx.speak_sent = False
            ctx.pending = bytearray()
            self._close_playback(ctx)
        try:
            ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------------
    # Renderer -> RTC client
    # ------------------------------------------------------------------------------------------------------------------------
    def _on_frame(self, ctx: AvatarContext, message: bytes) -> None:
        magic, version, flags, _state, seq, _pts, width, height, audio_len, video_len = HEADER.unpack_from(message, 0)
        if magic != MAGIC or version != VERSION:
            logger.warning(f"Avatar: unexpected frame header magic={magic:#x} version={version}")
            return
        body = HEADER.size
        video = memoryview(message)[body:body + video_len]
        audio = memoryview(message)[body + video_len:body + video_len + audio_len]

        if flags & FLAG_JPEG:
            import cv2
            frame = cv2.imdecode(np.frombuffer(video, np.uint8), cv2.IMREAD_COLOR)  # already BGR
        else:
            frame = np.frombuffer(video, np.uint8).reshape(height, width, 3)[..., ::-1]  # RGB -> BGR, which fastrtc expects
        h, w = frame.shape[0] & ~1, frame.shape[1] & ~1  # H.264 needs even dimensions
        frame = np.ascontiguousarray(frame[:h, :w])

        bundle = DataBundle(ctx.definitions[ChatDataType.AVATAR_VIDEO])
        bundle.set_main_data(frame[np.newaxis, ...])
        ctx.submit_data(ChatData(type=ChatDataType.AVATAR_VIDEO, data=bundle))

        if audio_len:
            pcm = (np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0)[np.newaxis, :]
        elif ctx.config.idle_audio == "silence":
            pcm = np.zeros((1, SILENCE_TICK_SAMPLES), dtype=np.float32)
        else:
            pcm = None
        if pcm is not None:
            bundle = DataBundle(ctx.definitions[ChatDataType.AVATAR_AUDIO])
            bundle.set_main_data(pcm)
            ctx.submit_data(ChatData(type=ChatDataType.AVATAR_AUDIO, data=bundle))

        if flags & FLAG_SPEAKING and not ctx.first_frame_traced:
            ctx.first_frame_traced = True
            trace("avatar_first_speaking_frame", reply=ctx.reply_id, seq=seq)

    def _on_event(self, ctx: AvatarContext, event: dict) -> None:
        kind = event.get("type")
        if kind == "speak_started":
            trace("renderer_speak_started", reply=event.get("id"), first_audio_to_first_frame_ms=event.get("first_audio_to_first_frame_ms"))
        elif kind == "speak_done":
            trace("renderer_speak_done", **{k: event.get(k) for k in (
                "id", "frames", "audio_s", "repeated", "late", "stale", "audio_underruns", "render_ms", "wall_s")})
            with ctx.lock:
                if event.get("id") == ctx.reply_id and ctx.tts_key is not None:
                    self._finish_reply(ctx)
        elif kind == "interrupted":
            trace("renderer_interrupted", reply=event.get("id"), frames_played=event.get("frames_played"))
        elif kind == "error":
            logger.warning(f"Avatar: renderer reported an error: {event}")
            with ctx.lock:
                if event.get("id") is not None and event.get("id") == ctx.reply_id and ctx.tts_key is not None:
                    self._finish_reply(ctx)

    def _watchdog(self, ctx: AvatarContext) -> None:
        """Return to listening when the turn that put the avatar into `thinking` will never produce a reply."""
        if ctx.state != "thinking":
            return
        cfg = ctx.config
        with ctx.lock:
            if ctx.state != "thinking" or ctx.tts_key is not None or ctx.reply_open:
                return
            waited = time.monotonic() - ctx.thinking_since
            if (not ctx.asr_seen and waited > cfg.asr_timeout_s) or waited > cfg.thinking_timeout_s:
                trace("avatar_watchdog", waited_s=f"{waited:.1f}", asr_seen=ctx.asr_seen)
                ctx.state = "listening"
                self._send_json(ctx, {"type": "state", "state": "listening"})
