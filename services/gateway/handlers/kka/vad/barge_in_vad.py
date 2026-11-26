"""Silero VAD with barge-in.

The stock handler is half-duplex: after the user stops speaking it ignores the microphone until the avatar has finished its reply,
so the user cannot interrupt, and an empty transcript can leave it deaf for good. This subclass keeps the stock endpointing
(including its reconnect logic for short pauses) and adds:

* barge-in: when the user starts talking while a reply is being generated or played, the in-flight turn is cancelled through the
  engine's INTERRUPT signal, which the LLM, TTS and avatar handlers already honour;
* a recovery for the half-duplex mode (`barge_in: false`), which re-enables the microphone when no reply is in progress.

Barge-in listens while the avatar speaks, so it relies on the browser's echo cancellation (or a headset). With loudspeakers and no
echo cancellation the avatar can interrupt itself; set `barge_in: false` there.
"""
import os
import time
from collections import deque
from typing import Dict, List, Optional, cast

import numpy as np
from loguru import logger
from pydantic import Field

from chat_engine.common.handler_base import HandlerBaseInfo, HandlerDataInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalSourceType, ChatSignalType
from engine_utils.audio_utils import create_mel_agc

# Imported as a module on purpose: the engine takes the first concrete handler class it finds in this module's namespace.
from handlers.vad.silerovad import vad_handler_silero as _silero

from kka.common.trace import trace

SAMPLE_RATE = 16000

# Streams that make up one conversation turn, leaf first. Cancelling the leaf also cancels its cancelable ancestors, so the whole
# chain (transcript, LLM reply, speech, playback) goes with it.
CHAIN_TYPES = (
    ChatDataType.CLIENT_PLAYBACK,
    ChatDataType.AVATAR_AUDIO,
    ChatDataType.AVATAR_TEXT,
    ChatDataType.HUMAN_TEXT,
)


class BargeInVADConfig(_silero.SileroVADConfigModel):
    barge_in: bool = Field(default=True, description="Let the user interrupt the avatar; false keeps the stock half-duplex behaviour")
    barge_in_window_ms: int = Field(default=512, description="Sliding window used to judge that the user is really speaking")
    barge_in_min_speech_ms: int = Field(default=256, description="Speech needed inside that window; short noises and clicks stay below it")
    barge_in_min_db: float = Field(default=-40.0, description="Clips quieter than this never count as speech for barge-in")
    stuck_disabled_timeout_s: float = Field(
        default=3.0, description="Half-duplex only: re-enable the microphone after this long with no reply in progress")


class _BargeIn:
    """Per-session barge-in state; only the handler's own thread touches it."""

    def __init__(self, config: BargeInVADConfig, clip_size: int):
        clips_per_ms = SAMPLE_RATE / 1000 / clip_size
        self.config = config
        self.window = deque(maxlen=max(1, round(config.barge_in_window_ms * clips_per_ms)))
        self.needed = max(1, round(config.barge_in_min_speech_ms * clips_per_ms))
        self.fired = False
        self.disabled_since: Optional[float] = None


class HandlerBargeInVAD(_silero.HandlerAudioVAD):
    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=BargeInVADConfig)

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        # The stock load() looks for the ONNX model next to the stock module, and this class lives elsewhere.
        own_root = self.handler_root
        self.handler_root = os.path.dirname(_silero.__file__)
        try:
            super().load(engine_config, handler_config)
        finally:
            self.handler_root = own_root
        self._prewarm_agc()

    @staticmethod
    def _prewarm_agc() -> None:
        """Compile librosa's numba kernels at start-up instead of in the first session.

        The stock handler's per-session warm-up does this on its pumper thread before it reads any microphone audio. In a fresh
        process that takes ~17 s (about 1.6 s with a populated NUMBA_CACHE_DIR), and the session's first utterance is answered
        that much late. The parameters mirror the AGC the stock create_context() builds.
        """
        started = time.perf_counter()
        agc = create_mel_agc(target_level_db=-5.0, max_gain_db=30.0, min_gain_db=-30.0, attack_time_ms=5.0, release_time_ms=50.0,
                             sample_rate=SAMPLE_RATE, n_mels=80, n_fft=1024, hop_length=256)
        agc.warmup()
        chunk = np.zeros(512, dtype=np.float32)
        agc.update_gain(chunk)
        agc.apply_gain(chunk)
        trace("vad_prewarm", ms=round((time.perf_counter() - started) * 1000))

    def create_context(self, session_context, handler_config=None) -> HandlerContext:
        context = super().create_context(session_context, handler_config)
        config = handler_config if isinstance(handler_config, BargeInVADConfig) else BargeInVADConfig()
        context.barge = _BargeIn(config, context.clip_size)
        return context

    def _inference(self, context, clip, sr: int = SAMPLE_RATE):
        # Called once per 32 ms clip, so this sees every clip's speech probability, not just the stock state transitions.
        prob = super()._inference(context, clip, sr)
        barge = getattr(context, "barge", None)
        if barge is not None:
            voiced = prob > context.config.speaking_threshold and context.current_db > barge.config.barge_in_min_db
            barge.window.append(1 if voiced else 0)
        return prob

    def handle(self, context: HandlerContext, inputs: ChatData, output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        ctx = cast(_silero.HumanAudioVADContext, context)
        barge: _BargeIn = ctx.barge
        if barge.config.barge_in:
            ctx.input_enabled = True  # the stock handler switches itself off at the end of every utterance
        else:
            self._recover_stuck_input(ctx, barge)
        super().handle(context, inputs, output_definitions)
        if barge.config.barge_in:
            ctx.input_enabled = True
            self._barge_in_if_needed(ctx, barge)

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        barge: _BargeIn = context.barge
        related = signal.related_stream
        playback_begins = (signal.type == ChatSignalType.STREAM_BEGIN and related is not None
                           and related.data_type == ChatDataType.CLIENT_PLAYBACK)
        if barge.config.barge_in and playback_begins:
            return  # the stock handler would mute the microphone here, which is exactly what barge-in must avoid
        super().on_signal(context, signal)

    @staticmethod
    def _active_chain(ctx: HandlerContext) -> List:
        manager = ctx.stream_manager
        if manager is None:
            return []
        # Streams without sources are the avatar's continuous video/audio output; they are not part of any turn.
        return [s for s in manager.get_active_streams() if s.source_streams and s.identity.data_type in CHAIN_TYPES]

    def _barge_in_if_needed(self, ctx: _silero.HumanAudioVADContext, barge: _BargeIn) -> None:
        voiced = sum(barge.window)
        if barge.fired:
            if voiced == 0:  # the user went quiet; the next burst of speech can interrupt again
                barge.fired = False
            return
        if voiced < barge.needed:
            return
        barge.fired = True  # one decision per burst of speech, even if nothing was playing
        active = self._active_chain(ctx)
        for data_type in CHAIN_TYPES:
            targets = [s for s in active if s.identity.data_type == data_type]
            if not targets:
                continue
            logger.info(f"VAD: barge-in, cancelling the {data_type.value} chain ({len(targets)} stream(s))")
            trace("barge_in", leaf=data_type.value, streams=len(targets), speech_ms=voiced * ctx.clip_size * 1000 // SAMPLE_RATE)
            for stream in targets:
                ctx.emit_signal(ChatSignal(
                    type=ChatSignalType.INTERRUPT,
                    source_type=ChatSignalSourceType.HANDLER,
                    source_name=ctx.owner,
                    related_stream=stream.identity,
                    signal_data={"reason": "barge_in"},
                ))
            return

    def _recover_stuck_input(self, ctx: _silero.HumanAudioVADContext, barge: _BargeIn) -> None:
        if ctx.input_enabled:
            barge.disabled_since = None
            return
        now = time.monotonic()
        if barge.disabled_since is None:
            barge.disabled_since = now
            return
        if now - barge.disabled_since > barge.config.stuck_disabled_timeout_s and not self._active_chain(ctx):
            # An empty transcript or a failed reply never opens a playback stream, which is what normally re-enables the input.
            logger.warning("VAD: microphone was disabled but no reply is in progress; re-enabling it")
            ctx.input_enabled = True
            barge.disabled_since = None
