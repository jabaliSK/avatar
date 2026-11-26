"""Faster-Whisper speech recognition for OpenAvatarChat (faster-whisper: MIT; whisper-large-v3-turbo weights: MIT).

One decode per utterance, triggered by the last HUMAN_AUDIO chunk from the VAD. A result with no speech emits nothing: the avatar
handler's watchdog returns the avatar to `listening`, and the VAD is never left waiting for a reply that will not come.
"""
import re
import time
from typing import Dict, List, Optional, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.chat_stream_status import ChatStreamStatus
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

from kka.common.trace import trace

SAMPLE_RATE = 16000

# Phrases Whisper invents for near-silence; only dropped when the model itself also leans towards "no speech".
_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye", "the end", "so"}


class FasterWhisperConfig(HandlerBaseConfigModel, BaseModel):
    model_path: str = Field(default="/models/faster-whisper-large-v3-turbo")
    device: str = Field(default="cuda")
    device_index: int = Field(default=0)
    compute_type: str = Field(default="float16")
    language: Optional[str] = Field(default="en", description="ISO 639-1 code; null lets the model detect the language")
    beam_size: int = Field(default=1)
    initial_prompt: Optional[str] = Field(default=None, description="Domain vocabulary that biases the spelling of product names")
    min_audio_s: float = Field(default=0.25, description="Shorter utterances are ignored")
    no_speech_threshold: float = Field(default=0.6)
    log_prob_threshold: float = Field(default=-1.0)


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = FasterWhisperConfig()
        self.stream_key: Optional[str] = None
        self.chunks: List[np.ndarray] = []


class HandlerFasterWhisperASR(HandlerBase):
    def __init__(self):
        super().__init__()
        self.model = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=FasterWhisperConfig)

    def get_handler_detail(self, session_context, context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("human_text"))
        return HandlerDetail(
            inputs={ChatDataType.HUMAN_AUDIO: HandlerDataInfo(type=ChatDataType.HUMAN_AUDIO)},
            outputs={ChatDataType.HUMAN_TEXT: HandlerDataInfo(type=ChatDataType.HUMAN_TEXT, definition=definition)},
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        from faster_whisper import WhisperModel

        cfg = handler_config if isinstance(handler_config, FasterWhisperConfig) else FasterWhisperConfig()
        t0 = time.perf_counter()
        self.model = WhisperModel(cfg.model_path, device=cfg.device, device_index=cfg.device_index, compute_type=cfg.compute_type)
        # The first decode initialises cuDNN/cuBLAS handles and kernels; do it here, not on the first user's turn.
        warm = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 2) * 0.01).astype(np.float32)
        for _ in range(2):
            list(self.model.transcribe(warm, language=cfg.language, beam_size=cfg.beam_size, without_timestamps=True)[0])
        logger.info(f"Faster-Whisper ready in {time.perf_counter() - t0:.1f}s ({cfg.model_path}, {cfg.compute_type})")

    def create_context(self, session_context, handler_config: Optional[BaseModel] = None) -> HandlerContext:
        context = ASRContext(session_context.session_info.session_id)
        if isinstance(handler_config, FasterWhisperConfig):
            context.config = handler_config
        return context

    def start_context(self, session_context, handler_context: HandlerContext):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData, output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        ctx = cast(ASRContext, context)
        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return
        key = inputs.stream_id.key if inputs.stream_id is not None else None
        if key != ctx.stream_key:
            # The VAD's reconnect path cancels a stream and replays the whole utterance into a new one.
            ctx.stream_key, ctx.chunks = key, []

        audio = inputs.data.get_main_data()
        if audio is not None:
            ctx.chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not inputs.is_last_data:
            return
        chunks, ctx.chunks, ctx.stream_key = ctx.chunks, [], None
        if not chunks:
            return
        sample_rate = inputs.data.get_main_definition_entry().sample_rate
        if sample_rate != SAMPLE_RATE:
            logger.warning(f"ASR expects {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        audio = np.concatenate(chunks)
        duration = audio.shape[0] / SAMPLE_RATE
        if duration < ctx.config.min_audio_s:
            trace("asr_skip", reason="too_short", audio_s=f"{duration:.2f}")
            return
        if self._cancelled(ctx, inputs):
            trace("asr_skip", reason="cancelled")
            return

        t0 = time.perf_counter()
        text = self._transcribe(audio, ctx.config)
        decode_ms = (time.perf_counter() - t0) * 1e3
        trace("asr_done", audio_s=f"{duration:.2f}", decode_ms=f"{decode_ms:.0f}", chars=len(text))
        if not text:
            logger.info("ASR: no speech recognised")
            return
        if self._cancelled(ctx, inputs):
            trace("asr_skip", reason="cancelled_during_decode")
            return
        logger.info(f"ASR: {text!r}")
        bundle = DataBundle(output_definitions[ChatDataType.HUMAN_TEXT].definition)
        bundle.set_main_data(text)
        ctx.submit_data(bundle, finish_stream=True)

    @staticmethod
    def _cancelled(ctx: ASRContext, inputs: ChatData) -> bool:
        if ctx.stream_manager is None or inputs.stream_id is None:
            return False
        stream = ctx.stream_manager.find_stream(inputs.stream_id)
        return stream is not None and stream.status == ChatStreamStatus.CANCELLED

    def _transcribe(self, audio: np.ndarray, cfg: FasterWhisperConfig) -> str:
        segments, _ = self.model.transcribe(
            audio, language=cfg.language, task="transcribe", beam_size=cfg.beam_size, best_of=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
            initial_prompt=cfg.initial_prompt, no_speech_threshold=cfg.no_speech_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
        )
        parts = []
        for seg in segments:
            text = seg.text.strip()
            if text.lower().strip(" .!?,") in _HALLUCINATIONS and seg.no_speech_prob > 0.3:
                continue
            parts.append(text)
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return text if len(re.findall(r"[A-Za-z0-9]", text)) >= 2 else ""

    def destroy_context(self, context: HandlerContext):
        pass
