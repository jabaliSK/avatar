"""Kokoro-82M streaming TTS for OpenAvatarChat (kokoro: Apache-2.0; Kokoro-82M weights: Apache-2.0).

The LLM's text deltas are cut into clauses (the first one as early as possible, later ones at sentence ends) and each clause is
synthesised as soon as it is complete. espeak-ng (GPL-3) is never loaded: words missing from misaki's dictionaries are skipped, so
product names that matter belong in `pronunciations`.

Every reply produces exactly one AVATAR_AUDIO stream and always closes it, even when nothing is speakable, so downstream handlers
(and the VAD's half-duplex gate) never wait for audio that will not come.
"""
import os
import re
import sys
import threading
import time
import types
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.chat_stream import ChatStreamIdentity, StreamKey
from chat_engine.data_models.chat_stream_config import ChatStreamConfig
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

from kka.common.trace import trace

KOKORO_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000


class KokoroTTSConfig(HandlerBaseConfigModel, BaseModel):
    model_dir: str = Field(default="/models/Kokoro-82M")
    voice: str = Field(default="am_michael", description="American (a*) or British (b*) voice; comma-separated names are blended")
    speed: float = Field(default=1.0)
    device: str = Field(default="cuda", description="cpu keeps this launch-bound model off a GPU that the LLM and renderer keep busy")
    cpu_threads: int = Field(default=0, description="torch intra-op threads when device is cpu; 0 keeps torch's default")
    first_flush_min_chars: int = Field(default=24, description="The first clause ends at the first punctuation after this many characters")
    first_soft_break_chars: int = Field(default=30, description="With no punctuation yet, the first clause ends before a conjunction once this long")
    first_flush_max_chars: int = Field(default=48, description="With no punctuation or conjunction, the first clause is cut at a space once this long")
    clause_min_chars: int = Field(default=40, description="Later clauses are cut at commas only beyond this length; sentence ends always cut")
    max_chunk_chars: int = Field(default=220)
    trim_silence: bool = Field(default=True, description="Trim the model's leading/trailing silence; pauses between clauses are re-added below")
    silence_threshold: float = Field(default=0.003, description="Peak amplitude below which a sample counts as silence")
    lead_keep_ms: int = Field(default=15)
    tail_keep_ms: int = Field(default=60)
    pause_sentence_ms: int = Field(default=160)
    pause_clause_ms: int = Field(default=70)
    pronunciations: Dict[str, str] = Field(default_factory=dict, description="Whole word (any case) -> spoken replacement; [word](/phonemes/) gives exact phonemes")


# ----------------------------------------------------------------------------------------------------------------------------
# Text handling (pure functions, unit-testable without a GPU)
# ----------------------------------------------------------------------------------------------------------------------------
_BOUNDARY = re.compile(r"([.!?\u2026]+|[;:,]+)([\"')\u201d\u2019]*)(?=\s)")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co", "corp", "no", "fig", "approx",
                  "e.g", "i.e", "u.s", "u.k", "a.m", "p.m"}
_LIST_MARKER = re.compile(r"^\s*\d{1,2}$")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
# Words a speaker would pause before: a break just ahead of one of these sounds like a clause boundary. Prepositions are left out on
# purpose ("look | for a server" would split a phrasal verb). The pattern needs whitespace after the word, so "help you with that."
# never matches.
_SOFT_BREAK_WORDS = ("and", "but", "or", "so", "that", "which", "who", "whom", "whose", "where", "when", "while", "because", "although",
                     "though", "unless", "until", "since", "if", "then", "including")
_SOFT_BREAK = re.compile(r"\s(?=(?:" + "|".join(_SOFT_BREAK_WORDS) + r")\s)", re.I)
_CLAUSE_END = ",;:.!?\u2026"


def _is_abbreviation(head: str) -> bool:
    """True when the '.' that follows `head` does not end a sentence ("Dr.", "J.", "1." at a list start)."""
    if _LIST_MARKER.match(head):
        return True
    m = _LAST_WORD.search(head)
    if not m:
        return False
    word = m.group(1)
    return word.lower().rstrip(".") in _ABBREVIATIONS or (len(word) == 1 and word.isupper())


def _find_cut(buffer: str, first: bool, cfg: KokoroTTSConfig) -> Optional[int]:
    for m in _BOUNDARY.finditer(buffer):
        punct = m.group(1)
        if punct == "." and _is_abbreviation(buffer[:m.start()]):
            continue
        end = m.end()
        n = len(buffer[:end].strip())
        if first:
            if n > cfg.first_flush_max_chars:
                return None  # too far for a first clause: the caller cuts earlier without punctuation
            if n >= cfg.first_flush_min_chars:
                return end
        elif punct[0] in ".!?\u2026" or n >= cfg.clause_min_chars:
            return end
    return None


def _find_soft_cut(buffer: str, cfg: KokoroTTSConfig) -> Optional[int]:
    """Index of a space before a conjunction or relative word that leaves a first clause long enough to speak."""
    for m in _SOFT_BREAK.finditer(buffer):
        n = len(buffer[:m.start()].strip())
        if n > cfg.first_flush_max_chars:
            return None
        if n >= cfg.first_soft_break_chars:
            return m.start()
    return None


def split_ready(buffer: str, first: bool, ended: bool, cfg: KokoroTTSConfig) -> Tuple[List[str], str, bool]:
    """Take every clause that is ready to be spoken out of `buffer`.

    A punctuation mark only counts once the character after it has arrived, because "3.5" and "Dr. Smith" look like sentence ends
    until then. The first clause is also cut without punctuation (before a conjunction, else at a space) so speech starts early;
    such a cut gets a trailing comma so the voice does not fall as if the sentence were over.
    Returns (clauses, remaining buffer, still waiting for the first clause).
    """
    chunks: List[str] = []
    while True:
        cut = _find_cut(buffer, first, cfg)
        forced = False
        if cut is None and first:
            cut = _find_soft_cut(buffer, cfg)
            forced = cut is not None
        if cut is None:
            limit = cfg.first_flush_max_chars if first else cfg.max_chunk_chars
            if len(buffer) <= limit:
                break
            cut = buffer.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            forced = True
        chunk, buffer = buffer[:cut].strip(), buffer[cut:].lstrip()
        if chunk:
            chunks.append(chunk + "," if forced and chunk[-1] not in _CLAUSE_END else chunk)
            first = False
    if ended:
        if buffer.strip():
            chunks.append(buffer.strip())
            first = False
        buffer = ""
    return chunks, buffer, first


_CONTROL = re.compile(r"<\|.*?\|>")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://)?(?:www\.)?((?:[a-z0-9-]+\.)+(?:com|org|net|io|edu|gov|co|ai|dev|info|biz|us|uk|de|in|ae|sa|qa)\b)(?:/\S*)?", re.I)
_DASHES = re.compile(r"\s*[\u2012-\u2015]\s*")
_BULLET = re.compile(r"(?m)^\s*[-*\u2022\u00b7]\s+")
_ELLIPSIS_MID = re.compile(r"\s*\.{2,}\s*(?=[a-z])")
_ELLIPSIS = re.compile(r"\.{2,}")
_HASH_NUMBER = re.compile(r"#(?=\d)")
_SPACES = re.compile(r"\s+")
_TRANSLATE = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ", "\u200b": ""})
_SYMBOLS = {"%": " percent", "&": " and ", "@": " at ", "+": " plus ", "=": " equals ", "<": " less than ", ">": " greater than ",
            "~": " about ", "/": " ", "\\": " ", "|": ", ", "_": " ", "*": " ", "#": " ", "^": " ", "`": " ", "[": " ", "]": " ",
            "{": " ", "}": " "}


def normalise(text: str) -> str:
    """Reduce LLM output to what Kokoro's English G2P can say: no markdown, emoji, CJK, URLs or stray symbols."""
    text = _CONTROL.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(lambda m: m.group(1).replace(".", " dot "), text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).translate(_TRANSLATE)
    text = _DASHES.sub(", ", text)
    text = _ELLIPSIS_MID.sub(", ", text)
    text = _ELLIPSIS.sub(".", text)
    text = _BULLET.sub("", text)
    text = _HASH_NUMBER.sub(" number ", text)
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = _SPACES.sub(" ", text).strip()
    return text if re.search(r"[A-Za-z0-9]", text) else ""


def compile_pronunciations(mapping: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    """One whole-word, case-insensitive pattern per `pronunciations` entry."""
    return [(re.compile(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", re.IGNORECASE), spoken) for word, spoken in mapping.items()]


def apply_pronunciations(text: str, patterns: List[Tuple[re.Pattern, str]]) -> str:
    """Apply the overrides to normalised text. The replacement is used literally (no backslash or group expansion)."""
    for pattern, spoken in patterns:
        text = pattern.sub(lambda _m, s=spoken: s, text)
    return text


def _block_espeak() -> None:
    """kokoro imports misaki.espeak at import time (GPL-3 espeak-ng) but only uses it for out-of-dictionary words.

    A stub module makes KPipeline fall back to "skip unknown words" without ever loading espeak.
    """
    if "misaki.espeak" in sys.modules:
        return
    stub = types.ModuleType("misaki.espeak")

    def _unavailable(*args, **kwargs):
        raise RuntimeError("espeak-ng (GPL-3) is not used; out-of-dictionary words are skipped")

    stub.EspeakFallback = _unavailable
    stub.EspeakG2P = _unavailable
    sys.modules["misaki.espeak"] = stub


# ----------------------------------------------------------------------------------------------------------------------------
# Handler
# ----------------------------------------------------------------------------------------------------------------------------
@dataclass
class _Session:
    """One reply: the text stream coming in and the audio stream going out."""
    input_stream_id: ChatStreamIdentity
    output_stream_key: Optional[StreamKey] = None
    buffer: str = ""
    first_clause: bool = True
    cancelled: bool = False
    first_text_at: Optional[float] = None
    first_audio_sent: bool = False
    prev_terminator: str = ""
    audio_s: float = 0.0


class KokoroContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = KokoroTTSConfig()
        self.sessions: Dict[StreamKey, _Session] = {}


class HandlerKokoroTTS(HandlerBase):
    def __init__(self):
        super().__init__()
        self.config = KokoroTTSConfig()
        self.pipeline = None
        self.voice_arg = ""
        self._pronunciations: List[Tuple[re.Pattern, str]] = []
        self._synth_lock = threading.Lock()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=KokoroTTSConfig)

    def get_handler_detail(self, session_context, context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, SAMPLE_RATE))
        return HandlerDetail(
            inputs=[HandlerDataInfo(type=ChatDataType.AVATAR_TEXT)],
            outputs=[HandlerDataInfo(type=ChatDataType.AVATAR_AUDIO, definition=definition)],
            signal_filters=[SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, None)],
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        cfg = handler_config if isinstance(handler_config, KokoroTTSConfig) else KokoroTTSConfig()
        self.config = cfg
        _block_espeak()
        import torch
        from kokoro import KModel, KPipeline

        t0 = time.perf_counter()
        voices = [v.strip() for v in cfg.voice.split(",") if v.strip()]
        families = {v[0] for v in voices}
        if not voices or len(families) != 1 or not families <= {"a", "b"}:
            raise ValueError(f"Kokoro voices must all be American (a*) or all British (b*); the others need espeak-ng. Got {cfg.voice!r}")
        if cfg.device == "cpu" and cfg.cpu_threads > 0:
            torch.set_num_threads(cfg.cpu_threads)
        paths = []
        for name in voices:
            path = name if name.endswith(".pt") else os.path.join(cfg.model_dir, "voices", f"{name}.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Kokoro voice file not found: {path}")
            paths.append(path)
        self.voice_arg = ",".join(paths)

        kmodel = KModel(repo_id=KOKORO_REPO, config=os.path.join(cfg.model_dir, "config.json"),
                        model=os.path.join(cfg.model_dir, "kokoro-v1_0.pth")).to(cfg.device).eval()
        self.pipeline = KPipeline(lang_code=voices[0][0], repo_id=KOKORO_REPO, model=kmodel)
        # Keep the voice pack on the GPU so every synthesis call skips a host-to-device copy.
        self.pipeline.voices[self.voice_arg] = self.pipeline.load_voice(self.voice_arg).to(kmodel.device)

        self._pronunciations = compile_pronunciations(cfg.pronunciations)
        with torch.inference_mode():
            for text in ("Hello, I am ready to help.", "This is a longer sentence, so the model also sees a bigger batch of phonemes."):
                for _ in self.pipeline(text, voice=self.voice_arg, speed=cfg.speed, split_pattern=None):
                    pass
        logger.info(f"Kokoro TTS ready in {time.perf_counter() - t0:.1f}s (voice {cfg.voice}, {cfg.device}, {torch.get_num_threads()} torch threads)")

    def create_context(self, session_context, handler_config: Optional[BaseModel] = None) -> HandlerContext:
        context = KokoroContext(session_context.session_info.session_id)
        if isinstance(handler_config, KokoroTTSConfig):
            context.config = handler_config
        return context

    def start_context(self, session_context, handler_context: HandlerContext):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData, output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        ctx = cast(KokoroContext, context)
        if inputs.type != ChatDataType.AVATAR_TEXT:
            return
        session = self._session_for(ctx, inputs)
        text = inputs.data.get_main_data()
        if text:
            if session.first_text_at is None and text.strip():
                session.first_text_at = time.perf_counter()
                trace("tts_first_text", chars=len(text))
            session.buffer += text
        ended = bool(inputs.is_last_data)
        clauses, session.buffer, session.first_clause = split_ready(session.buffer, session.first_clause, ended, self.config)
        definition = output_definitions[ChatDataType.AVATAR_AUDIO].definition
        for clause in clauses:
            if session.cancelled:
                return
            self._speak(ctx, session, clause, definition)
        if ended:
            self._finish(ctx, session, definition)

    def _session_for(self, ctx: KokoroContext, inputs: ChatData) -> _Session:
        key = inputs.stream_id.key
        session = ctx.sessions.get(key)
        if session is not None:
            return session
        # A new text stream replaces whatever was still being said.
        for old in list(ctx.sessions.values()):
            old.cancelled = True
        ctx.sessions.clear()
        session = _Session(input_stream_id=inputs.stream_id)
        ctx.sessions[key] = session
        streamer = ctx.data_submitter.get_streamer(ChatDataType.AVATAR_AUDIO)
        out_id = streamer.new_stream(sources=[inputs.stream_id], name="kokoro_tts", config=ChatStreamConfig(cancelable=True))
        session.output_stream_key = out_id.key
        return session

    def _speak(self, ctx: KokoroContext, session: _Session, clause: str, definition: DataBundleDefinition) -> None:
        import torch

        cfg = self.config
        text = apply_pronunciations(normalise(clause), self._pronunciations)
        if not text:
            logger.info(f"TTS: nothing speakable in {clause[:60]!r}")
            return
        logger.info(f"TTS: {text!r}")
        with self._synth_lock, torch.inference_mode():
            t0 = time.perf_counter()
            for result in self.pipeline(text, voice=self.voice_arg, speed=cfg.speed, split_pattern=None):
                if session.cancelled:
                    return
                if result.audio is None:
                    continue
                wav = result.audio.detach().float().cpu().numpy().reshape(-1)
                synth_ms = (time.perf_counter() - t0) * 1e3
                wav = self._shape(wav, session)
                session.prev_terminator = result.graphemes.rstrip(" \"')")[-1:]
                if wav.size == 0:
                    t0 = time.perf_counter()
                    continue
                bundle = DataBundle(definition)
                bundle.set_main_data(wav[np.newaxis, :])
                if session.cancelled:
                    return
                ctx.submit_data(bundle)
                audio_s = wav.size / SAMPLE_RATE
                session.audio_s += audio_s
                if not session.first_audio_sent:
                    session.first_audio_sent = True
                    ttfa = (time.perf_counter() - session.first_text_at) * 1e3 if session.first_text_at else -1
                    trace("tts_first_audio", ttfa_ms=f"{ttfa:.0f}", synth_ms=f"{synth_ms:.0f}", audio_s=f"{audio_s:.2f}", chars=len(text))
                else:
                    trace("tts_segment", synth_ms=f"{synth_ms:.0f}", audio_s=f"{audio_s:.2f}", chars=len(text))
                t0 = time.perf_counter()

    def _shape(self, wav: np.ndarray, session: _Session) -> np.ndarray:
        """Trim the model's own padding, then restore a pause that suits the punctuation before this segment."""
        cfg = self.config
        if cfg.trim_silence:
            loud = np.flatnonzero(np.abs(wav) > cfg.silence_threshold)
            if loud.size == 0:
                return wav[:0]
            start = max(0, int(loud[0]) - cfg.lead_keep_ms * SAMPLE_RATE // 1000)
            stop = min(wav.size, int(loud[-1]) + 1 + cfg.tail_keep_ms * SAMPLE_RATE // 1000)
            wav = wav[start:stop]
        pause_ms = 0
        if session.audio_s > 0 or session.first_audio_sent:
            pause_ms = cfg.pause_sentence_ms if session.prev_terminator in (".", "!", "?") else (
                cfg.pause_clause_ms if session.prev_terminator in (",", ";", ":") else 0)
        if pause_ms:
            wav = np.concatenate([np.zeros(pause_ms * SAMPLE_RATE // 1000, dtype=np.float32), wav])
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _finish(self, ctx: KokoroContext, session: _Session, definition: DataBundleDefinition) -> None:
        ctx.sessions.pop(session.input_stream_id.key, None)
        if session.cancelled:
            return
        # Closing frame: a few ms of silence. Sent even when nothing was spoken so the output stream always ends.
        end = DataBundle(definition)
        end.set_main_data(np.zeros((1, 240), dtype=np.float32))
        ctx.submit_data(end, finish_stream=True)
        trace("tts_done", audio_s=f"{session.audio_s:.2f}")

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        ctx = cast(KokoroContext, context)
        if signal.type != ChatSignalType.STREAM_CANCEL or signal.related_stream is None:
            return
        key = signal.related_stream.key
        for input_key, session in list(ctx.sessions.items()):
            if input_key == key or session.output_stream_key == key:
                session.cancelled = True
                ctx.sessions.pop(input_key, None)
                trace("tts_cancel", chars_pending=len(session.buffer))

    def destroy_context(self, context: HandlerContext):
        ctx = cast(KokoroContext, context)
        for session in ctx.sessions.values():
            session.cancelled = True
        ctx.sessions.clear()
