"""English-only guard around OpenAvatarChat's OpenAI-compatible LLM handler (Apache-2.0).

Everything the avatar says, and everything shown next to it, is English. Two things in the stock handler can break that:

* the model can slip into another script (Qwen is bilingual) and nothing filters what it streams;
* when a request fails, the handler answers with a Chinese error message, which would then be shown as a caption.

`HandlerEnglishLLM` is the stock handler with its client wrapped. CJK, kana and Hangul characters are removed from the streamed
text (full-width sentence punctuation becomes its ASCII form first, so the sentence chunker downstream still finds the ends).
A reply that had nothing but such characters is replaced by a short English sentence, and so is a request that failed, which
means the stock handler's own error branch is never reached. The stock behaviour is otherwise unchanged: the wrapped stream is
opened on first use and `close()` reaches the real one, which is what the handler relies on to cancel a reply on barge-in.

One more thing in the stock handler is closed here. The reply it is streaming is cancelled when the person speaks again (the
voice-activity handler restarts a turn on every pause), but the handler only learns of it from a signal that arrives a few tens
of milliseconds after the stream manager cancelled the stream. A token sent in between opens a new stream, which the next reply
then joins: the stray word shows up in front of the next caption and is spoken with the next sentence. So the wrapper reads the
reply stream's own status before it releases each chunk and, once it is cancelled, takes the stream out of the handler's active
set, which is what the signal would do a moment later; the handler then stops through its normal cancel branch.
"""
import re
from types import SimpleNamespace
from typing import Callable, Iterator, Optional, Tuple

from loguru import logger
from pydantic import Field

from chat_engine.common.handler_base import HandlerBaseInfo
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_stream_status import ChatStreamStatus

# Imported as a module on purpose: the engine takes the first concrete handler class it finds in this module's namespace.
from handlers.llm.openai_compatible import llm_handler_openai_compatible as _stock

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable virtual assistant. You appear as a video avatar and are talking with a person in real "
    "time, and everything you write is spoken aloud. Always answer in English, whatever language the question seems to be in; "
    "never write Chinese, Japanese, Korean or any other non-English script. Keep every reply short: at most two sentences and "
    "about twenty-five words, even for broad questions. Give the answer first and then stop; the person can ask for more detail. "
    "Reply the way people talk: plain conversational English with natural punctuation. Never use markdown, bullet points, "
    "numbered lists, headings, emojis, code or web addresses. Write numbers and abbreviations the way they are spoken. If a "
    "request is unclear, ask one short clarifying question. If you are not sure about something, say so briefly instead of "
    "guessing."
)
UNAVAILABLE = "Sorry, I could not reach my language model just now. Please try again in a moment."
ENGLISH_ONLY = "Sorry, I can only answer in English. Could you ask that again?"

_PUNCTUATION = str.maketrans({"\u3000": " ", "\uff0c": ", ", "\u3001": ", ", "\u3002": ". ", "\uff01": "! ", "\uff1f": "? ",
                              "\uff1a": ": ", "\uff1b": "; "})
# Hangul Jamo; CJK radicals; CJK symbols, kana, Bopomofo, enclosed and compatibility forms and Extension A; the unified
# ideographs; Hangul Jamo Extended-A and the syllables; compatibility ideographs and forms; full-width forms; the supplementary
# ideograph planes.
_FOREIGN = re.compile("[\u1100-\u11ff\u2e80-\u2fdf\u3000-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7ff\uf900-\ufaff\ufe30-\ufe4f"
                      "\uff00-\uffef\U00020000-\U0003ffff]")
_LETTER = re.compile(r"[A-Za-z]")


def english_only(text: str) -> Tuple[str, int]:
    """`text` without CJK, kana and Hangul characters, and how many characters were removed."""
    text = text.translate(_PUNCTUATION)
    clean = _FOREIGN.sub("", text)
    return clean, len(text) - len(clean)


class EnglishLLMConfig(_stock.LLMConfig):
    system_prompt: str = Field(default=SYSTEM_PROMPT)


def _chunk(text: str):
    """A completion chunk carrying `text`: the only part of a streamed chunk that the stock handler reads."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _GuardedStream:
    """A completion stream that is opened when first iterated, so that failing to connect is handled here as well."""

    def __init__(self, open_stream, reply_cancelled: Optional[Callable[[], bool]] = None):
        self._open_stream = open_stream
        self._reply_cancelled = reply_cancelled or (lambda: False)
        self._stream = None

    def __iter__(self) -> Iterator:
        removed, spoken, held = 0, False, []
        try:
            self._stream = self._open_stream()
            for chunk in self._stream:
                if self._reply_cancelled():
                    yield _chunk("")  # the stock loop finds its stream inactive now, closes this one and stops
                    return
                choices = getattr(chunk, "choices", None)
                text = getattr(choices[0].delta, "content", None) if choices else None
                if not text:
                    yield chunk
                    continue
                clean, count = english_only(text)
                removed += count
                spoken = spoken or bool(_LETTER.search(clean))
                out = _chunk(clean) if clean != text else chunk
                if spoken or not removed:
                    yield from held
                    held.clear()
                    yield out
                else:
                    # Foreign script and no English yet: show nothing until a word arrives, so that a reply that never has one
                    # is not preceded by leftover punctuation.
                    held.append(out)
            if removed and not spoken:
                logger.warning("the model answered without any English ({} characters dropped)", removed)
                yield _chunk("" if self._reply_cancelled() else ENGLISH_ONLY)
        except Exception as exc:
            logger.error("LLM request failed: {}", exc)
            yield _chunk(UNAVAILABLE)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()


class _GuardedClient:
    """Stands in for the OpenAI client: `chat.completions.create(...)` returns a guarded stream."""

    def __init__(self, client, context=None):
        self._client = client
        self._context = context
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _reply_cancelled(self) -> Callable[[], bool]:
        """A check that turns true once the avatar-text stream this request feeds is cancelled, and then ends the handler's use of it.

        The stock handler creates or reuses that stream and registers it as active just before it calls `create`, so the
        streamer's current stream is the reply's. Without a stream to watch, the check never fires (stock behaviour).
        """
        context = self._context
        submitter = getattr(context, "data_submitter", None)
        stream = submitter.get_streamer(ChatDataType.AVATAR_TEXT).current_stream if submitter is not None else None
        if stream is None:
            return lambda: False
        key = stream.identity.stream_key_str

        def cancelled() -> bool:
            if stream.status != ChatStreamStatus.CANCELLED:
                return False
            context.active_stream_keys.discard(key)
            return True

        return cancelled

    def _create(self, **kwargs) -> _GuardedStream:
        return _GuardedStream(lambda: self._client.chat.completions.create(**kwargs), self._reply_cancelled())

    def close(self) -> None:
        self._client.close()


class HandlerEnglishLLM(_stock.HandlerLLM):
    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=EnglishLLMConfig)

    def create_context(self, session_context, handler_config=None):
        context = super().create_context(session_context, handler_config or EnglishLLMConfig())
        context.client = _GuardedClient(context.client, context)
        return context
