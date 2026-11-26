"""The English-only LLM guard: foreign script is removed from the stream, failures are answered in English, and the wrapper is
transparent to the stock handler (lazy stream, `close()`, usage chunks). No model, no network.

The non-English sample text is written as escapes on purpose, so that no source file in the repository contains such characters.
"""
import inspect
import os
import sys
from types import SimpleNamespace

sys.path[:0] = ["/opt/oac/src", "/opt/oac/src/handlers", os.path.join(os.path.dirname(__file__), "..", "handlers")]

from chat_engine.common.handler_base import HandlerBase  # noqa: E402
from chat_engine.data_models.chat_stream_status import ChatStreamStatus  # noqa: E402
from kka.llm import english_llm  # noqa: E402
from kka.llm.english_llm import ENGLISH_ONLY, SYSTEM_PROMPT, UNAVAILABLE, _GuardedClient, english_only  # noqa: E402

HELLO_ZH = "\u4f60\u597d"  # Chinese ideographs
WORLD_ZH = "\u4e16\u754c"
COMMA_ZH = "\uff0c"  # full-width comma
STOP_ZH = "\u3002"  # ideographic full stop
KANA = "\u3053\u3093\u306b\u3061\u306f"
HANGUL = "\uc548\ub155"
ERROR_ZH = "\u8fde\u63a5\u9519\u8bef"


def delta(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeStream:
    def __init__(self, pieces, fail_at=None):
        self.pieces, self.fail_at, self.closed = pieces, fail_at, False

    def __iter__(self):
        for i, piece in enumerate(self.pieces):
            if i == self.fail_at:
                raise ConnectionError(ERROR_ZH)
            yield piece

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, stream=None, error=None):
        self.stream, self.error, self.calls, self.closed = stream, error, [], False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.stream

    def close(self):
        self.closed = True


def spoken(client, **kwargs):
    """What the stock handler would read from a reply: the non-empty text of every chunk that has any."""
    stream = _GuardedClient(client).chat.completions.create(model="m", messages=[], **kwargs)
    return [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]


def test_foreign_characters_are_removed_and_full_width_punctuation_becomes_ascii():
    clean, removed = english_only(f"Sure{COMMA_ZH}{HELLO_ZH} I can help{STOP_ZH}{KANA} {HANGUL}")
    assert " ".join(clean.split()) == "Sure, I can help."
    assert removed == 9


def test_plain_english_with_accents_and_typographic_punctuation_is_untouched():
    text = "Caf\u00e9 \u2014 it\u2019s 42% done!"
    assert english_only(text) == (text, 0)


def test_english_streams_through_unchanged_including_the_usage_chunk():
    chunks = [delta("Hello"), delta(" there."), SimpleNamespace(choices=[], usage={"total_tokens": 9})]
    stream = _GuardedClient(FakeClient(FakeStream(chunks))).chat.completions.create(model="m", messages=[])
    assert list(stream) == chunks


def test_a_mixed_reply_keeps_its_english_and_loses_the_rest():
    pieces = [delta(f"Sure{COMMA_ZH}"), delta(HELLO_ZH), delta(f" I can help{STOP_ZH}")]
    assert "".join(spoken(FakeClient(FakeStream(pieces)))) == "Sure,  I can help. "


def test_a_reply_with_no_english_is_replaced_without_leftover_punctuation():
    pieces = [delta(HELLO_ZH), delta(f"{WORLD_ZH}{STOP_ZH}"), delta(KANA)]
    assert spoken(FakeClient(FakeStream(pieces))) == [ENGLISH_ONLY]


def test_text_held_back_is_released_in_order_once_english_arrives():
    pieces = [delta(HELLO_ZH + STOP_ZH), delta("Hi"), delta(" there")]
    assert spoken(FakeClient(FakeStream(pieces))) == [". ", "Hi", " there"]


def test_a_request_that_fails_to_start_is_answered_in_english():
    client = FakeClient(error=ConnectionError(ERROR_ZH))
    assert spoken(client) == [UNAVAILABLE]
    assert english_only(UNAVAILABLE)[1] == 0


def test_a_stream_that_breaks_midway_ends_with_the_english_notice():
    pieces = [delta("One"), delta(" two"), delta(" three")]
    assert spoken(FakeClient(FakeStream(pieces, fail_at=2))) == ["One", " two", UNAVAILABLE]


def test_the_stream_is_opened_on_first_use_and_close_reaches_it():
    stream = FakeStream([delta("Hi")])
    client = FakeClient(stream)
    guarded = _GuardedClient(client).chat.completions.create(model="m", messages=[], stream=True)
    assert client.calls == []
    guarded.close()  # not opened yet: nothing to close
    assert not stream.closed
    next(iter(guarded))
    assert client.calls == [{"model": "m", "messages": [], "stream": True}]
    guarded.close()
    assert stream.closed


def reply_context(status=ChatStreamStatus.STARTED):
    """The parts of the handler context the guard reads: the avatar-text stream the reply feeds and the handler's active set."""
    reply = SimpleNamespace(status=status, identity=SimpleNamespace(stream_key_str="avatar_text:7:0"))
    streamer = SimpleNamespace(current_stream=reply)
    context = SimpleNamespace(data_submitter=SimpleNamespace(get_streamer=lambda data_type: streamer),
                              active_stream_keys={reply.identity.stream_key_str})
    return reply, context


def stock_loop(context, reply, pieces, cancel_after=None):
    """The stock handler's loop over a reply: it stops when its stream has left the active set, and delivers the rest.

    `cancel_after` cancels the reply stream after that many pieces, the way the stream manager does; the cancel signal that
    would take the stream out of the active set has not arrived yet.
    """
    stream = _GuardedClient(FakeClient(FakeStream(pieces)), context).chat.completions.create(model="m", messages=[], stream=True)
    delivered = []
    for chunk in stream:
        if reply.identity.stream_key_str not in context.active_stream_keys:
            break
        delivered.append(chunk.choices[0].delta.content)
        if cancel_after == len(delivered):
            reply.status = ChatStreamStatus.CANCELLED
    return delivered


def test_a_token_that_follows_a_cancel_is_not_delivered_before_the_cancel_signal_arrives():
    reply, context = reply_context()
    pieces = [delta("Of"), delta(" course"), delta(" to"), delta(" help")]
    assert stock_loop(context, reply, pieces, cancel_after=2) == ["Of", " course"]
    assert reply.identity.stream_key_str not in context.active_stream_keys


def test_a_reply_that_is_not_cancelled_is_delivered_whole_and_stays_active():
    reply, context = reply_context()
    pieces = [delta("Of"), delta(" course"), delta(" to"), delta(" help")]
    assert stock_loop(context, reply, pieces) == ["Of", " course", " to", " help"]
    assert reply.identity.stream_key_str in context.active_stream_keys


def test_the_system_prompt_is_english_and_asks_for_english():
    assert SYSTEM_PROMPT.isascii() and "Always answer in English" in SYSTEM_PROMPT
    assert english_llm.EnglishLLMConfig().system_prompt == SYSTEM_PROMPT


def test_the_engine_finds_exactly_one_handler_class_in_the_module():
    found = [obj for _, obj in inspect.getmembers(english_llm, inspect.isclass)
             if issubclass(obj, HandlerBase) and not inspect.isabstract(obj)]
    assert found == [english_llm.HandlerEnglishLLM]


def test_the_context_gets_a_guarded_client_and_the_configured_prompt(monkeypatch):
    built = {}

    class FakeOpenAI(FakeClient):
        def __init__(self, **kwargs):
            super().__init__(FakeStream([delta(HELLO_ZH + "Hi")]))
            built.update(kwargs)

    monkeypatch.setattr(english_llm._stock, "OpenAI", FakeOpenAI)
    config = english_llm.EnglishLLMConfig(api_key="EMPTY", api_url="http://llm.invalid/v1", model_name="m")
    session = SimpleNamespace(session_info=SimpleNamespace(session_id="s1"))
    context = english_llm.HandlerEnglishLLM().create_context(session, config)
    assert isinstance(context.client, _GuardedClient)
    assert built["base_url"] == "http://llm.invalid/v1"
    assert context.system_prompt == {"role": "system", "content": SYSTEM_PROMPT}
    reply = context.client.chat.completions.create(model="m", messages=[], stream=True)
    assert [c.choices[0].delta.content for c in reply] == ["Hi"]
    context.client.close()
