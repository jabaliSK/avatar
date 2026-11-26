"""Unit tests for the pure text logic of the Kokoro handler (no GPU, no model).

Run inside the gateway image:
  docker run --rm -v $PWD/third_party/OpenAvatarChat/src:/opt/oac/src:ro -v $PWD/services/gateway:/opt/kka-gw:ro -w /opt/oac \
      kka-gateway-deps:latest python -m pytest -q /opt/kka-gw/tests
"""
import os
import sys

sys.path[:0] = ["/opt/oac/src", "/opt/oac/src/handlers", os.path.join(os.path.dirname(__file__), "..", "handlers")]

from kka.tts.kokoro_tts import KokoroTTSConfig, apply_pronunciations, compile_pronunciations, normalise, split_ready  # noqa: E402

CFG = KokoroTTSConfig()


def feed(pieces, cfg=CFG):
    """Simulate LLM token streaming: returns the clauses in the order they would be sent to the synthesiser."""
    buffer, first, out = "", True, []
    for piece in pieces:
        buffer += piece
        chunks, buffer, first = split_ready(buffer, first, False, cfg)
        out += chunks
    chunks, buffer, first = split_ready(buffer, first, True, cfg)
    return out + chunks


def test_first_clause_is_sent_as_soon_as_it_is_complete():
    text = "Sure, I can help you with that. What would you like to know about our servers?"
    assert feed(list(text)) == ["Sure, I can help you with that.", "What would you like to know about our servers?"]


def test_first_clause_cuts_at_comma_once_long_enough():
    out = feed(["Well, ", "that depends on the workload you have, ", "so tell me more."])
    assert out[0] == "Well, that depends on the workload you have,"


def test_short_leading_interjection_is_not_sent_alone():
    assert feed(["Ok, ", "let me look at that for you."])[0].startswith("Ok, let me")


def test_decimal_and_abbreviation_are_not_sentence_ends():
    assert feed(["Version 3.5 is out now for all users. ", "Dr. Smith agrees."]) == ["Version 3.5 is out now for all users.", "Dr. Smith agrees."]


def test_punctuation_waits_for_next_character():
    # A trailing "." could still be the start of "3.5", so nothing is released until a space arrives.
    chunks, rest, first = split_ready("Version 3.", True, False, CFG)
    assert chunks == [] and rest == "Version 3." and first is True


def test_first_clause_is_cut_before_a_conjunction_when_no_punctuation_comes():
    text = "Sure! HPE GreenLake is a cloud service that lets you rent computing resources on demand."
    out = feed(list(text))
    assert out == ["Sure! HPE GreenLake is a cloud service,", "that lets you rent computing resources on demand."]


def test_no_soft_break_before_a_word_that_ends_the_sentence():
    out = feed(list("Hello there, I can help you with that. Anything else?"))
    assert out[0] == "Hello there, I can help you with that."


def test_forced_first_cut_is_short_and_ends_with_a_comma():
    chunks, rest, first = split_ready("word " * 60, True, False, CFG)
    assert chunks[0].endswith(",") and len(chunks[0]) <= CFG.first_flush_max_chars and first is False


def test_punctuation_beyond_the_first_clause_limit_is_not_waited_for():
    chunks, _, _ = split_ready("word " * 18 + "end. Next one follows.", True, False, CFG)
    assert len(chunks[0]) <= CFG.first_flush_max_chars


def test_long_text_without_punctuation_is_cut_at_a_space():
    chunks, rest, first = split_ready("word " * 120, True, False, CFG)
    assert len(chunks) > 1 and all(len(c) <= CFG.max_chunk_chars for c in chunks)


def test_remainder_is_flushed_at_end_of_stream():
    assert feed(["No punctuation at the end"]) == ["No punctuation at the end"]


def test_normalise_strips_markdown_emoji_and_cjk():
    assert normalise("**Hello** there \U0001F600 \u4f60\u597d!") == "Hello there !"
    assert normalise("See [our site](https://example.com/x) now") == "See our site now"
    assert normalise("- first\n- second") == "first second"


def test_normalise_symbols_and_urls():
    assert normalise("Save 20% at www.hpe.com") == "Save 20 percent at hpe dot com"
    assert normalise("Open https://www.hpe.com/us/en/home.html now") == "Open hpe dot com now"
    assert normalise("Write to jo@example.com.") == "Write to jo at example dot com."
    assert normalise("R&D") == "R and D"


def test_normalise_leaves_decimals_and_abbreviations_alone():
    assert normalise("It costs 3.5 times more, e.g. in Node.js") == "It costs 3.5 times more, e.g. in Node.js"


def test_normalise_empty_for_non_speech():
    assert normalise("\u8fde\u63a5\u9519\u8bef") == ""
    assert normalise("... --- !!!") == ""


def test_pronunciations_match_whole_words_in_any_case():
    patterns = compile_pronunciations({"HPE": "H P E", "iLO": "i L O"})
    assert apply_pronunciations("hpe and HPE, but not HPEs or xHPE; ILO.", patterns) == "H P E and H P E, but not HPEs or xHPE; i L O."


def test_pronunciation_replacement_is_used_literally():
    patterns = compile_pronunciations({"Alletra": r"[Alletra](/əlˈɛtɹə/) \1 \g<0>"})
    assert apply_pronunciations("Try Alletra.", patterns) == r"Try [Alletra](/əlˈɛtɹə/) \1 \g<0>."
