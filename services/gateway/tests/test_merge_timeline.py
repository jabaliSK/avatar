"""merge_timeline: client + gateway traces of one turn become a single latency budget (stdlib only, no GPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import merge_timeline as mt  # noqa: E402

# Trace lines and client timings copied from a real two-turn run (work/e2e/t9_two_sentences). In turn 2 the user paused long
# enough for the VAD to cut the utterance, then went on talking: the first decode was cancelled and the whole sentence decoded again.
LOG = """\
2026-11-25 15:01:34.540 | INFO | kka.asr:1 - KKA_T asr_done ts=1795624494540 audio_s=3.20 decode_ms=321 chars=48
KKA_T tts_first_text ts=1795624494730 chars=1
KKA_T tts_first_audio ts=1795624495477 ttfa_ms=748 synth_ms=565 audio_s=2.52 chars=36
KKA_T avatar_first_pcm ts=1795624495484 reply=1
KKA_T renderer_speak_started ts=1795624495966 reply=1 first_audio_to_first_frame_ms=475
KKA_T avatar_first_speaking_frame ts=1795624495978 reply=1 seq=175
KKA_T tts_segment ts=1795624496145 synth_ms=666 audio_s=3.85 chars=57
KKA_T renderer_speak_done ts=1795624506525 id=1 frames=265
KKA_T asr_done ts=1795624512628 audio_s=3.17 decode_ms=241 chars=61
KKA_T asr_skip ts=1795624512628 reason=cancelled_during_decode
KKA_T asr_done ts=1795624514101 audio_s=4.70 decode_ms=264 chars=64
KKA_T tts_first_text ts=1795624514198 chars=1
KKA_T tts_first_audio ts=1795624515090 ttfa_ms=891 synth_ms=679 audio_s=3.15 chars=46
unrelated log line
"""
TURNS = [
    {"clip": "/out/q_greenlake.wav", "how": "say", "speech_start": 1795624491.149, "speech_end": 1795624493.737,
     "asr_text": "What is HPE GreenLake?", "reply_text": "HPE GreenLake is a cloud service.", "first_audio_after_speech_end_s": 2.386},
    {"clip": "/out/q_servers.wav", "how": "say", "speech_start": 1795624509.449, "speech_end": 1795624513.507,
     "asr_text": "What is the difference between a ProServer and a Synergy system?", "reply_text": None},
]


def rows(index):
    window_end = TURNS[index + 1]["speech_start"] if index + 1 < len(TURNS) else float("inf")
    return {label: t for label, t, _ in mt.turn_timeline(TURNS[index], window_end, mt.parse_trace(LOG.splitlines()))}


def test_parse_ignores_other_lines_and_reads_fields():
    events = mt.parse_trace(LOG.splitlines())
    assert len(events) == 13
    assert events[0]["event"] == "asr_done" and events[0]["f"]["decode_ms"] == "321" and events[0]["t"] == 1795624494.54


def test_colour_codes_of_the_log_do_not_end_up_in_field_values():
    line = "\x1b[32m2026-11-26 02:11:00 | INFO\x1b[0m - KKA_T renderer_speak_started ts=1795659079400 reply=1 first_audio_to_first_frame_ms=521\x1b[0m"
    assert mt.parse_trace([line])[0]["f"] == {"reply": "1", "first_audio_to_first_frame_ms": "521"}


def test_turn_offsets_are_relative_to_the_end_of_speech():
    r = rows(0)
    assert round(r["ASR text ready"], 3) == 0.803
    assert round(r["first LLM token reaches TTS"], 3) == 0.993
    assert round(r["gateway receives the first video frame"], 3) == 2.241
    assert r["reply sound reaches the client"] == 2.386


def test_cancelled_decode_is_not_the_asr_result():
    assert round(rows(1)["ASR text ready"], 3) == 0.594


def test_stages_that_did_not_happen_are_left_out_and_report_still_prints():
    window_end = float("inf")
    only_asr = mt.parse_trace(["KKA_T asr_done ts=1795624514101 decode_ms=264"])
    labels = [r[0] for r in mt.turn_timeline(TURNS[1], window_end, only_asr)]
    assert labels == ["you stop speaking", "ASR text ready"]
    assert "ASR text ready" in mt.format_turn(2, TURNS[1], mt.turn_timeline(TURNS[1], window_end, only_asr))


def test_renderer_line_reports_repeats_render_time_and_underruns():
    log = ("KKA_T avatar_first_pcm ts=1795624495484 reply=1\n"
           "KKA_T renderer_speak_done ts=1795624506525 id=1 frames=301 audio_s=11.12 repeated=64 late=64 stale=0 "
           "audio_underruns=0 render_ms=43.1 wall_s=12.0\n"
           "KKA_T avatar_first_pcm ts=1795624514200 reply=2\n"
           "KKA_T renderer_speak_done ts=1795624520000 id=2 frames=100 audio_s=4 repeated=0 render_ms=30.0 audio_underruns=2\n")
    events = mt.parse_trace(log.splitlines())
    assert mt.renderer_summary(TURNS[0], TURNS[1]["speech_start"], events) == (
        "  renderer: 301 frames, 64 repeated (21%), 43.1 ms to render a frame (a 25 fps clock allows 40)")
    assert mt.renderer_summary(TURNS[1], float("inf"), events).endswith("30.0 ms to render a frame (a 25 fps clock allows 40), 2 audio underruns")
    assert mt.renderer_summary(TURNS[0], TURNS[1]["speech_start"], mt.parse_trace(["KKA_T asr_done ts=1795624494540"])) is None


# Real trace of a barge-in: reply 1 is cut 3 s in by the next question, which is answered by reply 2. The interruption is logged
# inside the window of turn 2, and the ids start again at 1 in a later session.
BARGE_LOG = """\
KKA_T avatar_reply_begin ts=1795634682707 reply=1
KKA_T avatar_first_pcm ts=1795634682708 reply=1
KKA_T renderer_speak_started ts=1795634683506 reply=1 first_audio_to_first_frame_ms=790
KKA_T barge_in ts=1795634687475 leaf=client_playback streams=1 speech_ms=256
KKA_T avatar_interrupt ts=1795634687477 reason=playback_cancelled
KKA_T renderer_interrupted ts=1795634687527 reply=1 frames_played=100
KKA_T avatar_reply_begin ts=1795634690373 reply=2
KKA_T renderer_speak_started ts=1795634690825 reply=2 first_audio_to_first_frame_ms=448
KKA_T renderer_speak_done ts=1795634693185 id=2 frames=60 audio_s=1.6 repeated=8 late=8 stale=0 audio_underruns=0 render_ms=43.0 wall_s=2.36
KKA_T avatar_reply_begin ts=1795634800000 reply=1
KKA_T renderer_speak_done ts=1795634805000 id=1 frames=999 repeated=0 render_ms=40.0
"""
BARGE_TURNS = [
    {"clip": "q_long.wav", "how": "say", "speech_start": 1795634668.0, "speech_end": 1795634680.626},
    {"clip": "q_interrupt.wav", "how": "barge_on_audio", "speech_start": 1795634686.835, "speech_end": 1795634689.6,
     "old_reply_stopped_after_s": 0.812},
]


def test_interrupted_reply_is_reported_on_the_turn_that_started_it():
    events = mt.parse_trace(BARGE_LOG.splitlines())
    assert mt.renderer_summary(BARGE_TURNS[0], BARGE_TURNS[1]["speech_start"], events) == "  renderer: the reply was interrupted after 100 frames"
    assert mt.renderer_summary(BARGE_TURNS[1], 1795634700.0, events) == (
        "  renderer: 60 frames, 8 repeated (13%), 43.0 ms to render a frame (a 25 fps clock allows 40)")


def test_barge_in_line_gives_the_gateway_renderer_and_client_delays():
    events = mt.parse_trace(BARGE_LOG.splitlines())
    assert mt.barge_in_summary(BARGE_TURNS[0], BARGE_TURNS[1]["speech_start"], events) is None
    assert mt.barge_in_summary(BARGE_TURNS[1], 1795634700.0, events) == (
        "  barge-in: the gateway cut the reply 0.64 s after you started talking (it waits for 256 ms of speech); "
        "the renderer stopped 52 ms later; the old reply's sound ended at the client 0.81 s after you started talking")
