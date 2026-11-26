#!/usr/bin/env python3
"""Line up an e2e_client.py run with the gateway's KKA_T trace lines and print the latency budget of every turn.

    docker logs kka-gateway 2>&1 | python merge_timeline.py <e2e out dir>
    python merge_timeline.py <e2e out dir> --log gateway.log

<e2e out dir> is the --out directory of e2e_client.py (it holds metrics.json). The client and the gateway stamp events with the
epoch clock of the machine they run on, so they have to share a machine (or have clocks synchronised to a few milliseconds).
Only the Python standard library is used.
"""
import argparse
import json
import os
import re
import sys

TRACE = re.compile(r"KKA_T (?P<event>\w+) ts=(?P<ts>\d+)(?P<rest>.*)")
FIELD = re.compile(r"(\w+)=(\S+)")
# The gateway's logger colours its lines; a reset code after the last field would otherwise become part of that field's value.
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# After the ASR text: (trace event, label, how to describe its fields). The first such event inside the turn is used.
STAGES = [
    ("tts_first_text", "first LLM token reaches TTS", lambda f: ""),
    ("tts_first_audio", "first clause synthesised", lambda f: f"TTS {f.get('synth_ms')} ms for the clause"),
    ("avatar_first_pcm", "renderer receives the first audio", lambda f: ""),
    ("renderer_speak_started", "renderer: first frame ready", lambda f: f"{f.get('first_audio_to_first_frame_ms')} ms after its first audio"),
    ("avatar_first_speaking_frame", "gateway receives the first video frame", lambda f: ""),
]
# Trace events that carry the id of a reply the moment it begins.
REPLY_STARTS = ("avatar_reply_begin", "avatar_first_pcm", "renderer_speak_started")


def parse_trace(lines):
    """KKA_T lines -> [{"event", "t" (epoch seconds), "f" (key=value fields)}], other lines are ignored."""
    events = []
    for line in lines:
        m = TRACE.search(ANSI.sub("", line))
        if m:
            events.append({"event": m["event"], "t": int(m["ts"]) / 1000.0, "f": dict(FIELD.findall(m["rest"]))})
    return events


def _first(events, name):
    return next((e for e in events if e["event"] == name), None)


def turn_timeline(turn, window_end, events):
    """[(label, seconds after the end of the user's speech, detail)] for one turn; stages that did not happen are left out."""
    t_end = turn["speech_end"]
    mine = [e for e in events if turn["speech_start"] <= e["t"] < window_end]
    rows = [("you stop speaking", 0.0, "")]

    # A decode that was cancelled because the user kept talking logs asr_done and asr_skip together; it did not start a reply.
    cancelled = [e["t"] for e in mine if e["event"] == "asr_skip" and e["f"].get("reason") == "cancelled_during_decode"]
    asr = [e for e in mine if e["event"] == "asr_done" and e["t"] >= t_end and not any(abs(e["t"] - c) <= 0.01 for c in cancelled)]
    first_text = _first(mine, "tts_first_text")
    if first_text:
        asr = [e for e in asr if e["t"] <= first_text["t"]]
    if asr:
        rows.append(("ASR text ready", asr[-1]["t"] - t_end, f"decode {asr[-1]['f'].get('decode_ms')} ms, includes the VAD end-of-speech wait"))

    for name, label, describe in STAGES:
        e = _first([m for m in mine if m["t"] >= t_end], name)
        if e:
            rows.append((label, e["t"] - t_end, describe(e["f"])))

    heard = turn.get("first_audio_after_speech_end_s")
    if heard is not None:
        rows.append(("reply sound reaches the client", heard, "WebRTC transport + jitter buffer"))
    return sorted(rows, key=lambda r: r[1])


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def renderer_summary(turn, window_end, events):
    """Lines on how the renderer kept up while it played the replies that began during this turn, or None.

    This is where repeated pictures show up: when a frame is not ready within its 40 ms slot the renderer shows the previous one again.
    A reply is followed to its end by its id, because a reply that was interrupted ends inside the next turn's window. Ids restart
    with every session, so the end event is the first one with that id after the reply began.
    """
    starts = {}
    for e in events:
        if turn["speech_start"] <= e["t"] < window_end and e["event"] in REPLY_STARTS and e["f"].get("reply") is not None:
            starts.setdefault(e["f"]["reply"], e)
    lines = []
    for rid, start in starts.items():
        end = next((e for e in events if e["t"] >= start["t"] and (
            (e["event"] == "renderer_speak_done" and e["f"].get("id") == rid)
            or (e["event"] == "renderer_interrupted" and e["f"].get("reply") == rid))), None)
        if end and end["event"] == "renderer_interrupted":
            lines.append(f"  renderer: the reply was interrupted after {end['f'].get('frames_played')} frames")
        elif end:
            f = end["f"]
            frames, repeated, render_ms, underruns = (_number(f.get(k)) for k in ("frames", "repeated", "render_ms", "audio_underruns"))
            parts = []
            if frames is not None:
                parts.append(f"{int(frames)} frames")
            if frames and repeated is not None:
                parts.append(f"{int(repeated)} repeated ({100 * repeated / frames:.0f}%)")
            if render_ms is not None:
                parts.append(f"{render_ms:.1f} ms to render a frame (a 25 fps clock allows 40)")
            if underruns:
                parts.append(f"{int(underruns)} audio underruns")
            if parts:
                lines.append("  renderer: " + ", ".join(parts))
    return "\n".join(lines) or None


def barge_in_summary(turn, window_end, events):
    """One line on how fast the avatar fell silent when this turn's speech interrupted it, or None when it interrupted nothing.

    The gateway waits for a minimum amount of speech before it cuts a reply (so a cough does not), which is most of the delay.
    """
    mine = [e for e in events if turn["speech_start"] <= e["t"] < window_end]
    barge = _first(mine, "barge_in")
    if not barge:
        return None
    parts = [f"the gateway cut the reply {barge['t'] - turn['speech_start']:.2f} s after you started talking "
             f"(it waits for {barge['f'].get('speech_ms')} ms of speech)"]
    stopped = next((e for e in mine if e["event"] == "renderer_interrupted" and e["t"] >= barge["t"]), None)
    if stopped:
        parts.append(f"the renderer stopped {1000 * (stopped['t'] - barge['t']):.0f} ms later")
    heard = turn.get("old_reply_stopped_after_s")
    if heard is not None:
        parts.append(f"the old reply's sound ended at the client {heard:.2f} s after you started talking")
    return "  barge-in: " + "; ".join(parts)


def format_turn(index, turn, rows):
    out = [f"turn {index}  {os.path.basename(turn['clip'])} ({turn['how']})"]
    if turn.get("asr_text"):
        out.append(f"  heard : {turn['asr_text']!r}")
    if turn.get("reply_text"):
        out.append(f"  reply : {turn['reply_text'][:100]!r}")
    out.append("  after speech end    step    stage")
    previous, steps = 0.0, []
    for label, t, detail in rows:
        step = t - previous
        steps.append((step, label))
        out.append(f"  {t:+8.3f} s      {step:+6.3f}  {label}" + (f"  ({detail})" if detail else ""))
        previous = t
    if len(rows) > 2:
        step, label = max(steps[1:])
        out.append(f"  -> first reply sound {rows[-1][1]:.2f} s after speech end; the longest single step is '{label}' ({step:.2f} s)")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="--out directory of an e2e_client.py run")
    ap.add_argument("--log", help="file with the gateway log (default: standard input)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        turns = json.load(f)["turns"]
    if args.log:
        with open(args.log, errors="replace") as f:
            events = parse_trace(f)
    elif not sys.stdin.isatty():
        events = parse_trace(sys.stdin)
    else:
        ap.error("pipe the gateway log in (docker logs kka-gateway 2>&1 | ...) or pass --log")
    if not events:
        print("no KKA_T lines found in the log; only the client side can be shown", file=sys.stderr)

    for i, turn in enumerate(turns, 1):
        window_end = turns[i]["speech_start"] if i < len(turns) else float("inf")
        extras = [s for s in (renderer_summary(turn, window_end, events), barge_in_summary(turn, window_end, events)) if s]
        print("\n".join([format_turn(i, turn, turn_timeline(turn, window_end, events)), *extras]), end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
