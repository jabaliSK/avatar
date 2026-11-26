"""Benchmark + smoke test for the two GPU speech handlers, using the handlers' own code (no engine, no WebRTC).

  docker run --rm --gpus '"device=0"' --entrypoint python -v $PWD/models:/models:ro -v $PWD/services/gateway/tools:/opt/kka-tools:ro \
      -v $PWD/work/e2e:/out kka-gateway:latest /opt/kka-tools/bench_asr_tts.py

Measures, against the targets in the stack spec:
  ASR  decode latency per utterance (target: < 120 ms) for 1-12 s utterances, and transcript accuracy on synthetic questions
  TTS  time-to-first-audio for the first clause (target: < 150 ms), real-time factor, leading/trailing silence, and a
       TTS -> ASR round trip over the HPE vocabulary to check that pronunciation works without espeak-ng
It also writes the spoken test questions (24 kHz mono wav) that tools/e2e_client.py plays into the gateway as microphone audio.
"""
import json
import os
import statistics
import sys
import time
import wave

import numpy as np

sys.path[:0] = ["/opt/oac/src", "/opt/oac/src/handlers", "/opt/kka-handlers"]

from kka.asr.faster_whisper_asr import FasterWhisperConfig, HandlerFasterWhisperASR  # noqa: E402
from kka.tts.kokoro_tts import SAMPLE_RATE, HandlerKokoroTTS, KokoroTTSConfig, normalise  # noqa: E402

OUT = os.environ.get("OUT_DIR", "/out")
VOICES = "/models/Kokoro-82M/voices"

QUESTIONS = {
    "q_greenlake": ("af_heart", "Hi there, can you tell me what HPE GreenLake is?"),
    "q_servers": ("bm_george", "What is the difference between a ProLiant server and a Synergy system?"),
    "q_help": ("af_nicole", "Hello, could you give me a quick summary of what you can help me with today?"),
    "q_short": ("am_adam", "Are you there?"),
    "q_long": ("bf_emma", "I'm planning to move our virtual machines from an old data center to a new platform next year, and I would like "
                          "to understand what options we have for storage and networking, and how long a migration like that usually takes."),
    "q_interrupt": ("af_heart", "Wait, stop. Let me ask you something else."),
}

ROUND_TRIP = [
    "HPE GreenLake is a cloud service.", "The ProLiant server supports iLO management.", "Aruba makes wireless access points.",
    "Alletra storage replaces Nimble Storage.", "HPE Synergy is a composable infrastructure platform.",
    "Cray builds supercomputers for Hewlett Packard Enterprise.", "OneView manages servers and firmware.",
]

FIRST_CLAUSES = ["Sure, I can help with that.", "Great question, the main difference is how they scale.",
                 "That depends on the workload you have in mind, so let me ask you a few things first."]


def save_wav(path, wav, rate):
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def resample(wav, src, dst):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(src, dst)
    return resample_poly(wav, dst // g, src // g).astype(np.float32)


def synth(tts, text, voice=None):
    """Returns (audio, time-to-first-audio ms). Mirrors HandlerKokoroTTS._speak minus the engine plumbing."""
    import torch
    v = tts.voice_arg if voice is None else os.path.join(VOICES, voice + ".pt")
    parts, first = [], None
    t0 = time.perf_counter()
    with torch.inference_mode():
        for r in tts.pipeline(text, voice=v, speed=tts.config.speed, split_pattern=None):
            if r.audio is None:
                continue
            wav = r.audio.detach().float().cpu().numpy().reshape(-1)
            if first is None:
                first = (time.perf_counter() - t0) * 1e3
            parts.append(wav)
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), first


def stats(values):
    values = sorted(values)
    return {"median": round(statistics.median(values), 1), "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
            "min": round(values[0], 1), "max": round(values[-1], 1), "n": len(values)}


def norm_words(text):
    import re
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main():
    os.makedirs(OUT, exist_ok=True)
    tts = HandlerKokoroTTS()
    tts.load(None, KokoroTTSConfig())
    asr = HandlerFasterWhisperASR()
    asr_cfg = FasterWhisperConfig(initial_prompt="HPE, Hewlett Packard Enterprise, ProLiant, Aruba, GreenLake, Synergy, Nimble Storage, "
                                                 "Alletra, Cray, OneView, iLO.")
    asr.load(None, asr_cfg)
    report = {}

    # ---- questions + ASR accuracy -------------------------------------------------------------------------------------------
    rows = []
    for name, (voice, text) in QUESTIONS.items():
        wav, _ = synth(tts, normalise(text), voice)
        save_wav(os.path.join(OUT, name + ".wav"), wav, SAMPLE_RATE)
        wav16 = resample(wav, SAMPLE_RATE, 16000)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            heard = asr._transcribe(wav16, asr_cfg)
            times.append((time.perf_counter() - t0) * 1e3)
        rows.append({"name": name, "voice": voice, "audio_s": round(wav.size / SAMPLE_RATE, 2), "asr_ms": stats(times),
                     "said": text, "heard": heard, "wer": round(wer(text, heard), 3)})
        print(f"[asr] {name:12s} {wav.size / SAMPLE_RATE:5.1f}s  decode median {statistics.median(times):6.0f} ms  WER {rows[-1]['wer']:.2f}  heard: {heard!r}")
    report["questions"] = rows

    # ---- ASR latency vs utterance length (same audio truncated/tiled) ----------------------------------------------------------
    base = resample(synth(tts, normalise(QUESTIONS["q_long"][1]), "bf_emma")[0], SAMPLE_RATE, 16000)
    length_rows = []
    for secs in (1, 2, 4, 6, 8, 12):
        clip = base[: int(secs * 16000)]
        times = []
        for _ in range(8):
            t0 = time.perf_counter()
            asr._transcribe(clip, asr_cfg)
            times.append((time.perf_counter() - t0) * 1e3)
        length_rows.append({"audio_s": secs, **stats(times)})
        print(f"[asr] {secs:2d}s clip   decode median {statistics.median(times):6.0f} ms  p95 {length_rows[-1]['p95']:6.0f} ms")
    report["asr_latency_by_length"] = length_rows

    # ---- TTS time-to-first-audio -----------------------------------------------------------------------------------------------
    tts_rows = []
    for clause in FIRST_CLAUSES:
        text = normalise(clause)
        ttfa, rtf, lead, tail = [], [], [], []
        for _ in range(10):
            t0 = time.perf_counter()
            wav, first = synth(tts, text)
            total = (time.perf_counter() - t0) * 1e3
            ttfa.append(first)
            rtf.append(total / 1e3 / (wav.size / SAMPLE_RATE))
            loud = np.flatnonzero(np.abs(wav) > tts.config.silence_threshold)
            lead.append(loud[0] / SAMPLE_RATE * 1e3)
            tail.append((wav.size - 1 - loud[-1]) / SAMPLE_RATE * 1e3)
        tts_rows.append({"clause": clause, "ttfa_ms": stats(ttfa), "rtf_median": round(statistics.median(rtf), 3),
                         "audio_s": round(wav.size / SAMPLE_RATE, 2), "lead_silence_ms": round(statistics.median(lead)),
                         "tail_silence_ms": round(statistics.median(tail))})
        print(f"[tts] {len(text):3d} chars  TTFA median {statistics.median(ttfa):5.0f} ms p95 {tts_rows[-1]['ttfa_ms']['p95']:5.0f}  "
              f"RTF {statistics.median(rtf):.3f}  silence lead/tail {tts_rows[-1]['lead_silence_ms']}/{tts_rows[-1]['tail_silence_ms']} ms")
    report["tts"] = tts_rows

    # ---- pronunciation round trip (TTS -> ASR) ---------------------------------------------------------------------------------
    pron = []
    for sentence in ROUND_TRIP:
        text = normalise(sentence)
        for pattern, spoken in tts._pronunciations:
            text = pattern.sub(lambda _m, s=spoken: s, text)
        wav, _ = synth(tts, text)
        heard = asr._transcribe(resample(wav, SAMPLE_RATE, 16000), asr_cfg)
        pron.append({"said": sentence, "spoken_text": text, "heard": heard, "wer": round(wer(sentence, heard), 3)})
        print(f"[rt ] WER {pron[-1]['wer']:.2f}  said {sentence!r}\n           heard {heard!r}")
    report["pronunciation_round_trip"] = pron

    with open(os.path.join(OUT, "bench_asr_tts.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", os.path.join(OUT, "bench_asr_tts.json"))


if __name__ == "__main__":
    main()
