"""Headless client for the renderer WebSocket: plays a WAV as a reply and records what a viewer would see and hear.

    python tools/ws_client.py --url ws://127.0.0.1:8290/ws --avatar aisha --audio reply.wav --out reply.mp4
        [--format raw|jpeg] [--tts-speed 3]   # the TTS delivers audio 3x faster than real time (0 = all at once)
        [--interrupt-after 2.0]               # barge-in this many seconds after the first speaking frame

The recorded video holds every frame the service sent (listening, thinking, speaking, cross-fades); its audio track is the
PCM that arrived with the frames, so what you watch is exactly the A/V alignment the service produced.
"""
import argparse
import asyncio
import json
import os
import queue
import statistics
import struct
import subprocess
import threading
import time
import wave
import zlib

import cv2
import librosa
import numpy as np
import websockets

HEADER = struct.Struct("<BBBBIIHHII")
STATES = ["idle", "listening", "thinking", "speaking"]


def load_pcm24(path, max_seconds=0.0):
    y, _ = librosa.load(path, sr=24000, mono=True, duration=max_seconds or None)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()


class Recorder(threading.Thread):
    """Decodes frames and feeds ffmpeg off the event loop, so a slow encoder can never throttle the socket reader."""

    def __init__(self, path, width, height):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.crc = {}
        self.ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                    "-r", "25", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                                    "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        self.start()

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            seq, video, jpeg, w, h = item
            if jpeg:
                video = cv2.cvtColor(cv2.imdecode(np.frombuffer(video, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).tobytes()
            self.crc[seq] = zlib.crc32(video)
            self.ff.stdin.write(video)
        self.ff.stdin.close()
        self.ff.wait()

    def finish(self):
        self.q.put(None)
        self.join()


async def run(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pcm = load_pcm24(args.audio, args.max_seconds)
    frames, events = [], []
    t_start = time.perf_counter()
    marks = {}

    async with websockets.connect(args.url, max_size=None, compression=None) as ws:
        await ws.send(json.dumps({"type": "hello", "avatar": args.avatar, "format": args.format, "quality": args.quality}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"service refused: {ready}")
        print("ready:", ready, flush=True)
        rec = Recorder(args.out + ".video.mp4", ready["width"], ready["height"])
        audio_track = bytearray()
        done = asyncio.Event()

        async def receive():
            async for msg in ws:
                now = time.perf_counter()
                if isinstance(msg, str):
                    ev = json.loads(msg)
                    ev["t"] = now - t_start
                    events.append(ev)
                    print(f"  [{ev['t']:6.2f}s] event {ev}", flush=True)
                    if ev["type"] == "interrupted":
                        marks["interrupted_event"] = now
                    if ev["type"] in ("speak_done", "interrupted", "error"):
                        done.set()
                    continue
                magic, ver, flags, state, seq, pts_ms, fw, fh, alen, vlen = HEADER.unpack_from(msg, 0)
                speaking = bool(flags & 2)
                if speaking:
                    marks.setdefault("first_speaking", now)
                elif "interrupt_sent" in marks:
                    marks.setdefault("first_calm_frame", now)
                rec.q.put((seq, msg[HEADER.size:HEADER.size + vlen], bool(flags & 1), fw, fh))
                audio_track.extend(msg[HEADER.size + vlen:HEADER.size + vlen + alen] if alen else bytes(1920))
                frames.append((seq, now, STATES[state], speaking, len(msg)))

        rx = asyncio.create_task(receive())
        await ws.send(json.dumps({"type": "state", "state": "listening"}))
        await asyncio.sleep(args.lead_in)
        await ws.send(json.dumps({"type": "state", "state": "thinking"}))
        await asyncio.sleep(0.5)

        await ws.send(json.dumps({"type": "speak_start"}))
        chunk = int(24000 * 0.1) * 2

        async def send_audio():
            t0 = time.perf_counter()
            for off in range(0, len(pcm), chunk):
                marks.setdefault("first_audio", time.perf_counter())
                await ws.send(pcm[off:off + chunk])
                if args.tts_speed > 0:
                    await asyncio.sleep(max(0.0, t0 + (off + chunk) / 2 / 24000 / args.tts_speed - time.perf_counter()))
            await ws.send(json.dumps({"type": "speak_end"}))

        sender = asyncio.create_task(send_audio())
        if args.interrupt_after:
            while "first_speaking" not in marks or time.perf_counter() - marks["first_speaking"] < args.interrupt_after:
                await asyncio.sleep(0.02)
            sender.cancel()
            marks["interrupt_sent"] = time.perf_counter()
            await ws.send(json.dumps({"type": "interrupt"}))
        await asyncio.wait_for(done.wait(), timeout=args.timeout)
        await asyncio.sleep(args.tail)
        rx.cancel()
        rec.finish()

    with wave.open(args.out + ".wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(bytes(audio_track))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.out + ".video.mp4", "-i", args.out + ".wav", "-c:v", "copy",
                    "-c:a", "aac", "-shortest", args.out], check=True)
    for tmp in (args.out + ".video.mp4", args.out + ".wav"):
        os.remove(tmp)

    if args.meta:
        with open(args.meta, "w") as f:  # one state per recorded video frame, for tools/motion_probe.py
            json.dump({"fps": 25, "states": [f[2] if not f[3] else "speaking" for f in frames], "events": events}, f)

    speak = [f for f in frames if f[3]]
    report = {"frames_received": len(frames), "speaking_frames": len(speak),
              "bytes_per_frame_mean": round(statistics.mean(f[4] for f in frames))}
    if "first_speaking" in marks:
        report["first_audio_to_first_speaking_frame_ms"] = round((marks["first_speaking"] - marks["first_audio"]) * 1e3)
    if "interrupt_sent" in marks and "interrupted_event" in marks:
        report["interrupt_to_event_ms"] = round((marks["interrupted_event"] - marks["interrupt_sent"]) * 1e3)
        if "first_calm_frame" in marks:
            report["interrupt_to_calm_frame_ms"] = round((marks["first_calm_frame"] - marks["interrupt_sent"]) * 1e3)
    if len(speak) > 2:
        gaps = np.diff([f[1] for f in speak]) * 1e3
        report["speaking_arrival_ms"] = {"mean": round(float(gaps.mean()), 1), "p95": round(float(np.percentile(gaps, 95)), 1),
                                         "max": round(float(gaps.max()), 1)}
        crc = [rec.crc[f[0]] for f in speak]
        report["distinct_frames_per_s"] = round(25 * len(set(crc)) / len(crc), 1)
    print(json.dumps(report, indent=2))
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8290/ws")
    ap.add_argument("--avatar", default="")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="", help="also write the per-frame state sequence and the events here (JSON)")
    ap.add_argument("--format", choices=["raw", "jpeg"], default="raw")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--tts-speed", type=float, default=3.0)
    ap.add_argument("--interrupt-after", type=float, default=0.0)
    ap.add_argument("--lead-in", type=float, default=1.5)
    ap.add_argument("--tail", type=float, default=1.5)
    ap.add_argument("--timeout", type=float, default=120.0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
