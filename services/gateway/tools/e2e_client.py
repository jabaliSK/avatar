#!/usr/bin/env python
"""Headless WebRTC client that holds a conversation with the gateway the way the browser UI does.

The microphone is a queue of clips played in real time; the avatar's audio and video are received, timed, and written out as an
mp4 together with the transcripts and playback signals the gateway sends on the `text` data channel. All times are epoch seconds
from the host clock, the same clock the gateway's `KKA_T` trace lines use, so tools/merge_timeline.py can line both up.
Lip-sync is not judged here: the picture and the sound only meet in the mp4, and a loudness-versus-picture-change estimate
scattered over +-400 ms on real runs. Run the renderer image's tools/lipsync_probe.py on the mp4 instead (it tracks the mouth and
recovered a 200 ms test delay to within 10 ms).

A run is a list of steps:
  say:<clip>                       play a clip (path, or noise:<seconds>) and return when its last voiced sample has been sent
  reply[:<timeout_s>]              wait for a complete reply to the last clip (sound, then 1.2 s of silence)
  noreply:<seconds>                pass only if no reply sound starts within <seconds> of the last clip
  sleep:<seconds>
  say_on_audio:<delay_s>:<clip>    barge-in: play the clip <delay_s> after the reply's sound starts
  say_after_end:<delay_s>:<clip>   barge-in while the reply is still being prepared: <delay_s> after the last clip ended
"""
import argparse
import asyncio
import fractions
import json
import os
import sys
import time
import uuid
import wave

import av
import httpx
import numpy as np
from aiortc import (RTCConfiguration, RTCIceServer, RTCIceTransportPolicy, RTCPeerConnection, RTCRtpReceiver,
                    RTCSessionDescription)
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, VideoStreamTrack

SR = 48000
FRAME = 960  # 20 ms of 48 kHz audio
AUDIBLE = 0.003  # RMS above which a received 20 ms audio frame counts as sound
REPLY_GAP_S = 1.2  # silence that ends a reply
VIDEO_CLOCK = 90000


# ----------------------------------------------------------------------------------------------------------------------------
# Microphone clips
# ----------------------------------------------------------------------------------------------------------------------------
def decode_audio(path: str) -> np.ndarray:
    container = av.open(path)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    parts = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            parts.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        parts.append(out.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(parts).astype(np.float32) / 32768.0


def synth_noise(seconds: float) -> np.ndarray:
    """Speech-band noise with a 4 Hz syllable-like envelope: energy a naive detector could mistake for speech."""
    rng = np.random.default_rng(7)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < 300) | (freqs > 3400)] = 0
    x = np.fft.irfft(spec, n).astype(np.float32)
    x *= (0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(n) / SR)) ** 2
    return x / (np.abs(x).max() + 1e-9)


def load_clip(spec: str, level_db: float):
    """Returns (samples, first voiced sample, end of voiced samples), trimmed to 150 ms before / 50 ms after the voiced part."""
    x = synth_noise(float(spec[len("noise:"):])) if spec.startswith("noise:") else decode_audio(spec)
    loud = np.flatnonzero(np.abs(x) > 0.01)
    if loud.size == 0:
        raise SystemExit(f"{spec}: no audible samples")
    a = max(0, int(loud[0]) - int(0.15 * SR))
    b = min(x.size, int(loud[-1]) + 1 + int(0.05 * SR))
    x = x[a:b]
    vs, ve = int(loud[0]) - a, int(loud[-1]) + 1 - a
    rms = float(np.sqrt(np.mean(x[vs:ve] ** 2)))
    x = np.clip(x * (10 ** (level_db / 20) / rms), -0.98, 0.98).astype(np.float32)
    return x, vs, ve


class MicTrack(AudioStreamTrack):
    """48 kHz mono microphone: silence, with queued clips mixed in. Paced like a real capture device."""

    def __init__(self):
        super().__init__()
        self._queue = []

    def play(self, clip) -> "asyncio.Future":
        x, vs, ve = clip
        future = asyncio.get_running_loop().create_future()
        self._queue.append({"x": x, "vs": vs, "ve": ve, "pos": 0, "future": future, "marks": {}})
        return future

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        if hasattr(self, "_timestamp"):
            self._timestamp += FRAME
            wait = self._start + self._timestamp / SR - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        t0 = time.time()
        mix = np.zeros(FRAME, np.float32)
        for clip in list(self._queue):
            lo = clip["pos"]
            seg = clip["x"][lo:lo + FRAME]
            mix[:seg.size] += seg
            marks = clip["marks"]
            if "t_start" not in marks and lo + seg.size > clip["vs"]:
                marks["t_start"] = t0 + max(clip["vs"] - lo, 0) / SR
            if "t_end" not in marks and lo + seg.size >= clip["ve"]:
                marks["t_end"] = t0 + (clip["ve"] - lo) / SR
                clip["future"].set_result(marks)
            clip["pos"] = lo + FRAME
            if clip["pos"] >= clip["x"].size:
                self._queue.remove(clip)
        frame = av.AudioFrame(format="s16", layout="mono", samples=FRAME)
        frame.planes[0].update((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
        frame.pts = self._timestamp
        frame.sample_rate = SR
        frame.time_base = fractions.Fraction(1, SR)
        return frame


class DummyVideo(VideoStreamTrack):
    """A small black camera at 5 fps. The server only creates its video track when the client sends one."""

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        step = VIDEO_CLOCK // 5
        if hasattr(self, "_timestamp"):
            self._timestamp += step
            wait = self._start + self._timestamp / VIDEO_CLOCK - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        frame = av.VideoFrame(width=160, height=120, format="yuv420p")
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
        return frame


# ----------------------------------------------------------------------------------------------------------------------------
# What the client receives
# ----------------------------------------------------------------------------------------------------------------------------
class Capture:
    def __init__(self):
        self.audio = []  # (arrival, rtp pts at 48 kHz, rms)
        self.pcm = []  # int16 mono samples of each audio frame
        self.video = []  # (arrival, rtp pts at 90 kHz, av.VideoFrame, small grey copy)
        self.rtp_bytes = {"audio": 0, "video": 0}
        self.events = []
        self.onsets = []  # arrival of the first sound after at least REPLY_GAP_S of silence
        self.last_audible = None

    def log(self, source: str, kind: str, **data) -> dict:
        event = {"t": round(time.time(), 3), "src": source, "kind": kind, **data}
        self.events.append(event)
        return event

    async def pump_audio(self, track):
        try:
            while True:
                frame = await track.recv()
                t = time.time()
                channels = len(frame.layout.channels)
                x = frame.to_ndarray().reshape(-1, channels).mean(axis=1)
                rms = float(np.sqrt(np.mean((x / 32768.0) ** 2)))
                self.audio.append((t, frame.pts, rms))
                self.pcm.append(x.astype(np.int16))
                if rms > AUDIBLE:
                    if self.last_audible is None or t - self.last_audible >= REPLY_GAP_S:
                        self.onsets.append(t)
                    self.last_audible = t
        except MediaStreamError:
            pass

    async def pump_video(self, track):
        try:
            while True:
                frame = await track.recv()
                t = time.time()
                plane = frame.planes[0]
                grey = np.frombuffer(plane, np.uint8)[:frame.height * plane.line_size].reshape(frame.height, plane.line_size)
                self.video.append((t, frame.pts, frame, grey[::8, :frame.width:8].astype(np.int16)))
        except MediaStreamError:
            pass


def count_rtp_bytes(cap: Capture) -> None:
    original = RTCRtpReceiver._handle_rtp_packet

    async def counting(self, packet, arrival_time_ms):
        try:
            cap.rtp_bytes[self.track.kind] += len(packet.payload)
        except Exception:
            pass
        return await original(self, packet, arrival_time_ms)

    RTCRtpReceiver._handle_rtp_packet = counting


def describe_message(msg: dict) -> str:
    name = msg.get("header", {}).get("name", "?")
    p = msg.get("payload") or {}
    if name in ("EchoHumanText", "EchoAvatarText"):
        return f"{name} end={p.get('end_of_speech')} {p.get('text', '')[:70]!r}"
    if name == "ChatSignal":
        return f"{name} {p.get('type')} {p.get('stream_type')}"
    return name


def ice_configuration(args) -> RTCConfiguration:
    """No ICE servers unless --turn-url is given. --relay-only then forbids every path except the TURN relay."""
    servers = []
    if args.turn_url:
        servers.append(RTCIceServer(urls=args.turn_url, username=args.turn_user, credential=args.turn_password))
    policy = RTCIceTransportPolicy.RELAY if args.relay_only else RTCIceTransportPolicy.ALL
    return RTCConfiguration(iceServers=servers, iceTransportPolicy=policy)


def selected_pair(pc) -> str:
    """The nominated ICE pair as 'local <type> host:port <-> remote <type> host:port'; a local 'relay' means the media goes through
    TURN. aiortc has no public accessor for this, so it reads aioice's private state and says 'unknown' if that ever changes."""
    try:
        pair = next(iter(pc.sctp.transport.transport._connection._nominated.values()))
        local, remote = pair.local_candidate, pair.remote_candidate
        return f"local {local.type} {local.host}:{local.port} <-> remote {remote.type} {remote.host}:{remote.port}"
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


async def connect(args, cap: Capture, mic: MicTrack, camera: DummyVideo):
    pc = RTCPeerConnection(ice_configuration(args))
    channel = pc.createDataChannel("text")
    tasks = []

    @channel.on("message")
    def on_message(message):
        try:
            msg = json.loads(message)
        except Exception:
            cap.log("channel", "raw", data=str(message)[:200])
            return
        event = cap.log("channel", (msg.get("header") or {}).get("name", "?"), payload=msg.get("payload"))
        print(f"  [{event['t'] - cap.t_conn:7.3f}] channel {describe_message(msg)}", flush=True)

    @pc.on("track")
    def on_track(track):
        pump = cap.pump_audio if track.kind == "audio" else cap.pump_video
        tasks.append(asyncio.ensure_future(pump(track)))

    pc.addTrack(mic)
    pc.addTrack(camera)
    if args.offer_codec != "default":
        wanted = {"h264": "video/H264", "vp8": "video/VP8"}[args.offer_codec]
        codecs = RTCRtpReceiver.getCapabilities("video").codecs
        preferred = [c for c in codecs if c.mimeType == wanted] + [c for c in codecs if c.mimeType != wanted]
        for transceiver in pc.getTransceivers():
            if transceiver.kind == "video":
                transceiver.setCodecPreferences(preferred)

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=30, verify=not args.insecure) as http:
        response = await http.post(f"{args.url}/webrtc/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "webrtc_id": uuid.uuid4().hex[:12]})
    body = response.json()
    if "sdp" not in body:
        raise SystemExit(f"offer rejected: {response.status_code} {body}")
    await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
    deadline = time.time() + 25
    while (pc.connectionState != "connected" or channel.readyState != "open") and time.time() < deadline:
        await asyncio.sleep(0.05)
    if pc.connectionState != "connected":
        raise SystemExit(f"WebRTC did not connect (state {pc.connectionState}, channel {channel.readyState})")
    return pc, channel, tasks, answer_codec(body["sdp"])


def answer_codec(sdp: str) -> str:
    """The codec the server will send video in: the first payload type of its video m-line."""
    lines = sdp.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("m=video"):
            first = line.split()[3]
            for other in lines[i:]:
                if other.startswith(f"a=rtpmap:{first} "):
                    return other.split(" ", 1)[1]
    return "?"


# ----------------------------------------------------------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------------------------------------------------------
class Script:
    def __init__(self, args, cap: Capture, mic: MicTrack):
        self.args, self.cap, self.mic = args, cap, mic
        self.says = []
        self.failed = []

    async def say(self, spec: str, how: str = "say") -> dict:
        marks = await self.mic.play(load_clip(spec, self.args.level_db))
        entry = {"clip": spec, "how": how, **{k: round(v, 3) for k, v in marks.items()}}
        self.says.append(entry)
        self.cap.log("client", "speech", **entry)
        print(f"  [{marks['t_start'] - self.cap.t_conn:7.3f}] client  {how} {spec} voiced {marks['t_end'] - marks['t_start']:.2f}s", flush=True)
        return entry

    def last_end(self) -> float:
        return self.says[-1]["t_end"] if self.says else self.cap.t_conn

    async def onset_after(self, ref: float, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            onset = next((o for o in self.cap.onsets if o > ref), None)
            if onset is not None:
                return onset
            await asyncio.sleep(0.03)
        return None

    async def reply(self, timeout: float) -> None:
        ref = self.last_end()
        onset = await self.onset_after(ref, timeout)
        if onset is None:
            self.failed.append(f"no reply within {timeout:.0f}s of {self.says[-1]['clip'] if self.says else 'start'}")
            print(f"  !! no reply within {timeout:.0f}s", flush=True)
            return
        print(f"  [{onset - self.cap.t_conn:7.3f}] client  reply sound starts, {onset - ref:.2f}s after speech end", flush=True)
        while time.time() - (self.cap.last_audible or 0) < REPLY_GAP_S:
            await asyncio.sleep(0.05)
            if time.time() - onset > 120:
                self.failed.append("reply still playing after 120 s")
                return
        print(f"  [{self.cap.last_audible - self.cap.t_conn:7.3f}] client  reply sound ends", flush=True)

    async def run(self, step: str) -> None:
        kind, _, rest = step.partition(":")
        print(f"step {step}", flush=True)
        if kind == "say":
            await self.say(rest)
        elif kind == "reply":
            await self.reply(float(rest) if rest else self.args.reply_timeout)
        elif kind == "noreply":
            ref = self.last_end()
            if await self.onset_after(ref, float(rest)) is not None:
                self.failed.append(f"unexpected reply after {self.says[-1]['clip']}")
                print("  !! a reply started", flush=True)
            else:
                print(f"  ok: no reply sound for {float(rest):.1f}s", flush=True)
        elif kind == "sleep":
            await asyncio.sleep(float(rest))
        elif kind == "say_on_audio":
            delay, _, clip = rest.partition(":")
            if await self.onset_after(self.last_end(), self.args.reply_timeout) is None:
                self.failed.append("no reply sound to barge into")
                return
            await asyncio.sleep(float(delay))
            await self.say(clip, "barge_on_audio")
        elif kind == "say_after_end":
            delay, _, clip = rest.partition(":")
            await asyncio.sleep(max(0.0, self.last_end() + float(delay) - time.time()))
            await self.say(clip, "barge_after_end")
        else:
            raise SystemExit(f"unknown step {step!r}")


# ----------------------------------------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------------------------------------
def quantiles(values, scale=1.0):
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: round(v[min(len(v) - 1, int(q * len(v)))] * scale, 2)
    return {"mean": round(sum(v) / len(v) * scale, 2), "p50": pick(0.5), "p95": pick(0.95), "max": round(v[-1] * scale, 2)}


def replies_from_audio(cap: Capture):
    """Groups sound into replies separated by >= REPLY_GAP_S of silence."""
    loud = [(t, pts) for (t, pts, rms) in cap.audio if rms > AUDIBLE]
    groups, cur = [], []
    for item in loud:
        if cur and (item[1] - cur[-1][1]) / SR >= REPLY_GAP_S:
            groups.append(cur)
            cur = []
        cur.append(item)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        pauses = [round((b[1] - a[1]) / SR - 0.02, 2) for a, b in zip(g, g[1:]) if (b[1] - a[1]) / SR - 0.02 >= 0.2]
        out.append({"onset": round(g[0][0], 3), "end": round(g[-1][0], 3), "span_s": round((g[-1][1] - g[0][1]) / SR + 0.02, 2),
                    "sound_s": round(len(g) * 0.02, 2), "pauses_ge_200ms": pauses})
    return out


def video_stats(video, lo: float, hi: float):
    """Cadence of the pictures received in [lo, hi].

    Pictures the renderer repeated (it does that when a frame is not ready in time) cannot be counted here: H.264 codes an identical
    picture a little differently each time, so after decoding they are not equal. The renderer's own count is printed by
    merge_timeline.py.
    """
    sel = [v for v in video if lo <= v[0] <= hi]
    if len(sel) < 3:
        return None
    diffs = [float(np.abs(b[3] - a[3]).mean()) for a, b in zip(sel, sel[1:])]
    span = sel[-1][0] - sel[0][0]
    return {
        "frames": len(sel), "fps": round((len(sel) - 1) / span, 2) if span > 0 else None,
        "arrival_interval_ms": quantiles([b[0] - a[0] for a, b in zip(sel, sel[1:])], 1000),
        "rtp_interval_ms": quantiles([(b[1] - a[1]) / VIDEO_CLOCK for a, b in zip(sel, sel[1:])], 1000),
        "frame_diff": quantiles(diffs),
    }


def analyse(cap: Capture, script: Script, codec: str) -> dict:
    says = script.says
    replies = replies_from_audio(cap)
    turns = []
    for i, say in enumerate(says):
        window_end = says[i + 1]["t_start"] if i + 1 < len(says) else float("inf")
        events = [e for e in cap.events if e["src"] == "channel" and say["t_start"] <= e["t"] < window_end]
        asr = next((e for e in events if e["kind"] == "EchoHumanText" and (e["payload"] or {}).get("end_of_speech")), None)
        first_text = next((e for e in events if e["kind"] == "EchoAvatarText"), None)
        text = "".join((e["payload"] or {}).get("text", "") for e in events if e["kind"] == "EchoAvatarText")
        signals = [(e["kind"], (e["payload"] or {}).get("type"), (e["payload"] or {}).get("stream_type"), round(e["t"] - say["t_end"], 3))
                   for e in events if e["kind"] == "ChatSignal"]
        reply = next((r for r in replies if r["onset"] > say["t_end"] and r["onset"] < window_end + 0.0), None)
        turn = {"clip": say["clip"], "how": say["how"], "speech_start": say["t_start"], "speech_end": say["t_end"],
                "asr_text": (asr or {}).get("payload", {}).get("text") if asr else None,
                "asr_text_after_speech_end_s": round(asr["t"] - say["t_end"], 3) if asr else None,
                "reply_text": text or None,
                "first_reply_text_after_speech_end_s": round(first_text["t"] - say["t_end"], 3) if first_text else None,
                "playback_signals": signals}
        if reply:
            turn.update(first_audio_after_speech_end_s=round(reply["onset"] - say["t_end"], 3), reply=reply)
        if say["how"].startswith("barge"):
            # The reply that was sounding when this clip began; its last audible frame is when the avatar fell silent.
            playing = [r for r in replies if r["onset"] < say["t_start"] <= r["end"]]
            if playing:
                turn["old_reply_stopped_after_s"] = round(playing[-1]["end"] - say["t_start"], 3)
        turns.append(turn)

    metrics = {"video_codec_answer": codec, "turns": turns, "replies": replies, "script_failures": script.failed}
    if cap.video:
        t0, t1 = cap.video[0][0], cap.video[-1][0]
        metrics["video_all"] = video_stats(cap.video, t0 + 1, t1)
        metrics["video_kbps"] = round(cap.rtp_bytes["video"] * 8 / 1000 / max(t1 - t0, 1e-3), 0)
        metrics["video_size"] = [cap.video[0][2].width, cap.video[0][2].height]
        for r in replies:
            r["video"] = video_stats(cap.video, r["onset"], r["end"])
    if cap.audio:
        arrivals = [b[0] - a[0] for a, b in zip(cap.audio, cap.audio[1:])]
        metrics["audio_arrival_interval_ms"] = quantiles(arrivals, 1000)
        metrics["audio_kbps"] = round(cap.rtp_bytes["audio"] * 8 / 1000 / max(cap.audio[-1][0] - cap.audio[0][0], 1e-3), 1)
    return metrics


# ----------------------------------------------------------------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------------------------------------------------------------
def write_wav(path: str, cap: Capture) -> None:
    pcm = np.concatenate(cap.pcm) if cap.pcm else np.zeros(0, np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def write_mp4(path: str, cap: Capture, fps: int = 25) -> None:
    """Video re-timed to a constant frame rate from the sender's RTP clock; each stream starts where it first arrived."""
    t_v0, pts_v0 = cap.video[0][0], cap.video[0][1]
    t_ref = min(t_v0, cap.audio[0][0]) if cap.audio else t_v0
    container = av.open(path, "w")
    vs = container.add_stream("libx264", rate=fps)
    first = cap.video[0][2]
    vs.width, vs.height, vs.pix_fmt = first.width, first.height, "yuv420p"
    vs.options = {"crf": "20", "preset": "veryfast"}
    ast = None
    if cap.audio:
        ast = container.add_stream("aac", rate=SR)
        ast.layout = "mono"

    times = [(v[1] - pts_v0) / VIDEO_CLOCK for v in cap.video]
    shift = int(round((t_v0 - t_ref) * fps))
    j = 0
    for i in range(int(times[-1] * fps) + 1):
        while j + 1 < len(times) and times[j + 1] <= i / fps + 1e-6:
            j += 1
        frame = cap.video[j][2].reformat(format="yuv420p")
        frame.pts, frame.time_base = shift + i, fractions.Fraction(1, fps)
        for packet in vs.encode(frame):
            container.mux(packet)
    for packet in vs.encode(None):
        container.mux(packet)

    if ast is not None:
        pts0 = cap.audio[0][1]
        signal = np.zeros(cap.audio[-1][1] - pts0 + FRAME, np.int16)
        for (_, pts, _), block in zip(cap.audio, cap.pcm):
            signal[pts - pts0:pts - pts0 + block.size] = block
        signal = np.concatenate([np.zeros(int(round((cap.audio[0][0] - t_ref) * SR)), np.int16), signal])
        for k in range(0, signal.size, 1024):
            frame = av.AudioFrame.from_ndarray(signal[k:k + 1024].reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = SR, k, fractions.Fraction(1, SR)
            for packet in ast.encode(frame):
                container.mux(packet)
        for packet in ast.encode(None):
            container.mux(packet)
    container.close()


def save_frames(out: str, cap: Capture, replies: list, count: int) -> None:
    """Full-resolution frames spread across the first reply, to judge what H.264 did to the picture."""
    if not replies or not cap.video:
        return
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    r = replies[0]
    for k in range(count):
        t = r["onset"] + 0.4 + k * max(r["span_s"] - 0.8, 1.0) / max(count - 1, 1)
        frame = min(cap.video, key=lambda v: abs(v[0] - t))[2]
        frame.to_image().save(os.path.join(out, "frames", f"reply1_{k}.jpg"), quality=93)


def print_summary(metrics: dict, t_conn: float) -> None:
    print("\n==== summary ====")
    print(f"ICE pair: {metrics.get('ice_pair')}")
    print(f"video codec (server answer): {metrics['video_codec_answer']}   frame {metrics.get('video_size')}   "
          f"video {metrics.get('video_kbps')} kbps   audio {metrics.get('audio_kbps')} kbps")
    for i, t in enumerate(metrics["turns"], 1):
        print(f"turn {i} [{t['how']}] {t['clip']}")
        if t.get("old_reply_stopped_after_s") is not None:
            print(f"   old reply stopped   : {t['old_reply_stopped_after_s']} s after you started talking over it")
        print(f"   ASR heard           : {t['asr_text']!r} ({t['asr_text_after_speech_end_s']} s after speech end)")
        print(f"   first reply text    : {t['first_reply_text_after_speech_end_s']} s after speech end")
        print(f"   first reply sound   : {t.get('first_audio_after_speech_end_s')} s after speech end")
        print(f"   reply text          : {(t['reply_text'] or '')[:160]!r}")
        if t.get("reply"):
            r = t["reply"]
            print(f"   reply sound         : span {r['span_s']} s, pauses >= 200 ms: {r['pauses_ge_200ms']}")
            if r.get("video"):
                v = r["video"]
                print(f"   video during reply  : {v['fps']} fps arriving, arrival interval {v['arrival_interval_ms']}")
        sig = [s for s in t["playback_signals"]]
        if sig:
            print(f"   playback signals    : {sig}")
    if metrics["script_failures"]:
        print("FAILURES:", metrics["script_failures"])


async def main(args) -> int:
    cap = Capture()
    cap.t_conn = time.time()
    count_rtp_bytes(cap)
    mic, camera = MicTrack(), DummyVideo()
    pc, channel, tasks, codec = await connect(args, cap, mic, camera)
    cap.t_conn = time.time()
    ice_pair = selected_pair(pc)
    print(f"connected; server answers video in {codec}; ICE {ice_pair}", flush=True)
    script = Script(args, cap, mic)
    await asyncio.sleep(args.lead_in)
    for step in args.step:
        await script.run(step)
    await asyncio.sleep(args.hold)
    stats = await pc.getStats()
    await pc.close()
    await asyncio.gather(*tasks, return_exceptions=True)

    os.makedirs(args.out, exist_ok=True)
    metrics = analyse(cap, script, codec)
    metrics["ice_pair"] = ice_pair
    metrics["webrtc_stats"] = {k: {a: v for a, v in vars(s).items() if isinstance(v, (int, float, str))}
                               for k, s in stats.items() if "inbound" in k}
    shown_args = {k: v for k, v in vars(args).items() if k != "turn_password"}
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump({"t_conn": cap.t_conn, "args": shown_args, "events": cap.events, "says": script.says}, f, indent=1)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    write_wav(os.path.join(args.out, "rx_audio.wav"), cap)
    if cap.video and not args.no_mp4:
        write_mp4(os.path.join(args.out, "session.mp4"), cap)
        save_frames(args.out, cap, metrics["replies"], args.save_frames)
    print_summary(metrics, cap.t_conn)
    return 1 if script.failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://kka-gateway:8282", help="https://... when the gateway has GATEWAY_TLS=1")
    parser.add_argument("--insecure", action="store_true", help="accept the gateway's self-signed certificate")
    parser.add_argument("--turn-url", action="append", help="TURN server, e.g. turn:10.0.0.5:3478?transport=udp; aiortc uses the first")
    parser.add_argument("--turn-user", default="kka")
    parser.add_argument("--turn-password", default=os.environ.get("KKA_TURN_PASSWORD"), help="default: $KKA_TURN_PASSWORD")
    parser.add_argument("--relay-only", action="store_true", help="send no media except through the TURN relay (needs --turn-url)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--step", action="append", required=True, help="see above; repeat for several steps")
    parser.add_argument("--level-db", type=float, default=-23.0, help="RMS level of the voiced part of every clip")
    parser.add_argument("--lead-in", type=float, default=2.0, help="seconds of silence before the first step")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to keep recording after the last step")
    parser.add_argument("--reply-timeout", type=float, default=60.0)
    parser.add_argument("--offer-codec", choices=("default", "h264", "vp8"), default="default",
                        help="video codec the client lists first in its offer (the gateway re-orders to H.264 itself)")
    parser.add_argument("--save-frames", type=int, default=6)
    parser.add_argument("--no-mp4", action="store_true")
    options = parser.parse_args()
    if options.relay_only and not options.turn_url:
        parser.error("--relay-only needs --turn-url")
    sys.exit(asyncio.run(main(options)))
