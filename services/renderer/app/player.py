"""Real-time avatar player: a 25 fps clock that turns "state + speech" into paced (frame, audio) outputs.

Every 40 ms tick emits exactly one video frame and, while a reply is playing, the 960 PCM samples (24 kHz) that belong to
it, so audio and video stay locked by construction no matter how the GPU behaves:

* not speaking: frames of the idle / listening / thinking body loop (a full cycle rendered through the live head, played round and
  round from the rest pose), no GPU work at all;
* speaking: the newest composited head frame that is due (idx <= tick), or a repeat of the previous one when the renderer
  is late. The renderer is told (gate) to skip motion frames that could no longer make their slot, so a GPU that cannot
  sustain 25 fps degrades to a lower number of distinct frames instead of drifting out of sync or accumulating delay;
* switches between sources (base state <-> speech) are cross-faded with an eased ramp. A reply ends with the body walking back
  to its rest pose (HeadEngine.set_tail) and the base loop restarts at its own rest pose, so the two ends of that fade show the
  same posture and the same face rendering.

Threads: the clock (this class), the SpeechPipeline motion and render threads. Public methods are thread-safe.
"""
import logging
import queue
import threading
import time

import cv2
import numpy as np
from scipy.signal import resample_poly

from .avatar_store import drive_kwargs

log = logging.getLogger("kka.player")

FPS = 25
TICK = 1.0 / FPS
AUDIO_RATE = 24000
TICK_SAMPLES = AUDIO_RATE // FPS  # 960 samples = 40 ms
TICK_BYTES = TICK_SAMPLES * 2
SAMPLES_16K_PER_TICK = 16000 // FPS  # what the head engine consumes per frame (640)
STATES = ("idle", "listening", "thinking")


class Output:
    __slots__ = ("seq", "rgb", "pcm", "state", "pts")

    def __init__(self, seq, rgb, pcm, state, pts):
        self.seq = seq
        self.rgb = rgb      # uint8 [H,W,3]; read-only for the consumer
        self.pcm = pcm      # s16le mono 24 kHz, TICK_BYTES while speaking, b"" otherwise
        self.state = state  # idle | listening | thinking | speaking
        self.pts = pts      # seconds since the player started


class _Reply:
    def __init__(self, rid=None):
        self.id = rid           # client-chosen tag, echoed in this reply's events so a late event can't close the next reply
        self.pcm = bytearray()  # everything received so far, s16le 24 kHz
        self.ended = False
        self.speech_frames = None  # frames carrying speech; set at speak_end, the rest of the reply is the walk back to rest
        self.tail = 0
        self.phase = "prebuffer"  # -> "play"
        self.k = 0                # next tick index of the reply (frame k plays with audio slice k)
        self.ready = {}           # idx -> composited frame
        self.last = None          # frame shown last (repeat source)
        self.pipe_done = False
        self.t_begin = time.perf_counter()
        self.t_first_audio = None
        self.t_last_audio = self.t_begin
        self.t_play = None
        self.repeats = 0
        self.late = 0
        self.stale = 0
        self.underruns = 0
        self.played = 0


class Player:
    def __init__(self, pipe, avatar, on_output, on_event=None, *, preroll=3, max_lead=10, crossfade=8, audio_timeout=10.0):
        self.pipe = pipe
        self.avatar = avatar
        self.on_output = on_output
        self.on_event = on_event or (lambda event: None)
        self.preroll, self.max_lead, self.crossfade, self.audio_timeout = preroll, max_lead, crossfade, audio_timeout
        self.state = "idle"
        self.reply = None
        self.replies = 0
        self._seq = 0
        self._stop = threading.Event()
        self._thread = None
        self._src = None
        self._last = None
        self._xf_prev = None
        self._xf_n = 0
        self._base_loop, self._base_pos = None, 0
        self._base_restart = False  # the next base frame is the first of its loop (the rest pose)
        self._t0 = 0.0
        pipe.gate = self._gate

    # ------------------------------------------------------------------ control (any thread)
    def start(self):
        self._thread = threading.Thread(target=self._run, name="clock", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        self.pipe.gate = None
        self.interrupt(quiet=True)

    def set_state(self, state):
        if state in STATES:
            self.state = state

    def speak_start(self, rid=None):
        """Begin a reply. Blocks for the engine/pipeline set-up (tens of ms)."""
        if self.reply is not None:
            self.interrupt(quiet=True)
        self.pipe.start(self.avatar.speak_loop, **drive_kwargs(self.avatar.settings))
        self.reply = _Reply(rid)

    def push_audio(self, pcm24):
        """Append s16le mono 24 kHz audio (bytes) to the current reply."""
        r = self.reply
        if r is None or r.ended or not pcm24:
            return
        pcm24 = bytes(pcm24[: len(pcm24) // 2 * 2])
        if r.t_first_audio is None:
            r.t_first_audio = time.perf_counter()
        r.t_last_audio = time.perf_counter()
        r.pcm += pcm24
        x = np.frombuffer(pcm24, np.int16).astype(np.float32) / 32768.0
        self.pipe.push(resample_poly(x, 2, 3).astype(np.float32))

    def speak_end(self):
        r = self.reply
        if r is None or r.ended:
            return
        r.speech_frames = -(-len(r.pcm) // TICK_BYTES)
        r.tail = self.pipe.set_tail(r.speech_frames)
        if r.tail:
            # The body is still walking back to its rest pose: keep the head rendering silence (and the client playing it) until
            # it is there, so a gesture finishes instead of being cut and cross-faded into the idle body.
            r.pcm += bytes(r.tail * TICK_BYTES)
            self.pipe.push(np.zeros(r.tail * SAMPLES_16K_PER_TICK, np.float32))
        r.ended = True
        self.pipe.finish()

    def interrupt(self, quiet=False):
        """Barge-in: stop the reply now and fall back to the listening loop. Blocks for at most one frame render."""
        r = self.reply
        if r is None:
            return False
        self.reply = None
        self.state = "listening"
        self._base_restart = True
        self.pipe.cancel()
        if not quiet:
            self.on_event({"type": "interrupted", "id": r.id, "frames_played": r.played})
        return True

    def stats(self):
        return {"state": "speaking" if self.reply is not None else self.state, "frames": self._seq, "replies": self.replies,
                "render_ms": None if self.pipe.render_ms is None else round(self.pipe.render_ms, 1)}

    # ------------------------------------------------------------------ renderer gate (render thread)
    def _gate(self, idx):
        r = self.reply
        if r is None:
            return 0
        if idx > r.k + self.max_lead:
            return 1  # far enough ahead: do not burn GPU on frames that are not needed yet (or never, after a barge-in)
        if r.phase == "play":
            # A frame needs about render_ms to come out; if its slot would have passed by then it is not worth rendering.
            behind = int((self.pipe.render_ms or 0.0) // (TICK * 1e3))
            if idx < r.k + behind:
                return -1
        return 0

    # ------------------------------------------------------------------ clock (own thread)
    def _run(self):
        self._t0 = t_next = time.perf_counter()
        while not self._stop.is_set():
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -4 * TICK:
                t_next = time.perf_counter()  # long stall (GC, host contention): resync instead of bursting frames
            pts = t_next - self._t0
            t_next += TICK
            try:
                out = self._tick(pts)
            except Exception:
                log.exception("player tick failed; dropping the current reply")
                self._abort("internal error")
                continue
            try:
                self.on_output(out)
            except Exception:
                log.exception("output consumer failed")

    def _tick(self, pts):
        r = self.reply
        frame, pcm = None, b""
        if r is not None:
            frame, pcm = self._speech_tick(r)
        if frame is not None:
            src, state = "speech", "speaking"
        else:
            loop = self.avatar.base[self.state]
            if loop is not self._base_loop or self._base_restart:
                self._base_loop, self._base_pos, self._base_restart = loop, 0, False
            frame = loop.at(self._base_pos)
            self._base_pos += 1
            src, state = id(loop), self.state
        if src != self._src:
            if self._last is not None and self.crossfade > 0:
                self._xf_prev, self._xf_n = self._last, 0
            self._src = src
        if self._xf_prev is not None:
            self._xf_n += 1
            alpha = self._xf_n / (self.crossfade + 1)
            if alpha >= 1.0:
                self._xf_prev = None
            else:
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # eased: no visible start or stop of the fade
                frame = cv2.addWeighted(self._xf_prev, 1.0 - alpha, frame, alpha, 0.0)
        self._last = frame
        self._seq += 1
        return Output(self._seq, frame, pcm, state, pts)

    def _speech_tick(self, r):
        """-> (frame or None while the reply is still pre-buffering, pcm for this tick)."""
        pipe, now = self.pipe, time.perf_counter()
        while True:
            try:
                f = pipe.frames.get_nowait()
            except queue.Empty:
                break
            if f is None:
                r.pipe_done = True
            else:
                r.ready[f.idx] = f.rgb
        if pipe.error is not None:
            log.error("render pipeline failed: %r", pipe.error)
            self._abort(f"render pipeline failed: {pipe.error!r}")
            return None, b""
        if not r.ended and now - r.t_last_audio > self.audio_timeout:
            self._abort("no audio received for %.0f s" % self.audio_timeout)
            return None, b""

        if r.phase == "prebuffer":
            if (0 in r.ready and len(r.ready) >= self.preroll) or (r.pipe_done and r.ready):
                r.phase, r.t_play = "play", now
                ref = r.t_first_audio or r.t_begin
                self.on_event({"type": "speak_started", "id": r.id, "first_audio_to_first_frame_ms": round((now - ref) * 1e3)})
            elif r.pipe_done and r.ended:
                self._finish(r)
                return None, b""
            else:
                return None, b""

        k = r.k
        chunk = bytes(r.pcm[k * TICK_BYTES:(k + 1) * TICK_BYTES])
        if len(chunk) < TICK_BYTES and not r.ended:
            r.underruns += 1  # the TTS has not delivered this slice yet: hold the position, play silence, retry next tick
            pcm, advance = bytes(TICK_BYTES), False
        else:
            pcm, advance = chunk.ljust(TICK_BYTES, b"\0"), True

        due = [i for i in r.ready if i <= k]
        if due:
            i = max(due)
            r.stale += len(due) - 1
            r.late += i < k
            r.last = r.ready[i]
            for j in due:
                del r.ready[j]
        else:
            r.repeats += 1
        frame = r.last
        if advance:
            r.k += 1
            r.played += 1
            if r.ended and r.k * TICK_BYTES >= len(r.pcm):
                self._finish(r)
        return frame, pcm

    def _finish(self, r):
        now = time.perf_counter()
        self.reply = None
        self.state = "listening"
        self._base_restart = True
        self.replies += 1
        self.on_event({
            "type": "speak_done", "id": r.id, "frames": r.played,
            "audio_s": round((r.speech_frames if r.speech_frames is not None else len(r.pcm) / TICK_BYTES) * TICK, 2),
            "tail_s": round(r.tail * TICK, 2),
            "repeated": r.repeats, "late": r.late, "dropped_by_renderer": self.pipe.dropped, "stale": r.stale,
            "audio_underruns": r.underruns, "render_ms": None if self.pipe.render_ms is None else round(self.pipe.render_ms, 1),
            "wall_s": round(now - (r.t_play or r.t_begin), 2),
        })

    def _abort(self, reason):
        r = self.reply
        self.reply = None
        self.state = "listening"
        self._base_restart = True
        self.pipe.cancel()  # inline: a deferred cancel could hit the next reply's pipeline
        self.on_event({"type": "error", "id": r.id if r else None, "reason": reason, "frames_played": r.played if r else 0})
