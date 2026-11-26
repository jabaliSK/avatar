"""Two-stage speech pipeline: audio -> driving motion (HuBERT + LMDM) -> composited frames (warp + decode + blend + paste).

HuBERT/LMDM are many small kernels that leave the GPU mostly idle, while warp+decode saturate it. Running them on
separate threads and CUDA streams lets the motion stage hide behind rendering, so the frame rate is set by the render
stage alone.

    pipe = SpeechPipeline(engine, reader)
    pipe.start(loop)                      # loop: app.avatar.Loop (template frames + GPU paste assets)
    pipe.push(pcm16k_float32)             # any number of times, from any thread
    pipe.finish()                         # flush the tail
    while (f := pipe.frames.get()) is not None:
        ...                               # Frame(idx, fidx, rgb, t_ready)
    pipe.cancel()                         # barge-in: drop everything still queued and stop both threads

gate(idx) -> int is an optional real-time hook called by the render stage before it starts frame `idx`:
0 = render it, -1 = skip it (it could no longer be shown on time), +1 = too early, wait. Skipping is safe because a frame's
render depends only on its own motion and the template, not on earlier frames.
"""
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from .compositor import PatchReader, blend_roi, paste
from .head_engine import HUBERT_STEP

_END = object()


@dataclass
class Frame:
    idx: int        # driving-frame index within the reply (frame k plays at k * 40 ms)
    fidx: int       # template frame it was composited onto
    rgb: np.ndarray  # uint8 [H, W, 3], a fresh array owned by the consumer
    t_ready: float  # time.perf_counter() when the frame left the render stage
    render: np.ndarray = None  # uint8 [512, 512, 3] raw head render, only when keep_render=True (scoring/debug)


def _to_u8(render01):
    return (render01[0].clamp(0, 1).permute(1, 2, 0).float() * 255.0).round().to(torch.uint8).cpu().numpy()


class SpeechPipeline:
    def __init__(self, engine, reader=None, motion_queue=32, keep_render=False, gate=None):
        self.engine = engine
        self.gate = gate
        self.render_ms = None  # moving average of the render stage per frame (warp + decode + blend + readback + paste)
        self.dropped = 0
        self.error = None
        self.reader = reader or PatchReader()
        self.motion_queue = motion_queue
        self.keep_render = keep_render
        self._motion_stream = torch.cuda.Stream()
        self._render_stream = torch.cuda.Stream()
        self._threads = []
        self.frames = queue.Queue()

    def start(self, loop, **begin_kw):
        self.join()
        self.loop = loop
        self.dropped = 0
        self._cancelled = threading.Event()
        self._audio = queue.Queue()
        self._motion = queue.Queue(maxsize=self.motion_queue)
        self.frames = queue.Queue()
        self.error = None
        torch.cuda.synchronize()  # everything created on the default stream (loop assets) is complete before the worker streams use it
        with torch.cuda.stream(self._motion_stream), torch.inference_mode():
            self.engine.begin(loop.info, **begin_kw)
        self._motion_stream.synchronize()
        self._threads = [threading.Thread(target=self._guard, args=(self._motion_main,), name="motion", daemon=True),
                         threading.Thread(target=self._guard, args=(self._render_main,), name="render", daemon=True)]
        for t in self._threads:
            t.start()

    def push(self, pcm16k):
        pcm = np.asarray(pcm16k, np.float32)
        # feed_audio() returns only once it has consumed everything it was given, so a multi-second push would hold back
        # the first frames until the whole buffer is processed. Hand it one HuBERT step (5 frames) at a time instead.
        for i in range(0, len(pcm), HUBERT_STEP):
            self._audio.put((pcm[i:i + HUBERT_STEP], False))

    def finish(self):
        self._audio.put((None, True))

    def set_tail(self, speech_frames):
        """The speech fills output frames [0, speech_frames): returns how many silent frames to push before finish() so the
        body template can walk back to its rest pose (see HeadEngine.set_tail)."""
        return self.engine.set_tail(speech_frames)

    def cancel(self):
        self._cancelled.set()
        self._audio.put(_END)
        try:  # unblock a motion thread waiting on a full queue
            while True:
                self._motion.get_nowait()
        except queue.Empty:
            pass
        try:  # and a render thread waiting on an empty one
            self._motion.put_nowait(_END)
        except queue.Full:
            pass
        self.join()

    def join(self, timeout=None):
        for t in self._threads:
            t.join(timeout)

    # ------------------------------------------------------------------ threads
    def _guard(self, fn):
        try:
            with torch.inference_mode():
                fn()
        except BaseException as e:  # surface the failure to the consumer instead of dying silently
            self.error = e
            self._cancelled.set()
            try:  # wake a render stage that is waiting for motion that will never come
                self._motion.put_nowait(_END)
            except queue.Full:
                pass
            self.frames.put(None)

    def _motion_main(self):
        with torch.cuda.stream(self._motion_stream):
            while not self._cancelled.is_set():
                item = self._audio.get()
                if item is _END:
                    break
                pcm, final = item
                for out in self.engine.feed_audio(pcm, final=final):
                    while not self._cancelled.is_set():
                        try:
                            self._motion.put(out, timeout=0.05)
                            break
                        except queue.Full:
                            continue
                if final:
                    break
        while not self._cancelled.is_set():
            try:
                self._motion.put(_END, timeout=0.05)
                break
            except queue.Full:
                continue

    def _render_main(self):
        eng, loop = self.engine, self.loop
        with torch.cuda.stream(self._render_stream):
            while True:
                item = self._motion.get()
                if item is _END or self._cancelled.is_set():
                    break
                idx, x_d = item
                if self.gate is not None:
                    verdict = self.gate(idx)
                    while verdict > 0 and not self._cancelled.is_set():
                        time.sleep(0.004)
                        verdict = self.gate(idx)
                    if self._cancelled.is_set():
                        break
                    if verdict < 0:
                        self.dropped += 1
                        continue
                t0 = time.perf_counter()
                render, fidx = eng.render_gpu(idx, x_d)
                roi = loop.rois[fidx]
                rgb = paste(loop.frames[fidx], self.reader.read(blend_roi(render, roi)), roi)
                t1 = time.perf_counter()
                eng.timer.add("render+blend+d2h+paste", t1 - t0)
                ms = (t1 - t0) * 1e3
                self.render_ms = ms if self.render_ms is None else 0.8 * self.render_ms + 0.2 * ms
                kept = None
                if self.keep_render:
                    kept = _to_u8(render)
                    eng.timer.add("keep_render", time.perf_counter() - t1)
                self.frames.put(Frame(idx, fidx, rgb, t1, kept))
        self.frames.put(None)
