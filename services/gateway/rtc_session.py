"""Session housekeeping for the gateway's WebRTC endpoint (FastRTC, as wired up by OpenAvatarChat); launch.py applies it at start-up.

The gateway serves one conversation at a time (`concurrent_limit: 1`: the renderer, the VAD and the avatar choice are single-session).

* Reconnecting. A browser that leaves without a goodbye (a closed or reloaded tab, a laptop that went to sleep) keeps its session until
  aiortc's ICE consent checks give up, about 30 s later. Meanwhile every new offer is answered with `concurrency_limit_reached`, which
  the web page swallows: clicking "start" does nothing. An offer from the same client address as a live session now replaces that
  session; another machine still gets the single-session refusal, which is written to the gateway log.
* FastRTC's `wait_for_time_limit` task sleeps for `connection_ttl` (an hour) while holding its peer connection, so every finished
  session stayed in memory (senders, tracks, RTP history; about 5 MB) for that long. It now ends together with the connection.
* A finished session leaves reference cycles (its handler contexts) that the interpreter collects only after many new objects, and glibc
  keeps the heap pages it freed (the 2.7 MB video frames, the TTS tensors): after 8 sessions the process had grown from 2.9 to 7.2 GB,
  5.2 GB of it free but retained. Ten seconds after a session ends, and only if no other connection is live by then, the cycles are
  collected and the free pages are returned with malloc_trim (0.1 to 0.4 s, in a worker thread).
"""
import asyncio
import ctypes
import gc
import os
import time
from typing import Dict

from fastapi import Request
from loguru import logger

# How long a replaced session may take to be unregistered before it is cleaned up by hand.
CLEANUP_WAIT_S = 3.0


class SessionOwners:
    """Which client address opened which WebRTC session (FastRTC itself only knows the ids)."""

    def __init__(self):
        self.owner: Dict[str, str] = {}

    async def release(self, stream, client: str, new_id: str) -> list:
        """Close the live sessions that `client` opened earlier and wait until FastRTC has torn them down."""
        live = stream.pcs
        self.owner = {wid: host for wid, host in self.owner.items() if wid in live}
        stale = [wid for wid, host in self.owner.items() if host == client and wid != new_id]
        for wid in stale:
            pc = live.get(wid)
            if pc is None:
                continue
            logger.info(f"Offer from {client}: closing its earlier session {wid} (reload or reconnect)")
            try:
                await pc.close()
            except Exception as e:
                logger.warning(f"Closing session {wid} failed: {e!r}")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CLEANUP_WAIT_S
        for wid in stale:
            while wid in live and loop.time() < deadline:
                await asyncio.sleep(0.05)
            if wid in live:
                # FastRTC unregisters a session from the pc's state-change handler; do what that handler does.
                logger.warning(f"Session {wid} was still registered after its connection closed; cleaning it up")
                for connection in stream.clean_up(wid):
                    connection.stop()
        return stale


def install_reconnect(stream_cls, body_cls, takeover: bool = True) -> SessionOwners:
    """Wrap `Stream.offer` (the `/webrtc/offer` route) so that it sees the client address. Call before the stream is mounted."""
    original = stream_cls.offer
    owners = SessionOwners()

    async def offer(self, body: body_cls, request: Request):
        client = request.client.host if request.client else None
        is_offer = body.type == "offer"
        if takeover and client and is_offer:
            await owners.release(self, client, body.webrtc_id)
        response = await original(self, body)
        if is_offer:
            if client and body.webrtc_id in self.pcs:
                owners.owner[body.webrtc_id] = client
            elif b"concurrency_limit_reached" in getattr(response, "body", b""):
                holders = ", ".join(sorted({owners.owner.get(wid, "unknown") for wid in self.pcs}))
                logger.warning(f"Offer from {client} refused: the session of {holders} is still active (concurrent_limit=1)")
        return response

    stream_cls.offer = offer
    return owners


def install_time_limit_release(mixin_cls) -> None:
    """Make FastRTC's connection time limit end together with the connection instead of holding it for the whole limit."""

    async def wait_for_time_limit(pc, time_limit: float):
        gone = asyncio.Event()

        def check():
            if pc.connectionState in ("closed", "failed"):
                gone.set()

        pc.on("connectionstatechange", check)
        check()
        try:
            await asyncio.wait_for(gone.wait(), time_limit)
        except asyncio.TimeoutError:
            await pc.close()

    mixin_cls.wait_for_time_limit = staticmethod(wait_for_time_limit)


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def install_idle_trim(mixin_cls, delay_s: float = 10.0, trim=None) -> None:
    """Free what a session left behind `delay_s` after it ended, unless another connection is live by then: its reference cycles (the
    interpreter only collects those after a lot of new objects, and the sessions' handler contexts are among them) and the heap pages."""
    if trim is None:
        try:
            trim = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            logger.warning("malloc_trim is not available; the process keeps the pages that sessions free")
            return
    original = mixin_cls.clean_up
    pending = {}

    def reclaim():
        gc.collect()
        trim(0)

    def run(stream, loop):
        pending.clear()
        if stream.pcs:
            return

        async def reclaim_now():
            before, started = _rss_mb(), time.monotonic()
            await loop.run_in_executor(None, reclaim)
            logger.info(f"Idle: garbage collected and freed heap pages returned to the OS, RSS {before:.0f} -> {_rss_mb():.0f} MB "
                        f"({(time.monotonic() - started) * 1000:.0f} ms)")

        loop.create_task(reclaim_now())

    def clean_up(self, webrtc_id):
        result = original(self, webrtc_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return result
        if "timer" in pending:
            pending["timer"].cancel()
        pending["timer"] = loop.call_later(delay_s, run, self, loop)
        return result

    mixin_cls.clean_up = clean_up
