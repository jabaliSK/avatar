"""rtc_session: a reconnecting client replaces its stale session, other clients are refused and logged, finished connections are
released (fake FastRTC stream behind the real FastAPI routing; no GPU, no network)."""
import asyncio
import os
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import rtc_session  # noqa: E402


class Body(BaseModel):
    sdp: str | None = None
    candidate: dict | None = None
    type: str
    webrtc_id: str


class FakePc:
    """Closes like aiortc: the state change is emitted first (FastRTC's handler then cleans up in a task), then the listeners go."""

    def __init__(self, stream, wid, cleans_up=True):
        self.stream, self.wid, self.cleans_up = stream, wid, cleans_up
        self.connectionState = "connected"
        self.listeners = []
        self.closed = 0

    def on(self, event, f):
        assert event == "connectionstatechange"
        self.listeners.append(f)
        return f

    async def close(self):
        if self.connectionState == "closed":
            return
        self.closed += 1
        self.connectionState = "closed"
        for f in self.listeners:
            f()
        self.listeners.clear()
        if self.cleans_up:
            asyncio.get_running_loop().call_soon(lambda: [c.stop() for c in self.stream.clean_up(self.wid)])


class FakeStream:
    """Just what rtc_session touches: `pcs`, `clean_up()` and the `offer` route with FastRTC's concurrency check."""

    cleans_up = True

    def __init__(self):
        self.pcs = {}
        self.stopped = []

    def clean_up(self, wid):
        if self.pcs.pop(wid, None) is None:
            return []
        return [SimpleNamespace(stop=lambda: self.stopped.append(wid))]

    async def offer(self, body: Body):
        if body.type != "offer":
            return {"status": "success"}
        if len(self.pcs) >= 1:
            return JSONResponse({"status": "failed", "meta": {"error": "concurrency_limit_reached", "limit": 1}})
        self.pcs[body.webrtc_id] = FakePc(self, body.webrtc_id, self.cleans_up)
        return {"sdp": "answer", "type": "answer"}


def rig(takeover=True, cleans_up=True):
    class Stream(FakeStream):
        pass

    Stream.cleans_up = cleans_up
    rtc_session.install_reconnect(Stream, Body, takeover)
    stream, app = Stream(), FastAPI()
    app.post("/webrtc/offer")(stream.offer)

    def post(client, wid, type="offer"):
        async def run():
            transport = httpx.ASGITransport(app=app, client=(client, 5000))
            async with httpx.AsyncClient(transport=transport, base_url="http://gw") as http:
                return (await http.post("/webrtc/offer", json={"type": type, "webrtc_id": wid, "sdp": "x"})).json()
        return asyncio.run(run())

    return stream, post


def test_same_client_replaces_its_live_session():
    stream, post = rig()
    assert post("10.0.0.5", "a")["sdp"] == "answer"
    old = stream.pcs["a"]
    assert post("10.0.0.5", "b")["sdp"] == "answer"
    assert list(stream.pcs) == ["b"] and old.closed == 1 and stream.stopped == ["a"]


def test_another_client_is_refused_and_the_refusal_is_logged():
    stream, post = rig()
    post("10.0.0.5", "a")
    lines = []
    sink = logger.add(lines.append, format="{message}")
    try:
        reply = post("10.0.0.9", "b")
    finally:
        logger.remove(sink)
    assert reply["meta"]["error"] == "concurrency_limit_reached"
    assert list(stream.pcs) == ["a"] and stream.pcs["a"].closed == 0
    assert any("10.0.0.9" in m and "10.0.0.5" in m for m in lines)


def test_ice_candidates_never_replace_a_session():
    stream, post = rig()
    post("10.0.0.5", "a")
    assert post("10.0.0.5", "a", type="ice-candidate") == {"status": "success"}
    assert post("10.0.0.5", "other", type="ice-candidate") == {"status": "success"}
    assert list(stream.pcs) == ["a"] and stream.pcs["a"].closed == 0


def test_a_session_whose_close_handler_never_runs_is_cleaned_up_by_hand(monkeypatch):
    monkeypatch.setattr(rtc_session, "CLEANUP_WAIT_S", 0.2)
    stream, post = rig(cleans_up=False)
    post("10.0.0.5", "a")
    assert post("10.0.0.5", "b")["sdp"] == "answer"
    assert list(stream.pcs) == ["b"] and stream.stopped == ["a"]


def test_takeover_can_be_switched_off():
    stream, post = rig(takeover=False)
    post("10.0.0.5", "a")
    assert post("10.0.0.5", "b")["meta"]["error"] == "concurrency_limit_reached"
    assert list(stream.pcs) == ["a"]


def test_time_limit_task_ends_with_the_connection():
    class Mixin:
        pass

    rtc_session.install_time_limit_release(Mixin)

    async def run():
        pc = FakePc(FakeStream(), "a")
        task = asyncio.create_task(Mixin.wait_for_time_limit(pc, 3600))
        await asyncio.sleep(0.01)
        assert not task.done()
        await pc.close()
        await asyncio.wait_for(task, 1)
        assert pc.closed == 1

        already_closed = FakePc(FakeStream(), "b")
        await already_closed.close()
        await asyncio.wait_for(Mixin.wait_for_time_limit(already_closed, 3600), 1)

    asyncio.run(run())


def test_connection_that_outlives_the_time_limit_is_closed():
    class Mixin:
        pass

    rtc_session.install_time_limit_release(Mixin)

    async def run():
        pc = FakePc(FakeStream(), "a")
        await Mixin.wait_for_time_limit(pc, 0.05)
        assert pc.closed == 1

    asyncio.run(run())


def idle_trim_rig(delay_s=0.05):
    calls = []

    class Stream:
        def __init__(self):
            self.pcs = {}

        def clean_up(self, wid):
            self.pcs.pop(wid, None)
            return []

    rtc_session.install_idle_trim(Stream, delay_s=delay_s, trim=calls.append)
    return Stream(), calls


def test_idle_trim_runs_once_after_the_last_session_ended():
    stream, calls = idle_trim_rig()

    async def run():
        stream.pcs = {"a": object(), "b": object()}
        stream.clean_up("a")
        await asyncio.sleep(0.02)
        stream.clean_up("b")  # the second teardown restarts the wait
        await asyncio.sleep(0.2)

    asyncio.run(run())
    assert calls == [0]


def test_idle_trim_is_skipped_while_a_connection_is_live():
    stream, calls = idle_trim_rig()

    async def run():
        stream.pcs = {"a": object(), "b": object()}
        stream.clean_up("a")
        await asyncio.sleep(0.2)

    asyncio.run(run())
    assert calls == []
