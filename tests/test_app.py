import asyncio
import io
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from avatar.app import create_app
from avatar.config import Settings
from avatar.providers import PipelineError, Providers


class FakeProviders:
    def __init__(self):
        self.fail_render = False
        self.ready = True
        self.histories = []
        self.transcriptions = 0

    async def warmup(self):
        pass

    def status(self):
        return {
            key: {"configured": self.ready, "mode": "local"}
            for key in ["llm", "stt", "tts", "renderer"]
        }

    async def transcribe(self, audio):
        assert audio.is_file()
        self.transcriptions += 1
        return "Hello from my microphone"

    async def chat(self, history):
        self.histories.append(history)
        return "Hello! How can I help?"

    async def speak(self, text, target):
        target.write_bytes(b"test-wave")

    async def render(self, portrait, audio, output):
        assert portrait.is_file() and audio.is_file()
        output.write_bytes(b"test-video")
        if self.fail_render:
            raise PipelineError("Renderer failed")


@pytest.fixture
def workspace(tmp_path):
    settings = replace(Settings.from_env(), data=tmp_path)
    providers = FakeProviders()
    with TestClient(create_app(settings, providers)) as client:
        yield client, providers, tmp_path


def portrait_bytes(size=(513, 769)):
    output = io.BytesIO()
    Image.new("RGB", size, "#927caa").save(output, "PNG")
    return output.getvalue()


def new_avatar(client, size=(513, 769)):
    response = client.post(
        "/api/avatars", data={"name": "Alex", "consent": "true"},
        files={"image": ("portrait.png", portrait_bytes(size), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def new_session(client, avatar):
    response = client.post("/api/sessions", json={"avatar_id": avatar["id"]})
    assert response.status_code == 201
    return response.json()["id"]


@pytest.mark.parametrize("size,expected", [
    ((513, 769), (512, 768)),
    ((2560, 1440), (1920, 1080)),
])
def test_portrait_has_persistent_browser_url_and_even_dimensions(workspace, size, expected):
    client, _, directory = workspace
    avatar = new_avatar(client, size)
    assert client.get(avatar["url"]).status_code == 200
    assert client.get(avatar["image"]).headers["content-type"] == "image/png"
    with Image.open(directory / "media" / f"{avatar['id']}.png") as image:
        assert image.size == expected
    assert client.get("/api/avatars").json()[0]["name"] == "Alex"
    with TestClient(create_app(replace(Settings.from_env(), data=directory), FakeProviders())) as reopened:
        assert reopened.get("/api/avatars").json()[0]["id"] == avatar["id"]


@pytest.mark.parametrize("consent,content,status", [
    ("false", portrait_bytes(), 422),
    ("true", b"not an image", 422),
    ("true", portrait_bytes((32, 32)), 422),
    ("true", b"x" * (15 * 1024 * 1024 + 1), 413),
])
def test_invalid_portrait_is_not_saved(workspace, consent, content, status):
    client, _, directory = workspace
    response = client.post(
        "/api/avatars", data={"name": "Alex", "consent": consent},
        files={"image": ("portrait.png", content, "image/png")},
    )
    assert response.status_code == status
    assert not list((directory / "media").iterdir())
    assert client.get("/api/avatars").json() == []


def test_text_and_voice_turns_use_history_and_cleanup_intermediate_media(workspace):
    client, providers, directory = workspace
    session = new_session(client, new_avatar(client))
    first = client.post(f"/api/sessions/{session}/turn", data={"text": "Hi"})
    assert first.status_code == 200, first.text
    assert client.get(first.json()["video"]).status_code == 200
    second = client.post(
        f"/api/sessions/{session}/turn",
        files={"audio": ("voice.webm", b"audio-content", "audio/webm;codecs=opus")},
    )
    assert second.status_code == 200, second.text
    assert second.json()["user_text"] == "Hello from my microphone"
    assert providers.transcriptions == 1
    assert providers.histories[1] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Hello from my microphone"},
    ]
    assert len(client.get(f"/api/sessions/{session}").json()["turns"]) == 2
    assert sorted(path.suffix for path in (directory / "media").iterdir()) == [".mp4", ".mp4", ".png", ".png"]


def test_failed_render_does_not_commit_history_or_leave_partial_media(workspace):
    client, providers, directory = workspace
    session = new_session(client, new_avatar(client))
    providers.fail_render = True
    response = client.post(f"/api/sessions/{session}/turn", data={"text": "Hi"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Renderer failed"
    assert client.get(f"/api/sessions/{session}").json()["turns"] == []
    assert len(list((directory / "media").iterdir())) == 2
    providers.fail_render = False
    assert client.post(f"/api/sessions/{session}/turn", data={"text": "Try again"}).status_code == 200
    assert providers.histories[-1] == [{"role": "user", "content": "Try again"}]


def test_display_photo_preserves_full_resolution_detail_and_alpha(workspace):
    client, _, directory = workspace
    source = Image.new("RGBA", (1081, 2201), (64, 80, 120, 255))
    for point, color in zip(
        [(0, 0), (1080, 0), (0, 2200), (1080, 2200)],
        [(240, 32, 12, 255), (15, 128, 48, 255), (8, 24, 240, 255), (40, 60, 80, 128)],
    ):
        source.putpixel(point, color)
    original = io.BytesIO()
    source.save(original, format="PNG")
    response = client.post(
        "/api/avatars", data={"name": "Full length", "consent": "true"},
        files={"image": ("portrait.png", original.getvalue(), "image/png")},
    )
    assert response.status_code == 201, response.text
    avatar = client.get("/api/avatars").json()[0]
    with Image.open(io.BytesIO(client.get(avatar["image"]).content)) as displayed:
        assert (displayed.mode, displayed.size) == (source.mode, source.size)
        assert displayed.tobytes() == source.tobytes()
    with Image.open(directory / "media" / f"{avatar['id']}.png") as render:
        assert render.height == 1920
    (directory / "media" / f"{avatar['id']}.display.png").unlink()
    assert client.get("/api/avatars").json()[0]["image"] == f"/media/{avatar['id']}.png"


def test_missing_models_and_invalid_input_are_explicit(workspace):
    client, providers, _ = workspace
    session = new_session(client, new_avatar(client))
    assert client.post(f"/api/sessions/{session}/turn").status_code == 422
    both = client.post(
        f"/api/sessions/{session}/turn", data={"text": "Hi"},
        files={"audio": ("a.wav", b"audio", "audio/wav")},
    )
    assert both.status_code == 422
    providers.ready = False
    response = client.post(f"/api/sessions/{session}/turn", data={"text": "Hi"})
    assert response.status_code == 503
    assert "Models not installed" in response.json()["detail"]


def test_media_does_not_expose_database_or_arbitrary_paths(workspace):
    client, _, _ = workspace
    assert client.get("/media/avatar.sqlite3").status_code == 404
    assert client.get("/avatar/nonexistent").status_code == 404
    assert client.get("/api/sessions/nonexistent").status_code == 404


def test_overlapping_inference_is_rejected_without_committing_a_second_turn(tmp_path):
    import httpx

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        providers = FakeProviders()

        async def blocked_chat(history):
            started.set()
            await release.wait()
            return "Hello"

        providers.chat = blocked_chat
        app = create_app(replace(Settings.from_env(), data=tmp_path), providers)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/avatars", data={"name": "Alex", "consent": "true"},
                files={"image": ("a.png", portrait_bytes(), "image/png")},
            )
            response = await client.post("/api/sessions", json={"avatar_id": response.json()["id"]})
            url = f"/api/sessions/{response.json()['id']}"
            first = asyncio.create_task(client.post(f"{url}/turn", data={"text": "Hi"}))
            await asyncio.wait_for(started.wait(), 2)
            try:
                second = await client.post(f"{url}/turn", data={"text": "Overlap"})
                assert second.status_code == 409
            finally:
                release.set()
            assert (await first).status_code == 200
            assert len((await client.get(url)).json()["turns"]) == 1

    asyncio.run(exercise())


def test_remote_providers_use_openai_compatible_contracts(tmp_path, monkeypatch):
    import httpx

    settings = replace(
        Settings.from_env(), data=tmp_path,
        llm_base="https://llm.example/v1", llm_key="test-key", llm_model="test-chat",
        stt_base="https://stt.example/v1", stt_key="", stt_model="test-stt",
        tts_base="https://tts.example/v1", tts_key="", tts_model="test-tts",
    )
    providers = Providers(settings)
    requests = []

    async def request(base, key, path, **kwargs):
        requests.append((base, key, path, kwargs))
        if path == "chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hi there."}}]})
        if path == "audio/transcriptions":
            return httpx.Response(200, json={"text": "Hello"})
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        return httpx.Response(200, content=buffer.getvalue())

    monkeypatch.setattr(providers, "request", request)
    source = tmp_path / "input.webm"
    import wave
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    async def exercise():
        assert await providers.transcribe(source) == "Hello"
        assert await providers.chat([{"role": "user", "content": "Hello"}]) == "Hi there."
        await providers.speak("Hi there.", tmp_path / "voice.wav")
    asyncio.run(exercise())
    chat = requests[1]
    assert chat[:3] == ("https://llm.example/v1", "test-key", "chat/completions")
    assert chat[3]["json"]["messages"][0]["role"] == "system"
    assert requests[0][3]["data"] == {"model": "test-stt"}
    assert requests[2][3]["json"]["response_format"] == "wav"


def test_recording_duration_and_decode_failures_are_explicit(tmp_path):
    import wave

    recording = tmp_path / "long.wav"
    with wave.open(str(recording), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * (16000 * 61))
    with pytest.raises(PipelineError, match="60 seconds"):
        Providers._validate_recording(recording)
    recording.write_bytes(b"not audio")
    with pytest.raises(PipelineError, match="could not be decoded"):
        Providers._validate_recording(recording)


def test_flashhead_interpreter_keeps_virtual_environment_symlink(tmp_path, monkeypatch):
    import sys

    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("FLASHHEAD_PYTHON", str(interpreter))
    assert Settings.from_env().flashhead_python == interpreter


@pytest.mark.parametrize("engine,script", [
    ("musetalk", "render.py"),
    ("flashhead", "render_flashhead.py"),
])
def test_selected_renderer_uses_its_own_environment_and_surfaces_failure(
    tmp_path, monkeypatch, engine, script
):
    import sys
    import avatar.providers as provider_module

    settings = replace(
        Settings.from_env(), renderer=engine, flashhead_python=tmp_path / "flashhead-python"
    )
    calls = []

    async def failing_process(command, timeout, cwd):
        calls.append((command, timeout, cwd))
        raise PipelineError("Inference failed")

    monkeypatch.setattr(provider_module, "run_process", failing_process)
    with pytest.raises(PipelineError, match="Inference failed"):
        asyncio.run(Providers(settings).render(
            tmp_path / "portrait.png", tmp_path / "speech.wav", tmp_path / "video.mp4"
        ))
    assert len(calls) == 1
    command, timeout, cwd = calls[0]
    assert command[0] == (
        str(settings.flashhead_python) if engine == "flashhead" else sys.executable
    )
    assert command[1].endswith(f"/scripts/{script}")
    assert cwd.name == ("FlashHead" if engine == "flashhead" else "MuseTalk")
    assert timeout == settings.render_timeout


@pytest.mark.parametrize("sample_count", [1, 28, 29, 56, 57])
def test_motion_audio_chunking_preserves_the_final_partial_chunk(sample_count):
    import numpy as np
    from scripts.render_flashhead import audio_chunks

    samples = np.arange(sample_count, dtype=np.float32) + 1
    chunks = list(audio_chunks(samples, 28))
    combined = np.concatenate(chunks)
    assert all(len(chunk) == 28 for chunk in chunks)
    np.testing.assert_array_equal(combined[:sample_count], samples)
    assert np.all(combined[sample_count:] == 0)


def test_invalid_renderer_configuration_is_not_silently_replaced(monkeypatch):
    monkeypatch.setenv("AVATAR_RENDERER", "unknown")
    with pytest.raises(ValueError, match="AVATAR_RENDERER"):
        Settings.from_env()


class FakeLiveWorker:
    def __init__(self, fail_start=False):
        self.ready = False
        self.error = None
        self.fail_start = fail_start
        self.fail_stream = False
        self.finished = False
        self.closed_streams = 0

    async def start(self):
        if self.fail_start:
            raise PipelineError("GPU startup failed")
        self.ready = True

    async def close(self):
        self.ready = False

    async def prepare(self, avatar, image, idle):
        assert image.is_file()
        idle.write_bytes(b"idle-video")

    async def stream(self, audio, output):
        assert audio.is_file()
        self.finished = False
        output.write_bytes(b"partial-video")
        try:
            yield {"event": "segment", "index": 0, "frames": ["jpeg"], "audio": "pcm", "rate": 16000, "fps": 25}
            if self.fail_stream:
                raise PipelineError("Live generation failed")
            self.finished = True
            output.write_bytes(b"complete-video")
            yield {"event": "rendered", "render_seconds": 0.1, "audio_seconds": 1}
        finally:
            self.closed_streams += 1


def receive_event(socket, name):
    for _ in range(10):
        event = socket.receive_json()
        if event["event"] == name:
            return event
        assert event["event"] != "error", event
    pytest.fail(f"Missing event: {name}")


@pytest.mark.parametrize("renderer,custom_idle", [("flashhead", False), ("flashhead", True), ("livetalk", True)])
def test_live_media_arrives_before_completion_and_reuses_history(tmp_path, renderer, custom_idle):
    worker, providers = FakeLiveWorker(), FakeProviders()
    app = create_app(replace(Settings.from_env(), data=tmp_path, renderer=renderer), providers, worker)
    with TestClient(app) as client:
        avatar = new_avatar(client)
        if custom_idle:
            (tmp_path / "media" / f"{avatar['id']}.lite-head-idle.mp4").write_bytes(b"custom-idle")
        session = new_session(client, avatar)
        with client.websocket_connect(f"/api/sessions/{session}/live") as socket:
            prepared = receive_event(socket, "prepared")
            expected_idle = b"custom-idle" if custom_idle and renderer == "flashhead" else b"idle-video"
            assert client.get(prepared["idle"]).content == expected_idle
            for text in ["Hi", "Tell me more"]:
                socket.send_json({"type": "message", "text": text})
                segment = receive_event(socket, "segment")
                assert worker.finished is False
                assert len(client.get(f"/api/sessions/{session}").json()["turns"]) == len(providers.histories) - 1
                socket.send_json({"type": "ack", "index": segment["index"]})
                complete = receive_event(socket, "complete")
                assert client.get(complete["turn"]["video"]).content == b"complete-video"
            assert len(providers.histories[-1]) == 3
            assert not list((tmp_path / "media").glob("*.wav"))
        assert client.get("/api/status").json()["live"]["ready"]
    assert not worker.ready


@pytest.mark.parametrize("cancel", [True, False])
def test_live_cancel_or_failure_cleans_media_and_allows_retry(tmp_path, cancel):
    worker, providers = FakeLiveWorker(), FakeProviders()
    app = create_app(replace(Settings.from_env(), data=tmp_path), providers, worker)
    with TestClient(app) as client:
        session = new_session(client, new_avatar(client))
        with client.websocket_connect(f"/api/sessions/{session}/live") as socket:
            receive_event(socket, "prepared")
            worker.fail_stream = not cancel
            socket.send_json({"type": "message", "text": "Hi"})
            receive_event(socket, "segment")
            socket.send_json({"type": "cancel"} if cancel else {"type": "ack", "index": 0})
            event = socket.receive_json()
            assert event["event"] == ("cancelled" if cancel else "error")
            assert client.get(f"/api/sessions/{session}").json()["turns"] == []
            assert worker.closed_streams == 1
            assert not list((tmp_path / "media").glob("*.wav"))
            assert not [
                p for p in (tmp_path / "media").glob("*.mp4")
                if not p.name.endswith((".lite-listening-idle.mp4", ".fullbody-idle.mp4"))
            ]
            worker.fail_stream = False
            socket.send_json({"type": "message", "text": "Retry"})
            receive_event(socket, "segment")
            socket.send_json({"type": "ack", "index": 0})
            receive_event(socket, "complete")
            assert providers.histories[-1] == [{"role": "user", "content": "Retry"}]


def test_live_startup_failure_is_not_reported_as_ready(tmp_path):
    worker = FakeLiveWorker(fail_start=True)
    with TestClient(create_app(
        replace(Settings.from_env(), data=tmp_path), FakeProviders(), worker
    )) as client:
        session = new_session(client, new_avatar(client))
        with client.websocket_connect(f"/api/sessions/{session}/live") as socket:
            assert socket.receive_json()["event"] == "stage"
            assert socket.receive_json() == {"event": "error", "message": "GPU startup failed"}
        status = client.get("/api/status").json()
        assert status["mode"] == "live"
        assert status["live"] == {"enabled": True, "ready": False, "error": "GPU startup failed"}
        assert client.post(f"/api/sessions/{session}/turn", data={"text": "Hi"}).status_code == 503


def test_worker_drains_cancelled_job_before_accepting_next_command(tmp_path):
    import json
    from types import SimpleNamespace
    from avatar.live import LiveWorker

    async def exercise():
        reader = asyncio.StreamReader()
        commands = []

        class Input:
            def write(self, data):
                commands.append(json.loads(data))
                if commands[-1]["command"] == "prepare":
                    reader.feed_data(b'{"event":"prepared","avatar":"next"}\n{"event":"done"}\n')

            async def drain(self):
                pass

        worker = LiveWorker(replace(Settings.from_env(), data=tmp_path))
        worker.process = SimpleNamespace(stdin=Input(), stdout=reader, returncode=None)
        worker._ready = True
        first = worker.command({"command": "stream"})
        reader.feed_data(b'{"event":"segment","index":0}\n')
        assert (await first.__anext__())["index"] == 0
        closing = asyncio.create_task(first.aclose())
        second = worker.command({"command": "prepare"})
        next_event = asyncio.create_task(second.__anext__())
        await asyncio.sleep(0)
        assert [command["command"] for command in commands] == ["stream", "cancel"]
        reader.feed_data(b'{"event":"segment","index":1}\n{"event":"done"}\n')
        await closing
        assert await next_event == {"event": "prepared", "avatar": "next"}
        with pytest.raises(StopAsyncIteration):
            await second.__anext__()

    asyncio.run(exercise())


def test_cancelled_local_inference_finishes_before_cleanup(tmp_path):
    import threading
    from avatar.providers import run_in_thread

    async def exercise():
        entered, release = threading.Event(), threading.Event()
        output = tmp_path / "speech.wav"

        def inference():
            entered.set()
            assert release.wait(5)
            output.write_bytes(b"wave")

        async def job():
            try:
                await run_in_thread(inference)
            finally:
                output.unlink(missing_ok=True)

        task = asyncio.create_task(job())
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not output.exists()

    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(480, 1600), (1600, 480), (512, 512)])
def test_fullbody_preparation_keeps_all_image_edges_without_a_face_crop(tmp_path, size):
    import threading
    from types import SimpleNamespace
    from PIL import ImageDraw
    from scripts.render_livetalk import Engine

    reference = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(reference)
    colors = ["red", "green", "blue", "yellow"]
    corners = [(0, 0), (size[0] - 64, 0), (0, size[1] - 64), (size[0] - 64, size[1] - 64)]
    for (x, y), color in zip(corners, colors):
        draw.rectangle((x, y, x + 63, y + 63), fill=color)
    source, idle = tmp_path / "source.png", tmp_path / "idle.mp4"
    reference.save(source)
    idle.write_bytes(b"already-cached-idle")
    engine = Engine.__new__(Engine)
    engine.avatar, engine.warmed = None, True
    engine.cancelled = threading.Event()
    engine.image = tmp_path / "prepared.png"
    engine.pipeline = SimpleNamespace(args=SimpleNamespace(), kv_cache1=None)
    engine.prepare("avatar", source, idle, lambda event: None)
    assert engine.original_size == size
    with Image.open(engine.image) as prepared:
        content = prepared.crop(engine.crop)
        assert content.getpixel((0, 0)) == reference.getpixel((0, 0))
        assert content.getpixel((content.width - 1, 0)) == reference.getpixel((size[0] - 1, 0))
        assert content.getpixel((0, content.height - 1)) == reference.getpixel((0, size[1] - 1))
        assert content.getpixel((content.width - 1, content.height - 1)) == reference.getpixel((size[0] - 1, size[1] - 1))
        assert abs(content.width / content.height - size[0] / size[1]) <= 1 / content.height
        assert prepared.width * prepared.height <= 384 * 384
