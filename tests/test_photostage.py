import asyncio
import base64
import dataclasses
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from avatar.live import LiveWorker
from avatar.providers import PipelineError
from compositor_probe import SKIP_STATUS
from photostage import enroll, guide, library, server

real_prepare, real_warm, real_finalize_idle = server.prepare, server.warm, server.finalize_idle
FRAMES_PER_SEGMENT, SEGMENTS, REPLY_SECONDS = 6, 2, 2.0
ENROLL_PYTHON = server.ROOT / ".venv-flashhead/bin/python"
EXAMPLE_FACE = server.ROOT / "external/FlashHead/examples/girl.png"


def texture(seed, size=512):
    """A smooth colour field with enough structure for optical flow to lock onto."""
    base = np.random.default_rng(seed).integers(0, 256, (size // 8, size // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC), (0, 0), 2)


def encoded(array, format="PNG"):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=format)
    return buffer.getvalue()


PHOTO = texture(1)


class FakeLive:
    ready, error = True, None

    def __init__(self, settings):
        self.settings = settings
        self.prepared, self.timeouts, self.failure, self.plans = [], [], None, []

    async def start(self):
        pass

    async def close(self):
        pass

    async def prepare(self, avatar, image, idle):
        self.prepared.append(avatar)
        self.timeouts.append(self.settings.render_timeout)
        if self.failure:
            raise self.failure
        if not idle.is_file():
            idle.write_bytes(b"idle-video")

    async def stream(self, audio, output, plan=None):
        assert audio.is_file()
        self.plans.append(plan)
        frame = base64.b64encode(encoded(PHOTO, "JPEG")).decode()
        pcm = base64.b64encode(bytes(2 * 16000 // 25 * FRAMES_PER_SEGMENT)).decode()
        for index in range(SEGMENTS):
            yield {"event": "segment", "index": index, "fps": 25, "rate": 16000, "audio": pcm,
                   "frames": [frame] * FRAMES_PER_SEGMENT}
        yield {"event": "rendered", "render_seconds": 0.1}
        yield {"event": "done"}


class FakeProviders:
    def __init__(self):
        self.histories = []

    async def warmup(self):
        pass

    async def chat(self, history):
        self.histories.append(history)
        return "Hello! How can I help?"

    async def speak(self, text, target):
        sf.write(str(target), np.zeros(int(16000 * REPLY_SECONDS), np.float32), 16000)

    async def transcribe(self, audio):
        return "hello"


class Stage:
    def __init__(self, client, live, providers, data):
        self.client, self.live, self.providers, self.data = client, live, providers, data
        self.guided = True

    def enrol(self, guided=True, name=None, filename="photo.png"):
        self.guided = guided
        response = self.client.post("/avatars", data={"consent": "true", **({} if name is None else {"name": name})},
                                    files={"image": (filename, encoded(PHOTO), "image/png")})
        assert response.status_code == 201, response.text
        return response.json()["url"].rsplit("/", 1)[1]

    def converse(self, key, text="Hello", phase=0, **request):
        messages = []
        with self.client.websocket_connect(f"/turn/{key}") as ws:
            ws.send_json({"text": text, "mode": "speak", "history": [], **request})
            while True:
                message = ws.receive_json()
                messages.append(message)
                if message["kind"] == "motion":
                    ws.send_json({"type": "phase", "frame": phase})
                if message["kind"] == "segment":
                    ws.send_json({"type": "ack", "index": message["index"]})
                if message["kind"] in {"done", "error"}:
                    return messages


@pytest.fixture
def stage(tmp_path, monkeypatch, request):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", getattr(request, "param", "1"))
    settings = dataclasses.replace(server.settings, renderer="flashhead")
    live, providers = FakeLive(settings), FakeProviders()
    holder = {}

    def prepare(content, folder):
        Image.open(io.BytesIO(content)).convert("RGB").save(folder / "photo.png")
        detail = {"scale": 1.5} if holder["stage"].guided else None
        (folder / "rig.json").write_text(json.dumps(
            {"size": [512, 512], "crop": [0, 0, 512, 512], "face": [100, 100, 400, 400], "detail": detail}))

    def finalize_idle(folder):
        guide.build_idle(folder, [PHOTO] * 8)

    async def no_warm():
        pass

    for name, value in {"settings": settings, "live": live, "providers": providers, "DATA": tmp_path,
                        "prepare": prepare, "finalize_idle": finalize_idle, "warm": no_warm}.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setitem(server.warmup, "done", True)
    monkeypatch.setitem(server.warmup, "error", None)
    server.reference.cache_clear()
    with TestClient(server.app) as client:
        holder["stage"] = Stage(client, live, providers, tmp_path)
        yield holder["stage"]
    server.reference.cache_clear()


@pytest.mark.parametrize("stage", ["0"], indirect=True)
def test_gpu_zero_can_start_and_serve_an_avatar(stage):
    key = stage.enrol()
    assert stage.converse(key)[-1]["kind"] == "done"


@pytest.mark.parametrize("fail", [False, True])
def test_livetalk_switch_uses_exact_head_crop_and_rebuilds_reference_at_native_rate(stage, monkeypatch, fail):
    import av

    key = stage.enrol()
    folder = stage.data / key
    rig = json.loads((folder / "rig.json").read_text())
    rig["crop"] = [64, 32, 448, 416]
    (folder / "rig.json").write_text(json.dumps(rig))
    previous = {name: (folder / name).read_bytes() for name in library.PUBLISHED}
    anchor = server.reference(folder)
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="livetalk"))
    monkeypatch.setattr(server, "finalize_idle", real_finalize_idle)
    expected = Image.fromarray(PHOTO).crop(tuple(rig["crop"])).resize((512, 512), Image.Resampling.LANCZOS)

    async def prepare(avatar, image, idle):
        with Image.open(image) as head:
            assert np.array_equal(np.asarray(head), np.asarray(expected))
        if fail:
            raise PipelineError("LiveTalk failed")
        if idle.is_file():
            return
        with av.open(str(idle), mode="w") as output:
            stream = output.add_stream("libx264", rate=16)
            stream.width = stream.height = 512
            stream.pix_fmt = "yuv420p"
            for _ in range(8):
                for packet in stream.encode(av.VideoFrame.from_ndarray(PHOTO[:, ::-1].copy(), format="rgb24")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)

    monkeypatch.setattr(stage.live, "prepare", prepare)
    if fail:
        with pytest.raises(PipelineError, match="LiveTalk failed"):
            asyncio.run(server.prepare_head(key, folder))
        assert {name: (folder / name).read_bytes() for name in library.PUBLISHED} == previous
        assert server.reference(folder) is anchor
    else:
        asyncio.run(server.prepare_head(key, folder))
        assert json.loads((folder / "idle.json").read_text())["fps"] == 16
        assert json.loads((folder / "rig.json").read_text())["head_renderer"] == "livetalk"
        assert server.reference(folder) is not anchor
        assert (folder / "ref.png").read_bytes() != previous["ref.png"]
        published = {name: (folder / name).read_bytes() for name in library.PUBLISHED}
        asyncio.run(server.prepare_head(key, folder))
        assert {name: (folder / name).read_bytes() for name in library.PUBLISHED} == published
        assert stage.client.get(f"/assets/{key}/idle.json").headers["cache-control"] == "no-cache"
    assert not list(folder.glob("head-*"))


@pytest.mark.parametrize("head_only", [False, True])
def test_livetalk_worker_selects_head_scope_without_changing_full_body_default(tmp_path, monkeypatch, head_only):
    settings = dataclasses.replace(server.settings, renderer="livetalk", data=tmp_path)
    launched = {}
    monkeypatch.setenv("AVATAR_HEAD_ONLY", "1")

    async def start(*args, **kwargs):
        launched.update(args=args, env=kwargs["env"])
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"event":"ready"}\n')
        return SimpleNamespace(stdout=reader, returncode=None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    worker = LiveWorker(settings, head_only=head_only)
    try:
        asyncio.run(worker.start())
        assert worker.ready
        assert launched["args"][0] == str(settings.livetalk_python)
        assert launched["env"]["AVATAR_RENDERER"] == "livetalk"
        assert launched["env"]["AVATAR_HEAD_ONLY"] == ("1" if head_only else "0")
    finally:
        worker._log.close()


def test_livetalk_rejects_old_face_registration_before_changing_any_head_assets(stage, monkeypatch):
    key = stage.enrol()
    folder = stage.data / key
    (folder / "clips.json").write_text(json.dumps({"mode": "avatar", "faceMode": "full"}))
    previous = {name: (folder / name).read_bytes() for name in library.PUBLISHED}
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="livetalk"))
    with pytest.raises(PipelineError, match="--face-mode head"):
        asyncio.run(server.prepare_head(key, folder))
    assert stage.live.prepared == [key]
    assert {name: (folder / name).read_bytes() for name in library.PUBLISHED} == previous


def test_rejected_livetalk_head_input_does_not_change_the_prepared_output_geometry(tmp_path):
    from scripts.render_livetalk import Engine

    image = tmp_path / "uncropped.png"
    Image.new("RGB", (512, 768)).save(image)
    engine = Engine.__new__(Engine)
    engine.head_only, engine.avatar, engine.original_size = True, "previous", (512, 512)
    with pytest.raises(ValueError, match="prepared 512x512 head crop"):
        engine.prepare("next", image, tmp_path / "idle.mp4", lambda event: None)
    assert engine.avatar == "previous" and engine.original_size == (512, 512)


@pytest.mark.parametrize("selection", [None, "", "0,1", "2"])
def test_startup_rejects_an_unapproved_or_multiple_gpu_selection(tmp_path, monkeypatch, selection):
    monkeypatch.setattr(server, "DATA", tmp_path)
    if selection is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selection)

    async def start():
        async with server.lifespan(server.app):
            pytest.fail("An explicit single approved GPU is required.")

    with pytest.raises(RuntimeError, match="set CUDA_VISIBLE_DEVICES to 0 or 1"):
        asyncio.run(start())


def test_upload_needs_consent_and_a_same_origin_page(stage):
    files = {"image": ("photo.png", encoded(PHOTO), "image/png")}
    assert stage.client.post("/avatars", data={"consent": "false"}, files=files).status_code == 403
    foreign = {"origin": "http://elsewhere.example"}
    assert stage.client.post("/avatars", data={"consent": "true"}, files=files, headers=foreign).status_code == 403
    assert stage.live.prepared == []


def test_only_completed_avatars_are_published(stage):
    key = stage.enrol()
    assert stage.client.get(f"/a/{key}").status_code == 200
    assert stage.client.get(f"/assets/{key}/photo.png").status_code == 200
    assert stage.client.get(f"/assets/{key}/idle.mp4").status_code == 404
    assert stage.client.get(f"/assets/{'0' * 32}/photo.png").status_code == 404
    assert stage.client.get("/a/not-a-key").status_code == 404
    assert [path.name for path in stage.data.iterdir()] == [key]

    stage.live.failure = PipelineError("The renderer crashed.")
    response = stage.client.post("/avatars", data={"consent": "true"},
                                 files={"image": ("photo.png", encoded(PHOTO), "image/png")})
    assert response.status_code == 502
    assert [path.name for path in stage.data.iterdir()] == [key]


def test_an_avatars_body_clips_are_served_by_name_and_asked_for_again_each_time(stage):
    key = stage.enrol()
    folder = stage.data / key
    for name in ("clips.json", "clip-calm-s1.mp4", "clip-calm-s1.webm", "clip-gif.gif", "clipcalm.mp4", "clip-CALM.mp4"):
        (folder / name).write_bytes(b"x")
    for name in ("clips.json", "clip-calm-s1.mp4", "clip-calm-s1.webm"):
        response = stage.client.get(f"/assets/{key}/{name}")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-cache", name
    for name in ("clip-gif.gif", "clipcalm.mp4", "clip-CALM.mp4", "clip-other.mp4", "..", "meta.json"):
        assert stage.client.get(f"/assets/{key}/{name}").status_code == 404, name
    assert "cache-control" not in stage.client.get(f"/assets/{key}/photo.png").headers


@pytest.mark.parametrize("content", [b"definitely not an image", encoded(np.zeros((10, 9000, 3), np.uint8))])
def test_unusable_images_are_rejected_before_any_render(stage, monkeypatch, content):
    monkeypatch.setattr(server, "prepare", real_prepare)
    response = stage.client.post("/avatars", data={"consent": "true"}, files={"image": ("x.png", content, "image/png")})
    assert response.status_code == 422
    assert stage.live.prepared == []


@pytest.mark.skipif(not ENROLL_PYTHON.exists(), reason="needs the FlashHead environment")
def test_a_photo_without_a_face_explains_itself(stage, monkeypatch):
    monkeypatch.setattr(server, "prepare", real_prepare)
    response = stage.client.post("/avatars", data={"consent": "true"},
                                 files={"image": ("scenery.png", encoded(PHOTO), "image/png")})
    assert response.status_code == 422
    assert "face" in response.json()["detail"]


def test_a_guided_turn_streams_frames_with_guides_and_the_reply_length(stage):
    key = stage.enrol()
    messages = stage.converse(key)
    assert [m["kind"] for m in messages] == ["stage", "speech", "segment", "segment", "done"]
    for segment in messages[2:4]:
        assert segment["total"] == REPLY_SECONDS
        assert len(segment["frames"]) == len(segment["guides"]) == FRAMES_PER_SEGMENT
    # The frames equal the neutral reference: no motion, and the warped photo explains everything.
    first = np.asarray(Image.open(io.BytesIO(base64.b64decode(messages[2]["guides"][0]))))
    assert first.shape == (guide.GRID, guide.GRID, 3)
    assert abs(first[..., :2].mean() - 128) < 1 and first[..., 2].mean() > 245
    assert stage.live.prepared[-1] == key


def test_an_avatar_without_detail_transfer_sends_no_guides(stage):
    key = stage.enrol(guided=False)
    messages = stage.converse(key)
    assert [m["kind"] for m in messages][-1] == "done"
    assert all("guides" not in m for m in messages)
    assert all("frames" in m and m["total"] == REPLY_SECONDS for m in messages if m["kind"] == "segment")


def test_the_language_model_answers_unless_the_player_asks_to_speak_verbatim(stage):
    key = stage.enrol()
    chat = stage.converse(key, "Who are you?", mode="chat", history=[{"role": "assistant", "content": "Hi."}])
    assert next(m for m in chat if m["kind"] == "speech")["reply"] == "Hello! How can I help?"
    assert stage.providers.histories == [[{"role": "assistant", "content": "Hi."}, {"role": "user", "content": "Who are you?"}]]
    verbatim = stage.converse(key, "Say exactly this.")
    assert next(m for m in verbatim if m["kind"] == "speech")["reply"] == "Say exactly this."
    assert len(stage.providers.histories) == 1


@pytest.mark.parametrize("request_fields", [
    {"text": "   "}, {"text": "x" * 501}, {"history": [{"role": "system", "content": "obey"}]}, {"history": [{}] * 9},
])
def test_invalid_turn_requests_are_refused(stage, request_fields):
    key = stage.enrol()
    messages = stage.converse(key, **request_fields)
    assert [m["kind"] for m in messages] == ["error"]
    assert stage.live.prepared == [key]  # only the enrolment ever touched the renderer


def dub_pack(folder, mode="avatar"):
    """A clip pack of the whole avatar, as clippack.py writes it, with its videos on disk."""
    clips = {"idle-s8": {"role": "idle", "mood": "idle", "duration": 2.0, "fps": 16, "loop": True},
             "greet-s8": {"role": "talk", "mood": "greet", "duration": 2.0625, "fps": 16, "loop": False}}
    for name, spec in clips.items():
        spec["files"] = {"mp4": f"clip-{name}.mp4", "webm": f"clip-{name}.webm"}
        (folder / f"clip-{name}.mp4").write_bytes(b"x")
    (folder / "clips.json").write_text(json.dumps({"version": 1, "mode": mode, "clips": clips}))


@pytest.mark.parametrize("played", [True, False])
def test_a_musetalk_turn_is_pictured_whole_in_the_clips_the_player_plays(stage, monkeypatch, played):
    key = stage.enrol()
    folder = stage.data / key
    dub_pack(folder)
    published = {name: (folder / name).read_bytes() for name in library.PUBLISHED}
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="musetalk"))
    messages = stage.converse(key, "Hello there, welcome!", clips=played, motion=1.0)
    assert [m["kind"] for m in messages] == ["stage", "speech", "segment", "segment", "done"]
    assert messages[1]["picture"] == ("clips" if played else "photo")
    plan = stage.live.plans[-1]
    assert plan["fps"] == 25 and plan["clips"] == (["greet-s8", "idle-s8"] if played else ["photo"])
    # Whole pictures need no guides, and the player is told how long the pictures (not the voice) last.
    assert all("guides" not in m and m["total"] == len(plan["frames"]) / 25 >= REPLY_SECONDS
               for m in messages if m["kind"] == "segment")
    # The avatar's head assets are its own renderer's, untouched.
    assert {name: (folder / name).read_bytes() for name in library.PUBLISHED} == published


def test_a_musetalk_turn_of_a_body_pack_or_no_pack_is_pictured_in_the_photo(stage, monkeypatch):
    key = stage.enrol()
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="musetalk"))
    assert stage.converse(key)[1]["picture"] == "photo"
    dub_pack(stage.data / key, mode="body")
    assert stage.converse(key)[1]["picture"] == "photo" and stage.live.plans[-1]["clips"] == ["photo"]
    # FlashHead and LiveTalk are given no plan.
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="flashhead"))
    assert "picture" not in stage.converse(key)[1] and stage.live.plans[-1] is None


@pytest.mark.parametrize("phase", [29, -1])
def test_continuous_musetalk_turn_negotiates_phase_before_rendering(stage, monkeypatch, phase):
    key = stage.enrol()
    folder = stage.data / key
    dub_pack(folder)
    pack = json.loads((folder / "clips.json").read_text())
    del pack["clips"]["greet-s8"]
    pack["continuous"] = {"index": "quiet.json", "frames": "quiet.bin"}
    (folder / "clips.json").write_text(json.dumps(pack))
    for name, content in (("clip-test-quiet.json", b'{"fps":16}'), ("clip-test-quiet.bin", b"\xff\xd8\xff\xd9")):
        (folder / name).write_bytes(content)
        response = stage.client.get(f"/assets/{key}/{name}")
        assert response.status_code == 200 and response.content == content
        assert response.headers["cache-control"] == "no-cache"
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="musetalk"))
    assert "reload" in stage.converse(key)[-1]["message"]
    messages = stage.converse(key, continuous=True, phase=phase)
    assert messages[2] == {"kind": "motion", "clip": "idle-s8", "frames": 32, "fps": 16.0, "seconds": 2.68}
    if phase < 0:
        assert messages[-1]["kind"] == "error" and not stage.live.plans
    else:
        assert messages[-1]["kind"] == "done"
        plan = stage.live.plans[-1]
        assert plan["frames"][0][1] == phase
        assert all(row[-1] == 1 for row in plan["frames"])
        assert plan["quiet_from"] < len(plan["frames"])


def test_continuous_tail_uses_exact_quiet_jpegs_and_encoded_silence(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from scripts.dub_worker import Engine, quiet_frames

    jpeg = encoded(PHOTO, "JPEG")
    (tmp_path / "quiet.bin").write_bytes(jpeg)
    (tmp_path / "quiet.json").write_text(json.dumps({"frames": [[0, len(jpeg)]]}))
    quiet = quiet_frames(tmp_path, {"index": "quiet.json", "frames": "quiet.bin"}, 1)
    audio = tmp_path / "voice.wav"
    sf.write(audio, np.full(640, 0.2, np.float32), 16000)
    heard, events = [], []

    def feature(path):
        data, rate = sf.read(path)
        heard.append(Path(path))
        assert len(data) == 14720 and np.all(data[640:] == 0)
        return None, len(data)

    engine = Engine.__new__(Engine)
    engine.materials = {"loop": SimpleNamespace(frames=[PHOTO], quiet=quiet)}
    engine.cancelled = threading.Event()
    engine.processor = SimpleNamespace(get_audio_feature=feature, get_whisper_chunk=lambda *a, **k: None)
    engine.device, engine.whisper = "cpu", None
    engine.faces = lambda *a: {}
    engine.picture = lambda *a: "live"
    plan = {"fps": 25, "clips": ["loop"], "frames": [[0, 0, -1, 0, 0, 1]] * 20, "quiet_from": 10}
    with ThreadPoolExecutor(1) as pool:
        engine.pool = pool
        engine.render(plan, str(audio), events.append)
    assert events[0]["frames"] == ["live"] * 10 + quiet * 10
    assert events[-1]["audio_seconds"] == 0.04
    assert not heard[0].exists(), "the temporary padded conditioning audio must be removed"
    (tmp_path / "quiet.bin").write_bytes(jpeg[:-1])
    with pytest.raises(ValueError, match="offsets"):
        quiet_frames(tmp_path, {"index": "quiet.json", "frames": "quiet.bin"}, 1)


def test_an_avatar_made_under_musetalk_gets_a_still_head_for_the_player(stage, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(stage.live.settings, renderer="musetalk"))
    key = stage.enrol()
    folder = stage.data / key
    for name in ("ref.png", "bias.png", "idle.json", "idle.bin"):
        assert (folder / name).is_file(), name
    assert len(json.loads((folder / "idle.json").read_text())["frames"]) == 8
    assert "head_renderer" not in json.loads((folder / "rig.json").read_text())
    assert stage.live.prepared == [key]


@pytest.mark.parametrize("options", [{"motion": 3}, {"motion": True}, {"motion": "1"}, {"clips": "yes"}, {"continuous": "yes"}])
def test_invalid_player_options_are_refused(stage, options):
    key = stage.enrol()
    assert [m["kind"] for m in stage.converse(key, **options)] == ["error"]


def test_turns_from_other_origins_are_refused(stage):
    key = stage.enrol()
    with pytest.raises(WebSocketDisconnect):
        with stage.client.websocket_connect(f"/turn/{key}", headers={"origin": "http://elsewhere.example"}):
            pass


def test_nothing_is_accepted_until_the_renderer_is_warm(stage, monkeypatch):
    key = stage.enrol()
    monkeypatch.setitem(server.warmup, "done", False)
    status = stage.client.get("/status").json()
    assert not status["ready"] and "warming up" in status["message"]
    response = stage.client.post("/avatars", data={"consent": "true"},
                                 files={"image": ("photo.png", encoded(PHOTO), "image/png")})
    assert response.status_code == 503
    assert stage.converse(key)[0]["kind"] == "error"
    monkeypatch.setitem(server.warmup, "error", "CUDA out of memory")
    assert stage.client.get("/status").json()["message"] == "CUDA out of memory"


def test_the_renderer_warmup_pays_the_compile_once_without_leaving_the_long_timeout(stage, tmp_path, monkeypatch):
    portrait = tmp_path / "warm.png"
    portrait.write_bytes(b"portrait")
    monkeypatch.setattr(server, "WARMUP_PORTRAIT", portrait)
    patient = stage.live.settings.render_timeout
    asyncio.run(server.start_renderer())
    assert stage.live.prepared == ["warmup"] and stage.live.timeouts == [1800]
    assert stage.live.settings.render_timeout == patient

    stage.live.prepared.clear()
    monkeypatch.setattr(server, "WARMUP_PORTRAIT", tmp_path / "missing.png")
    asyncio.run(server.start_renderer())
    assert stage.live.prepared == []


def test_warmup_compiles_the_renderer_unless_told_not_to(stage, monkeypatch):
    monkeypatch.setenv("AVATAR_COMPILE", "unset-me")
    monkeypatch.delenv("AVATAR_COMPILE")
    monkeypatch.setitem(server.warmup, "done", False)
    asyncio.run(real_warm())
    assert server.warmup["done"] and os.environ["AVATAR_COMPILE"] == "1"

    monkeypatch.setenv("AVATAR_COMPILE", "0")
    asyncio.run(real_warm())
    assert os.environ["AVATAR_COMPILE"] == "0"


def test_a_failed_warmup_is_reported_through_status(stage, monkeypatch):
    monkeypatch.setitem(server.warmup, "done", False)
    stage.live.failure = PipelineError("no GPU")
    asyncio.run(real_warm())
    assert server.warmup["error"] == "no GPU" and not stage.client.get("/status").json()["ready"]


def shelve(stage, key, *, name=None, created=None, options=None, body=None, photo=None):
    """A finished avatar put on disk the way the renderer leaves it, without making one (which takes seconds): enough for the
    console to list, show, change and delete."""
    folder = stage.data / key
    folder.mkdir()
    Image.fromarray(PHOTO if photo is None else photo).save(folder / "photo.png")
    (folder / "rig.json").write_text(json.dumps({"size": [512, 512], "crop": [0, 0, 512, 512], "face": [100, 100, 400, 400],
                                                 "detail": None}))
    for published in set(library.PUBLISHED) - {"rig.json"}:
        (folder / published).write_bytes(b"x")
    if body is not None:
        (folder / "body.json").write_text(json.dumps(body))
    meta = {"name": name, "created": created, "options": options}
    if any(value is not None for value in meta.values()):
        (folder / "meta.json").write_text(json.dumps({field: value for field, value in meta.items() if value is not None}))
    return key


def meta_of(stage, key):
    return json.loads((stage.data / key / "meta.json").read_text(encoding="utf-8"))


A, B, C = "a" * 32, "b" * 32, "c" * 32
FOREIGN = {"origin": "http://elsewhere.example"}


def test_the_console_and_the_player_are_pages_that_are_never_cached(stage):
    shelve(stage, A)
    for path, status in (("/", 200), (f"/a/{A}", 200), (f"/a/{B}", 404), ("/a/not-a-key", 404)):
        response = stage.client.get(path)
        # An avatar that is not there still gets the page, which explains it to the viewer.
        assert response.status_code == status, path
        assert response.headers["content-type"].startswith("text/html"), path
        assert response.headers["cache-control"] == "no-store", path


def test_the_player_reads_the_same_option_definitions_as_the_console(stage):
    assert stage.client.get("/api/options").json() == {"options": library.schema()}


def test_the_console_lists_finished_avatars_newest_first_with_what_the_cards_show(stage):
    shelve(stage, A, name="Anna", created="2024-01-01T00:00:00Z")
    shelve(stage, B, name="Ben", created="2024-02-01T00:00:00Z", options={"chat": False, "motion": 1.5},
           body={"version": 1, "sway": 17.7, "hands": [{}, {}]})
    (stage.data / C).mkdir()  # an avatar still being made
    (stage.data / C / "rig.json").write_text("{}")

    listed = stage.client.get("/api/avatars").json()["avatars"]
    assert [item["key"] for item in listed] == [B, A]
    ben, anna = listed
    assert (ben["name"], ben["created"], ben["size"], ben["url"]) == ("Ben", "2024-02-01T00:00:00Z", [512, 512], f"/a/{B}")
    assert ben["motion"] == {"hands": 2, "sway": True} and anna["motion"] == {"hands": 0, "sway": False}
    assert ben["options"] == {**library.resolve({}), "chat": False, "motion": 1.5} and anna["options"] == library.resolve({})
    for item in listed:
        assert stage.client.get(item["url"]).status_code == 200
        picture = stage.client.get(item["thumb"])
        assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    assert stage.client.get(f"/api/avatars/{B}").json() == ben
    for unknown in (C, "f" * 32, "not-a-key"):
        assert stage.client.get(f"/api/avatars/{unknown}").status_code == 404
    assert stage.client.get(f"/assets/{C}/thumb.jpg").status_code == 404


def test_an_avatar_is_renamed_and_only_the_options_that_differ_from_the_defaults_are_saved(stage):
    shelve(stage, A, name="Old", created="2024-01-01T00:00:00Z", options={"stop": False})

    renamed = stage.client.patch(f"/api/avatars/{A}", json={"name": "  Dr.\n Rao "})
    assert renamed.status_code == 200 and renamed.json()["name"] == "Dr. Rao"
    assert meta_of(stage, A) == {"name": "Dr. Rao", "created": "2024-01-01T00:00:00Z", "options": {"stop": False}}

    changed = stage.client.patch(f"/api/avatars/{A}", json={"options": {"chat": False, "motion": 0.5}})
    assert changed.json()["options"] == {**library.resolve({}), "stop": False, "chat": False, "motion": 0.5}
    assert meta_of(stage, A)["options"] == {"stop": False, "chat": False, "motion": 0.5}
    assert stage.client.get(f"/api/avatars/{A}").json() == changed.json()

    back = stage.client.patch(f"/api/avatars/{A}", json={"name": "Dr. Rao II", "options": {"chat": True, "stop": True}})
    assert meta_of(stage, A) == {"name": "Dr. Rao II", "created": "2024-01-01T00:00:00Z", "options": {"motion": 0.5}}
    assert back.json()["options"]["chat"] is True and back.json()["options"]["motion"] == 0.5

    before = meta_of(stage, A)
    assert stage.client.patch(f"/api/avatars/{A}", json={}).status_code == 200
    assert meta_of(stage, A) == before


def test_an_avatar_made_before_names_existed_gets_its_meta_the_first_time_it_is_changed(stage):
    shelve(stage, A)
    assert not (stage.data / A / "meta.json").exists()
    assert stage.client.get(f"/api/avatars/{A}").json()["name"] == "Avatar aaaaaa"
    assert stage.client.patch(f"/api/avatars/{A}", json={"options": {"mic": False}}).json()["name"] == "Avatar aaaaaa"
    assert meta_of(stage, A)["options"] == {"mic": False}


@pytest.mark.parametrize("change, mentions", [
    ({"name": ""}, "Give the avatar a name."), ({"name": " \n\t "}, "Give the avatar a name."),
    ({"options": {"motion": 3}}, "motion must be a number from 0 to 2."),
    ({"options": {"chat": "yes"}}, "chat must be true or false."),
    ({"options": {"mode": "talk"}}, "mode must be one of chat, speak."),
    ({"options": {"colour": "red"}}, "Unknown option: colour."),
    ({"options": {"chat": False, "colour": "red"}}, "Unknown option: colour."),
    ({"name": "Fine", "options": {"motion": 9}}, "motion must be"),
    ({"options": ["chat"]}, "options"), ({"name": 5}, "name"), ({"title": "Fine"}, "title"),
])
def test_an_invalid_change_is_refused_and_changes_nothing(stage, change, mentions):
    shelve(stage, A, name="Old", created="2024-01-01T00:00:00Z", options={"stop": False})
    before = (stage.data / A / "meta.json").read_bytes()
    response = stage.client.patch(f"/api/avatars/{A}", json=change)
    assert response.status_code == 422 and mentions in response.text
    assert (stage.data / A / "meta.json").read_bytes() == before


def test_an_avatar_that_is_not_there_cannot_be_changed_or_deleted(stage):
    shelve(stage, A)
    (stage.data / C).mkdir()  # not finished
    for key in (B, C, "not-a-key"):
        assert stage.client.patch(f"/api/avatars/{key}", json={"name": "Ghost"}).status_code == 404
        assert stage.client.delete(f"/api/avatars/{key}").status_code == 404
    assert [path.name for path in stage.data.iterdir() if path.name != A] == [C]


def test_changes_and_deletions_from_another_site_are_refused(stage):
    shelve(stage, A, name="Old", created="2024-01-01T00:00:00Z")
    before = (stage.data / A / "meta.json").read_bytes()
    assert stage.client.patch(f"/api/avatars/{A}", json={"name": "Hacked"}, headers=FOREIGN).status_code == 403
    assert stage.client.delete(f"/api/avatars/{A}", headers=FOREIGN).status_code == 403
    assert (stage.data / A / "meta.json").read_bytes() == before
    # A page of this server names itself as the origin, which is accepted.
    ours = {"origin": "http://testserver"}
    assert stage.client.patch(f"/api/avatars/{A}", json={"name": "Fine"}, headers=ours).status_code == 200
    assert stage.client.delete(f"/api/avatars/{A}", headers=ours).status_code == 204


def test_a_deleted_avatar_is_gone_everywhere_and_the_others_stay(stage):
    shelve(stage, A, name="Anna")
    shelve(stage, B, name="Ben")
    gone = stage.client.delete(f"/api/avatars/{A}")
    assert gone.status_code == 204 and gone.content == b""
    assert not (stage.data / A).exists()
    for path in (f"/a/{A}", f"/api/avatars/{A}", f"/assets/{A}/photo.png", f"/assets/{A}/thumb.jpg"):
        assert stage.client.get(path).status_code == 404, path
    assert stage.client.delete(f"/api/avatars/{A}").status_code == 404
    assert [item["key"] for item in stage.client.get("/api/avatars").json()["avatars"]] == [B]
    assert stage.client.get(f"/assets/{B}/photo.png").status_code == 200


def test_an_avatar_cannot_be_deleted_while_it_is_speaking_but_can_once_it_has_finished(stage):
    shelve(stage, A)
    with stage.client.websocket_connect(f"/turn/{A}") as ws:
        ws.send_json({"text": "Hello", "mode": "speak", "history": []})
        message = ws.receive_json()
        while message["kind"] != "segment":
            message = ws.receive_json()
        # The turn now waits for this segment to be acknowledged, so it is certainly still going.
        refused = stage.client.delete(f"/api/avatars/{A}")
        while message["kind"] != "done":
            if message["kind"] == "segment":
                ws.send_json({"type": "ack", "index": message["index"]})
            message = ws.receive_json()
    assert refused.status_code == 409 and "speaking" in refused.json()["detail"]
    assert stage.client.delete(f"/api/avatars/{A}").status_code == 204


def test_with_an_admin_token_only_managing_avatars_asks_for_it(stage, monkeypatch):
    monkeypatch.setattr(server, "ADMIN_TOKEN", "s3cret")
    shelve(stage, A, name="Anna")
    broken_upload = {"image": ("x.png", b"not an image", "image/png")}
    managing = {
        "list": lambda headers: stage.client.get("/api/avatars", headers=headers),
        "rename": lambda headers: stage.client.patch(f"/api/avatars/{A}", json={"name": "Renamed"}, headers=headers),
        "upload": lambda headers: stage.client.post("/avatars", data={"consent": "true"}, files=broken_upload, headers=headers),
        "delete": lambda headers: stage.client.delete(f"/api/avatars/{A}", headers=headers),
    }
    for given in ({}, {"x-admin-token": ""}, {"x-admin-token": "nope"}, {"x-admin-token": "S3CRET"},
                  {"x-admin-token": "s3cre"}, {"x-admin-token": "s3cret-and-more"}):
        for what, call in managing.items():
            assert call(given).status_code == 401, (what, given)
    assert meta_of(stage, A)["name"] == "Anna" and stage.live.prepared == []

    right = {"x-admin-token": "s3cret"}
    assert [item["key"] for item in managing["list"](right).json()["avatars"]] == [A]
    assert managing["rename"](right).json()["name"] == "Renamed"
    assert managing["upload"](right).status_code == 422  # past the token, then refused as not an image
    # What a viewer does needs no token: the pages, the pictures, the options, the status and talking to the avatar.
    for path in ("/", f"/a/{A}", "/api/options", f"/api/avatars/{A}", f"/assets/{A}/photo.png", f"/assets/{A}/thumb.jpg",
                 "/status"):
        assert stage.client.get(path).status_code == 200, path
    assert stage.converse(A)[-1]["kind"] == "done"
    assert managing["delete"](right).status_code == 204


def test_the_thumbnail_is_made_when_first_asked_for_and_an_unreadable_photo_has_none(stage):
    shelve(stage, A, photo=texture(2, 1600))
    thumb = stage.data / A / "thumb.jpg"
    for _ in range(2):  # the second time it is made again, as when it was lost
        assert not thumb.exists()
        response = stage.client.get(f"/assets/{A}/thumb.jpg")
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        with Image.open(io.BytesIO(response.content)) as picture:
            assert picture.format == "JPEG" and max(picture.size) == library.THUMB_SIDE
        assert thumb.is_file()
        thumb.unlink()
    (stage.data / A / "photo.png").write_bytes(b"damaged")
    assert stage.client.get(f"/assets/{A}/thumb.jpg").status_code == 404
    assert not thumb.exists()


def test_an_upload_is_named_by_its_owner_or_else_after_its_file(stage):
    named = stage.enrol(name="  Dr.\n  Rao  ", filename="IMG_0001.PNG")
    unnamed = stage.enrol(filename="Rashid 2.0.PNG")
    listed = {item["key"]: item for item in stage.client.get("/api/avatars").json()["avatars"]}
    assert listed[named]["name"] == "Dr. Rao" and listed[unnamed]["name"] == "Rashid 2.0"
    for key in (named, unnamed):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta_of(stage, key)["created"])
        assert meta_of(stage, key)["options"] == {}
        assert (stage.data / key / "thumb.jpg").is_file()  # made on upload, so the console's first view is quick


def test_guide_encodes_which_way_the_face_moved_and_how_far_to_trust_it():
    reference = texture(3)
    anchor = guide.Reference(reference)

    still = anchor.guide(reference)
    assert (still[..., 0] == 128).all() and (still[..., 1] == 128).all() and (still[..., 2] == 255).all()

    # The frame shows the reference moved 3 px right and 2 px down; the flow points back at the source:
    # frame(x) = reference(x + flow), stored as 128 + 4 * flow.
    moved = anchor.guide(np.roll(reference, (2, 3), (0, 1)))[24:104, 24:104]
    assert abs(moved[..., 0].mean() - (128 - 4 * 3)) < 1 and abs(moved[..., 1].mean() - (128 - 4 * 2)) < 1
    # Warping the reference by that flow reproduces the frame, so the motion is fully trusted.
    assert moved[..., 2].mean() > 240

    replaced = reference.copy()
    replaced[200:300, 200:300] = texture(9)[200:300, 200:300]
    trust = anchor.guide(replaced)[..., 2]
    assert trust[50:75, 50:75].mean() < 60 and trust[10:20, 10:20].min() > 240


def test_the_reference_must_be_a_flashhead_frame():
    with pytest.raises(ValueError):
        guide.Reference(np.zeros((256, 256, 3), np.uint8))


@pytest.mark.parametrize("guided", [True, False])
def test_the_idle_pack_is_a_contiguous_index_into_one_blob(tmp_path, guided):
    Image.fromarray(PHOTO).save(tmp_path / "photo.png")
    detail = {"scale": 1.5} if guided else None
    (tmp_path / "rig.json").write_text(json.dumps(
        {"size": [512, 512], "crop": [0, 0, 512, 512], "face": [0, 0, 512, 512], "detail": detail}))
    guide.build_idle(tmp_path, [PHOTO] * 10)

    pack, blob = json.loads((tmp_path / "idle.json").read_text()), (tmp_path / "idle.bin").read_bytes()
    assert pack["fps"] == 25 and pack["guided"] is guided and len(pack["frames"]) == 10
    cursor = 0
    for entry in pack["frames"]:
        assert len(entry) == (4 if guided else 2)
        for offset, length, size in zip(entry[::2], entry[1::2], [(512, 512), (guide.GRID, guide.GRID)]):
            assert offset == cursor
            assert Image.open(io.BytesIO(blob[offset:offset + length])).size == size
            cursor += length
    assert cursor == len(blob)
    # The photo is the reference, so there is no tone difference for the player to correct.
    assert np.abs(np.asarray(Image.open(tmp_path / "bias.png"), dtype=int) - 128).max() <= 1


@pytest.mark.parametrize("crop, stored", [
    ([40, 30, 316, 306], (276, 276)),    # smaller than FlashHead's 512 px: kept at its own size, not upsampled
    ([0, 0, 1200, 900], (1200, 900)),
    ([0, 0, 3000, 2000], (2048, 1365)),  # capped on the long side
])
def test_the_photos_detail_is_stored_at_the_crops_own_resolution_up_to_a_cap(crop, stored):
    assert enroll.detail_layer(Image.new("RGB", (3000, 2000), (90, 90, 90)), crop).size == stored


def test_the_photos_detail_holds_fine_structure_and_leaves_tone_to_flashhead():
    ramp = np.tile(np.linspace(40, 200, 276, dtype=np.float32), (276, 1))
    ramp[:, 100] += 60
    layer = np.asarray(enroll.detail_layer(Image.fromarray(np.dstack([ramp] * 3).astype(np.uint8)), [0, 0, 276, 276]), dtype=int) - 128
    assert np.abs(layer[:, 150:250]).max() <= 1
    assert layer[100, 100] > 20 and layer[100, 98] <= 0 and layer[100, 102] <= 0


@pytest.mark.skipif(not (ENROLL_PYTHON.exists() and EXAMPLE_FACE.exists()), reason="needs the FlashHead environment")
def test_enrolling_a_head_smaller_than_flashhead_renders_still_saves_the_photos_detail(tmp_path):
    shutil.copy(EXAMPLE_FACE, tmp_path / "photo.png")
    subprocess.run([ENROLL_PYTHON, server.BASE / "enroll.py", tmp_path], check=True, capture_output=True, timeout=120,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    rig = json.loads((tmp_path / "rig.json").read_text())
    x1, y1, x2, y2 = rig["crop"]
    assert max(x2 - x1, y2 - y1) < 512 * 1.08 and rig["detail"] is not None
    assert Image.open(tmp_path / "detail.png").size == (x2 - x1, y2 - y1)


@pytest.mark.skipif(not (ENROLL_PYTHON.exists() and EXAMPLE_FACE.exists()), reason="needs the FlashHead environment")
def test_the_body_rig_opens_no_gpu_not_even_for_mediapipes_gl_context(tmp_path):
    # On a shared machine the first GPU that the driver offers may be someone else's.
    shutil.copy(EXAMPLE_FACE, tmp_path / "photo.png")
    width, height = Image.open(tmp_path / "photo.png").size
    face = [width * 0.3, height * 0.2, width * 0.7, height * 0.6]
    (tmp_path / "rig.json").write_text(json.dumps({"size": [width, height], "face": face, "crop": face}))
    probe = ("import sys; from photostage import body; body.main(['body', sys.argv[1]]); "
             "print('gpu libraries:', sorted({line.split()[-1] for line in open('/proc/self/maps') if 'nvidia' in line}))")
    done = subprocess.run([ENROLL_PYTHON, "-c", probe, tmp_path], cwd=server.ROOT, capture_output=True, text=True,
                          timeout=120, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "body.json").is_file() and "gpu libraries: []" in done.stdout


@pytest.fixture(scope="module")
def cross_fade():
    """What the real compositor draws, in headless Chromium, when it fades between two frames of one bar 16 px apart."""
    done = subprocess.run([shutil.which("python3") or "python3", server.ROOT / "tests/compositor_probe.py"],
                          capture_output=True, text=True, timeout=120)
    if done.returncode == SKIP_STATUS:
        pytest.skip("needs a python3 with Playwright and Chromium")
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


# The two frames hold one bar 16 px apart (at 200 and 216, 32 px wide) and the guides say which pixels are the same
# part of the head. A head drawn twice shows as half-bright pixels, so the "ghost" count is the thing to watch.
@pytest.mark.parametrize("mix, left", [(0, 200), (0.25, 204), (0.5, 208), (0.75, 212), (1, 216)])
def test_a_cross_fade_moves_one_whole_head_from_the_pose_of_one_frame_to_the_other(cross_fade, mix, left):
    fade = cross_fade["aligned"][str(mix)]
    assert (fade["left"], fade["right"]) == (left, left + 31) and fade["ghost"] <= 2


@pytest.mark.parametrize("case", ["switchedOff", "unguidedIdle", "unguidedLive"])
def test_a_cross_fade_is_a_plain_dissolve_when_switched_off_or_a_frame_has_no_guide(cross_fade, case):
    assert cross_fade[case]["ghost"] == 32


# The clip layer: a photo of grey with a patch of 1 px checks and a head patch (orange, with a white line), and flat clips
# over its lower part (see CLIP_SCRIPT for the geometry).
@pytest.fixture(scope="module")
def clip_layer(cross_fade):
    return cross_fade["clips"]


def test_head_motion_is_registered_to_the_collar_without_changing_the_face(cross_fade):
    for frame in cross_fade["join"]["frames"]:
        assert frame["unjoinedCollar"] > 5
        assert frame["collar"] < 1
        assert frame["face"] == 0
        assert frame["outside"] == 0


def test_head_join_leaves_unguided_frames_unchanged(cross_fade):
    assert cross_fade["join"]["unguided"] == 0


def test_video_avatar_keeps_its_own_idle_face_and_only_reenacts_the_face_while_talking(cross_fade):
    avatar = cross_fade["avatar"]
    assert avatar["idle"]["face"] == [50, 100, 150]
    assert avatar["talk"]["face"] == [224, 128, 64]
    assert avatar["cleared"]["face"] == [224, 128, 64]
    for phase in ("idle", "talk", "carried", "registered"):
        for region in ("hair", "neck", "body"):
            assert avatar[phase][region] == [50, 100, 150]


def test_video_pose_not_independent_renderer_pose_carries_the_reenacted_face(cross_fade):
    avatar = cross_fade["avatar"]
    assert avatar["talk"]["line"] == pytest.approx(256, abs=0.1)
    assert avatar["carried"]["line"] == pytest.approx(276, abs=0.1)


def test_face_pose_fit_rejects_local_expression_motion(cross_fade):
    avatar = cross_fade["avatar"]
    assert avatar["fitted"] == pytest.approx(avatar["expected"], abs=0.003)


def test_registered_face_follows_video_shear_instead_of_the_old_neck_transform(cross_fade):
    assert cross_fade["avatar"]["registered"]["line"] == pytest.approx(267.47, abs=0.2)


def test_fixed_reference_anchors_do_not_turn_expression_or_confidence_changes_into_head_motion(cross_fade):
    avatar = cross_fade["avatar"]
    for frame in avatar["changingExpressions"]:
        assert frame["anchored"] == pytest.approx(avatar["expected"], abs=0.003)
    assert any(np.max(np.abs(np.array(frame["legacy"]) - avatar["expected"])) > 0.01
               for frame in avatar["changingExpressions"]), "the fixture must expose the old drifting fit"
    assert avatar["singularAnchors"] is None
    assert avatar["anchored"]["line"] == pytest.approx(267.47, abs=0.2)
    for region in ("face", "eyes", "mouth", "hair", "neck", "body"):
        assert avatar["anchored"][region] == avatar["registered"][region]


def test_mouth_only_reenactment_keeps_the_video_face_eyes_cheeks_and_neck(cross_fade):
    avatar = cross_fade["avatar"]
    assert avatar["mouthOnly"]["mouth"] == [224, 128, 64]
    assert avatar["mouthIdle"]["mouth"] == [50, 100, 150]
    for region in ("face", "eyes", "cheek", "hair", "neck", "body"):
        assert avatar["mouthOnly"][region] == [50, 100, 150]

def test_full_head_masks_the_generated_face_and_hair_but_keeps_video_body_and_background(cross_fade):
    avatar = cross_fade["avatar"]
    for phase in ("wholeIdle", "wholeTalk", "wholeCarried"):
        for region in ("face", "eyes", "cheek", "hair"):
            assert avatar[phase][region] == [224, 128, 64], (phase, region)
        for region in ("body", "background"):
            assert avatar[phase][region] == [50, 100, 150]
    assert avatar["wholeIdle"]["line"] == pytest.approx(256, abs=0.1)
    assert avatar["wholeTalk"]["line"] == pytest.approx(264, abs=0.1), "the earlier head retains its own motion"
    assert avatar["wholeCarried"]["line"] == pytest.approx(284, abs=0.1), "the whole head follows the collar anchor"


def test_dubbed_pictures_bypass_the_old_head_and_cover_the_whole_avatar(cross_fade):
    dubbed = cross_fade["avatar"]["dubbed"]
    assert dubbed["initial"] == [[51, 51, 51]] * 2, "MuseTalk idle must not show the old animated head"
    assert dubbed["speaking"] == [[128, 64, 224], [160, 224, 64]]
    assert dubbed["photo"] == [[224, 128, 64], [64, 160, 224]], "photo replies do not use clip colour correction"
    assert not dubbed["poseWarning"], "whole frames need no separate live-face registration"


def test_dubbed_pictures_and_local_clips_use_the_same_tone_during_crossfade(cross_fade):
    dubbed = cross_fade["avatar"]["dubbed"]
    assert dubbed["idle"] == [[100, 150, 50]] * 2
    for actual, expected in zip(dubbed["fade"], ([114, 107, 137], [130, 187, 57])):
        assert actual == pytest.approx(expected, abs=1)


def test_a_body_clip_with_no_weight_leaves_the_picture_exactly_as_it_was(clip_layer):
    assert clip_layer["off"] == 0


def test_a_body_clip_is_drawn_only_where_it_differs_from_the_photo(clip_layer):
    square = clip_layer["square"]
    assert square["inside"] == [255, 0, 0] and square["outside"] == clip_layer["photoFar"]
    assert clip_layer["above"] == clip_layer["photoFar"]


def test_a_body_clip_keeps_the_photos_own_sharpness_where_it_has_not_moved(clip_layer):
    # A clip is a softer picture of the photo: drawn whole it would blur the checks, and the matte leaves them as they were.
    assert clip_layer["sharp"] == {"kept": 1, "whole": 0, "photo": 1}


def test_a_body_clip_fades_with_its_weight_and_at_the_edge_of_its_box_but_not_at_the_pictures_border(clip_layer):
    assert all(abs(a - b) <= 1 for a, b in zip(clip_layer["square"]["half"], [192, 64, 64]))
    assert 150 < clip_layer["top"][0] < 230 and clip_layer["top"][1] > 20
    assert clip_layer["border"] == [[255, 0, 0]] * 3


def test_a_body_clip_never_covers_the_live_head_but_shows_under_the_patch_below_the_chin(clip_layer):
    assert clip_layer["face"] == [224, 128, 64]
    assert clip_layer["below"] == [255, 0, 0]
    # The patch's bottom band is the head over the clip's own body. Over the static photo it would stay grey (128).
    assert clip_layer["bottomBand"][1] < 64


def test_the_head_follows_the_clips_neck_from_the_chin_down_and_stays_on_the_photo_at_the_top_of_its_patch(clip_layer):
    follow = clip_layer["follow"]
    assert follow["rest"] == {"top": pytest.approx(256, abs=0.01), "chin": pytest.approx(256, abs=0.01)}
    assert follow["slid"]["chin"] == pytest.approx(276, abs=1) and follow["slid"]["top"] == pytest.approx(256, abs=1.5)
    assert follow["turned"]["chin"] == pytest.approx(272, abs=1.5) and follow["turned"]["top"] == pytest.approx(256, abs=2)
    assert follow["half"]["chin"] == pytest.approx(266, abs=1)


def test_a_body_clips_colours_are_put_right_by_the_tone_matrix_and_offset(clip_layer):
    assert clip_layer["tone"] == pytest.approx([126, 200, 50], abs=1)


def test_closing_the_live_worker_copes_with_a_worker_that_already_lost_its_pipe():
    def write(data):
        raise RuntimeError("unable to perform operation on <_UnixWritePipeTransport closed=True>; the handler is closed")

    async def wait():
        process.returncode = 0

    # What Ctrl-C leaves behind: the worker got the signal too, and its pipe closed before it was reaped.
    process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(is_closing=lambda: True, write=write), wait=wait)
    worker = LiveWorker(server.settings)
    worker.process = process
    asyncio.run(worker.close())
    assert process.returncode == 0 and not worker.ready
