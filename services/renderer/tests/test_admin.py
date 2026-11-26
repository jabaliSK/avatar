"""The avatar administration API: library, preview, rename, settings, deletion, warm-up and registration (no GPU: the registrar is faked)."""
import json
import os
import shutil
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import admin
from app.avatar_store import DEFAULT_SETTINGS, SETTING_RANGES, avatar_stamp


class FakeRuntime:
    def __init__(self, root, status="ready"):
        self.root, self.status, self.detail = str(root), status, "ok"
        self.session, self.load_lock = threading.Lock(), threading.Lock()
        self.loaded, self.calls = {}, []
        self.engine, self.registrar = SimpleNamespace(kw={"max_size": 1280}), object()

    def get_avatar(self, name):
        self.calls.append(name)
        self.loaded[name] = SimpleNamespace(stamp=avatar_stamp(self.root, name))


def make_avatar(root, name, states=(), title=None, size=(1600, 900)):
    directory = root / name
    (directory / "still").mkdir(parents=True)
    cv2.imwrite(str(directory / "source.png"), np.full((size[1], size[0], 3), 90, np.uint8))
    (directory / "meta.json").write_text(json.dumps({"name": name, "source": "source.png", "frame_size": [704, 1280], "states": list(states)}))
    if title:
        (directory / "profile.json").write_text(json.dumps({"title": title, "subject": "A presenter", "created": "2025-01-02T03:04:05Z"}))
    return directory


@pytest.fixture
def env(tmp_path, monkeypatch):
    root, studio = tmp_path / "avatars", tmp_path / "studio"
    root.mkdir(), studio.mkdir()
    monkeypatch.setattr(admin, "STUDIO_ROOTS", (str(studio),))
    rt = FakeRuntime(root)
    app = FastAPI()
    app.include_router(admin.build_router(rt, str(root)))
    return SimpleNamespace(root=root, studio=studio, rt=rt, client=TestClient(app))


def wait_for(condition, timeout=5.0):
    end = time.time() + timeout
    while not condition():
        assert time.time() < end, "timed out"
        time.sleep(0.01)


# ------------------------------------------------------------------------------------------------------------------ library
def test_the_library_lists_titles_modes_and_sizes(env):
    make_avatar(env.root, "gulf-man-white", ("idle", "speaking"), title="Gulf Man")
    make_avatar(env.root, "old-one")  # made with onboard.py: no profile, no loops
    (env.root / ".staging" / "half-done").mkdir(parents=True)
    body = env.client.get("/v1/avatars").json()
    by_id = {a["id"]: a for a in body["avatars"]}
    assert set(by_id) == {"gulf-man-white", "old-one"}
    full, quick = by_id["gulf-man-white"], by_id["old-one"]
    assert (full["title"], full["mode"], full["states"], full["created"]) == ("Gulf Man", "full", ["idle", "speaking"], "2025-01-02T03:04:05Z")
    assert (quick["title"], quick["mode"], quick["states"]) == ("Old One", "quick", [])
    assert full["settings"] == DEFAULT_SETTINGS and full["size_bytes"] > 0 and full["thumbnail"].startswith("/v1/avatars/gulf-man-white/thumbnail?v=")
    assert body["busy"] is False and body["renderer"]["status"] == "ready"
    # the same facts without walking the avatar folders, for a frequent health check
    assert env.client.get("/v1/status").json() == {k: v for k, v in body.items() if k != "avatars"}


def test_the_preview_is_made_once_and_again_after_a_new_registration(env):
    directory = make_avatar(env.root, "a")
    first = env.client.get("/v1/avatars/a/thumbnail")
    assert first.status_code == 200 and first.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(first.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(image.shape[:2]) == admin.THUMBNAIL_SIDE
    thumb = directory / "thumb.jpg"
    os.utime(directory / "meta.json", (1000, 1000))
    os.utime(thumb, (2000, 2000))
    env.client.get("/v1/avatars/a/thumbnail")
    assert os.path.getmtime(thumb) == 2000  # up to date: served from the file
    os.utime(directory / "meta.json", (3000, 3000))  # registered again since
    env.client.get("/v1/avatars/a/thumbnail")
    assert os.path.getmtime(thumb) > 3000


def test_names_that_are_not_avatars_are_not_found(env):
    make_avatar(env.root, "a")
    for name in ("missing", "Bad.Name", "a..b", "x" * 80):
        assert env.client.get(f"/v1/avatars/{name}").status_code == 404
        assert env.client.delete(f"/v1/avatars/{name}").status_code == 404
    assert (env.root / "a").is_dir()


# ------------------------------------------------------------------------------------------------------------------ editing
def test_renaming_does_not_change_what_makes_the_renderer_reload(env):
    make_avatar(env.root, "a")
    before = avatar_stamp(str(env.root), "a")
    out = env.client.patch("/v1/avatars/a", json={"title": "  Aisha\n  Khan\t", "subject": "A calm\x07 presenter  "}).json()
    assert (out["title"], out["subject"]) == ("Aisha Khan", "A calm presenter")
    assert avatar_stamp(str(env.root), "a") == before
    assert env.client.patch("/v1/avatars/a", json={"title": "   "}).status_code == 422
    assert env.client.patch("/v1/avatars/a", json={"title": "x" * 81}).status_code == 422
    assert env.client.get("/v1/avatars/a").json()["title"] == "Aisha Khan"


def test_settings_are_clamped_and_make_the_renderer_reload(env):
    make_avatar(env.root, "a")
    before = avatar_stamp(str(env.root), "a")
    time.sleep(0.01)
    saved = env.client.put("/v1/avatars/a/settings", json={"expression": 9, "head_motion": -1}).json()
    assert saved == {"expression": SETTING_RANGES["expression"][1], "head_motion": SETTING_RANGES["head_motion"][0]}
    assert avatar_stamp(str(env.root), "a") > before
    assert env.client.put("/v1/avatars/a/settings", json={"expression": "loud"}).status_code == 422
    assert env.client.get("/v1/avatars/a").json()["settings"] == saved


def test_an_avatar_is_deleted_unless_a_conversation_is_running(env):
    make_avatar(env.root, "a")
    env.rt.loaded["a"] = object()
    with env.rt.session:
        assert env.client.delete("/v1/avatars/a").status_code == 409
    assert (env.root / "a").is_dir()
    assert env.client.delete("/v1/avatars/a").status_code == 200
    assert not (env.root / "a").exists() and "a" not in env.rt.loaded and not env.rt.session.locked()


# ------------------------------------------------------------------------------------------------------------------ warm-up
def test_warm_up_loads_an_avatar_once_and_leaves_a_busy_renderer_alone(env):
    make_avatar(env.root, "a")
    with env.rt.session:  # a conversation is running
        assert env.client.post("/v1/avatars/a/warm").json()["busy"] is True
    assert env.rt.calls == []
    first = env.client.post("/v1/avatars/a/warm").json()
    assert first["loaded"] or first["loading"]
    wait_for(lambda: env.client.get("/v1/avatars/a/status").json()["loaded"])
    env.client.post("/v1/avatars/a/warm")
    assert env.rt.calls == ["a"]
    env.client.put("/v1/avatars/a/settings", json={"expression": 0.7})  # the loaded copy is out of date now
    assert env.client.get("/v1/avatars/a/status").json()["loaded"] is False
    env.rt.status = "loading"
    env.client.post("/v1/avatars/a/warm")
    assert env.rt.calls == ["a"]


# ------------------------------------------------------------------------------------------------------------------ registration
@pytest.fixture
def fake_registrar(monkeypatch):
    """Stands in for the GPU work: writes the files that onboard_image / add_state_loop write."""
    sizes = {}

    def onboard(root, name, image, registrar, engine, kw, target_roi=560, size=None):
        directory = os.path.join(root, name)
        os.makedirs(os.path.join(directory, "still"))
        shutil.copy(image, os.path.join(directory, "source.png"))
        meta = {"name": name, "source": "source.png", "frame_size": list(size or (704, 1280)), "states": []}
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)

    def add(root, name, state, video, registrar, engine, kw, pin=True):
        if state == "speaking" and sizes.get("fail"):
            raise RuntimeError("out of memory")
        path = os.path.join(root, name, "meta.json")
        with open(path) as f:
            meta = json.load(f)
        meta["states"] = sorted(set(meta["states"]) | {state})
        with open(path, "w") as f:
            json.dump(meta, f)

    monkeypatch.setattr(admin, "onboard_image", onboard)
    monkeypatch.setattr(admin, "add_state_loop", add)
    monkeypatch.setattr(admin, "video_size", lambda path: sizes.get(os.path.basename(path), (704, 1280)))
    return sizes


def upload(env, name="photo.png", content=b"x"):
    path = env.studio / name
    cv2.imwrite(str(path), np.full((64, 48, 3), 120, np.uint8))
    return str(path)


def finish(env, started):
    assert started.status_code == 202, started.text
    rid = started.json()["id"]
    wait_for(lambda: env.client.get(f"/v1/registrations/{rid}").json()["state"] != "running")
    return env.client.get(f"/v1/registrations/{rid}").json()


def test_registration_is_refused_for_bad_requests_and_a_busy_renderer(env, fake_registrar):
    image = upload(env)
    clip = env.studio / "idle.mp4"
    clip.write_bytes(b"")
    body = {"id": "new-one", "image": image, "loops": {"idle": str(clip)}}
    for change in ({"id": "Bad_Id"}, {"id": "-x"}, {"image": "/etc/passwd"}, {"image": str(env.studio / "missing.png")},
                   {"loops": {"flying": str(clip)}}, {"loops": {"idle": "/etc/hostname"}}, {"image": str(env.studio / ".." / "avatars")}):
        assert env.client.post("/v1/registrations", json={**body, **change}).status_code == 422, change
    link = env.studio / "link.png"
    link.symlink_to("/etc/hostname")  # a link out of the studio folder is no way around the check
    assert env.client.post("/v1/registrations", json={**body, "image": str(link)}).status_code == 422
    env.rt.status = "loading"
    assert env.client.post("/v1/registrations", json=body).status_code == 503
    env.rt.status = "ready"
    with env.rt.session:
        assert env.client.post("/v1/registrations", json=body).status_code == 409
    assert env.rt.session.acquire(blocking=False)  # nothing is left locked by the refusals
    env.rt.session.release()
    assert list_ids(env) == []


def list_ids(env):
    return [a["id"] for a in env.client.get("/v1/avatars").json()["avatars"]]


def test_registration_builds_in_staging_then_moves_into_place(env, fake_registrar):
    image = upload(env)
    clips = {}
    for state in ("idle", "speaking"):
        clips[state] = str(env.studio / f"{state}.mp4")
        open(clips[state], "wb").close()
    done = finish(env, env.client.post("/v1/registrations", json={"id": "aisha", "image": image, "loops": clips,
                                                                   "title": "Aisha", "subject": "A presenter"}))
    assert (done["state"], done["done"], done["steps"], done["error"]) == ("done", 3, 3, None)
    assert not env.rt.session.locked() and list(env.root.iterdir()) == [env.root / ".staging", env.root / "aisha"]
    assert os.listdir(env.root / ".staging") == []
    avatar = env.client.get("/v1/avatars/aisha").json()
    assert (avatar["title"], avatar["subject"], avatar["mode"], avatar["frame_size"]) == ("Aisha", "A presenter", "full", [704, 1280])

    # registering the same id again replaces it but keeps what the user tuned
    env.client.put("/v1/avatars/aisha/settings", json={"expression": 0.9})
    env.client.patch("/v1/avatars/aisha", json={"title": "Aisha K."})
    created = avatar["created"]
    env.rt.loaded["aisha"] = object()
    done = finish(env, env.client.post("/v1/registrations", json={"id": "aisha", "image": image}))
    assert done["state"] == "done" and (done["done"], done["steps"]) == (1, 1)
    again = env.client.get("/v1/avatars/aisha").json()
    assert (again["title"], again["settings"]["expression"], again["mode"], again["created"]) == ("Aisha K.", 0.9, "quick", created)
    assert "aisha" not in env.rt.loaded


def test_a_failed_registration_keeps_the_previous_avatar(env, fake_registrar):
    make_avatar(env.root, "aisha", ("idle",), title="Aisha")
    clip = env.studio / "speaking.mp4"
    clip.write_bytes(b"")
    fake_registrar["fail"] = True
    done = finish(env, env.client.post("/v1/registrations", json={"id": "aisha", "image": upload(env), "loops": {"speaking": str(clip)}}))
    assert done["state"] == "failed" and "out of memory" in done["error"]
    assert not env.rt.session.locked() and os.listdir(env.root / ".staging") == []
    assert env.client.get("/v1/avatars/aisha").json()["states"] == ["idle"]


def test_clips_of_different_sizes_are_refused(env, fake_registrar):
    for state, size in (("idle", (704, 1280)), ("speaking", (720, 1280))):
        (env.studio / f"{state}.mp4").write_bytes(b"")
        fake_registrar[f"{state}.mp4"] = size
    clips = {s: str(env.studio / f"{s}.mp4") for s in ("idle", "speaking")}
    done = finish(env, env.client.post("/v1/registrations", json={"id": "x", "image": upload(env), "loops": clips}))
    assert done["state"] == "failed" and "one size" in done["error"]
    assert list_ids(env) == []
