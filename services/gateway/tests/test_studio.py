"""The Avatar Studio: what it accepts from a user (photo, text, key), how it drives the renderer and the job runner (replaced by fakes here),
and that it replaces the stock page without touching the engine's own endpoints. No GPU, no network.

Non-English sample text is written as escapes on purpose, so that no source file in the repository contains such characters.
"""
import io
import json
import os
import re
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GATEWAY = os.path.join(HERE, "..")
sys.path[:0] = ["/opt/oac", "/opt/oac/src", GATEWAY]

import studio  # noqa: E402
from studio import photos  # noqa: E402
from studio.guard import KeyGuard  # noqa: E402
from studio.settings import Settings  # noqa: E402

ZH = "\u4f60\u597d"  # Chinese ideographs
KANA = "\u3053\u3093\u306b\u3061\u306f"
HANGUL = "\uc548\ub155"
FULL_WIDTH_A = "\uff21"
THUMB = b"\xff\xd8\xff\xe0 a preview"
ACTIVE = ("queued", "generating", "waiting", "registering")


# ------------------------------------------------------------------------------------------------------------------------------
# Photos and text
# ------------------------------------------------------------------------------------------------------------------------------
def encoded(size=(640, 960), fmt="PNG", mode="RGB", color=(200, 120, 90), **save):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, fmt, **save)
    return buf.getvalue()


def noise_png(size=(640, 960)):
    """A photo that does not compress well: about as large as a real one."""
    buf = io.BytesIO()
    Image.effect_noise(size, 80).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def test_a_photo_is_rewritten_as_a_clean_png_with_a_preview():
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: the camera says "rotate"
    photo = photos.prepare_photo(encoded((800, 400), "JPEG", exif=exif))
    assert photo.size == (400, 800)
    with Image.open(io.BytesIO(photo.png)) as image:
        assert image.format == "PNG" and image.mode == "RGB" and image.size == (400, 800)
    with Image.open(io.BytesIO(photo.thumbnail)) as preview:
        assert preview.format == "JPEG" and preview.size == (320, 640)


def test_a_transparent_photo_is_put_on_white_and_a_large_one_is_scaled_down():
    clear = photos.prepare_photo(encoded((400, 400), "PNG", "RGBA", (255, 0, 0, 0)))
    with Image.open(io.BytesIO(clear.png)) as image:
        assert image.mode == "RGB" and image.getpixel((10, 10)) == (255, 255, 255)
    big = photos.prepare_photo(encoded((4000, 3000)))
    assert big.size == (3072, 2304)
    with Image.open(io.BytesIO(big.thumbnail)) as preview:
        assert preview.size == (640, 480)


@pytest.mark.parametrize("data, message", [
    (b"this is not a picture", "not a photo"),
    (encoded((400, 400), "GIF"), "not a photo"),
    (encoded((400, 400), "BMP"), "not a photo"),
    (noise_png()[:20000], "damaged"),
    (encoded((200, 600)), "too small"),
    (encoded((300, 1500)), "too narrow"),
])
def test_a_file_that_is_not_a_usable_photo_is_refused_with_a_reason(data, message):
    with pytest.raises(photos.InputError, match=message):
        photos.prepare_photo(data)


def test_a_photo_with_too_many_pixels_is_refused_before_it_is_decoded(monkeypatch):
    monkeypatch.setattr(photos, "MAX_PIXELS", 1_000_000)
    with pytest.raises(photos.InputError, match="larger than"):
        photos.prepare_photo(encoded((1200, 1200)))


def test_text_becomes_one_plain_line():
    assert photos.clean_text("  Ada \r\n\t Lovelace\x00 ", "The name", 80) == "Ada Lovelace"
    assert photos.clean_text("Ada\u200bLove\u202elace", "The name", 80) == "AdaLovelace"  # zero-width and direction-override characters
    assert photos.clean_text(None, "The name", 80) == ""


@pytest.mark.parametrize("text", ["Hi " + ZH, KANA, HANGUL, FULL_WIDTH_A + "da"])
def test_text_in_another_script_is_refused(text):
    with pytest.raises(photos.InputError, match="must be written in English"):
        photos.clean_text(text, "The description", 600)


def test_text_that_is_too_long_or_missing_is_refused():
    with pytest.raises(photos.InputError, match="longer than 80"):
        photos.clean_text("x" * 81, "The name", 80)
    with pytest.raises(photos.InputError, match="is required"):
        photos.clean_text(" \n ", "The name", 80, required=True)


def test_avatar_ids_are_short_ascii_slugs_and_never_collide():
    assert photos.slugify("Ada Lovelace!") == "ada-lovelace"
    assert photos.slugify("Zo\u00eb \u00d1and\u00fa") == "zoe-nandu"
    assert photos.slugify(ZH) == "avatar"
    assert photos.slugify("x" * 100) == "x" * photos.ID_MAX
    assert photos.unique_id("ada", {"ada", "ada-2"}) == "ada-3"
    assert photos.unique_id("ada", {"bob"}) == "ada"
    long = "x" * photos.ID_MAX
    assert photos.unique_id(long, {long}) == "x" * (photos.ID_MAX - 2) + "-2"
    assert photos.title_from_filename("gulf_man-white.JPG") == "Gulf Man White"
    assert photos.title_from_filename("") == "New Avatar"


# ------------------------------------------------------------------------------------------------------------------------------
# The key and cross-site requests
# ------------------------------------------------------------------------------------------------------------------------------
def fake_request(key=None, host="10.0.0.5"):
    return SimpleNamespace(client=SimpleNamespace(host=host), headers={} if key is None else {"x-studio-key": key}, method="POST")


def test_the_key_is_checked_and_guessing_is_slowed_down():
    now = [0.0]
    guard = KeyGuard("s3cret", attempts=3, window_s=60, clock=lambda: now[0])
    guard.check(fake_request("s3cret"))
    for _ in range(3):
        with pytest.raises(HTTPException) as refused:
            guard.check(fake_request("wrong"))
        assert refused.value.status_code == 401
    with pytest.raises(HTTPException) as locked_out:
        guard.check(fake_request("s3cret"))  # too many wrong keys from this address: even the right one waits
    assert locked_out.value.status_code == 429
    guard.check(fake_request("s3cret", host="10.0.0.9"))
    now[0] = 61
    guard.check(fake_request("s3cret"))
    assert guard.locked and not KeyGuard("").locked
    KeyGuard("").check(fake_request())


# ------------------------------------------------------------------------------------------------------------------------------
# The Studio against a fake renderer, job runner and LLM server
# ------------------------------------------------------------------------------------------------------------------------------
def job_view(job_id="00000001", avatar="ada", status="queued", **extra):
    """A job as the job runner returns it, including the fields that must not reach the browser."""
    view = {"id": job_id, "avatar": avatar, "title": avatar.title(), "subject": "", "mode": "full", "quality": "hd",
            "image": f"/studio/uploads/{avatar}/source.png", "status": status, "phase": "", "stage": None, "fraction": 0.0,
            "finished_stages": [], "error": None, "hint": None, "attempts": 1, "clean": True, "registration": "r1",
            "log": ["output of the generator"], "created": 1700000000.0, "started": None, "finished": None,
            "stages": [], "progress": 0.0, "eta_s": 780, "queue_ahead": 0}
    view.update(extra)
    return view


class Fakes:
    def __init__(self):
        self.avatars, self.jobs = {}, {}
        self.calls, self.bodies, self.created_jobs = [], {}, []
        self.down, self.slow, self.fail = set(), set(), {}
        self.busy = False
        self.renderer_state = ("ready", "ready")

    def transport(self, service):
        def handle(request):
            path = request.url.path
            self.calls.append((service, request.method, path))
            if request.content:
                self.bodies[(service, request.method, path)] = json.loads(request.content)
            if service in self.down:
                raise httpx.ConnectError("refused")
            if service in self.slow:
                raise httpx.ReadTimeout("too slow")
            failure = self.fail.get((service, request.method, path))
            if failure:
                return httpx.Response(failure[0], json={"detail": failure[1]})
            return getattr(self, service)(request, path)
        return httpx.MockTransport(handle)

    def add_avatar(self, name, mode="full", **extra):
        states = {"full": ["idle", "speaking"], "partial": ["speaking"], "quick": []}[mode]
        self.avatars[name] = {"id": name, "title": name.title(), "subject": "", "frame_size": [704, 1280], "states": states, "mode": mode,
                              "settings": {"expression": 0.6, "head_motion": 0.5}, "size_bytes": 5000, "created": "2025-01-01T00:00:00Z",
                              "updated": "2025-01-02T00:00:00Z", "stamp": 111, "thumbnail": f"/v1/avatars/{name}/thumbnail?v=111", **extra}

    def renderer(self, request, path):
        status, detail = self.renderer_state
        state = {"renderer": {"status": status, "detail": detail}, "busy": self.busy, "registering": None, "loaded": []}
        if path == "/v1/status":
            return httpx.Response(200, json=state)
        if path == "/v1/avatars":
            return httpx.Response(200, json={"avatars": list(self.avatars.values()), **state})
        name, tail = re.fullmatch(r"/v1/avatars/([^/]+)(?:/(\w+))?", path).groups()
        if name not in self.avatars:
            return httpx.Response(404, json={"detail": "Unknown avatar."})
        avatar = self.avatars[name]
        if tail == "thumbnail":
            return httpx.Response(200, content=THUMB, headers={"content-type": "image/jpeg"})
        if tail in ("status", "warm"):
            return httpx.Response(200, json={"id": name, "renderer": status, "loaded": True, "loading": False, "busy": self.busy, "error": None})
        if tail == "settings":
            avatar["settings"].update(json.loads(request.content))
            return httpx.Response(200, json=avatar)
        if request.method == "PATCH":
            avatar.update(json.loads(request.content))
        if request.method == "DELETE":
            del self.avatars[name]
            return httpx.Response(200, json={"deleted": name})
        return httpx.Response(200, json=avatar)

    def worker(self, request, path):
        if path == "/healthz":
            active = sum(j["status"] in ACTIVE for j in self.jobs.values())
            return httpx.Response(200, json={"status": "ok", "active": active, "current": None})
        if path == "/v1/jobs" and request.method == "GET":
            return httpx.Response(200, json={"jobs": list(self.jobs.values())})
        if path == "/v1/jobs":
            body = json.loads(request.content)
            self.created_jobs.append(body)
            job = job_view(f"{len(self.jobs) + 1:08x}", body["avatar"], title=body["title"], mode=body["mode"], quality=body["quality"])
            self.jobs[job["id"]] = job
            return httpx.Response(202, json=job)
        if path.startswith("/v1/avatars/"):
            avatar = path.rsplit("/", 1)[1]
            self.jobs = {k: v for k, v in self.jobs.items() if v["avatar"] != avatar}
            return httpx.Response(200, json={"deleted": avatar})
        job_id, _, action = path[len("/v1/jobs/"):].partition("/")
        if job_id not in self.jobs:
            return httpx.Response(404, json={"detail": "Unknown job."})
        job = self.jobs[job_id]
        if request.method == "DELETE":
            del self.jobs[job_id]
            return httpx.Response(200, json={"deleted": job_id})
        job["status"] = {"cancel": "cancelled", "retry": "queued"}.get(action, job["status"])
        return httpx.Response(200, json=job)

    def llm(self, request, path):
        return httpx.Response(200, content=b"")


def build(tmp_path, key="", **overrides):
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "index.html").write_text("<title>Studio test page</title>", encoding="utf-8")
    (static / "js" / "app.js").write_text("export {};", encoding="utf-8")
    (tmp_path / "studio").mkdir()
    settings = Settings(studio_dir=str(tmp_path / "studio"), control_file=str(tmp_path / "run" / "avatar"), static_dir=str(static), key=key,
                        **overrides)
    fakes = Fakes()
    app = FastAPI()
    instance = studio.mount(app, settings, {name: fakes.transport(name) for name in ("renderer", "worker", "llm")})
    return SimpleNamespace(client=TestClient(app), fakes=fakes, studio=instance, settings=settings, root=tmp_path)


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


def uploads(env):
    return os.path.join(env.settings.studio_dir, "uploads")


def kept(env):
    """The avatars that have a stored upload."""
    return os.listdir(uploads(env)) if os.path.isdir(uploads(env)) else []


def store_photo(env, avatar, data=b"photo"):
    directory = os.path.join(uploads(env), avatar)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "source.png"), "wb") as f:
        f.write(data)
    return directory


def upload(env, name="Ada Lovelace", subject="A mathematician", mode="quick", quality="hd", filename="ada.png", data=None, **kwargs):
    return env.client.post("/api/avatars", files={"image": (filename, encoded() if data is None else data, "image/png")},
                           data={"name": name, "subject": subject, "mode": mode, "quality": quality}, **kwargs)


def mutations(env):
    return [call for call in env.fakes.calls if call[1] != "GET"]


# --- creating -----------------------------------------------------------------------------------------------------------------
def test_a_photo_becomes_a_queued_job_and_a_stored_upload(env):
    response = upload(env)
    assert response.status_code == 202
    body = response.json()
    assert body["id"] == "ada-lovelace"
    assert not {"image", "log", "registration"} & set(body["job"])  # server paths and generator output stay on the server
    assert body["job"]["thumbnail"].startswith("/api/avatars/ada-lovelace/thumbnail")
    directory = os.path.join(uploads(env), "ada-lovelace")
    source = os.path.join(directory, "source.png")
    assert env.fakes.created_jobs == [{"avatar": "ada-lovelace", "image": source, "title": "Ada Lovelace", "subject": "A mathematician",
                                       "mode": "quick", "quality": "hd"}]
    with Image.open(source) as image:
        assert image.format == "PNG" and image.size == (640, 960)
    assert sorted(os.listdir(directory)) == ["source.png", "thumb.jpg"]
    assert upload(env).json()["id"] == "ada-lovelace-2"  # the same name again: the first one is not replaced


def test_an_upload_without_a_name_is_named_after_its_file(env):
    response = upload(env, name="", filename="gulf_man-white.JPG")
    assert response.json()["id"] == "gulf-man-white"
    assert env.fakes.created_jobs[0]["title"] == "Gulf Man White"


@pytest.mark.parametrize("kwargs, message", [
    ({"name": "Hi " + ZH}, "must be written in English"),
    ({"subject": KANA}, "must be written in English"),
    ({"name": "x" * 81}, "longer than 80"),
    ({"mode": "instant"}, "full motion or a quick preview"),
    ({"quality": "ultra"}, "full motion or a quick preview"),
    ({"data": b"hello"}, "not a photo"),
])
def test_bad_input_is_refused_with_a_reason_and_nothing_is_kept(env, kwargs, message):
    response = upload(env, **kwargs)
    assert response.status_code == 422 and message in response.json()["detail"]
    assert env.fakes.created_jobs == []
    assert kept(env) == []


def test_an_upload_without_a_photo_is_refused(env):
    assert env.client.post("/api/avatars", data={"name": "Ada"}).status_code == 422
    assert env.client.post("/api/avatars", files={"image": (None, "just text")}).status_code == 422


def test_an_upload_over_the_size_limit_is_refused_however_it_arrives(tmp_path):
    env = build(tmp_path, max_upload_bytes=1000)
    assert upload(env).status_code == 413  # the photo itself is over the limit
    assert upload(env, data=noise_png()).status_code == 413  # the whole body is, as announced in its headers
    chunks = (b"0" * 100_000 for _ in range(4))  # no announced length: counted as it arrives
    assert env.client.post("/api/avatars", content=chunks, headers={"content-type": "multipart/form-data; boundary=x"}).status_code == 413
    assert env.fakes.created_jobs == []


def test_a_new_avatar_is_refused_when_the_disk_is_nearly_full(tmp_path):
    env = build(tmp_path, min_free_bytes=10**18)
    response = upload(env)
    assert response.status_code == 507 and "out of disk space" in response.json()["detail"]


def test_a_refusal_by_the_job_runner_is_passed_on_and_the_upload_is_removed(env):
    env.fakes.fail[("worker", "POST", "/v1/jobs")] = (409, "An avatar with that id is already being made.")
    response = upload(env)
    assert response.status_code == 409 and "already being made" in response.json()["detail"]
    assert kept(env) == []


@pytest.mark.parametrize("service, text", [("renderer", "The renderer is not reachable."), ("worker", "The job runner is not reachable.")])
def test_an_upload_fails_cleanly_when_a_service_is_down(env, service, text):
    env.fakes.down.add(service)
    response = upload(env)
    assert response.status_code == 503 and response.json()["detail"] == text
    assert kept(env) == []


# --- the library --------------------------------------------------------------------------------------------------------------
def test_the_library_shows_studio_urls_and_what_can_be_done_with_each_avatar(env):
    env.fakes.add_avatar("ada", mode="quick")
    env.fakes.add_avatar("bob", mode="full")
    store_photo(env, "ada")
    env.client.post("/api/stage", json={"avatar": "bob"})
    data = env.client.get("/api/avatars").json()
    by_id = {a["id"]: a for a in data["avatars"]}
    assert by_id["ada"]["thumbnail"] == "/api/avatars/ada/thumbnail?v=111"
    assert by_id["ada"]["upgradable"] is True and by_id["bob"]["upgradable"] is False  # full motion cannot be added to full motion
    assert by_id["bob"]["selected"] is True and by_id["ada"]["selected"] is False and data["stage"] == "bob"
    assert data["renderer"]["status"] == "ready"
    assert env.client.get("/api/avatars/ada").json()["upgradable"] is True


def test_an_upgrade_needs_the_stored_photo(env):
    env.fakes.add_avatar("ada", mode="quick")  # made before the Studio existed: no photo of it is stored
    assert env.client.get("/api/avatars").json()["avatars"][0]["upgradable"] is False


def test_thumbnails_come_from_the_renderer_or_else_from_the_upload(env):
    env.fakes.add_avatar("ada")
    response = env.client.get("/api/avatars/ada/thumbnail")
    assert response.content == THUMB and response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-content-type-options"] == "nosniff"
    directory = store_photo(env, "bob")  # still being made: only the preview of the upload exists
    with open(os.path.join(directory, "thumb.jpg"), "wb") as f:
        f.write(b"preview of the upload")
    assert env.client.get("/api/avatars/bob/thumbnail").content == b"preview of the upload"
    assert env.client.get("/api/avatars/nobody/thumbnail").status_code == 404
    assert env.client.get("/api/avatars/-bad/thumbnail").status_code == 404


# --- changing -----------------------------------------------------------------------------------------------------------------
def test_editing_cleans_the_text_and_passes_on_only_what_was_given(env):
    env.fakes.add_avatar("ada")
    assert env.client.patch("/api/avatars/ada", json={"title": "  Ada\n Lovelace "}).status_code == 200
    assert env.fakes.bodies[("renderer", "PATCH", "/v1/avatars/ada")] == {"title": "Ada Lovelace"}
    assert "must be written in English" in env.client.patch("/api/avatars/ada", json={"title": "Ada " + ZH}).json()["detail"]
    assert "is required" in env.client.patch("/api/avatars/ada", json={"title": "  "}).json()["detail"]
    assert env.client.put("/api/avatars/ada/settings", json={"expression": 0.5}).status_code == 200
    assert env.fakes.bodies[("renderer", "PUT", "/v1/avatars/ada/settings")] == {"expression": 0.5}
    assert env.client.put("/api/avatars/ada/settings", json={"expression": 9}).status_code == 422
    assert env.fakes.avatars["ada"]["title"] == "Ada Lovelace" and env.fakes.avatars["ada"]["settings"]["expression"] == 0.5


def test_upgrade_starts_a_full_motion_job_from_the_stored_photo(env):
    env.fakes.add_avatar("ada", mode="quick", title="Ada L.", subject="A mathematician")
    store_photo(env, "ada")
    response = env.client.post("/api/avatars/ada/upgrade", json={"quality": "standard"})
    assert response.status_code == 202 and response.json()["id"] == "ada"
    assert env.fakes.created_jobs == [{"avatar": "ada", "image": os.path.join(uploads(env), "ada", "source.png"), "title": "Ada L.",
                                       "subject": "A mathematician", "mode": "full", "quality": "standard"}]


def test_an_upgrade_is_refused_when_there_is_nothing_to_add_or_nothing_to_add_it_to(env):
    env.fakes.add_avatar("full", mode="full")
    store_photo(env, "full")
    env.fakes.add_avatar("quick", mode="quick")
    assert env.client.post("/api/avatars/full/upgrade", json={}).status_code == 409
    assert env.client.post("/api/avatars/quick/upgrade", json={}).status_code == 409
    assert env.fakes.created_jobs == []


# --- deleting -----------------------------------------------------------------------------------------------------------------
def test_delete_removes_the_avatar_then_its_files_and_clears_the_selection(env):
    env.fakes.add_avatar("ada")
    env.client.post("/api/stage", json={"avatar": "ada"})
    assert env.client.delete("/api/avatars/ada").json() == {"deleted": "ada", "cleaned": True}
    assert mutations(env)[-2:] == [("renderer", "DELETE", "/v1/avatars/ada"), ("worker", "DELETE", "/v1/avatars/ada")]
    assert not os.path.exists(env.settings.control_file) and env.client.get("/api/stage").json() == {"avatar": ""}


def test_delete_is_refused_while_the_avatar_is_being_made(env):
    env.fakes.add_avatar("ada")
    env.fakes.jobs["00000001"] = job_view(avatar="ada", status="generating")
    response = env.client.delete("/api/avatars/ada")
    assert response.status_code == 409 and "still being made" in response.json()["detail"]
    assert "ada" in env.fakes.avatars and mutations(env) == []


def test_delete_passes_on_the_renderers_refusal_and_keeps_the_files(env):
    env.fakes.add_avatar("ada")
    env.client.post("/api/stage", json={"avatar": "ada"})
    env.fakes.fail[("renderer", "DELETE", "/v1/avatars/ada")] = (409, "A conversation is running with this avatar.")
    response = env.client.delete("/api/avatars/ada")
    assert response.status_code == 409 and "conversation is running" in response.json()["detail"]
    assert ("worker", "DELETE", "/v1/avatars/ada") not in env.fakes.calls
    assert env.client.get("/api/stage").json() == {"avatar": "ada"}


def test_delete_says_when_the_files_could_not_be_removed(env):
    env.fakes.add_avatar("ada")
    env.fakes.down.add("worker")
    assert env.client.delete("/api/avatars/ada").json() == {"deleted": "ada", "cleaned": False}
    assert "ada" not in env.fakes.avatars


# --- talking to an avatar -----------------------------------------------------------------------------------------------------
def test_choosing_an_avatar_writes_the_control_file_whole_and_only_for_real_avatars(env):
    env.fakes.add_avatar("ada")
    assert env.client.post("/api/stage", json={"avatar": "ada"}).json() == {"avatar": "ada"}
    control = env.settings.control_file
    with open(control, encoding="utf-8") as f:
        assert f.read() == "ada\n"
    assert os.listdir(os.path.dirname(control)) == ["avatar"]  # no temporary file left behind
    assert env.client.get("/api/stage").json() == {"avatar": "ada"}
    assert env.client.post("/api/stage", json={"avatar": "nobody"}).status_code == 404
    assert env.client.post("/api/stage", json={"avatar": "../etc/passwd"}).status_code == 404
    with open(control, encoding="utf-8") as f:
        assert f.read() == "ada\n"


def test_the_stage_page_can_ask_whether_an_avatar_is_loaded_and_warm_it_up(env):
    env.fakes.add_avatar("ada")
    assert env.client.get("/api/avatars/ada/status").json()["loaded"] is True
    assert env.client.post("/api/avatars/ada/warm").json()["id"] == "ada"
    assert env.client.get("/api/avatars/nobody/status").status_code == 404


# --- health and jobs ----------------------------------------------------------------------------------------------------------
def test_health_reports_each_service_and_the_disk(env):
    health = env.client.get("/api/health").json()
    assert health["renderer"] == {"state": "ready", "detail": "ready", "busy": False, "registering": None, "loaded": []}
    assert health["worker"] == {"state": "ready", "active": 0, "current": None}
    assert health["llm"] == {"state": "ready"} and health["locked"] is False and health["stage"] == ""
    assert health["disk"]["free_bytes"] > 0 and health["disk"]["low"] is False and health["disk"]["full"] is False


def test_health_when_services_are_down_or_still_loading(env):
    env.fakes.renderer_state = ("loading", "Loading the models")
    assert env.client.get("/api/health").json()["renderer"]["state"] == "loading"
    env.fakes.down |= {"renderer", "worker", "llm"}
    env.studio.health_cache = (0.0, None)  # the answer is kept for a moment
    health = env.client.get("/api/health").json()
    assert health["renderer"] == {"state": "offline"} and health["worker"] == {"state": "offline"} and health["llm"] == {"state": "offline"}


def test_a_failing_or_slow_service_gives_the_browser_a_clear_error(env):
    env.fakes.slow.add("renderer")
    response = env.client.get("/api/avatars")
    assert response.status_code == 504 and "did not answer in time" in response.json()["detail"]
    env.fakes.slow.clear()
    env.fakes.fail[("renderer", "GET", "/v1/avatars")] = (500, "Traceback: the secret details of a crash")
    response = env.client.get("/api/avatars")
    assert response.status_code == 502 and "secret" not in response.text  # the service's own error text is not passed on


def test_jobs_are_listed_without_server_paths_or_generator_output(env):
    env.fakes.jobs["00000001"] = job_view(avatar="ada", status="generating")
    data = env.client.get("/api/jobs").json()
    job = data["jobs"][0]
    assert data["worker"] == "ready" and job["status"] == "generating" and job["eta_s"] == 780
    assert not {"image", "log", "registration"} & set(job)
    assert job["thumbnail"].startswith("/api/avatars/ada/thumbnail?v=")
    env.fakes.down.add("worker")
    assert env.client.get("/api/jobs").json() == {"worker": "offline", "jobs": []}


def test_jobs_can_be_cancelled_retried_and_removed(env):
    env.fakes.jobs["00000001"] = job_view("00000001", "ada", "generating")
    assert env.client.post("/api/jobs/00000001/cancel").json()["status"] == "cancelled"
    assert env.client.post("/api/jobs/00000001/retry", json={"quality": "standard"}).json()["status"] == "queued"
    assert env.fakes.bodies[("worker", "POST", "/v1/jobs/00000001/retry")] == {"quality": "standard"}
    assert env.client.post("/api/jobs/not-a-job/cancel").status_code == 404
    # an avatar that was never made leaves nothing behind (its upload goes too); a failed upgrade keeps the avatar and its photo
    env.fakes.jobs["00000001"]["status"] = "failed"
    assert env.client.delete("/api/jobs/00000001").json() == {"deleted": "00000001"}
    assert ("worker", "DELETE", "/v1/avatars/ada") in env.fakes.calls
    env.fakes.add_avatar("bob")
    env.fakes.jobs["00000002"] = job_view("00000002", "bob", "failed")
    env.client.delete("/api/jobs/00000002")
    assert ("worker", "DELETE", "/v1/jobs/00000002") in env.fakes.calls and ("worker", "DELETE", "/v1/avatars/bob") not in env.fakes.calls


# --- who may do what ----------------------------------------------------------------------------------------------------------
def test_changes_need_the_key_when_one_is_set_but_talking_and_reading_do_not(tmp_path):
    env = build(tmp_path, key="s3cret")
    env.fakes.add_avatar("ada")
    assert env.client.get("/api/avatars").status_code == 200
    assert env.client.post("/api/stage", json={"avatar": "ada"}).status_code == 200
    refused = [upload(env), env.client.delete("/api/avatars/ada"), env.client.patch("/api/avatars/ada", json={"title": "X"}),
               env.client.put("/api/avatars/ada/settings", json={"expression": 0.5}), env.client.post("/api/avatars/ada/upgrade", json={}),
               env.client.post("/api/jobs/00000001/cancel"), env.client.post("/api/jobs/00000001/retry", json={}),
               env.client.delete("/api/jobs/00000001"), upload(env, headers={"X-Studio-Key": "wrong"})]
    assert [r.status_code for r in refused] == [401] * len(refused)
    assert mutations(env) == [] and "ada" in env.fakes.avatars  # not even the first step reached another service
    key = {"X-Studio-Key": "s3cret"}
    assert env.client.post("/api/unlock", headers=key).json() == {"ok": True, "locked": True}
    assert env.client.post("/api/unlock").status_code == 401
    assert env.client.get("/api/health").json()["locked"] is True
    assert upload(env, headers=key).status_code == 202
    assert env.client.delete("/api/avatars/ada", headers=key).status_code == 200


def test_other_sites_cannot_change_anything(env):
    env.fakes.add_avatar("ada")
    for headers in ({"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}):
        assert env.client.delete("/api/avatars/ada", headers=headers).status_code == 403
        assert upload(env, headers=headers).status_code == 403
    assert "ada" in env.fakes.avatars and env.fakes.created_jobs == []
    assert env.client.get("/api/avatars", headers={"Origin": "https://evil.example"}).status_code == 200  # reading is never refused
    own = {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}  # what the page itself sends
    assert env.client.post("/api/stage", json={"avatar": "ada"}, headers=own).status_code == 200


# --- the pages ----------------------------------------------------------------------------------------------------------------
def test_the_pages_are_served_with_strict_headers(env):
    for path in ("/", "/a/ada", "/a/anything-at-all"):
        response = env.client.get(path)
        assert response.status_code == 200 and "Studio test page" in response.text
        policy = response.headers["content-security-policy"]
        assert "default-src 'none'" in policy and "script-src 'self'" in policy and "unsafe" not in policy
        assert response.headers["x-content-type-options"] == "nosniff" and response.headers["x-frame-options"] == "DENY"
        assert response.headers["cache-control"] == "no-cache" and "microphone=(self)" in response.headers["permissions-policy"]
    script = env.client.get("/ui/js/app.js")
    assert script.status_code == 200 and script.headers["cache-control"] == "no-cache"
    assert env.client.get("/ui/js/missing.js").status_code == 404
    assert env.client.get("/ui/%2e%2e/index.html").status_code in (400, 404)


# ------------------------------------------------------------------------------------------------------------------------------
# Taking the place of the stock page inside the real engine
# ------------------------------------------------------------------------------------------------------------------------------
def test_attach_puts_the_studio_in_front_of_the_stock_page_and_keeps_the_engine_endpoint(tmp_path, monkeypatch):
    import gradio
    from service.frontend_service import register_frontend as stock

    monkeypatch.setenv("KKA_STUDIO_DIR", str(tmp_path / "studio"))
    monkeypatch.setenv("KKA_AVATAR_CONTROL", str(tmp_path / "avatar"))
    client_module = SimpleNamespace(register_frontend=stock)
    studio.attach(client_module)
    assert client_module.register_frontend is not stock
    app = FastAPI()
    # called the way the engine's RTC client calls it
    client_module.register_frontend(app=app, ui=gradio.Blocks(), parent_block=None, init_config=lambda: {"rtc_configuration": None, "marker": 7})
    http = TestClient(app)
    root = http.get("/", follow_redirects=False)
    assert root.status_code == 200 and "Avatar Studio" in root.text  # the Studio, not the stock page's redirect
    assert http.get("/a/ada").text == root.text
    assert http.get("/ui/index.html").text == root.text
    assert http.get("/openavatarchat/initconfig").json() == {"rtc_configuration": None, "marker": 7}  # still served for the WebRTC client
    assert http.get("/api/stage").json() == {"avatar": ""}


# ------------------------------------------------------------------------------------------------------------------------------
# What the shipped files contain
# ------------------------------------------------------------------------------------------------------------------------------
def shipped_files(*suffixes):
    for directory, _, names in os.walk(os.path.join(GATEWAY, "studio")):
        for name in names:
            if name.endswith(suffixes):
                yield os.path.join(directory, name)


def test_the_studio_is_written_entirely_in_english():
    files = list(shipped_files(".py", ".html", ".css", ".js", ".svg", ".json"))
    assert len(files) > 10
    for path in files:
        with open(path, encoding="utf-8") as f:
            assert not photos.CJK.search(f.read()), path


def test_the_page_loads_nothing_from_other_sites():
    namespaces = ("http://www.w3.org/2000/svg",)  # an XML namespace name, not an address that is fetched
    for path in shipped_files(".html", ".css", ".js", ".svg"):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        found = [u for u in re.findall(r"https?://[^\s\"'`)]+", text) if not u.startswith(namespaces)]
        assert not found, (path, found)
