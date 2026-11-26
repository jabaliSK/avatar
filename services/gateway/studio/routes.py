"""The Studio's pages and its API (/api/...).

The browser never talks to the renderer or the job runner and never names a path: every request here is checked, and an avatar is
always referred to by its id. Reading is open to everyone who can open the page; creating, changing and deleting asks for the
Studio key when one is configured.
"""
import asyncio
import os
import re
import shutil
import time
from typing import Literal, NamedTuple, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from .guard import KeyGuard, reject_cross_site
from .photos import MIN_SIDE, InputError, clean_text, prepare_photo, slugify, title_from_filename, unique_id
from .upstream import Upstream

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # what the renderer accepts as an avatar name
JOB = re.compile(r"^[0-9a-f]{8}$")                      # a job id of the worker
ACTIVE = ("queued", "generating", "waiting", "registering")
HIDDEN_JOB_FIELDS = ("image", "log", "registration")    # paths and the generator's output stay on the server
FORM_OVERHEAD = 256 * 1024                              # what a multipart upload adds to the photo itself
HEALTH_TTL_S = 1.5

# The page loads only its own scripts, styles and images; a conversation is WebRTC and same-origin requests. Microphone access is the one
# permission it asks for.
PAGE_HEADERS = {
    "Cache-Control": "no-cache",
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                                "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=(self)",
}


class ProfileBody(BaseModel):
    title: Optional[str] = Field(None, max_length=400)
    subject: Optional[str] = Field(None, max_length=4000)


class SettingsBody(BaseModel):
    expression: Optional[float] = Field(None, ge=0, le=5)
    head_motion: Optional[float] = Field(None, ge=0, le=5)


class StageBody(BaseModel):
    avatar: str = Field(max_length=100)


class UpgradeBody(BaseModel):
    quality: Literal["hd", "standard"] = "hd"


class RetryBody(BaseModel):
    quality: Optional[Literal["hd", "standard"]] = None


class LimitedRequest(Request):
    """A request whose body may not exceed `limit` bytes, checked as the body arrives (not from the announced length alone)."""

    def __init__(self, scope, receive, limit):
        super().__init__(scope, receive)
        self.limit = limit

    async def stream(self):
        received = 0
        async for chunk in super().stream():
            received += len(chunk)
            if received > self.limit:
                raise HTTPException(413, "The upload is too large.")
            yield chunk


class Upload(NamedTuple):
    data: bytes
    filename: str
    name: str
    subject: str
    mode: str
    quality: str


def field(form, key):
    """A text field of a form; empty when it is missing or was sent as a file."""
    value = form.get(key)
    return value if isinstance(value, str) else ""


def valid(avatar_id):
    if not NAME.match(avatar_id):
        raise HTTPException(404, "Unknown avatar.")
    return avatar_id


def valid_job(job_id):
    if not JOB.match(job_id):
        raise HTTPException(404, "Unknown job.")
    return job_id


class Studio:
    def __init__(self, settings, transports=None):
        transports = transports or {}
        self.settings = settings
        self.guard = KeyGuard(settings.key)
        self.renderer = Upstream("renderer", settings.renderer_url, transport=transports.get("renderer"))
        self.worker = Upstream("job runner", settings.worker_url, transport=transports.get("worker"))
        self.llm = Upstream("language model", settings.llm_url, transport=transports.get("llm"))
        self.health_cache = (0.0, None)

    # ------------------------------------------------------------------------------------------------------------ files and state
    def upload_dir(self, avatar):
        return os.path.join(self.settings.studio_dir, "uploads", avatar)

    def source_photo(self, avatar):
        return os.path.join(self.upload_dir(avatar), "source.png")

    def free_bytes(self):
        try:
            return shutil.disk_usage(self.settings.studio_dir).free
        except OSError:
            return shutil.disk_usage("/").free

    def selected(self):
        """The avatar the next conversation uses."""
        try:
            with open(self.settings.control_file, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def select(self, avatar):
        path = self.settings.control_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(avatar + "\n")
        os.replace(tmp, path)  # the avatar handler reads this file when a conversation starts: never a half-written name

    def deselect(self, avatar):
        if self.selected() == avatar:
            try:
                os.remove(self.settings.control_file)
            except OSError:
                pass

    def annotate(self, avatar, selected):
        """An avatar as the library shows it: preview address through the Studio, whether motion can still be added, whether it is selected."""
        name = avatar["id"]
        avatar["thumbnail"] = f"/api/avatars/{name}/thumbnail?v={avatar.get('stamp', 0)}"
        avatar["upgradable"] = avatar.get("mode") != "full" and os.path.isfile(self.source_photo(name))
        avatar["selected"] = name == selected
        return avatar

    @staticmethod
    def public_job(job):
        view = {k: v for k, v in job.items() if k not in HIDDEN_JOB_FIELDS}
        view["thumbnail"] = f"/api/avatars/{job['avatar']}/thumbnail?v={int(job.get('created') or 0)}"
        return view

    async def taken_ids(self):
        """Ids that a new avatar must not take: the renderer's avatars, the job runner's jobs and the stored uploads."""
        taken = {a["id"] for a in (await self.renderer.json("GET", "/v1/avatars"))["avatars"]}
        try:
            taken |= {j["avatar"] for j in (await self.worker.json("GET", "/v1/jobs"))["jobs"]}
        except HTTPException:
            pass  # the request to the job runner that follows fails with a clear message
        uploads = os.path.join(self.settings.studio_dir, "uploads")
        if os.path.isdir(uploads):
            taken |= set(os.listdir(uploads))
        return taken

    def claim(self, base, taken):
        """Create the upload folder of a new avatar under the first free id (mkdir decides when two uploads race)."""
        os.makedirs(os.path.join(self.settings.studio_dir, "uploads"), exist_ok=True)
        while True:
            avatar = unique_id(base, taken)
            try:
                os.mkdir(self.upload_dir(avatar))
                return avatar
            except FileExistsError:
                taken.add(avatar)

    def store(self, avatar, photo):
        directory = self.upload_dir(avatar)
        for name, data in (("thumb.jpg", photo.thumbnail), ("source.png", photo.png)):  # the photo last: its presence means complete
            with open(os.path.join(directory, name + ".part"), "wb") as f:
                f.write(data)
            os.replace(os.path.join(directory, name + ".part"), os.path.join(directory, name))

    def page(self):
        return FileResponse(os.path.join(self.settings.static_dir, "index.html"), headers=PAGE_HEADERS)

    async def read_upload(self, request):
        """The photo and the text fields of an upload. The size limit is applied while the body arrives, not only from its announced length."""
        limit = self.settings.max_upload_bytes + FORM_OVERHEAD
        too_large = HTTPException(413, f"The photo is larger than {self.settings.max_upload_bytes // 2**20} MB.")
        announced = request.headers.get("content-length", "")
        if announced.isdigit() and int(announced) > limit:
            raise too_large
        try:
            form = await LimitedRequest(request.scope, request.receive, limit).form()
        except HTTPException as exc:
            raise (too_large if exc.status_code == 413 else exc) from None
        except Exception:
            logger.exception("Studio: an upload could not be read")
            raise HTTPException(400, "The upload could not be read.") from None
        try:
            image = form.get("image")
            if not isinstance(image, UploadFile):
                raise HTTPException(422, "Choose a photo to upload.")
            data = await image.read(self.settings.max_upload_bytes + 1)
            if len(data) > self.settings.max_upload_bytes:
                raise too_large
            return Upload(data, image.filename or "", *(field(form, key) for key in ("name", "subject", "mode", "quality")))
        finally:
            await form.close()

    # ------------------------------------------------------------------------------------------------------------ health
    async def health(self):
        expires, cached = self.health_cache
        if cached is not None and time.monotonic() < expires:
            return cached
        (r_ok, r_body), (w_ok, w_body), (l_ok, _) = await asyncio.gather(
            self.renderer.probe("/v1/status"), self.worker.probe("/healthz"), self.llm.probe("/health"))
        renderer = {"state": "offline"}
        if r_ok and isinstance(r_body, dict):
            try:
                renderer = {"state": r_body["renderer"]["status"], "detail": r_body["renderer"]["detail"], "busy": bool(r_body["busy"]),
                            "registering": r_body["registering"], "loaded": r_body["loaded"]}
            except (KeyError, TypeError):
                pass
        worker = {"state": "offline"}
        if w_ok and isinstance(w_body, dict):
            worker = {"state": "ready", "active": w_body.get("active", 0), "current": w_body.get("current")}
        free = self.free_bytes()
        health = {
            "renderer": renderer,
            "worker": worker,
            "llm": {"state": "ready" if l_ok else "offline"},
            "disk": {"free_bytes": free, "low": free < self.settings.low_disk_bytes, "full": free < self.settings.min_free_bytes},
            "limits": {"upload_mb": self.settings.max_upload_bytes // 2**20, "min_side": MIN_SIDE},
            "locked": self.guard.locked,
            "stage": self.selected(),
        }
        self.health_cache = (time.monotonic() + HEALTH_TTL_S, health)
        return health

    # ------------------------------------------------------------------------------------------------------------ the routes
    def router(self):
        router = APIRouter(dependencies=[Depends(reject_cross_site)])
        guard = self.guard
        settings = self.settings

        @router.get("/", include_in_schema=False)
        async def library_page():
            return self.page()

        @router.get("/a/{avatar_id}", include_in_schema=False)
        async def stage_page(avatar_id: str):
            return self.page()

        @router.get("/api/health")
        async def health():
            return await self.health()

        @router.post("/api/unlock")
        async def unlock(request: Request):
            guard.check(request)
            return {"ok": True, "locked": guard.locked}

        # --- the library
        @router.get("/api/avatars")
        async def list_avatars():
            data = await self.renderer.json("GET", "/v1/avatars")
            selected = self.selected()
            data["avatars"] = [self.annotate(a, selected) for a in data["avatars"]]
            data["stage"] = selected
            return data

        @router.get("/api/avatars/{avatar_id}")
        async def one_avatar(avatar_id: str):
            avatar = await self.renderer.json("GET", f"/v1/avatars/{valid(avatar_id)}")
            return self.annotate(avatar, self.selected())

        @router.get("/api/avatars/{avatar_id}/thumbnail")
        async def thumbnail(avatar_id: str):
            valid(avatar_id)
            try:
                data = (await self.renderer.call("GET", f"/v1/avatars/{avatar_id}/thumbnail")).content
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                try:  # an avatar that is still being made has only the preview of its upload
                    with open(os.path.join(self.upload_dir(avatar_id), "thumb.jpg"), "rb") as f:
                        data = f.read()
                except OSError:
                    raise exc from None
            return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff"})

        # --- creating
        @router.post("/api/avatars", status_code=202)
        async def create(request: Request):
            guard.check(request)
            upload = await self.read_upload(request)
            mode, quality = upload.mode or "full", upload.quality or "hd"
            if mode not in ("full", "quick") or quality not in ("hd", "standard"):
                raise HTTPException(422, "Choose full motion or a quick preview, and HD or Standard quality.")
            try:
                title = clean_text(upload.name, "The name", 80) or clean_text(title_from_filename(upload.filename), "The name", 80)
                subject = clean_text(upload.subject, "The description", 600)
            except InputError as exc:
                raise HTTPException(422, str(exc)) from None
            if self.free_bytes() < settings.min_free_bytes:
                raise HTTPException(507, "The server is almost out of disk space. Delete an avatar you no longer need and try again.")
            try:
                photo = await run_in_threadpool(prepare_photo, upload.data)
            except InputError as exc:
                raise HTTPException(422, str(exc)) from None
            avatar = self.claim(slugify(title), await self.taken_ids())
            try:
                self.store(avatar, photo)
                job = await self.worker.json("POST", "/v1/jobs", json={
                    "avatar": avatar, "image": self.source_photo(avatar), "title": title, "subject": subject, "mode": mode, "quality": quality})
            except BaseException:
                shutil.rmtree(self.upload_dir(avatar), ignore_errors=True)
                raise
            logger.info(f"Studio: avatar {avatar} queued ({mode}, {quality}, photo {photo.size[0]}x{photo.size[1]})")
            return JSONResponse({"id": avatar, "job": self.public_job(job)}, status_code=202)

        @router.post("/api/avatars/{avatar_id}/upgrade", status_code=202)
        async def upgrade(avatar_id: str, body: UpgradeBody, request: Request):
            guard.check(request)
            info = await self.renderer.json("GET", f"/v1/avatars/{valid(avatar_id)}")
            if info.get("mode") == "full":
                raise HTTPException(409, "This avatar already has full-body motion.")
            if not os.path.isfile(self.source_photo(avatar_id)):
                raise HTTPException(409, "The original photo of this avatar is not stored, so its motion cannot be generated here.")
            job = await self.worker.json("POST", "/v1/jobs", json={
                "avatar": avatar_id, "image": self.source_photo(avatar_id), "title": info.get("title", ""), "subject": info.get("subject", ""),
                "mode": "full", "quality": body.quality})
            logger.info(f"Studio: motion for avatar {avatar_id} queued ({body.quality})")
            return JSONResponse({"id": avatar_id, "job": self.public_job(job)}, status_code=202)

        # --- changing and deleting
        @router.patch("/api/avatars/{avatar_id}")
        async def edit(avatar_id: str, body: ProfileBody, request: Request):
            guard.check(request)
            fields = {}
            try:
                if body.title is not None:
                    fields["title"] = clean_text(body.title, "The name", 80, required=True)
                if body.subject is not None:
                    fields["subject"] = clean_text(body.subject, "The description", 600)
            except InputError as exc:
                raise HTTPException(422, str(exc)) from None
            avatar = await self.renderer.json("PATCH", f"/v1/avatars/{valid(avatar_id)}", json=fields)
            return self.annotate(avatar, self.selected())

        @router.put("/api/avatars/{avatar_id}/settings")
        async def change_settings(avatar_id: str, body: SettingsBody, request: Request):
            guard.check(request)
            return await self.renderer.json("PUT", f"/v1/avatars/{valid(avatar_id)}/settings", json=body.model_dump(exclude_none=True))

        @router.delete("/api/avatars/{avatar_id}")
        async def delete(avatar_id: str, request: Request):
            guard.check(request)
            valid(avatar_id)
            try:
                jobs = (await self.worker.json("GET", "/v1/jobs"))["jobs"]
            except HTTPException:
                jobs = []
            if any(j["avatar"] == avatar_id and j["status"] in ACTIVE for j in jobs):
                raise HTTPException(409, "This avatar is still being made. Cancel that first.")
            await self.renderer.json("DELETE", f"/v1/avatars/{avatar_id}")  # refused while a conversation is running
            cleaned = True
            try:
                await self.worker.json("DELETE", f"/v1/avatars/{avatar_id}")  # its clips, its uploaded photo, its finished jobs
            except HTTPException:
                cleaned = False
            self.deselect(avatar_id)
            logger.info(f"Studio: avatar {avatar_id} deleted (files removed: {cleaned})")
            return {"deleted": avatar_id, "cleaned": cleaned}

        # --- talking to an avatar
        @router.get("/api/stage")
        async def stage():
            return {"avatar": self.selected()}

        @router.post("/api/stage")
        async def select(body: StageBody):
            avatar = valid(body.avatar)
            await self.renderer.json("GET", f"/v1/avatars/{avatar}")  # unknown: 404
            self.select(avatar)
            return {"avatar": avatar}

        @router.post("/api/avatars/{avatar_id}/warm")
        async def warm(avatar_id: str):
            return await self.renderer.json("POST", f"/v1/avatars/{valid(avatar_id)}/warm")

        @router.get("/api/avatars/{avatar_id}/status")
        async def status(avatar_id: str):
            return await self.renderer.json("GET", f"/v1/avatars/{valid(avatar_id)}/status")

        # --- jobs
        @router.get("/api/jobs")
        async def jobs():
            try:
                data = await self.worker.json("GET", "/v1/jobs")
            except HTTPException:
                return {"worker": "offline", "jobs": []}
            return {"worker": "ready", "jobs": [self.public_job(j) for j in data["jobs"]]}

        @router.post("/api/jobs/{job_id}/cancel")
        async def cancel(job_id: str, request: Request):
            guard.check(request)
            return self.public_job(await self.worker.json("POST", f"/v1/jobs/{valid_job(job_id)}/cancel"))

        @router.post("/api/jobs/{job_id}/retry")
        async def retry(job_id: str, body: RetryBody, request: Request):
            guard.check(request)
            return self.public_job(await self.worker.json("POST", f"/v1/jobs/{valid_job(job_id)}/retry", json=body.model_dump(exclude_none=True)))

        @router.delete("/api/jobs/{job_id}")
        async def remove_job(job_id: str, request: Request):
            guard.check(request)
            job = await self.worker.json("GET", f"/v1/jobs/{valid_job(job_id)}")
            avatar = job["avatar"]
            try:
                await self.renderer.json("GET", f"/v1/avatars/{avatar}")
                exists = True
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                exists = False
            if exists:  # an upgrade that failed: the avatar and its photo stay
                await self.worker.json("DELETE", f"/v1/jobs/{job_id}")
            else:  # an avatar that was never made: nothing of it is kept
                await self.worker.json("DELETE", f"/v1/avatars/{avatar}")
            return {"deleted": job_id}

        return router
