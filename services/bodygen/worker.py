"""Avatar generation worker: Stage A on demand, for the Studio.

Runs in the kka-bodygen image next to the live stack and handles one job at a time:

    queued -> generating -> waiting -> registering -> ready          (failed and cancelled can end a job early)

    generating   generate_loops.py makes the idle and speaking body loops from the photo. It runs as a subprocess, so the GPU
                 memory it needs (26-30 GB) is given back when it exits: an idle worker holds none.
    waiting      the GPU is short of memory, or the renderer is busy with a conversation; tried again until it frees up.
    registering  the renderer registers the photo and the clips as an avatar (POST /v1/registrations on the renderer).

A "quick" job skips the first step: the photo alone is registered (still image plus live face, no body video).
Jobs are small JSON files in $KKA_STUDIO_DIR/jobs and survive a restart.

    POST   /v1/jobs                {avatar, image, title, subject, mode: full|quick, quality: hd|standard}      -> 202
    GET    /v1/jobs                every job, newest first
    GET    /v1/jobs/{id}
    POST   /v1/jobs/{id}/cancel
    POST   /v1/jobs/{id}/retry     {quality?}   a failed or cancelled job again, e.g. at standard quality after an out-of-memory
    DELETE /v1/jobs/{id}           a finished job
    DELETE /v1/avatars/{avatar}    what was made for an avatar (clips, uploaded photo, jobs); 409 while one of its jobs is active
    GET    /healthz

There is no authentication: the port is only reachable on the compose network, behind the Studio in the gateway.
"""
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests
from PIL import Image

log = logging.getLogger("kka.worker")

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
JOB_RE = re.compile(r"^[0-9a-f]{8}$")

STUDIO = os.environ.get("KKA_STUDIO_DIR", "/studio")
LOOPS = os.environ.get("KKA_LOOPS_DIR", "/out")
RENDERER = os.environ.get("KKA_RENDERER_URL", "http://renderer:8290")
RENDERER_LOOPS = os.environ.get("KKA_RENDERER_LOOPS", "/loops")      # where the renderer sees LOOPS
RENDERER_STUDIO = os.environ.get("KKA_RENDERER_STUDIO", "/studio")   # ... and STUDIO
SPEECH = os.environ.get("KKA_SPEECH", "/speech/sample_en.wav")
GENERATOR = os.environ.get("KKA_GENERATOR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_loops.py"))

ACTIVE = ("queued", "generating", "waiting", "registering")
CLIP_STATES = ("idle", "speaking")
STEPS = 8  # sampling steps of generate_loops.py; its progress bar counts to this
# sample_area: --sample-area of generate_loops.py; clip_s: seconds per clip; gpu_gb: free GPU memory needed (peak 25.9 / 22.7 GB + headroom)
QUALITY = {"hd": {"sample_area": 960, "clip_s": 330, "gpu_gb": 30}, "standard": {"sample_area": 768, "clip_s": 165, "gpu_gb": 26}}
LOAD_S, STILL_S, LOOP_S = 125, 25, 50  # measured: loading the models, registering the photo, registering one clip
MAX_ACTIVE, MAX_KEPT = 6, 40
MAX_IMAGE_BYTES = 30 * 2**20
WAIT_LIMIT_S = 6 * 3600
GPU_POLL_S = 15
POLL_S = float(os.environ.get("KKA_POLL_S", "5"))

STEP_RE = re.compile(r"\b(\d+)/(\d+) \[")
CLIP_RE = re.compile(r"kka\.bodygen: (?P<avatar>[a-z0-9-]+)/(?P<state>\w+): (?P<what>generating|[\d.]+ s,)")
OOM_RE = re.compile(r"out of GPU memory")
LOG_RE = re.compile(r" (INFO|WARNING|ERROR|CRITICAL) |Error|Traceback|Exception")  # lines worth showing when something goes wrong
CLIP_PHASE = {"idle": "Making the idle loop", "speaking": "Making the speaking gestures"}
_CLEAN = {**dict.fromkeys([*range(0x20), *range(0x7F, 0xA0)], " "), **dict.fromkeys([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])}


class JobError(Exception):
    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint


class Cancelled(Exception):
    pass


class Refused(Exception):
    """A request that cannot be served: (HTTP status, message)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def text(value, limit):
    return " ".join(str(value or "").translate(_CLEAN).split())[:limit]


# ----------------------------------------------------------------------------------------------------------------------------
# Plan, progress and the generator's output
# ----------------------------------------------------------------------------------------------------------------------------
def stages(job):
    """[(key, label, estimated seconds)] of the stages a job goes through."""
    if job["mode"] != "full":
        return [("register", "Preparing the live face", STILL_S)]
    clip = QUALITY[job["quality"]]["clip_s"]
    return [("load", "Loading the motion model", LOAD_S), ("idle", "Idle loop: breathing and posture", clip),
            ("speaking", "Speaking gestures", clip), ("register", "Preparing the live face", STILL_S + LOOP_S * len(CLIP_STATES))]


def view(job, now=None, ahead=0):
    """The job as the API returns it: the stored record plus the checklist, overall progress and the time still needed."""
    now = now or time.time()
    plan = stages(job)
    total = sum(secs for _, _, secs in plan)
    ready, running = job["status"] == "ready", job["status"] in ("generating", "waiting", "registering")
    done, rows = 0.0, []
    for key, label, secs in plan:
        if ready or key in job["finished_stages"]:
            state, fraction = "done", 1.0
        elif key == job.get("stage") and running:
            state, fraction = "running", job.get("fraction", 0.0)
            if key == "load":  # no progress output while the models load: assume the usual time
                fraction = max(fraction, min(0.95, (now - job.get("stage_started", now)) / secs))
        elif key == job.get("stage") and job["status"] == "failed":
            state, fraction = "failed", job.get("fraction", 0.0)
        else:
            state, fraction = "pending", 0.0
        done += secs * fraction
        rows.append({"key": key, "label": label, "state": state, "fraction": round(fraction, 3)})
    finished = job["status"] in ("ready", "failed", "cancelled")
    return {**job, "stages": rows, "progress": 1.0 if ready else round(done / total, 3),
            "eta_s": 0 if finished else max(0, round(total - done)), "queue_ahead": ahead if job["status"] == "queued" else 0}


def parse_line(line, job, steps=STEPS):
    """What one line of generate_loops.py output says about the job: a dict of changes (empty when it is not about progress)."""
    clip = CLIP_RE.search(line)
    if clip and clip["avatar"] == job["avatar"]:
        state = clip["state"]
        if clip["what"] == "generating":
            return {"stage": state, "fraction": 0.0, "stage_started": time.time(), "finished_stages": sorted({*job["finished_stages"], "load"})}
        return {"stage": state, "fraction": 1.0, "finished_stages": sorted({*job["finished_stages"], state})}
    if job.get("stage") in CLIP_STATES:
        step = STEP_RE.search(line)
        if step and int(step[2]) == steps:
            return {"fraction": min(1.0, int(step[1]) / steps)}
    return {}


def complete(directory, state):
    """A clip counts when generate_loops.py got as far as writing its report, the last file it writes."""
    return all(os.path.isfile(os.path.join(directory, f"{state}.{ext}")) for ext in ("mp4", "json"))


def clean_incomplete(directory):
    """Remove clips whose report is missing (an interrupted run): generate_loops.py would skip them as done."""
    for state in CLIP_STATES:
        path = os.path.join(directory, f"{state}.mp4")
        if os.path.isfile(path) and not complete(directory, state):
            os.remove(path)


def free_gpu_gb():
    """Free memory of the visible GPU in GB, None when it cannot be told (then the check is skipped)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
        return float(out.split()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def upload_path(path):
    """The real path of an uploaded photo (a file below $STUDIO/uploads); None for anything else."""
    real, base = os.path.realpath(path), os.path.realpath(os.path.join(STUDIO, "uploads"))
    if os.path.commonpath([real, base]) != base or not os.path.isfile(real):
        return None
    return real


# ----------------------------------------------------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------------------------------------------------
class Worker:
    def __init__(self, directory=None):
        self.directory = os.path.join(directory or STUDIO, "jobs")
        os.makedirs(self.directory, exist_ok=True)
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.jobs, self.flags = {}, {}
        self.current = None
        self._load()

    # --- storage
    def _path(self, job_id):
        return os.path.join(self.directory, f"{job_id}.json")

    def _save(self, job):
        tmp = self._path(job["id"]) + f".{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(job, f, indent=1)
        os.replace(tmp, self._path(job["id"]))

    def _load(self):
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.directory, name)) as f:
                    job = json.load(f)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                log.warning("ignoring unreadable job file %s", name)
        for job in self.jobs.values():
            if job["status"] in ACTIVE:  # the worker stopped while it was running
                if job.get("attempts", 0) >= 2:
                    job.update(status="failed", finished=time.time(), error="The worker stopped twice while making this avatar.",
                               hint="Retry it; if it happens again check the worker's log.")
                else:
                    job.update(status="queued", phase="Waiting in the queue (restarted)", stage=None, fraction=0.0, registration=None)
                self._save(job)

    def update(self, job, **fields):
        with self.lock:
            job.update(fields)
            self._save(job)

    def _prune(self):
        with self.lock:
            done = sorted((j for j in self.jobs.values() if j["status"] not in ACTIVE), key=lambda j: j["created"])
            for job in done[:-MAX_KEPT]:
                self._forget(job)

    def _forget(self, job):
        self.jobs.pop(job["id"], None)
        try:
            os.remove(self._path(job["id"]))
        except FileNotFoundError:
            pass

    # --- requests
    def create(self, body):
        avatar = str(body.get("avatar", ""))
        if not ID_RE.match(avatar):
            raise Refused(422, "The avatar id must be 1-40 characters: lower-case letters, digits and hyphens.")
        mode, quality = body.get("mode", "full"), body.get("quality", "hd")
        if mode not in ("full", "quick") or quality not in QUALITY:
            raise Refused(422, "mode is full or quick, quality is hd or standard.")
        image = upload_path(str(body.get("image", "")))
        if image is None:
            raise Refused(422, "The image is not an uploaded photo.")
        try:
            if os.path.getsize(image) > MAX_IMAGE_BYTES:
                raise Refused(422, "The photo is larger than 30 MB.")
            with Image.open(image) as img:
                img.verify()
        except (OSError, SyntaxError, Image.DecompressionBombError):
            raise Refused(422, "The photo cannot be read as an image.")
        title = text(body.get("title"), 80)
        with self.lock:
            active = [j for j in self.jobs.values() if j["status"] in ACTIVE]
            if any(j["avatar"] == avatar for j in active):
                raise Refused(409, f"An avatar named {avatar} is already being made.")
            if len(active) >= MAX_ACTIVE:
                raise Refused(429, "Too many avatars are being made; wait for one to finish.")
            job = {"id": uuid.uuid4().hex[:8], "avatar": avatar, "title": title or avatar.replace("-", " ").title(),
                   "subject": text(body.get("subject"), 600), "mode": mode, "quality": quality, "image": image,
                   "status": "queued", "phase": "Waiting in the queue", "stage": None, "fraction": 0.0, "finished_stages": [],
                   "error": None, "hint": None, "attempts": 0, "clean": True, "registration": None, "log": [],
                   "created": time.time(), "started": None, "finished": None}
            self.jobs[job["id"]] = job
            self._save(job)
            self._prune()
            self.wake.notify_all()
            return self.view(job)

    def view(self, job):
        with self.lock:
            ahead = sum(1 for j in self.jobs.values() if j["status"] == "queued" and j["created"] < job["created"]) + (1 if self.current else 0)
            return view(job, ahead=ahead)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise Refused(404, "Unknown job.")
            return job

    def listing(self):
        with self.lock:
            return [self.view(j) for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)]

    def health(self):
        with self.lock:
            return {"status": "ok", "active": sum(1 for j in self.jobs.values() if j["status"] in ACTIVE), "current": self.current}

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] not in ACTIVE:
                raise Refused(409, "The job has finished.")
            if job.get("registration"):
                raise Refused(409, "The avatar is being finalised; this takes about a minute and cannot be cancelled.")
            self.flags.setdefault(job_id, threading.Event()).set()
            if job["status"] == "queued":
                self.update(job, status="cancelled", phase="Cancelled", finished=time.time())
            return self.view(job)

    def retry(self, job_id, body):
        with self.lock:
            job = self.get(job_id)
            if job["status"] not in ("failed", "cancelled"):
                raise Refused(409, "Only a failed or cancelled job can be retried.")
            if any(j["avatar"] == job["avatar"] and j["status"] in ACTIVE for j in self.jobs.values()):
                raise Refused(409, f"An avatar named {job['avatar']} is already being made.")
            quality = body.get("quality", job["quality"])
            if quality not in QUALITY:
                raise Refused(422, "quality is hd or standard.")
            if upload_path(job["image"]) is None:
                raise Refused(409, "The uploaded photo is gone; upload it again.")
            self.flags.pop(job_id, None)
            # clips made at another resolution cannot be combined with new ones
            self.update(job, status="queued", phase="Waiting in the queue", stage=None, fraction=0.0, finished_stages=[], error=None, hint=None,
                        attempts=0, registration=None, quality=quality, clean=quality != job["quality"] or job["clean"],
                        created=time.time(), started=None, finished=None)
            self.wake.notify_all()
            return self.view(job)

    def remove(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] in ACTIVE:
                raise Refused(409, "Cancel the job first.")
            self._forget(job)
            self.flags.pop(job_id, None)
            return {"deleted": job_id}

    def remove_avatar(self, avatar):
        """Delete what was made for an avatar: clips, the uploaded photo, finished jobs."""
        if not ID_RE.match(avatar):
            raise Refused(422, "Invalid avatar id.")
        with self.lock:
            if any(j["avatar"] == avatar and j["status"] in ACTIVE for j in self.jobs.values()):
                raise Refused(409, "The avatar is still being made; cancel that first.")
            for job in [j for j in self.jobs.values() if j["avatar"] == avatar]:
                self._forget(job)
        for base in (os.path.join(LOOPS, avatar), os.path.join(STUDIO, "uploads", avatar)):
            shutil.rmtree(base, ignore_errors=True)
        return {"deleted": avatar}

    # --- running
    def next_job(self):
        with self.lock:
            while True:
                queued = sorted((j for j in self.jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
                if queued:
                    self.current = queued[0]["id"]
                    return queued[0]
                self.current = None
                self.wake.wait()

    def run_forever(self):
        while True:
            job = self.next_job()
            try:
                self.execute(job)
            finally:
                with self.lock:
                    self.current = None
                self._prune()

    def execute(self, job):
        flag = self.flags.setdefault(job["id"], threading.Event())
        full = job["mode"] == "full"
        log.info("job %s (%s): %s, %s", job["id"], job["avatar"], job["mode"], job["quality"])
        try:
            if flag.is_set():  # cancelled between being picked and starting
                raise Cancelled()
            self.update(job, status="generating" if full else "registering", attempts=job.get("attempts", 0) + 1, started=time.time(),
                        phase="Starting" if full else "Preparing the live face", error=None, hint=None,
                        stage=None if full else "register", stage_started=time.time())
            if full:
                self.generate(job, flag)
            self.register(job, flag)
            self.update(job, status="ready", phase="Ready", stage="register", fraction=1.0, finished=time.time())
            log.info("job %s (%s) is ready after %.0f s", job["id"], job["avatar"], time.time() - job["started"])
        except Cancelled:
            self.update(job, status="cancelled", phase="Cancelled", finished=time.time())
        except JobError as exc:
            log.error("job %s failed: %s", job["id"], exc)
            self.update(job, status="failed", phase="Failed", error=str(exc), hint=exc.hint, finished=time.time())
        except Exception as exc:
            log.exception("job %s failed", job["id"])
            self.update(job, status="failed", phase="Failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())

    def pause(self, flag, seconds):
        """Sleep, but wake on a cancel request (raises Cancelled)."""
        if flag.wait(seconds):
            raise Cancelled()

    def wait_for_gpu(self, job, flag):
        need = QUALITY[job["quality"]]["gpu_gb"]
        t0 = time.time()
        while True:
            free = free_gpu_gb()
            if free is None or free >= need:
                return
            if time.time() - t0 > WAIT_LIMIT_S:
                raise JobError(f"The GPU stayed too busy (needs {need} GB free, has {free:.0f} GB).", "Try again when no other work is running.")
            self.update(job, status="waiting", phase=f"Waiting for GPU memory ({free:.0f} of {need} GB free)")
            self.pause(flag, GPU_POLL_S)

    def command(self, job, todo):
        cmd = [sys.executable, GENERATOR, "--job", f"{job['avatar']}={job['image']}", "--states", ",".join(todo), "--speech", SPEECH,
               "--sample-area", str(QUALITY[job["quality"]]["sample_area"]), "--end-frame", "auto", "--out", LOOPS]
        if job["subject"]:
            cmd += ["--subject", f"{job['avatar']}={job['subject']}"]
        return cmd

    def generate(self, job, flag):
        out = os.path.join(LOOPS, job["avatar"])
        if job.get("clean"):
            shutil.rmtree(out, ignore_errors=True)
            self.update(job, clean=False)
        clean_incomplete(out)
        todo = [s for s in CLIP_STATES if not complete(out, s)]
        have = [s for s in CLIP_STATES if s not in todo]
        if not todo:
            self.update(job, finished_stages=["load", *have])
            return
        self.wait_for_gpu(job, flag)
        self.update(job, status="generating", phase="Loading the motion model", stage="load", fraction=0.0, stage_started=time.time(),
                    finished_stages=have)
        proc = subprocess.Popen(self.command(job, todo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
        threading.Thread(target=self._stop_on_cancel, args=(proc, flag), daemon=True).start()
        oom = False
        try:
            for line in self.lines(proc.stdout):
                oom = oom or bool(OOM_RE.search(line))
                changes = parse_line(line, job)
                if changes.get("stage") in CLIP_STATES:
                    changes["phase"] = CLIP_PHASE[changes["stage"]]
                if changes:
                    if any(job.get(key) != value for key, value in changes.items()):
                        self.update(job, **changes)
                elif LOG_RE.search(line):
                    self.update(job, log=[*job["log"], line.strip()[:300]][-30:])
        except BaseException:
            self.terminate(proc)
            raise
        code = proc.wait()
        clean_incomplete(out)
        if flag.is_set():
            raise Cancelled()
        missing = [s for s in todo if not complete(out, s)]
        if code != 0 or missing:
            if oom:
                raise JobError("The GPU ran out of memory while making the motion clips.",
                               "Retry at Standard quality, or when fewer other programs are using the GPU.")
            raise JobError(f"The motion generator stopped without finishing ({', '.join(missing) or 'exit code ' + str(code)}).",
                           "Retry; the clips that were finished are kept.")

    @staticmethod
    def lines(stream):
        """Lines of a process's output; tqdm separates its updates with carriage returns."""
        buf = b""
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            buf += chunk
            *done, buf = re.split(rb"[\r\n]+", buf)
            for piece in done:
                yield piece.decode("utf-8", "replace")
        if buf:
            yield buf.decode("utf-8", "replace")

    @staticmethod
    def terminate(proc):
        """Stop the generator and anything it started; its GPU memory is free once it has exited."""
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

    @classmethod
    def _stop_on_cancel(cls, proc, flag):
        while proc.poll() is None:
            if flag.wait(1.0):
                cls.terminate(proc)
                return

    def register(self, job, flag):
        avatar = job["avatar"]
        body = {"id": avatar, "image": job["image"].replace(os.path.realpath(STUDIO), RENDERER_STUDIO, 1), "title": job["title"],
                "subject": job["subject"], "loops": {}}
        if job["mode"] == "full":
            body["loops"] = {s: f"{RENDERER_LOOPS}/{avatar}/{s}.mp4" for s in CLIP_STATES}
        t0 = time.time()
        self.update(job, status="waiting", phase="Waiting for the renderer", stage="register", fraction=0.0, stage_started=t0,
                    finished_stages=sorted({*job["finished_stages"], *(["load", *CLIP_STATES] if job["mode"] == "full" else [])}))
        while True:
            if flag.is_set():
                raise Cancelled()
            try:
                answer = requests.post(f"{RENDERER}/v1/registrations", json=body, timeout=30)
            except requests.RequestException as exc:
                answer, why = None, f"the renderer is not reachable ({type(exc).__name__})"
            else:
                why = "a conversation or another registration is running" if answer.status_code == 409 else f"the renderer answered {answer.status_code}"
            if answer is not None and answer.status_code == 202:
                break
            if answer is not None and answer.status_code not in (409, 503):
                raise JobError(f"The renderer refused the avatar: {self.detail(answer)}")
            if time.time() - t0 > WAIT_LIMIT_S:
                raise JobError(f"The renderer stayed unavailable: {why}.", "Retry when no conversation is running.")
            self.update(job, phase=f"Waiting: {why}")
            self.pause(flag, POLL_S)
        rid = answer.json()["id"]
        self.update(job, status="registering", phase="Preparing the live face", registration=rid)
        while True:
            time.sleep(min(POLL_S, 2.0))
            try:
                state = requests.get(f"{RENDERER}/v1/registrations/{rid}", timeout=30).json()
            except (requests.RequestException, ValueError):
                continue  # the renderer is busy rendering; it answers again shortly
            self.update(job, fraction=state["done"] / max(1, state["steps"]), phase=f"Preparing the live face ({state['step']})")
            if state["state"] == "failed":
                raise JobError(f"The renderer could not register the avatar: {state['error']}", "Retry; if it persists check the renderer's log.")
            if state["state"] == "done":
                return

    @staticmethod
    def detail(answer):
        try:
            return str(answer.json().get("detail") or answer.text)[:300]
        except ValueError:
            return answer.text[:300]


# ----------------------------------------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------------------------------------
def make_handler(worker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > 65536:
                raise Refused(413, "The request is too large.")
            if not length:
                return {}
            try:
                data = json.loads(self.rfile.read(length))
            except ValueError:
                raise Refused(400, "The body is not JSON.")
            if not isinstance(data, dict):
                raise Refused(400, "The body must be a JSON object.")
            return data

        def route(self, method, path, data):
            if method == "GET" and path == "/healthz":
                return worker.health()
            if method == "GET" and path == "/v1/jobs":
                return {"jobs": worker.listing()}
            if method == "POST" and path == "/v1/jobs":
                return worker.create(data)
            parts = path.split("/")
            if len(parts) >= 4 and parts[:3] == ["", "v1", "jobs"] and JOB_RE.match(parts[3]):
                job_id, tail = parts[3], parts[4:]
                if method == "GET" and not tail:
                    return worker.view(worker.get(job_id))
                if method == "POST" and tail == ["cancel"]:
                    return worker.cancel(job_id)
                if method == "POST" and tail == ["retry"]:
                    return worker.retry(job_id, data)
                if method == "DELETE" and not tail:
                    return worker.remove(job_id)
            if method == "DELETE" and len(parts) == 4 and parts[:3] == ["", "v1", "avatars"]:
                return worker.remove_avatar(parts[3])
            raise Refused(404, "Not found.")

        def handle_any(self, method):
            path = urlparse(self.path).path.rstrip("/")
            try:
                data = self.body()  # always read it: an unread body would be taken for the next request on this connection
                self.reply(202 if (method, path) == ("POST", "/v1/jobs") else 200, self.route(method, path, data))
            except Refused as exc:
                self.reply(exc.status, {"detail": str(exc)})
            except Exception:
                log.exception("request failed")
                self.reply(500, {"detail": "Internal error."})

        def do_GET(self):
            self.handle_any("GET")

        def do_POST(self):
            self.handle_any("POST")

        def do_DELETE(self):
            self.handle_any("DELETE")

    return Handler


def main():
    logging.basicConfig(level=os.environ.get("KKA_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = Worker()
    threading.Thread(target=worker.run_forever, name="runner", daemon=True).start()
    server = ThreadingHTTPServer((os.environ.get("KKA_WORKER_HOST", "0.0.0.0"), int(os.environ.get("KKA_WORKER_PORT", "8300"))), make_handler(worker))
    server.daemon_threads = True
    log.info("worker listening on %s:%s (studio %s, loops %s)", *server.server_address, STUDIO, LOOPS)
    server.serve_forever()


if __name__ == "__main__":
    main()
