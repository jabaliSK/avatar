"""The generation worker: progress parsing, job validation, the full job flow against a fake generator and renderer, cancel, retry,
restart recovery and the HTTP API."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from PIL import Image

import worker

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_generator.py")


def job_record(**changes):
    job = {"id": "abcd1234", "avatar": "aisha", "mode": "full", "quality": "hd", "status": "generating", "stage": None, "fraction": 0.0,
           "finished_stages": [], "created": 1.0}
    return {**job, **changes}


# ------------------------------------------------------------------------------------------------------------------ pure logic
def test_progress_follows_the_generators_output():
    job = job_record()
    bar = " 12%|#         | 1/8 [00:36<04:12, 36.08s/it]"
    assert worker.parse_line("Loading checkpoint shards: 100%|####| 3/3 [00:02<00:00, 1.1it/s]", job) == {}  # loading the models is no clip
    assert worker.parse_line(bar, job) == {}
    started = worker.parse_line("2026-11-26 06:52:00,550 INFO kka.bodygen: aisha/idle: generating (8 steps, image 1882x1168, seed 11)", job)
    assert (started["stage"], started["fraction"], started["finished_stages"]) == ("idle", 0.0, ["load"])
    job.update(started)
    assert worker.parse_line(bar, job) == {"fraction": 0.125}
    assert worker.parse_line(" 50%|#####     | 4/8 [00:36<04:12, 36.08s/it]", job) == {"fraction": 0.5}
    assert worker.parse_line(" 50%|#####     | 4/7 [00:36<04:12, 36.08s/it]", job) == {}  # another bar
    done = worker.parse_line("INFO kka.bodygen: aisha/idle: 318 s, peak 25.9 GB, {'first_last_mae': 4.8}", job)
    assert (done["stage"], done["fraction"], done["finished_stages"]) == ("idle", 1.0, ["idle", "load"])
    assert worker.parse_line("INFO kka.bodygen: someone-else/idle: generating (8 steps", job) == {}  # not this job
    assert worker.parse_line("ERROR kka.bodygen: aisha/speaking: out of GPU memory; lower --sample-area", job) == {}


def test_the_checklist_progress_and_time_left_move_one_way():
    now = time.time()
    queued = worker.view(job_record(status="queued"), now=now, ahead=2)
    assert (queued["progress"], queued["queue_ahead"], [s["state"] for s in queued["stages"]]) == (0.0, 2, ["pending"] * 4)
    assert queued["eta_s"] == worker.LOAD_S + 2 * worker.QUALITY["hd"]["clip_s"] + worker.STILL_S + 2 * worker.LOOP_S
    seen = []
    for stage, fraction, finished in (("load", 0.5, []), ("idle", 0.0, ["load"]), ("idle", 0.5, ["load"]), ("speaking", 0.5, ["load", "idle"]),
                                      ("register", 0.5, ["load", "idle", "speaking"])):
        v = worker.view(job_record(stage=stage, fraction=fraction, finished_stages=finished, stage_started=now), now=now)
        seen.append((v["progress"], v["eta_s"]))
    assert [p for p, _ in seen] == sorted(p for p, _ in seen) and [e for _, e in seen] == sorted((e for _, e in seen), reverse=True)
    ready = worker.view(job_record(status="ready"))
    assert (ready["progress"], ready["eta_s"], {s["state"] for s in ready["stages"]}) == (1.0, 0, {"done"})
    assert [s["key"] for s in worker.view(job_record(mode="quick", status="queued"))["stages"]] == ["register"]
    failed = worker.view(job_record(status="failed", stage="speaking", fraction=0.3, finished_stages=["load", "idle"]))
    assert [s["state"] for s in failed["stages"]] == ["done", "done", "failed", "pending"] and failed["eta_s"] == 0


def test_clips_without_their_report_are_not_complete(tmp_path):
    for name in ("idle.mp4", "idle.json", "speaking.mp4"):
        (tmp_path / name).write_bytes(b"x")
    assert worker.complete(tmp_path, "idle") and not worker.complete(tmp_path, "speaking")
    worker.clean_incomplete(tmp_path)
    assert sorted(os.listdir(tmp_path)) == ["idle.json", "idle.mp4"]


# ------------------------------------------------------------------------------------------------------------------ the job flow
class FakeRenderer:
    """The renderer's registration endpoints: refuses (409) `busy` times, then accepts and finishes after a few polls."""

    def __init__(self):
        self.busy, self.posts, self.polls, self.fail = 0, [], 0, None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                outer.posts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if outer.busy > 0:
                    outer.busy -= 1
                    self.send(409, {"detail": "busy"})
                else:
                    self.send(202, {"id": "r1"})

            def do_GET(self):
                outer.polls += 1
                end = "failed" if outer.fail else "done"
                self.send(200, {"id": "r1", "state": end if outer.polls >= 3 else "running", "step": "idle", "done": min(outer.polls, 3),
                                "steps": 3, "error": outer.fail})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.server.server_port


class Env:
    def __init__(self, tmp_path, monkeypatch):
        self.studio, self.loops, self.args_log = tmp_path / "studio", tmp_path / "loops", tmp_path / "args.jsonl"
        (self.studio / "uploads").mkdir(parents=True)
        self.loops.mkdir()
        self.renderer = FakeRenderer()
        for name, value in (("STUDIO", str(self.studio)), ("LOOPS", str(self.loops)), ("GENERATOR", FAKE), ("SPEECH", "/dev/null"),
                            ("POLL_S", 0.02), ("GPU_POLL_S", 0.02), ("RENDERER", self.renderer.url), ("free_gpu_gb", lambda: None)):
            monkeypatch.setattr(worker, name, value)
        monkeypatch.setenv("FAKE_ARGS_LOG", str(self.args_log))
        monkeypatch.setenv("FAKE_MODE", "ok")
        self.monkeypatch = monkeypatch
        self.worker = worker.Worker(str(self.studio))

    def start(self):
        threading.Thread(target=self.worker.run_forever, daemon=True).start()
        return self

    def photo(self, avatar):
        path = self.studio / "uploads" / avatar / "source.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (120, 90, 60)).save(path)
        return str(path)

    def create(self, avatar="aisha", **body):
        return self.worker.create({"avatar": avatar, "image": self.photo(avatar), **body})

    def wait(self, job_id, *statuses, timeout=20.0):
        end = time.time() + timeout
        while True:
            job = self.worker.view(self.worker.get(job_id))
            if job["status"] in statuses:
                return job
            assert time.time() < end, f"timed out in {job['status']}: {job['phase']}"
            time.sleep(0.02)

    def runs(self):
        return [json.loads(line) for line in self.args_log.read_text().splitlines()] if self.args_log.exists() else []

    def mode(self, value):
        self.monkeypatch.setenv("FAKE_MODE", value)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def test_a_full_job_makes_the_clips_then_registers_the_avatar(env):
    env.start()
    job = env.create(title="  Aisha\tKhan ", subject="A woman in a blue jacket")
    assert (job["status"], job["title"], job["mode"], job["quality"]) == ("queued", "Aisha Khan", "full", "hd")
    done = env.wait(job["id"], "ready", "failed")
    assert done["status"] == "ready", done["error"]
    assert [s["state"] for s in done["stages"]] == ["done"] * 4 and (done["progress"], done["eta_s"]) == (1.0, 0)
    assert all(worker.complete(env.loops / "aisha", s) for s in ("idle", "speaking"))
    (run,) = env.runs()
    assert (run["states"], run["sample_area"], run["end_frame"], run["subject"]) == ("idle,speaking", 960, "auto", ["aisha=A woman in a blue jacket"])
    (post,) = env.renderer.posts
    assert post == {"id": "aisha", "image": "/studio/uploads/aisha/source.png", "title": "Aisha Khan", "subject": "A woman in a blue jacket",
                    "loops": {"idle": "/loops/aisha/idle.mp4", "speaking": "/loops/aisha/speaking.mp4"}}
    assert any("models ready" in line for line in done["log"])  # progress lines are kept for diagnosis,
    assert not any("Namespace" in line or "checkpoint shards" in line for line in done["log"]) and all(len(l) <= 300 for l in done["log"])  # the noise is not


def test_a_quick_job_registers_the_photo_alone(env):
    env.start()
    done = env.wait(env.create("quick-one", mode="quick")["id"], "ready", "failed")
    assert done["status"] == "ready" and [s["key"] for s in done["stages"]] == ["register"]
    assert env.runs() == [] and env.renderer.posts[0]["loops"] == {}


def test_a_job_waits_while_the_renderer_is_busy(env):
    env.renderer.busy = 3
    env.start()
    job = env.create(mode="quick")
    seen = set()
    end = time.time() + 20
    while (current := env.worker.get(job["id"]))["status"] != "ready":
        seen.add((current["status"], current["phase"].split(":")[0]))
        assert time.time() < end and current["status"] != "failed"
        time.sleep(0.005)
    assert ("waiting", "Waiting") in seen and len(env.renderer.posts) == 4


def test_a_job_waits_for_gpu_memory(env):
    free = iter([4.0, 4.0, 80.0])
    env.monkeypatch.setattr(worker, "free_gpu_gb", lambda: next(free, 80.0))
    env.start()
    job = env.create()
    phases = set()
    while (current := env.worker.get(job["id"]))["status"] not in ("ready", "failed"):
        phases.add(current["phase"])
        time.sleep(0.002)
    assert current["status"] == "ready" and any(p.startswith("Waiting for GPU memory (4 of 30 GB free") for p in phases)


def test_an_out_of_memory_clip_fails_the_job_and_standard_quality_makes_it_work(env):
    env.start()
    env.mode("oom")
    failed = env.wait(env.create()["id"], "ready", "failed")
    assert failed["status"] == "failed" and "ran out of memory" in failed["error"] and "Standard" in failed["hint"]
    assert worker.complete(env.loops / "aisha", "idle") and not (env.loops / "aisha" / "speaking.mp4").exists()
    assert env.renderer.posts == []

    env.mode("ok")  # the same quality again: the finished idle clip is kept
    again = env.worker.retry(failed["id"], {})
    assert env.wait(again["id"], "ready", "failed")["status"] == "ready"
    assert [r["states"] for r in env.runs()] == ["idle,speaking", "speaking"]

    env.mode("oom")  # another resolution cannot be mixed with clips made before: everything is made again
    second = env.create("aisha-two")
    failed = env.wait(second["id"], "failed")
    env.mode("ok")
    redone = env.wait(env.worker.retry(failed["id"], {"quality": "standard"})["id"], "ready", "failed")
    assert redone["status"] == "ready" and redone["quality"] == "standard"
    assert [(r["states"], r["sample_area"]) for r in env.runs()[-2:]] == [("idle,speaking", 960), ("idle,speaking", 768)]


def test_a_crashing_generator_fails_the_job(env):
    env.start()
    env.mode("crash")
    failed = env.wait(env.create()["id"], "ready", "failed")
    assert failed["status"] == "failed" and "idle, speaking" in failed["error"] and failed["finished"]


def test_a_registration_the_renderer_cannot_finish_fails_the_job_and_retry_registers_again(env):
    env.renderer.fail = "NoFaceDetectedError: MediaPipe FaceLandmarker found no face in the source frame"
    env.start()
    failed = env.wait(env.create("no-face", mode="quick")["id"], "ready", "failed")
    assert failed["status"] == "failed" and failed["error"] == f"The renderer could not register the avatar: {env.renderer.fail}"
    assert "renderer's log" in failed["hint"]
    env.renderer.fail, env.renderer.polls = None, 0
    again = env.wait(env.worker.retry(failed["id"], {})["id"], "ready", "failed")
    assert again["status"] == "ready" and len(env.renderer.posts) == 2


def test_cancelling_stops_the_generator_and_keeps_the_finished_clips(env):
    env.start()
    env.mode("slow")
    first, second = env.create("first-one"), env.create("second-one")
    end = time.time() + 20  # the speaking clip is being made: its mp4 is on disk without a report
    while not (env.loops / "first-one" / "speaking.mp4").exists():
        assert time.time() < end
        time.sleep(0.02)
    assert env.worker.cancel(second["id"])["status"] == "cancelled"  # still queued: gone at once
    env.worker.cancel(first["id"])
    cancelled = env.wait(first["id"], "cancelled", "failed", "ready", timeout=40)
    assert cancelled["status"] == "cancelled"
    assert worker.complete(env.loops / "first-one", "idle") and not (env.loops / "first-one" / "speaking.mp4").exists()
    assert len(env.runs()) == 1 and not (env.loops / "second-one").exists()
    with pytest.raises(worker.Refused) as err:
        env.worker.cancel(first["id"])
    assert err.value.status == 409


# ------------------------------------------------------------------------------------------------------------------ requests and storage
def test_a_job_needs_a_valid_id_an_uploaded_photo_and_a_free_name(env):
    photo = env.photo("aisha")
    outside = env.studio / "elsewhere.png"
    outside.write_bytes(open(photo, "rb").read())
    link = env.studio / "uploads" / "link.png"
    link.symlink_to("/etc/hostname")
    notimage = env.studio / "uploads" / "text.png"
    notimage.write_text("not an image")
    for change, status in (({"avatar": "Aisha"}, 422), ({"avatar": "-x"}, 422), ({"avatar": "a" * 41}, 422), ({"image": str(outside)}, 422),
                           ({"image": str(link)}, 422), ({"image": str(notimage)}, 422), ({"image": "/etc/passwd"}, 422), ({"mode": "fast"}, 422),
                           ({"quality": "ultra"}, 422)):
        with pytest.raises(worker.Refused) as err:
            env.worker.create({"avatar": "aisha", "image": photo, **change})
        assert err.value.status == status, change
    env.worker.create({"avatar": "aisha", "image": photo})
    with pytest.raises(worker.Refused) as err:  # nobody is running the queue here, so the first one is still active
        env.worker.create({"avatar": "aisha", "image": photo})
    assert err.value.status == 409
    assert len(os.listdir(env.studio / "jobs")) == 1


def test_jobs_left_running_by_a_stopped_worker_start_again_twice_at_most(env):
    for job_id, attempts in (("aaaaaaaa", 1), ("bbbbbbbb", 2)):
        record = job_record(id=job_id, avatar=f"av-{job_id}", status="generating", attempts=attempts, quality="hd", clean=False, log=[], registration=None)
        with open(env.studio / "jobs" / f"{job_id}.json", "w") as f:
            json.dump(record, f)
    revived = worker.Worker(str(env.studio))
    assert revived.jobs["aaaaaaaa"]["status"] == "queued"
    assert revived.jobs["bbbbbbbb"]["status"] == "failed" and revived.jobs["bbbbbbbb"]["hint"]


def test_the_files_of_an_avatar_are_removed_unless_it_is_being_made(env):
    env.create("aisha")
    (env.loops / "aisha").mkdir()
    (env.loops / "aisha" / "idle.mp4").write_bytes(b"x")
    with pytest.raises(worker.Refused) as err:
        env.worker.remove_avatar("aisha")
    assert err.value.status == 409 and (env.loops / "aisha").exists()
    job = next(iter(env.worker.jobs.values()))
    env.worker.cancel(job["id"])
    assert env.worker.remove_avatar("aisha") == {"deleted": "aisha"}
    assert not (env.loops / "aisha").exists() and not (env.studio / "uploads" / "aisha").exists() and env.worker.jobs == {}
    assert os.listdir(env.studio / "jobs") == []


def test_the_http_api(env):
    server = ThreadingHTTPServer(("127.0.0.1", 0), worker.make_handler(env.worker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_port
    with requests.Session() as http:  # one connection: an unread request body would break the second request
        created = http.post(f"{base}/v1/jobs", json={"avatar": "aisha", "image": env.photo("aisha")})
        assert created.status_code == 202 and created.json()["status"] == "queued"
        job_id = created.json()["id"]
        assert http.post(f"{base}/v1/jobs/{job_id}/cancel", json={"unused": "x" * 100}).json()["status"] == "cancelled"
        assert http.get(f"{base}/v1/jobs/{job_id}").json()["stages"][0]["key"] == "load"
        assert [j["id"] for j in http.get(f"{base}/v1/jobs").json()["jobs"]] == [job_id]
        assert http.post(f"{base}/v1/jobs", json={"avatar": "Bad", "image": "/x"}).status_code == 422
        assert http.post(f"{base}/v1/jobs", data="not json").status_code == 400
        assert http.get(f"{base}/v1/jobs/{'0' * 8}").status_code == 404
        assert http.get(f"{base}/nothing").status_code == 404
        assert http.delete(f"{base}/v1/jobs/{job_id}").json() == {"deleted": job_id}
        assert http.get(f"{base}/healthz").json()["status"] == "ok"
    server.shutdown()
