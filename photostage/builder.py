"""Builds a new avatar's motion pack in the background (see photostage/autopack.py for the stages that need a model of
their own), one avatar at a time, and keeps its progress in <avatar>/build.json for the console and the player.

The GPU belongs to the conversations first: a stage's process is paused (SIGSTOP) while any avatar is speaking, and
the MuseTalk stages, which run in the live renderer itself, take it in short turns between conversations.
"""

import asyncio
import base64
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from avatar.providers import run_in_thread
from photostage import dubplan

log = logging.getLogger("uvicorn.error")
STATE = "build.json"
CLOSEUP = 0.25  # as photostage.autopack.CLOSEUP
# What a close-up says, in its own voice, while LiveTalk makes its speaking motion (photostage.autopack.stage_livetalk).
LIVETALK_LINE = ("Thank you for asking. Let me explain how this works. First we look at what you need, and then we find "
                 "the best way forward together.")
FPS = 16
REST_EVERY = 64
PHASES = tuple(range(0, REST_EVERY, 8))  # rest frames a filler may start on
FILLER_VERSION = 2
FILLERS = ("Let me check that.", "One moment, please.", "Sure, let me see.", "Good question, let me think.")
FILLER_STEM = "clip-fillers"
QUALITY = 90
DIFFERENT = 6  # levels (after a little blur) by which a dubbed picture differs from its silent one where the mouth moved
PAD = 8
FADE = 6  # filler pictures over which the rest dissolves into the speaking loop


class BuildError(RuntimeError):
    pass


def venv(root, name):
    return root / name / "bin"


def kind(rig):
    face, size = rig["face"], rig["size"]
    return "closeup" if (face[3] - face[1]) / size[1] >= CLOSEUP else "wide"


# name, label, expected seconds, where it runs
STEPS = {
    "wide": [("wan", "Animating the body and face", 1800, "flashhead"),
             ("quiet", "Matching the lips to silence", 60, "musetalk"),
             ("rest", "Building the idle animation", 60, "cpu"),
             ("fillers", "Recording the filler phrases", 150, "musetalk"),
             ("install", "Publishing the animations", 5, "local")],
    "closeup": [("livetalk", "Animating the speaking motion", 300, "livetalk"),
                ("bridge", "Closing the loop and animating the idle breathing", 250, "flashhead"),
                ("quiet", "Matching the lips to silence", 60, "musetalk"),
                ("rest", "Building the idle animation", 60, "cpu"),
                ("fillers", "Recording the filler phrases", 150, "musetalk"),
                ("install", "Publishing the animations", 5, "local")],
    "fillers": [("fillers", "Recording the filler phrases", 150, "musetalk"),
                ("install-fillers", "Publishing the filler phrases", 5, "local")],
}
OUTPUTS = {
    "wan": ("loop.npy", "rest-take.npy", "source.mp4"),
    "livetalk": ("talk.json",),
    "liveavatar": ("liveavatar.npy", "liveavatar.json"),
    "bridge": ("loop.npy", "rest-take.npy", "source.mp4"),
    "quiet": ("quiet.json", "quiet.bin"),
    "rest": ("rest.json", "rest.bin", "quiet.mp4"),
    "fillers": ("fillers.json", "fillers.bin"),
}


def read_state(folder):
    try:
        value = json.loads((folder / STATE).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_state(folder, state):
    temporary = folder / f"{STATE}.tmp"
    temporary.write_text(json.dumps(state))
    os.replace(temporary, folder / STATE)


def write_bank(folder, stem, jpegs, ids, fps=FPS):
    offsets, position = [], 0
    with (folder / f"{stem}.bin").open("wb") as output:
        for data in jpegs:
            offsets.append([position, len(data)])
            output.write(data)
            position += len(data)
    (folder / f"{stem}.json").write_text(json.dumps({"fps": fps, "frames": [offsets[i] for i in ids]}))


def read_bank(folder, stem):
    info = json.loads((folder / f"{stem}.json").read_text())
    data = (folder / f"{stem}.bin").read_bytes()
    return [data[a:a + n] for a, n in info["frames"]], info


def decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def silence_rows(count, pad=25):
    """Every frame of a loop once, with `pad` frames of it either side so that Whisper hears silence all round."""
    return [[0, i % count, -1, 0, 0.0, 1.0] for i in range(-pad, count + pad)]


def filler_box(takes, base, region):
    """The part of the picture a filler changes: where any dubbed picture differs from the silent loop frame under it,
    inside `region` [x1, y1, x2, y2] (the lower face; elsewhere only JPEG noise differs), padded and even. `takes`: lists
    of (loop frame, BGR picture)."""
    x1, y1, x2, y2 = region
    changed = np.zeros((y2 - y1, x2 - x1), bool)
    blur = lambda image: cv2.GaussianBlur(image[y1:y2, x1:x2].astype(np.float32), (0, 0), 1.5)
    base_blurred = {}
    for take in takes:
        for index, picture in take:
            if index not in base_blurred:
                base_blurred[index] = blur(base[index])
            changed |= np.abs(blur(picture) - base_blurred[index]).max(2) > DIFFERENT
    ys, xs = np.nonzero(changed)
    if not len(xs):
        raise BuildError("The filler phrases did not move the mouth.")
    height, width = base[0].shape[:2]
    left, top = max(0, x1 + xs.min() - PAD) // 2 * 2, max(0, y1 + ys.min() - PAD) // 2 * 2
    right, bottom = min(width, x1 + xs.max() + 1 + PAD), min(height, y1 + ys.max() + 1 + PAD)
    return [int(left), int(top), int(right - left), int(bottom - top)]


def mouth_region(face, size):
    """The lower face (nose to below the chin) of a face box, widened for the head's motion in the speaking loop, in a
    picture of `size` (width, height)."""
    x1, y1, x2, y2 = face
    w, h = x2 - x1, y2 - y1
    return [max(0, int(x1 - 0.5 * w)), max(0, int(y1 + 0.2 * h)), min(size[0], int(x2 + 0.5 * w)),
            min(size[1], int(y2 + 0.6 * h))]


def pose_key(jpeg):
    grey = cv2.cvtColor(decode(jpeg), cv2.COLOR_BGR2GRAY)
    small = (96, max(1, round(96 * grey.shape[0] / grey.shape[1])))
    return cv2.GaussianBlur(cv2.resize(grey, small, interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1)


def nearest_loop_frames(rest_jpegs, quiet_jpegs, phases=PHASES):
    """{rest phase: the silent loop frame that looks most like it}, where a filler that begins on that rest frame takes
    up the speaking motion (the speaking loop's mouth is at ease, as MuseTalk needs it; the rest's smile is not)."""
    loop = [pose_key(j) for j in quiet_jpegs]
    found = {}
    for phase in phases:
        rest = pose_key(rest_jpegs[phase])
        found[phase] = int(np.argmin([np.abs(rest - frame).mean() for frame in loop]))
    return found


def pack_fillers(build, quiet_jpegs, face):
    """build/filler-loop-<n>.json (each phrase's takes as MuseTalk made them, in the speaking loop) as build/fillers.bin
    and fillers.json: the phrases' audio and, per take, the part of each picture that the mouth changes over the silent
    loop frame under it, as JPEGs."""
    quiet = [decode(j) for j in quiet_jpegs]
    size = quiet[0].shape[1], quiet[0].shape[0]
    phrases = []
    for n in range(len(FILLERS)):
        raw = json.loads((build / f"filler-loop-{n}.json").read_text())
        takes = {p: [(index, decode(base64.b64decode(frame))) for index, frame in zip(raw["loop"][p], raw["frames"][p])]
                 for p in raw["frames"]}
        phrases.append((raw, takes))
    box = filler_box([t for _, takes in phrases for t in takes.values()], quiet, mouth_region(face, size))
    x, y, w, h = box
    out, position = [], 0
    with (build / "fillers.bin").open("wb") as data:
        def put(blob):
            nonlocal position
            data.write(blob)
            position += len(blob)
            return [position - len(blob), len(blob)]

        for raw, takes in phrases:
            audio = put(base64.b64decode(raw["pcm"]))
            coded = []
            for p, take in takes.items():
                rows = []
                for index, picture in take:
                    ok, jpeg = cv2.imencode(".jpg", picture[y:y + h, x:x + w], [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                    if not ok:
                        raise BuildError("A filler picture could not be encoded.")
                    rows.append([int(index), *put(jpeg.tobytes())])
                coded.append({"phase": int(p), "loop": rows[0][0], "rows": rows})
            out.append({"text": raw["text"], "rate": raw["rate"], "audio": audio, "takes": coded})
    (build / "fillers.json").write_text(json.dumps({"version": FILLER_VERSION, "fps": dubplan.FPS, "restFps": FPS,
                                                    "loopFps": FPS, "fade": FADE, "box": box, "fillers": out}))
    return box


class Builder:
    def __init__(self, data, root, live, gate, providers, speaking, ready=lambda: True):
        self.data, self.root, self.live, self.gate, self.ready = data, root, live, gate, ready
        self.providers, self.speaking = providers, speaking
        self.queue = asyncio.Queue()
        self.queued = set()
        self.holds = False  # whether the gate is held by a build (a conversation then waits for it, briefly)
        self.process = None
        self.task = None

    # Queue --------------------------------------------------------------------------------------------------------------

    def start(self):
        self.task = asyncio.create_task(self.run())
        self.backfill()
        for folder in sorted(self.data.iterdir()):
            state = read_state(folder) if folder.is_dir() else None
            if state and state.get("state") == "building":
                self.enqueue(folder.name)
        return self.task

    def enqueue(self, key):
        if key not in self.queued:
            self.queued.add(key)
            self.queue.put_nowait(key)

    def begin(self, folder, job):
        """Mark an avatar as being built (before it is published, so that nobody talks to it unfinished)."""
        if job == "fillers":
            steps = STEPS["fillers"]
        else:
            job = kind(json.loads((folder / "rig.json").read_text()))
            steps = STEPS[job]
        write_state(folder, {"state": "building", "job": job, "kind": job if job != "fillers" else None,
                             "step": 1, "steps": len(steps), "label": "Waiting to start", "percent": 0,
                             "eta": sum(s[2] for s in steps), "started": time.time()})

    def backfill(self):
        """Fillers for a finished continuous pack that has an idle bank but no fillers of the current kind (one made by
        hand, or before the fillers were dubbed into the speaking loop)."""
        for folder in sorted(self.data.iterdir()):
            pack = folder / "clips.json"
            if not pack.is_file() or (read_state(folder) or {}).get("state") in ("building", "failed"):
                continue
            continuous = json.loads(pack.read_text()).get("continuous") or {}
            fillers = continuous.get("fillers")
            current = False
            if fillers and (folder / fillers["index"]).is_file():
                current = json.loads((folder / fillers["index"]).read_text()).get("version") == FILLER_VERSION
            if continuous.get("rest") and not current:
                for name in OUTPUTS["fillers"]:
                    (folder / "build" / name).unlink(missing_ok=True)
                self.begin(folder, "fillers")
                self.enqueue(folder.name)

    async def run(self):
        while True:
            key = await self.queue.get()
            folder = self.data / key
            try:
                if folder.is_dir():
                    await self.build(folder)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # the avatar stays usable as a talking photo
                log.exception("Building %s failed", key)
                state = read_state(folder) or {}
                write_state(folder, {**state, "state": "failed", "error": str(error) or type(error).__name__,
                                     "finished": time.time()})
            finally:
                self.queued.discard(key)

    async def build(self, folder):
        state = read_state(folder)
        steps = STEPS[state["job"]]
        build = folder / "build"
        build.mkdir(exist_ok=True)
        total = sum(s[2] for s in steps)
        done = 0
        for number, (name, label, expected, where) in enumerate(steps, 1):
            outputs = OUTPUTS.get(name, ())
            if outputs and all((build / o).is_file() for o in outputs):
                done += expected
                continue

            def report(part, text=None, done=done, expected=expected, number=number, label=label):
                current = read_state(folder) or state
                percent = (done + part * expected) / total
                write_state(folder, {**current, "step": number, "label": text or label,
                                     "percent": round(100 * percent, 1), "eta": round(total - done - part * expected)})

            report(0)
            log.info("Building %s: %s", folder.name, label)
            started = time.monotonic()
            if name == "livetalk" and not (build / "drive.wav").is_file():
                report(0, "Recording the line the motion follows")
                await self.providers.speak(LIVETALK_LINE, build / "drive.wav")
            if where in ("flashhead", "liveavatar", "livetalk", "cpu"):
                await self.subprocess(folder, name, where, report)
            elif name == "quiet":
                await self.quiet(folder, build)
            elif name == "fillers":
                await self.fillers(folder, build, report)
            elif name == "install":
                await run_in_thread(self.install, folder, build)
            elif name == "install-fillers":
                await run_in_thread(self.install_fillers, folder, build)
            log.info("Built %s: %s in %.0f s", folder.name, name, time.monotonic() - started)
            done += expected
        write_state(folder, {**(read_state(folder) or state), "state": "ready", "percent": 100, "eta": 0,
                             "label": "Ready", "finished": time.time()})
        log.info("%s is ready", folder.name)

    # Stages in processes of their own ---------------------------------------------------------------------------------

    async def subprocess(self, folder, stage, where, report):
        bin_dir = venv(self.root, {"liveavatar": ".venv-liveavatar", "livetalk": ".venv-livetalk"}.get(where, ".venv-flashhead"))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if where == "cpu" else "0", "HF_HUB_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
               "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
        logfile = (folder / "build" / f"{stage}.log").open("a")
        process = await asyncio.create_subprocess_exec(
            str(bin_dir / "python"), "-m", "photostage.autopack", stage, str(folder), cwd=self.root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.process = process
        pauser = asyncio.create_task(self.pause_while_speaking(process))
        tail = []
        try:
            async for raw in process.stdout:
                line = raw.decode(errors="replace").rstrip()
                logfile.write(line + "\n")
                logfile.flush()
                tail = (tail + [line])[-5:]
                if line.startswith("PROGRESS "):
                    with_progress = json.loads(line[9:])
                    report(float(with_progress.get("part", 0)), with_progress.get("label"))
            code = await process.wait()
        finally:
            pauser.cancel()
            await asyncio.gather(pauser, return_exceptions=True)
            if process.returncode is None:
                with_group(process, signal.SIGCONT)
                with_group(process, signal.SIGKILL)
                await process.wait()
            self.process = None
            logfile.close()
        if code:
            raise BuildError(f"The {stage} stage failed (exit {code}): {' | '.join(tail[-3:])}")

    async def pause_while_speaking(self, process):
        paused = False
        try:
            while process.returncode is None:
                want = bool(self.speaking)
                if want != paused:
                    with_group(process, signal.SIGSTOP if want else signal.SIGCONT)
                    paused = want
                await asyncio.sleep(0.2)
        finally:
            if paused and process.returncode is None:
                with_group(process, signal.SIGCONT)

    # Stages in the live renderer ----------------------------------------------------------------------------------------

    @asynccontextmanager
    async def renderer(self):
        """The live renderer, between conversations."""
        while not (self.live.ready and self.ready()):
            await asyncio.sleep(2)
        while True:
            while self.speaking:
                await asyncio.sleep(0.5)
            await self.gate.acquire()
            if not self.speaking:
                break
            self.gate.release()
        self.holds = True
        try:
            yield
        finally:
            self.holds = False
            self.gate.release()

    async def stream(self, stage_key, stage, wav, plan):
        await self.live.prepare(stage_key, stage / "photo.png", stage / "idle.mp4")
        segments = []
        async for event in self.live.stream(wav, stage / "out.mp4", plan):
            if event["event"] == "segment":
                segments.append(event)
        return segments

    def staging(self, folder, build, clip, source):
        stage = Path(tempfile.mkdtemp(prefix="stage-", dir=build))
        (stage / "photo.png").symlink_to((folder / "photo.png").resolve())
        (stage / "clips.json").write_text(json.dumps({"mode": "avatar", "clips": {clip: {"files": {"mp4": str(source)}}}}))
        return stage

    async def quiet(self, folder, build):
        count = int(np.load(build / "loop.npy", mmap_mode="r").shape[0])
        stage = self.staging(folder, build, "loop", build / "source.mp4")
        try:
            rows = silence_rows(count)
            wav = stage / "silence.wav"
            sf.write(wav, np.zeros(math.ceil((len(rows) + 3) * 16000 / dubplan.FPS), np.float32), 16000)
            async with self.renderer():
                segments = await self.stream(f"build-{folder.name}", stage, wav,
                                             {"fps": dubplan.FPS, "clips": ["loop"], "frames": rows})
            frames = [base64.b64decode(f) for s in segments for f in s["frames"]]
            if len(frames) != len(rows):
                raise BuildError("The renderer made the wrong number of silent frames.")
            write_bank(build, "quiet", frames[25:25 + count], list(range(count)))
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    async def fillers(self, folder, build, report):
        source, quiet_jpegs, rest_jpegs, face = self.loop_source(folder, build)
        starts = await run_in_thread(nearest_loop_frames, rest_jpegs, quiet_jpegs)
        count = len(quiet_jpegs)
        clip = dubplan.Clip("loop", "idle", "calm", count, float(FPS), True)
        stage = self.staging(folder, build, "loop", source)
        try:
            for n, text in enumerate(FILLERS):
                target = build / f"filler-loop-{n}.json"
                if target.is_file():
                    continue
                report(n / (len(FILLERS) + 0.5), f"Recording “{text}”")
                wav = stage / f"filler-{n}.wav"
                await self.providers.speak(text, wav)
                speech = sf.info(str(wav)).duration
                result = {"text": text, "frames": {}, "loop": {}}
                async with self.renderer():
                    for phase in PHASES:
                        plan = dubplan.continuous(clip, speech, starts[phase])
                        rows = plan.frames[:plan.quiet_from]
                        segments = await self.stream(f"build-{folder.name}", stage, wav,
                                                     {"fps": dubplan.FPS, "clips": ["loop"], "frames": rows})
                        result["frames"][str(phase)] = [f for s in segments for f in s["frames"]]
                        result["loop"][str(phase)] = [row[1] for row in rows]
                        if phase == PHASES[0]:
                            result["rate"] = segments[0]["rate"]
                            result["pcm"] = base64.b64encode(b"".join(base64.b64decode(s["audio"]) for s in segments)).decode()
                target.write_text(json.dumps(result))
            report(len(FILLERS) / (len(FILLERS) + 0.5), "Packing the filler phrases")
            await run_in_thread(pack_fillers, build, quiet_jpegs, face)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        # The renderer now has the staging avatar; the next conversation prepares its own again (prepare_head).

    def loop_source(self, folder, build):
        """The speaking loop a filler is dubbed into (its video and silent frames), the open-eyed rest frames it begins
        from, and the face box in their pixels."""
        rig = json.loads((folder / "rig.json").read_text())
        if (build / "rest.json").is_file():
            source = build / "source.mp4"
            quiet, _ = read_bank(build, "quiet")
            rest, _ = read_bank(build, "rest")
            anchor = 0
        else:
            manifest = json.loads((folder / "clips.json").read_text())
            continuous = manifest["continuous"]
            source = folder / next(iter(manifest["clips"].values()))["source"]
            quiet, _ = read_bank(folder, Path(continuous["index"]).stem)
            spec = continuous["rest"]
            if Path(spec["frames"]).stem != Path(spec["index"]).stem:
                raise BuildError("The idle bank's index and frames must share a name.")
            rest, _ = read_bank(folder, Path(spec["index"]).stem)
            anchor = spec.get("anchor", 0)
        if not source.is_file():
            raise BuildError("The speaking loop's video is missing.")
        if anchor >= len(quiet):
            raise BuildError("The idle bank's anchor is not a frame of the speaking loop.")
        rest = rest[:REST_EVERY]
        height, width = decode(quiet[0]).shape[:2]
        sx, sy = width / rig["size"][0], height / rig["size"][1]
        face = [rig["face"][0] * sx, rig["face"][1] * sy, rig["face"][2] * sx, rig["face"][3] * sy]
        return source, quiet, rest, face

    # Publishing -------------------------------------------------------------------------------------------------------

    def install(self, folder, build):
        rig = json.loads((folder / "rig.json").read_text())
        width, height = rig["size"]
        face = rig["face"]
        _, quiet = read_bank(build, "quiet")
        count = len(quiet["frames"])
        names = {"source.mp4": "clip-auto-source.mp4", "quiet.mp4": "clip-auto-quiet.mp4",
                 "quiet.json": "clip-auto-quiet.json", "quiet.bin": "clip-auto-quiet.bin",
                 "rest.json": "clip-auto-rest.json", "rest.bin": "clip-auto-rest.bin",
                 "fillers.json": f"{FILLER_STEM}.json", "fillers.bin": f"{FILLER_STEM}.bin"}
        for source, target in names.items():
            shutil.copyfile(build / source, folder / f"{target}.tmp")
            os.replace(folder / f"{target}.tmp", folder / target)
        rest_info = json.loads((build / "rest.json").read_text())
        rest = {"index": "clip-auto-rest.json", "frames": "clip-auto-rest.bin", "anchor": 0, "anchorEvery": REST_EVERY}
        if len(rest_info["frames"]) == 2 * REST_EVERY:
            rest["randomBlink"] = {"every": 16, "offset": 8, "length": 4}
        manifest = {
            "version": 1, "mode": "avatar", "size": [width, height], "box": [0, 0, width, height],
            "pivot": [round((face[0] + face[2]) / 2, 2), round(face[1] + 0.5 * (face[3] - face[1]), 2)],
            "follow": [face[1], face[3]], "face": face,
            "continuous": {"index": "clip-auto-quiet.json", "frames": "clip-auto-quiet.bin", "rest": rest,
                           "fillers": {"index": f"{FILLER_STEM}.json", "frames": f"{FILLER_STEM}.bin"}},
            "types": {"mp4": 'video/mp4; codecs="avc1.640029"'},
            "clips": {"auto-loop": {"role": "idle", "mood": "calm", "loop": True, "fps": float(FPS),
                                    "duration": count / FPS, "source": "clip-auto-source.mp4",
                                    "files": {"mp4": "clip-auto-quiet.mp4"}}},
        }
        (folder / "clips.json.tmp").write_text(json.dumps(manifest, indent=1))
        os.replace(folder / "clips.json.tmp", folder / "clips.json")

    def install_fillers(self, folder, build):
        stem = f"{FILLER_STEM}-v{FILLER_VERSION}"
        for suffix in ("json", "bin"):
            shutil.copyfile(build / f"fillers.{suffix}", folder / f"{stem}.{suffix}.tmp")
            os.replace(folder / f"{stem}.{suffix}.tmp", folder / f"{stem}.{suffix}")
        pack = folder / "clips.json"
        backup = folder / "clips.before-fillers.json"
        if not backup.is_file():
            shutil.copyfile(pack, backup)
        manifest = json.loads(pack.read_text())
        manifest["continuous"]["fillers"] = {"index": f"{stem}.json", "frames": f"{stem}.bin"}
        (folder / "clips.json.tmp").write_text(json.dumps(manifest, indent=1))
        os.replace(folder / "clips.json.tmp", pack)


def with_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
