import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from avatar.config import ROOT, Settings
from avatar.providers import PipelineError

logger = logging.getLogger(__name__)


class LiveWorker:
    def __init__(self, settings: Settings, *, head_only: bool = False):
        self.settings = settings
        self.head_only = head_only
        self.process = None
        self._ready = False
        self.error = None
        self._lock = asyncio.Lock()
        self._log = None

    @property
    def ready(self):
        if self._ready and self.process and self.process.returncode is not None:
            self._ready = False
            self.error = "Live renderer stopped unexpectedly. Restart the server and check live-worker.log."
            logger.error(self.error)
        return bool(self._ready and self.process and self.process.returncode is None)

    def interpreter(self):
        """LiveTalk and FlashHead have environments of their own; MuseTalk runs in the main one, this server's."""
        renderer = self.settings.renderer
        if renderer == "musetalk":
            return Path(sys.executable)
        return self.settings.livetalk_python if renderer == "livetalk" else self.settings.flashhead_python

    async def start(self):
        self._log = (self.settings.data / "live-worker.log").open("ab")
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(self.interpreter()),
                "-u", str(ROOT / "scripts/live_worker.py"),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=self._log, limit=16 * 1024 * 1024,
                env={**os.environ, "AVATAR_RENDERER": self.settings.renderer,
                     "AVATAR_HEAD_ONLY": "1" if self.head_only else "0"},
            )
            event = await asyncio.wait_for(self._read(), self.settings.render_timeout)
            if event.get("event") != "ready":
                raise PipelineError("Live worker failed to initialize. See live-worker.log.")
            self._ready = True
        except (OSError, TimeoutError, PipelineError, ValueError) as error:
            self.error = str(error)
            logger.exception("Live worker startup failed")
            raise

    async def _read(self):
        line = await self.process.stdout.readline()
        if not line:
            self._ready = False
            self.error = "Live renderer stopped unexpectedly. See live-worker.log."
            raise PipelineError(self.error)
        try:
            return json.loads(line)
        except ValueError as error:
            self._ready = False
            self.error = "Live renderer sent an invalid protocol message. See live-worker.log."
            raise PipelineError(self.error) from error

    async def command(self, command):
        if not self.ready or self.process.returncode is not None:
            raise PipelineError(self.error or "Live renderer is still warming up.")
        async with self._lock:
            self.process.stdin.write((json.dumps(command) + "\n").encode())
            await self.process.stdin.drain()
            finished = False
            failure = None
            try:
                while True:
                    event = await asyncio.wait_for(self._read(), self.settings.render_timeout)
                    if event["event"] == "done":
                        finished = True
                        break
                    if event["event"] == "error":
                        failure = event["message"]
                    else:
                        yield event
                if failure:
                    raise PipelineError(failure)
            except TimeoutError as error:
                self._ready = False
                self.error = "Live renderer timed out. Restart the server and check live-worker.log."
                raise PipelineError(self.error) from error
            finally:
                # Cancellation must not leave the next request reading this job's frames.
                if not finished and self.ready:
                    self.process.stdin.write(b'{"command":"cancel"}\n')
                    await self.process.stdin.drain()
                    async def drain():
                        while (await self._read())["event"] != "done":
                            pass
                    await asyncio.shield(asyncio.wait_for(drain(), self.settings.render_timeout))

    async def prepare(self, avatar: str, image: Path, idle: Path):
        async for _ in self.command({
            "command": "prepare", "avatar": avatar, "image": str(image), "idle": str(idle),
        }):
            pass

    def stream(self, audio: Path, output: Path, plan: dict | None = None):
        command = {"command": "stream", "audio": str(audio), "output": str(output)}
        if plan is not None:
            command["plan"] = plan
        return self.command(command)

    async def close(self):
        self._ready = False
        if self.process and self.process.returncode is None:
            try:
                # Ctrl-C reaches the worker too, so its pipe can be closed before the
                # child is reaped and returncode is still None.
                if not self.process.stdin.is_closing():
                    self.process.stdin.write(b'{"command":"quit"}\n')
                    await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(), 10)
            except (TimeoutError, BrokenPipeError, ConnectionResetError):
                if self.process.returncode is None:
                    self.process.kill()
                    await self.process.wait()
        if self._log:
            self._log.close()
