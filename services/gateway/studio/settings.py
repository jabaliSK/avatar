"""The Studio's configuration, read from the environment once at start-up."""
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Settings:
    renderer_url: str = "http://kka-renderer:8290"   # the renderer's /v1 admin API (internal: it has no login)
    worker_url: str = "http://kka-worker:8300"       # the job runner that makes the body clips
    llm_url: str = "http://kka-vllm:8000"            # only asked whether it is up
    studio_dir: str = "/studio"                      # shared with the worker and the renderer: uploads/<avatar>/source.png
    control_file: str = "/run/kka/avatar"            # the avatar the next conversation uses (read by the avatar handler)
    static_dir: str = os.path.join(HERE, "static")
    key: str = ""                                    # empty: nobody has to unlock the actions that change something
    max_upload_bytes: int = 20 * 2**20
    min_free_bytes: int = 4 * 2**30                  # refuse new avatars below this much free disk
    low_disk_bytes: int = 12 * 2**30                 # the page warns below this

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls(
            renderer_url=env.get("KKA_STUDIO_RENDERER", cls.renderer_url).rstrip("/"),
            worker_url=env.get("KKA_STUDIO_WORKER", cls.worker_url).rstrip("/"),
            llm_url=env.get("KKA_STUDIO_LLM", cls.llm_url).rstrip("/"),
            studio_dir=env.get("KKA_STUDIO_DIR", cls.studio_dir),
            control_file=env.get("KKA_AVATAR_CONTROL", cls.control_file),
            key=env.get("KKA_STUDIO_KEY", "").strip(),
        )
