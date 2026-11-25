import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data: Path
    llm_base: str
    llm_key: str
    llm_model: str
    stt_base: str
    stt_key: str
    stt_model: str
    tts_base: str
    tts_key: str
    tts_model: str
    tts_voice: str
    system_prompt: str
    provider_timeout: float
    render_timeout: float
    renderer: str
    flashhead_python: Path
    livetalk_python: Path
    live_streaming: bool

    @classmethod
    def from_env(cls):
        renderer = os.getenv("AVATAR_RENDERER", "musetalk")
        if renderer not in {"musetalk", "flashhead", "livetalk"}:
            raise ValueError("AVATAR_RENDERER must be 'musetalk', 'flashhead', or 'livetalk'.")
        live_streaming = os.getenv("LIVE_STREAMING", "false").lower()
        if live_streaming not in {"true", "false"}:
            raise ValueError("LIVE_STREAMING must be 'true' or 'false'.")
        if live_streaming == "true" and renderer not in {"flashhead", "livetalk"}:
            raise ValueError("LIVE_STREAMING requires AVATAR_RENDERER=flashhead or livetalk.")
        if renderer == "livetalk" and live_streaming != "true":
            raise ValueError("AVATAR_RENDERER=livetalk requires LIVE_STREAMING=true.")
        return cls(
            data=Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve(),
            llm_base=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            llm_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            stt_base=os.getenv("STT_BASE_URL", "").rstrip("/"),
            stt_key=os.getenv("STT_API_KEY", ""),
            stt_model=os.getenv("STT_MODEL", "Systran/faster-whisper-tiny.en"),
            tts_base=os.getenv("TTS_BASE_URL", "").rstrip("/"),
            tts_key=os.getenv("TTS_API_KEY", ""),
            tts_model=os.getenv("TTS_MODEL", "tts-1"),
            tts_voice=os.getenv("TTS_VOICE", "alloy"),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "You are a friendly conversational assistant. Reply naturally in one "
                "or two short sentences. Do not use markdown.",
            ),
            provider_timeout=float(os.getenv("PROVIDER_TIMEOUT", "180")),
            render_timeout=float(os.getenv("RENDER_TIMEOUT", "300")),
            renderer=renderer,
            flashhead_python=Path(os.getenv(
                "FLASHHEAD_PYTHON", str(ROOT / ".venv-flashhead" / "bin" / "python")
            )).absolute(),
            livetalk_python=Path(os.getenv(
                "LIVETALK_PYTHON", str(ROOT / ".venv-livetalk/bin/python")
            )).absolute(),
            live_streaming=live_streaming == "true",
        )
