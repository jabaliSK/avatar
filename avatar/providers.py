import asyncio
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import httpx

from avatar.config import ROOT, Settings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


async def run_in_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def run_process(command: list[str], timeout: float, cwd: Path | None = None):
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError):
        # The renderer can spawn FFmpeg; terminate only this job's process group.
        import signal

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        raise
    if process.returncode:
        logger.error("Inference subprocess failed:\n%s", output.decode(errors="replace")[-8000:])
        raise PipelineError("Local inference failed. See the server log for details.")


class Providers:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None
        self._tokenizer = None
        self._stt = None
        self._tts = None

    async def warmup(self):
        if not self.settings.llm_base:
            await run_in_thread(self._local_chat, [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": "Hello."},
            ])
        if not self.settings.tts_base:
            with tempfile.TemporaryDirectory(prefix="avatar-tts-warmup-") as directory:
                await run_in_thread(self._local_speak, "Ready.", Path(directory) / "warmup.wav")

    async def request(self, base, key, path, **kwargs):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            async with httpx.AsyncClient(timeout=self.settings.provider_timeout) as client:
                response = await client.post(f"{base}/{path}", headers=headers, **kwargs)
                response.raise_for_status()
                return response
        except httpx.HTTPError as error:
            logger.exception("Provider request failed: %s", path)
            raise PipelineError(
                f"The {path.split('/')[0]} provider request failed. Check its URL, credentials, and server log."
            ) from error

    async def transcribe(self, audio: Path) -> str:
        await run_in_thread(self._validate_recording, audio)
        s = self.settings
        if s.stt_base:
            response = await self.request(
                s.stt_base, s.stt_key, "audio/transcriptions",
                files={"file": (audio.name, audio.read_bytes(), "application/octet-stream")},
                data={"model": s.stt_model},
            )
            payload = response.json()
            if not isinstance(payload, dict):
                raise PipelineError("The speech provider returned an invalid transcription response.")
            text = payload.get("text", "")
        else:
            text = await run_in_thread(self._local_transcribe, audio)
        if not isinstance(text, str) or not text.strip():
            raise PipelineError("No speech was recognized. Record a clear sentence and try again.")
        return text.strip()

    @staticmethod
    def _validate_recording(audio):
        import av

        try:
            with av.open(str(audio)) as container:
                if not container.streams.audio:
                    raise PipelineError("The uploaded recording has no audio stream.")
                duration = 0
                for frame in container.decode(audio=0):
                    duration += frame.samples / frame.sample_rate
                    if duration > 60:
                        raise PipelineError("Recordings must be 60 seconds or shorter.")
                if duration == 0:
                    raise PipelineError("The uploaded recording contains no audio samples.")
        except (av.FFmpegError, OSError, ValueError) as error:
            raise PipelineError("The uploaded recording could not be decoded.") from error

    def _local_transcribe(self, audio):
        from faster_whisper import WhisperModel

        if self._stt is None:
            self._stt = WhisperModel(
                str(ROOT / "models" / "stt"), device="cpu", compute_type="int8",
                cpu_threads=4, local_files_only=True,
            )
        segments, _ = self._stt.transcribe(str(audio), vad_filter=False)
        return " ".join(segment.text for segment in segments)

    async def chat(self, history: list[dict[str, str]]) -> str:
        s = self.settings
        messages = [{"role": "system", "content": s.system_prompt}, *history]
        if s.llm_base:
            response = await self.request(
                s.llm_base, s.llm_key, "chat/completions",
                json={"model": s.llm_model, "messages": messages, "max_tokens": 160},
            )
            payload = response.json()
            try:
                text = payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise PipelineError("The LLM returned an invalid chat-completions response.") from error
        else:
            text = await run_in_thread(self._local_chat, messages)
        if not isinstance(text, str) or not text.strip():
            raise PipelineError("The language model returned an empty response.")
        if len(text) > 1600:
            raise PipelineError("The response is too long to render. Configure your LLM for shorter replies.")
        return text.strip()

    def _load_local_llm(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        if self._llm is None:
            model_dir = ROOT / "models" / "llm"
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            device = "cuda" if self.settings.live_streaming else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise PipelineError("The live local LLM requires the selected CUDA GPU.")
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_dir, local_files_only=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device).eval()

    def _local_chat(self, messages):
        import torch

        self._load_local_llm()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
        if inputs.input_ids.shape[1] > 6000:
            raise PipelineError("Conversation context is too long. Start a new conversation.")
        with torch.inference_mode():
            result = self._llm.generate(
                **inputs, max_new_tokens=100, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

    async def speak(self, text: str, target: Path):
        s = self.settings
        if s.tts_base:
            response = await self.request(
                s.tts_base, s.tts_key, "audio/speech",
                json={
                    "model": s.tts_model, "voice": s.tts_voice,
                    "input": text, "response_format": "wav",
                },
            )
            target.write_bytes(response.content)
        else:
            await run_in_thread(self._local_speak, text, target)
        import soundfile as sf

        try:
            audio, rate = sf.read(io.BytesIO(target.read_bytes()), always_2d=True)
        except (RuntimeError, ValueError) as error:
            raise PipelineError("TTS did not return valid WAV audio.") from error
        if not rate or not 0 < len(audio) / rate <= 45:
            raise PipelineError("TTS audio must be between 0 and 45 seconds. Request a shorter reply.")

    def _load_local_tts(self):
        from piper import PiperVoice

        if self._tts is None:
            self._tts = PiperVoice.load(str(ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"))

    def _local_speak(self, text, target):
        self._load_local_tts()
        import wave

        with wave.open(str(target), "wb") as output:
            self._tts.synthesize_wav(text, output)

    async def render(self, portrait: Path, audio: Path, output: Path):
        if self.settings.renderer == "livetalk":
            raise PipelineError("Full-body rendering requires the resident live worker.")
        expressive = self.settings.renderer == "flashhead"
        interpreter = self.settings.flashhead_python if expressive else Path(sys.executable)
        script = "render_flashhead.py" if expressive else "render.py"
        source = "FlashHead" if expressive else "MuseTalk"
        try:
            await run_process(
                [
                    str(interpreter), str(ROOT / "scripts" / script),
                    "--image", str(portrait), "--audio", str(audio),
                    "--output", str(output),
                ],
                timeout=self.settings.render_timeout,
                cwd=ROOT / "external" / source,
            )
        except TimeoutError as error:
            raise PipelineError("Avatar animation timed out. Try a shorter reply.") from error
        if not output.is_file() or output.stat().st_size < 1000:
            raise PipelineError(f"{source} did not produce a playable video. See the server log.")

    def status(self):
        root = ROOT / "external" / "MuseTalk"
        checks = {
            "renderer": (
                root / "models/musetalkV15/unet.pth",
                root / "models/musetalkV15/musetalk.json",
                root / "models/sd-vae/diffusion_pytorch_model.bin",
                root / "models/whisper/pytorch_model.bin",
                root / "models/face-parse-bisent/79999_iter.pth",
                root / "models/face-parse-bisent/resnet18-f37072fd.pth",
                ROOT / "models/torch/hub/checkpoints/2DFAN4-cd938726ad.zip",
                ROOT / "models/torch/hub/checkpoints/s3fd-619a316812.pth",
            ),
            "llm": () if self.settings.llm_base else (ROOT / "models/llm/config.json",),
            "stt": () if self.settings.stt_base else (ROOT / "models/stt/model.bin",),
            "tts": () if self.settings.tts_base else (
                ROOT / "models/tts/en_US-lessac-medium.onnx",
                ROOT / "models/tts/en_US-lessac-medium.onnx.json",
            ),
        }
        expressive = self.settings.renderer == "flashhead"
        if expressive:
            checks["renderer"] = (
                self.settings.flashhead_python,
                ROOT / "external/FlashHead/flash_head/inference.py",
                ROOT / "models/flashhead/Model_Pro/config.json",
                ROOT / "models/flashhead/Model_Pro/diffusion_pytorch_model.safetensors",
                ROOT / "models/flashhead/VAE_Wan/Wan2.1_VAE.pth",
                ROOT / "models/wav2vec2/config.json",
                ROOT / "models/wav2vec2/preprocessor_config.json",
                ROOT / "models/wav2vec2/model.safetensors",
            )
            if self.settings.live_streaming:
                checks["renderer"] = (
                    self.settings.flashhead_python,
                    ROOT / "external/FlashHead/flash_head/inference.py",
                    ROOT / "models/flashhead/Model_Lite/config.json",
                    ROOT / "models/flashhead/Model_Lite/diffusion_pytorch_model.safetensors",
                    ROOT / "models/flashhead/VAE_LTX/config.json",
                    ROOT / "models/flashhead/VAE_LTX/diffusion_pytorch_model.safetensors",
                    ROOT / "models/wav2vec2/config.json",
                    ROOT / "models/wav2vec2/preprocessor_config.json",
                    ROOT / "models/wav2vec2/model.safetensors",
                )
        result = {
            key: {
                "configured": all(path.is_file() for path in paths),
                "mode": "endpoint" if key != "renderer" and not paths else "local",
            }
            for key, paths in checks.items()
        }
        result["renderer"].update({
            "engine": self.settings.renderer,
            "label": "FlashHead Pro / full-face motion" if expressive else "MuseTalk 1.5 / lip-sync",
            "description": (
                "Audio-driven head motion, eyes, facial expressions and lip-sync."
                if expressive else "Mouth-region animation blended into the original portrait."
            ),
        })
        if expressive and self.settings.live_streaming:
            result["renderer"].update({
                "label": "FlashHead Lite / live motion",
                "description": "Warm GPU, streaming audio/video, and silent idle animation.",
            })
        if self.settings.renderer == "livetalk":
            paths = [
                self.settings.livetalk_python,
                ROOT / "external/LiveTalk/scripts/inference_example.py",
                ROOT / "external/LiveTalk/OmniAvatar/OmniAvatar/models/wav2vec.py",
                ROOT / "models/livetalk/model/model.safetensors",
                ROOT / "models/livetalk/base/models_t5_umt5-xxl-enc-bf16.pth",
                ROOT / "models/livetalk/base/Wan2.1_VAE.pth",
                ROOT / "models/livetalk/base/google/umt5-xxl/spiece.model",
                ROOT / "models/wav2vec2/config.json",
                ROOT / "models/wav2vec2/preprocessor_config.json",
                ROOT / "models/wav2vec2/model.safetensors",
            ]
            result["renderer"].update({
                "configured": all(path.is_file() for path in paths),
                "label": "LiveTalk / full-body animation",
                "description": "Full image, facial motion and generated body/hand gestures. Not a static-body face overlay.",
                "fps": 16, "initial_buffer_seconds": 0.5,
            })
        return result
