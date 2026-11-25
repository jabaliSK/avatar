"""Optional real-model integration check against a running local server."""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from avatar.config import Settings
from avatar.providers import Providers


async def main(base: str, portrait: Path):
    providers = Providers(Settings.from_env())
    with tempfile.TemporaryDirectory(prefix="avatar-smoke-") as directory:
        speech = Path(directory) / "input.wav"
        await providers.speak("Hello. My name is Alex. What can you help me with?", speech)
        async with httpx.AsyncClient(base_url=base, timeout=600) as client:
            response = await client.get("/api/status")
            response.raise_for_status()
            expected_fps = response.json()["pipeline"]["renderer"].get("fps", 25)
            response = await client.post(
                "/api/avatars", data={"name": "NASA demo avatar", "consent": "true"},
                files={"image": (portrait.name, portrait.read_bytes(), "image/png")},
            )
            response.raise_for_status()
            avatar = response.json()
            response = await client.post("/api/sessions", json={"avatar_id": avatar["id"]})
            response.raise_for_status()
            session = response.json()
            response = await client.post(
                f"/api/sessions/{session['id']}/turn",
                files={"audio": ("input.wav", speech.read_bytes(), "audio/wav")},
            )
            if response.is_error:
                print(response.text, file=sys.stderr)
            response.raise_for_status()
            turn = response.json()
            assert turn["user_text"].strip() and turn["assistant_text"].strip()
            response = await client.get(turn["video"])
            response.raise_for_status()
            assert response.headers["content-type"] == "video/mp4"
            video_file = Path(directory) / "reply.mp4"
            video_file.write_bytes(response.content)
            import av
            with av.open(str(video_file)) as clip:
                video, sound = clip.streams.video[0], clip.streams.audio[0]
                difference = abs(
                    float(video.duration * video.time_base) - float(sound.duration * sound.time_base)
                )
                assert video.average_rate == expected_fps
                assert video.codec_context.name == "h264" and sound.codec_context.name == "aac"
                assert difference <= 0.1, f"Audio/video durations differ by {difference}s"
                metadata = {
                    "fps": float(video.average_rate), "width": video.width,
                    "height": video.height, "audio_video_duration_delta": difference,
                }
            print(json.dumps({
                "avatar": avatar, "turn": turn, "video_bytes": len(response.content),
                "media": metadata,
            }, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Creates a persistent demo avatar/conversation using real models.")
    cli.add_argument("--base-url", default="http://127.0.0.1:8000")
    cli.add_argument("--portrait", required=True, type=Path, help="A portrait you have permission to animate")
    args = cli.parse_args()
    asyncio.run(main(args.base_url, args.portrait))
