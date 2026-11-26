"""Latency trace lines shared by the kka handlers.

Every line has the form `KKA_T <event> ts=<epoch ms> key=value ...` so the stages of one turn (VAD end, ASR, TTS, renderer) can be
lined up from `docker logs` and from the end-to-end test client, which stamps its own events with the same clock.
"""
import time

from loguru import logger


def trace(event: str, **fields) -> None:
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    logger.info(f"KKA_T {event} ts={time.time() * 1000:.0f}{extra}")
