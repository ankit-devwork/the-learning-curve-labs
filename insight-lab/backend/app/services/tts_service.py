"""Text-to-speech for audio overviews (MP3 via edge-tts)."""

from __future__ import annotations

import io
from typing import Any

from pycorekit.core_logging.logger import get_logger

log = get_logger("tts")

DEFAULT_VOICE = "en-US-AriaNeural"
MAX_TTS_CHARS = 12000


async def synthesize_mp3(text: str, *, voice: str = DEFAULT_VOICE) -> bytes:
    """Generate MP3 bytes from plain text. Raises on failure."""
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed") from exc

    trimmed = text.strip()[:MAX_TTS_CHARS]
    if not trimmed:
        raise ValueError("Script is empty")

    communicate = edge_tts.Communicate(trimmed, voice)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    data = buffer.getvalue()
    if not data:
        raise RuntimeError("TTS produced no audio data")
    log.info("TTS synthesized", bytes=len(data), voice=voice)
    return data


def tts_voice_from_env() -> str:
    import os

    return os.getenv("TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
