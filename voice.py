"""
Voice Engine — Edge TTS integration.

Converts LLM text responses to natural-sounding audio using
Microsoft Edge TTS. Completely free, no API key required.

Voice used: en-US-JennyNeural — natural, clear, professional.
"""

import asyncio
import logging
import os
import re
import tempfile

import edge_tts

logger = logging.getLogger(__name__)

# Best free natural voice — change to any Edge TTS voice you prefer
VOICE = "en-US-JennyNeural"


def strip_html(text: str) -> str:
    """Remove HTML tags and emojis so TTS reads clean natural text."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove emoji (non-BMP unicode — everything above U+FFFF)
    text = re.sub(r"[^\u0000-\uFFFF]", "", text)
    # Remove common symbols TTS reads aloud as words
    text = re.sub(r"[•·→↳◆◇▸►»«★☆✓✗🔹🔎💬🎙]", "", text)
    # Remove misc unicode symbol blocks
    text = re.sub(r"[\u2000-\u27FF\u2900-\u2BFF]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def text_to_speech(text: str, output_path: str) -> bool:
    """
    Convert text to speech and save as MP3 to output_path.
    Returns True on success, False on failure.
    """
    clean = strip_html(text)
    if not clean:
        return False

    try:
        communicate = edge_tts.Communicate(clean, VOICE)
        await communicate.save(output_path)
        logger.info("TTS generated: %s (%d chars)", output_path, len(clean))
        return True
    except Exception as exc:
        logger.error("Edge TTS failed: %s", exc)
        return False


def text_to_speech_sync(text: str, output_path: str) -> bool:
    """Synchronous wrapper for use outside async context."""
    try:
        return asyncio.get_event_loop().run_until_complete(
            text_to_speech(text, output_path)
        )
    except RuntimeError:
        # If no event loop exists
        return asyncio.run(text_to_speech(text, output_path))
