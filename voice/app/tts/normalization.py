"""Language-dispatched normalization for local TTS backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .abbreviation_normalization import normalize_russian_tts_text
from .english_text_normalization import normalize_english_text

if TYPE_CHECKING:
    from ..config import Settings


def normalize_tts_text(text: str, language: str, settings: Settings) -> str:
    """Apply only the normalization rules belonging to ``language``."""
    if language == "ru" and settings.tts_normalize_ru_numbers:
        return normalize_russian_tts_text(text)
    if language == "en" and settings.tts_normalize_en_numbers:
        return normalize_english_text(text)
    return text


__all__ = ["normalize_tts_text"]
