"""Canonical language contracts shared by the main FastAPI application."""

from __future__ import annotations

from typing import Literal, TypeGuard

LanguageCode = Literal["ru", "kk", "en"]
SpeechRecognitionLanguage = Literal["ru", "kk", "en", "auto"]

SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("ru", "kk", "en")
SUPPORTED_SPEECH_RECOGNITION_LANGUAGES: tuple[
    SpeechRecognitionLanguage, ...
] = (*SUPPORTED_LANGUAGES, "auto")

LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "ru": "Russian",
    "kk": "Kazakh",
    "en": "English",
}


def is_language_code(value: object) -> TypeGuard[LanguageCode]:
    """Return whether ``value`` is one of the canonical response languages."""
    return isinstance(value, str) and value in SUPPORTED_LANGUAGES


def normalize_language_code(value: str, *, field: str = "language") -> LanguageCode:
    """Normalize and validate a canonical response or corpus language code."""
    normalized = value.strip().lower()
    if not is_language_code(normalized):
        choices = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"{field} must be one of: {choices}")
    return normalized


def normalize_speech_language(
    value: str, *, field: str = "language"
) -> SpeechRecognitionLanguage:
    """Normalize and validate an STT language, including automatic detection."""
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SPEECH_RECOGNITION_LANGUAGES:
        choices = ", ".join(SUPPORTED_SPEECH_RECOGNITION_LANGUAGES)
        raise ValueError(f"{field} must be one of: {choices}")
    return normalized  # type: ignore[return-value]


__all__ = [
    "LANGUAGE_NAMES",
    "LanguageCode",
    "SpeechRecognitionLanguage",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_SPEECH_RECOGNITION_LANGUAGES",
    "is_language_code",
    "normalize_language_code",
    "normalize_speech_language",
]
