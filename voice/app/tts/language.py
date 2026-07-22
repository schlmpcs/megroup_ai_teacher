from typing import Literal

from fastapi import HTTPException

TtsLanguage = Literal["kk", "ru", "en"]
SUPPORTED_TTS_LANGUAGES: tuple[TtsLanguage, ...] = ("ru", "kk", "en")


def normalize_tts_language(language: str) -> TtsLanguage:
    normalized = language.strip().lower()
    if normalized not in SUPPORTED_TTS_LANGUAGES:
        raise HTTPException(
            status_code=422, detail="language must be one of: ru, kk, en"
        )
    return normalized  # type: ignore[return-value]
