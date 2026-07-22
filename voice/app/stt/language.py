from typing import Literal

from fastapi import HTTPException

SttLanguage = Literal["kk", "ru", "en", "auto"]
SUPPORTED_STT_LANGUAGES: tuple[SttLanguage, ...] = ("ru", "kk", "en", "auto")


def normalize_stt_language(language: str) -> SttLanguage:
    normalized = language.strip().lower()
    if normalized not in SUPPORTED_STT_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail="language must be one of: ru, kk, en, auto",
        )
    return normalized  # type: ignore[return-value]
