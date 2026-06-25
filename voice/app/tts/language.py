from typing import Literal

from fastapi import HTTPException

TtsLanguage = Literal["kk", "ru"]


def normalize_tts_language(language: str) -> TtsLanguage:
    normalized = language.strip().lower()
    if normalized not in {"kk", "ru"}:
        raise HTTPException(status_code=422, detail="language must be one of: kk, ru")
    return normalized  # type: ignore[return-value]
