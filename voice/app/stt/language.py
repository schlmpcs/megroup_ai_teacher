from typing import Literal

from fastapi import HTTPException

SttLanguage = Literal["kk", "ru", "auto"]


def normalize_stt_language(language: str) -> SttLanguage:
    normalized = language.strip().lower()
    if normalized not in {"kk", "ru", "auto"}:
        raise HTTPException(status_code=422, detail="language must be one of: kk, ru, auto")
    return normalized  # type: ignore[return-value]
