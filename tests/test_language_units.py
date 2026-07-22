"""Canonical application language-contract tests."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.languages import (
    SUPPORTED_LANGUAGES,
    normalize_language_code,
    normalize_speech_language,
)


def test_canonical_language_contracts_include_english():
    assert SUPPORTED_LANGUAGES == ("ru", "kk", "en")
    assert normalize_language_code(" EN ") == "en"
    assert normalize_speech_language("auto") == "auto"
    assert normalize_speech_language("en") == "en"


def test_default_language_accepts_english_but_remains_russian_by_default():
    base = {
        "INTERNAL_API_KEY": "test-language-key",
        "OPENAI_API_KEY": "sk-test",
    }
    assert Settings(**base).DEFAULT_LANGUAGE == "ru"
    assert Settings(**base, DEFAULT_LANGUAGE="en").DEFAULT_LANGUAGE == "en"
    with pytest.raises(ValidationError, match="DEFAULT_LANGUAGE must be one of"):
        Settings(**base, DEFAULT_LANGUAGE="fr")
    with pytest.raises(
        ValidationError, match="VOICE_TTS_EN_DEFAULT_BACKEND must be one of"
    ):
        Settings(**base, VOICE_TTS_EN_DEFAULT_BACKEND="mms")
