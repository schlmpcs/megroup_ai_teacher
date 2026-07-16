"""Backend integration tests for language-specific TTS normalization."""

from types import SimpleNamespace

import pytest

from voice.app.config import Settings as RussianVoiceSettings
from voice.app.tts.model import LocalTtsBackend
from voice_omnivoice.app import main as omnivoice_main


class _FakeOmniVoiceModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [[0.0]]


def test_normalization_flags_are_enabled_by_default():
    assert RussianVoiceSettings().tts_normalize_ru_numbers is True
    assert omnivoice_main.Settings().normalize_kk_numbers is True


def test_omnivoice_model_receives_normalized_kazakh_text(monkeypatch):
    settings = omnivoice_main.Settings(device="cpu", normalize_kk_numbers=True)
    backend = omnivoice_main.OmniVoiceBackend(settings)
    backend.model = _FakeOmniVoiceModel()
    monkeypatch.setattr(omnivoice_main, "_encode_wav", lambda audio: b"WAV")

    result = backend.synthesize("H2O молекуласында 2 сутек атомы бар.", 1.0)

    assert result == b"WAV"
    call = backend.model.calls[0]
    assert call["text"] == "H2O молекуласында екі сутек атомы бар."
    assert call["language"] == "Kazakh"
    assert call["instruct"] == "male, young adult, moderate pitch"
    assert "normalize_text" not in call


def test_omnivoice_normalization_can_be_disabled(monkeypatch):
    settings = omnivoice_main.Settings(device="cpu", normalize_kk_numbers=False)
    backend = omnivoice_main.OmniVoiceBackend(settings)
    backend.model = _FakeOmniVoiceModel()
    monkeypatch.setattr(omnivoice_main, "_encode_wav", lambda audio: b"WAV")

    backend.synthesize("25 °C", 1.0)

    assert backend.model.calls[0]["text"] == "25 °C"


def _russian_backend(monkeypatch, selected_backend: str, *, enabled: bool = True):
    backend = LocalTtsBackend(
        SimpleNamespace(tts_normalize_ru_numbers=enabled, tts_ru_backend="supertonic")
    )
    backend._ru_models = {
        "supertonic": object(),
        "qwen": object(),
        "mms": object(),
    }
    captured = []

    if selected_backend == "qwen":
        monkeypatch.setattr(
            backend,
            "_synthesize_qwen_ru",
            lambda text, speed, voice: captured.append(text) or b"WAV",
        )
    elif selected_backend == "supertonic":
        monkeypatch.setattr(
            backend,
            "_synthesize_supertonic_ru",
            lambda text, speed, voice: captured.append(text) or b"WAV",
        )
    else:
        monkeypatch.setattr(
            backend,
            "_synthesize_mms",
            lambda text, language, speed: captured.append(text) or b"WAV",
        )
    return backend, captured


@pytest.mark.parametrize("selected_backend", ["supertonic", "qwen", "mms"])
def test_every_russian_backend_receives_normalized_text(
    monkeypatch, selected_backend
):
    backend, captured = _russian_backend(monkeypatch, selected_backend)

    result = backend.synthesize(
        "В молекуле H2O есть 2 атома водорода.",
        language="ru",
        backend=selected_backend,
    )

    assert result == b"WAV"
    assert captured == ["В молекуле H2O есть два атома водорода."]


def test_russian_normalization_can_be_disabled(monkeypatch):
    backend, captured = _russian_backend(monkeypatch, "qwen", enabled=False)

    backend.synthesize("25 °C", language="ru", backend="qwen")

    assert captured == ["25 °C"]


def test_russian_transliteration_preserves_non_linguistic_tokens(monkeypatch):
    backend, captured = _russian_backend(monkeypatch, "supertonic")

    backend.synthesize(
        "H2O x2 LAB-204 report2.csv test 2",
        language="ru",
        backend="supertonic",
    )

    assert captured == ["H2O x2 LAB-204 report2.csv тест два"]
