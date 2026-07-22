"""Hermetic English STT/TTS tests for the local voice sidecar."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice.app.config import Settings
from voice.app.main import create_app
from voice.app.stt.language import normalize_stt_language
from voice.app.stt.model import LocalWhisperSttBackend, normalize_detected_language
from voice.app.tts.english_text_normalization import normalize_english_text
from voice.app.tts.language import normalize_tts_language
from voice.app.tts.model import LocalTtsBackend


class _Context:
    def __init__(self, enter=None):
        self._enter = enter

    def __enter__(self):
        if self._enter:
            self._enter()
        return self

    def __exit__(self, *args):
        return False


def test_voice_language_normalizers_accept_english():
    assert normalize_stt_language(" EN ") == "en"
    assert normalize_tts_language("EN") == "en"
    assert normalize_detected_language("english") == "en"
    assert normalize_detected_language("<|en|>") == "en"
    assert normalize_detected_language("<|ru|>") == "ru"
    assert normalize_detected_language("kazakh") == "kk"
    with pytest.raises(ValueError, match="unsupported detected speech language"):
        normalize_detected_language("spanish")


def test_english_whisper_generation_disables_kazakh_adapter(monkeypatch):
    calls = {"disabled": 0, "generate": []}

    class _Feature:
        def to(self, *args, **kwargs):
            return self

    class _Processor:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_features=_Feature())

        def batch_decode(self, predicted_ids, skip_special_tokens=True):
            return ["What is next?"]

    class _Model:
        def parameters(self):
            return iter([SimpleNamespace(dtype="float32")])

        def disable_adapter(self):
            return _Context(lambda: calls.__setitem__("disabled", calls["disabled"] + 1))

        def generate(self, input_features, **kwargs):
            calls["generate"].append(kwargs)
            return [[1, 2, 3]]

    fake_torch = SimpleNamespace(no_grad=lambda: _Context())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = LocalWhisperSttBackend(
        SimpleNamespace(device="cpu", max_audio_duration_s=120)
    )
    backend._processors = {"en": _Processor()}
    backend._model = _Model()
    backend._inference_lock = threading.Lock()
    backend._decode_audio = lambda audio: (np.zeros(1600, dtype=np.float32), 16000)

    result = backend.transcribe(b"audio", language="en")

    assert result["language"] == "en"
    assert result["text"] == "What is next?"
    assert calls["disabled"] == 1
    assert calls["generate"][0]["language"] == "english"


def test_whisper_load_builds_english_processor_from_shared_base(monkeypatch):
    processor_calls = []

    class _BaseModel:
        pass

    class _SharedModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class _WhisperModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "shared-whisper"
            return _BaseModel()

    class _Peft:
        @classmethod
        def from_pretrained(cls, base, adapter_id, **kwargs):
            assert isinstance(base, _BaseModel)
            assert adapter_id == "kazakh-adapter"
            return _SharedModel()

    class _Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            processor_calls.append((model_id, kwargs["language"]))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float16="float16", float32="float32"),
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=_Peft))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperForConditionalGeneration=_WhisperModel,
            WhisperProcessor=_Processor,
        ),
    )

    backend = LocalWhisperSttBackend(
        SimpleNamespace(
            device="cpu",
            stt_kk_base_model="shared-whisper",
            stt_kk_model="kazakh-adapter",
            stt_ru_model="shared-whisper",
            hf_cache="/tmp/test-cache",
        )
    )
    backend.load_models()

    assert ("shared-whisper", "english") in processor_calls
    assert set(backend._processors) == {"ru", "kk", "en"}
    assert backend.loaded == ["stt_kk", "stt_ru", "stt_en"]


def _tts_settings(**overrides):
    values = {
        "tts_normalize_ru_numbers": True,
        "tts_normalize_en_numbers": True,
        "tts_ru_backend": "supertonic",
        "tts_en_backend": "supertonic",
        "tts_ru_qwen_speaker": "Aiden",
        "tts_ru_qwen_max_new_tokens": 2048,
        "tts_ru_supertonic_voice_style": "M3",
        "tts_ru_supertonic_steps": 8,
        "tts_ru_supertonic_sample_rate": 44100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_english_synthesis_never_uses_cyrillic_transliteration(monkeypatch):
    backend = LocalTtsBackend(_tts_settings())
    backend._shared_models = {"supertonic": object(), "qwen": object()}
    captured = {}
    monkeypatch.setattr(
        backend,
        "_synthesize_supertonic",
        lambda text, language, speed, voice: captured.update(
            text=text, language=language
        )
        or b"WAV",
    )

    result = backend.synthesize(
        "Measure H2O in the glass.", language="en", backend="supertonic"
    )

    assert result == b"WAV"
    assert captured == {"text": "Measure H2O in the glass.", "language": "en"}


def test_supertonic_and_qwen_receive_english_language_arguments(monkeypatch):
    calls = {}

    class _Qwen:
        def generate_custom_voice(self, **kwargs):
            calls["qwen"] = kwargs
            return [[0.0]], 24000

    class _Supertonic:
        def synthesize(self, **kwargs):
            calls["supertonic"] = kwargs
            return [0.0], None

    backend = LocalTtsBackend(_tts_settings())
    backend._shared_models = {"qwen": _Qwen(), "supertonic": _Supertonic()}
    backend._supertonic_styles["M3"] = object()
    monkeypatch.setattr(backend, "_wav_bytes", lambda *args, **kwargs: b"WAV")

    assert backend._synthesize_qwen("Hello", "en") == b"WAV"
    assert backend._synthesize_supertonic("Hello", "en") == b"WAV"
    assert calls["qwen"]["language"] == "English"
    assert calls["supertonic"]["lang"] == "en"


def test_supertonic_load_uses_persistent_model_directory(monkeypatch):
    captured = {}

    class _Supertonic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_voice_style(self, voice_name):
            return object()

    monkeypatch.setitem(sys.modules, "supertonic", SimpleNamespace(TTS=_Supertonic))
    backend = LocalTtsBackend(
        _tts_settings(tts_supertonic_model_dir="/models/hf_cache/supertonic3")
    )

    backend._load_supertonic_model()

    assert captured == {
        "auto_download": True,
        "model_dir": "/models/hf_cache/supertonic3",
    }


def test_english_normalization_handles_classroom_values_and_protects_tokens():
    text = normalize_english_text(
        "Heat 250 mL to 25 °C at 09:05 on 17.07.2026. "
        "Use 12.5%, H2O, x = 2, LAB-204, report2.csv, and https://example.com/a2."
    )

    assert "two hundred fifty milliliters" in text
    assert "twenty five degrees Celsius" in text
    assert "nine oh five" in text
    assert "July seventeenth two thousand twenty six" in text
    assert "twelve point five percent" in text
    for protected in ("H2O", "x = 2", "LAB-204", "report2.csv", "https://example.com/a2"):
        assert protected in text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Cool to -1 °C.", "Cool to minus one degree Celsius."),
        ("Add -1 mL.", "Add minus one milliliter."),
        ("Measure 1,000 mL.", "Measure one thousand milliliters."),
        ("Use 5-10 cm.", "Use five to ten centimeters."),
    ],
)
def test_english_normalization_handles_signed_grouped_and_range_values(
    source, expected
):
    assert normalize_english_text(source) == expected


class _FakeStt:
    loaded = ["stt_ru", "stt_kk", "stt_en"]

    def load_models(self):
        return None

    def warm_up(self, probes, **kwargs):
        return True

    def transcribe(self, audio_bytes, language):
        return {"text": "Hello", "language": "en"}


class _FakeTts:
    loaded = ["tts_ru", "tts_kk", "tts_en"]
    available_backends = {
        "ru": ["supertonic", "qwen"],
        "kk": ["mms"],
        "en": ["supertonic", "qwen"],
    }
    default_backends = {"ru": "supertonic", "kk": "mms", "en": "supertonic"}

    def load_models(self):
        return None

    def select_backend(self, language, backend=None):
        selected = backend or self.default_backends[language]
        if selected not in self.available_backends[language]:
            raise ValueError(f"backend {selected} is unavailable for {language}")
        return selected

    def synthesize(self, text, language, speed, backend=None, voice=None):
        return b"WAV"


def test_voice_health_reports_english_models_and_backends():
    app = create_app(
        stt_backend=_FakeStt(),
        tts_backend=_FakeTts(),
        settings=Settings(device="cpu"),
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "en" in body["supported_languages"]
    assert "en" in body["stt_models"]
    assert "en" in body["tts_models"]
    assert body["tts_backends"]["en"] == ["supertonic", "qwen"]
    assert body["tts_default_backends"]["en"] == "supertonic"
