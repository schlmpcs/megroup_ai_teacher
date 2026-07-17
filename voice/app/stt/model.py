import io
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings


def normalize_detected_language(value: str) -> str:
    """Map Whisper language names or tokens to supported canonical codes."""
    normalized = str(value or "").strip().lower()
    aliases = {
        "kazakh": "kk",
        "<|kk|>": "kk",
        "kk": "kk",
        "russian": "ru",
        "<|ru|>": "ru",
        "ru": "ru",
        "english": "en",
        "<|en|>": "en",
        "en": "en",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"unsupported detected speech language: {value or 'unknown'}")


class LocalWhisperSttBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded: list[str] = []
        self._processors: dict[str, Any] = {}
        self._model: Any = None
        self._inference_lock: Any = None

    def load_models(self) -> None:
        import threading

        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype = torch.float16 if self.settings.device == "cuda" else torch.float32

        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.settings.stt_kk_base_model,
            cache_dir=self.settings.hf_cache,
            torch_dtype=dtype,
        )
        shared_model = PeftModel.from_pretrained(
            base_model,
            self.settings.stt_kk_model,
            cache_dir=self.settings.hf_cache,
        )
        shared_model.to(self.settings.device)
        shared_model.eval()

        kk_processor = WhisperProcessor.from_pretrained(
            self.settings.stt_kk_base_model,
            cache_dir=self.settings.hf_cache,
            language="kazakh",
            task="transcribe",
        )
        ru_processor = WhisperProcessor.from_pretrained(
            self.settings.stt_ru_model,
            cache_dir=self.settings.hf_cache,
            language="russian",
            task="transcribe",
        )
        en_processor = WhisperProcessor.from_pretrained(
            self.settings.stt_kk_base_model,
            cache_dir=self.settings.hf_cache,
            language="english",
            task="transcribe",
        )

        self._processors = {"kk": kk_processor, "ru": ru_processor, "en": en_processor}
        self._model = shared_model
        self._inference_lock = threading.Lock()
        self.loaded = ["stt_kk", "stt_ru", "stt_en"]

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> dict:
        import torch

        audio, sampling_rate = self._decode_audio(audio_bytes)
        if audio.shape[0] > self.settings.max_audio_duration_s * sampling_rate:
            raise ValueError(f"audio duration exceeds {self.settings.max_audio_duration_s} seconds")

        resolved_language = (
            self._detect_language(audio, sampling_rate)
            if language == "auto"
            else language
        )
        processor = self._processors[resolved_language]
        torch_device = self.settings.device

        inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        model_dtype = next(self._model.parameters()).dtype
        input_features = inputs.input_features.to(torch_device, dtype=model_dtype)

        t0 = time.time()
        with self._inference_lock:
            if resolved_language == "kk":
                with torch.no_grad():
                    predicted_ids = self._model.generate(
                        input_features,
                        language="kazakh",
                        task="transcribe",
                        max_new_tokens=225,
                    )
            else:
                with self._model.disable_adapter():
                    with torch.no_grad():
                        predicted_ids = self._model.generate(
                            input_features,
                            language={"ru": "russian", "en": "english"}[
                                resolved_language
                            ],
                            task="transcribe",
                            max_new_tokens=225,
                        )
        elapsed_ms = int((time.time() - t0) * 1000)

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return {
            "text": text,
            "language": resolved_language,
            "confidence": None,
            "duration_ms": elapsed_ms,
        }

    def _detect_language(self, audio: np.ndarray, sampling_rate: int) -> str:
        import torch

        processor = self._processors["ru"]
        torch_device = self.settings.device
        audio_16k = audio
        if sampling_rate != 16000:
            import librosa

            audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sampling_rate, target_sr=16000)

        model_dtype = next(self._model.parameters()).dtype
        inputs = processor(
            audio_16k[: 5 * 16000],
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(torch_device, dtype=model_dtype)
        with self._inference_lock:
            with self._model.disable_adapter():
                with torch.no_grad():
                    detected = self._model.detect_language(inputs)

        lang_str = ""
        if torch.is_tensor(detected) and detected.numel() > 0:
            lang_id = int(detected[0].item())
            lang_to_id = getattr(getattr(self._model, "generation_config", None), "lang_to_id", {}) or {}
            lang_str = next(
                (lang for lang, token_id in lang_to_id.items() if token_id == lang_id),
                lang_str,
            )
        elif detected:
            first = detected[0]
            if isinstance(first, tuple) and len(first) > 1:
                lang_str = str(first[1])
            elif first and isinstance(first[0], tuple) and len(first[0]) > 1:
                lang_str = str(first[0][1])

        return normalize_detected_language(lang_str)

    def _decode_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        import librosa
        import soundfile as sf

        try:
            audio, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = Path(tmp.name)
            try:
                audio, sampling_rate = librosa.load(str(tmp_path), sr=16000, mono=True)
            finally:
                tmp_path.unlink(missing_ok=True)

        audio = audio.astype(np.float32)
        if audio.size and np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        if audio.ndim > 1:
            # soundfile returns stereo as (samples, channels), so average channels per sample.
            audio = audio.mean(axis=1)
        if sampling_rate != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
        return audio.astype(np.float32), sampling_rate
