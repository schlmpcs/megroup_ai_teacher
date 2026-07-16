from __future__ import annotations

import io
import re
import threading
from typing import TYPE_CHECKING

import numpy as np

from .abbreviation_normalization import normalize_russian_tts_text
from .text_normalization import transform_unprotected

if TYPE_CHECKING:
    from ..config import Settings

_LATIN_TO_CYR = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "й",
    "z": "з",
    "A": "А",
    "B": "Б",
    "C": "К",
    "D": "Д",
    "E": "Е",
    "F": "Ф",
    "G": "Г",
    "H": "Х",
    "I": "И",
    "J": "Й",
    "K": "К",
    "L": "Л",
    "M": "М",
    "N": "Н",
    "O": "О",
    "P": "П",
    "Q": "К",
    "R": "Р",
    "S": "С",
    "T": "Т",
    "U": "У",
    "V": "В",
    "W": "В",
    "X": "КС",
    "Y": "Й",
    "Z": "З",
}


def _transliterate_latin(text: str) -> str:
    """Replace Latin characters in text with Cyrillic equivalents."""

    def transliterate(unprotected: str) -> str:
        def replace_word(m: re.Match) -> str:
            return "".join(_LATIN_TO_CYR.get(ch, ch) for ch in m.group())

        return re.sub(r"[A-Za-z]+", replace_word, unprotected)

    return transform_unprotected(text, transliterate)


class LocalTtsBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded: list[str] = []
        self._models = {}
        self._tokenizers = {}
        self._ru_models = {}
        self._supertonic_styles = {}
        self._synthesis_lock = threading.Lock()

    def load_models(self) -> None:
        allowed = {"mms", "qwen", "supertonic"}
        enabled = tuple(dict.fromkeys(self.settings.tts_ru_backends))
        unknown = sorted(set(enabled) - allowed)
        if unknown:
            raise ValueError(
                f"Unsupported TTS_RU_BACKENDS values: {', '.join(unknown)}"
            )
        if not enabled:
            raise ValueError("TTS_RU_BACKENDS must enable at least one Russian backend")
        if self.settings.tts_ru_backend not in enabled:
            raise ValueError("TTS_RU_BACKEND must be included in TTS_RU_BACKENDS")

        self._load_mms_model("kk", self.settings.tts_kk_model)
        self.loaded = ["tts_kk"]

        for backend in enabled:
            if backend == "mms":
                self._load_mms_model("ru", self.settings.tts_ru_model)
                self._ru_models[backend] = self._models["ru"]
            elif backend == "qwen":
                self._load_qwen_ru_model()
            elif backend == "supertonic":
                self._load_supertonic_ru_model()

        self.loaded.append("tts_ru")

    @property
    def available_backends(self) -> dict[str, list[str]]:
        return {"kk": ["mms"], "ru": list(self._ru_models)}

    @property
    def default_backends(self) -> dict[str, str]:
        return {"kk": "mms", "ru": self.settings.tts_ru_backend}

    def _load_mms_model(self, language: str, model_id: str) -> None:
        from transformers import AutoTokenizer, VitsModel

        model = VitsModel.from_pretrained(
            model_id, cache_dir=self.settings.hf_cache
        ).to(self.settings.device)
        model.eval()
        self._models[language] = model
        self._tokenizers[language] = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.settings.hf_cache
        )

    def _load_qwen_ru_model(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype_name = self.settings.tts_ru_qwen_dtype
        if self.settings.device == "cpu":
            dtype = torch.float32
        elif dtype_name == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtypes = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if dtype_name not in dtypes:
                raise ValueError(
                    "TTS_RU_QWEN_DTYPE must be one of: auto, bfloat16, float16, float32"
                )
            dtype = dtypes[dtype_name]

        device_map = "cuda:0" if self.settings.device.startswith("cuda") else "cpu"
        load_kwargs = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if self.settings.tts_ru_qwen_attention:
            load_kwargs["attn_implementation"] = self.settings.tts_ru_qwen_attention

        self._ru_models["qwen"] = Qwen3TTSModel.from_pretrained(
            self.settings.tts_ru_qwen_model,
            **load_kwargs,
        )

    def _load_supertonic_ru_model(self) -> None:
        from supertonic import TTS

        model = TTS(auto_download=True)
        self._ru_models["supertonic"] = model
        voice = self.settings.tts_ru_supertonic_voice_style
        self._supertonic_styles[voice] = model.get_voice_style(voice_name=voice)

    def synthesize(
        self,
        text: str,
        language: str = "ru",
        speed: float = 1.0,
        backend: str | None = None,
        voice: str | None = None,
    ) -> bytes:
        selected = self.select_backend(language, backend)
        synthesis_text = (
            normalize_russian_tts_text(text)
            if language == "ru" and self.settings.tts_normalize_ru_numbers
            else text
        )

        with self._synthesis_lock:
            if selected == "qwen":
                return self._synthesize_qwen_ru(synthesis_text, speed, voice)
            if selected == "supertonic":
                return self._synthesize_supertonic_ru(
                    _transliterate_latin(synthesis_text), speed, voice
                )
            if selected == "mms":
                normalized_text = (
                    _transliterate_latin(synthesis_text)
                    if language == "ru"
                    else synthesis_text
                )
                return self._synthesize_mms(normalized_text, language, speed)
        raise ValueError(f"TTS backend is not loaded: {selected}")

    def select_backend(self, language: str, backend: str | None = None) -> str:
        selected = (backend or self.default_backends.get(language, "")).strip().lower()
        available = self.available_backends.get(language, [])
        if selected not in available:
            choices = ", ".join(available) or "none"
            raise ValueError(
                f"TTS backend {selected or '<empty>'} is unavailable for {language}; available: {choices}"
            )
        return selected

    def _synthesize_mms(self, text: str, language: str, speed: float = 1.0) -> bytes:
        import torch

        model = self._models[language]
        tokenizer = self._tokenizers[language]
        inputs = tokenizer(text, return_tensors="pt").to(self.settings.device)

        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0].detach().cpu().numpy()
        sample_rate = model.config.sampling_rate
        return self._wav_bytes(waveform, sample_rate, speed)

    def _synthesize_qwen_ru(
        self, text: str, speed: float = 1.0, voice: str | None = None
    ) -> bytes:
        model = self._ru_models["qwen"]
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language="Russian",
            speaker=voice or self.settings.tts_ru_qwen_speaker,
            max_new_tokens=self.settings.tts_ru_qwen_max_new_tokens,
        )
        waveform = np.asarray(wavs[0], dtype=np.float32).squeeze()
        return self._wav_bytes(waveform, sample_rate, speed)

    def _synthesize_supertonic_ru(
        self, text: str, speed: float = 1.0, voice: str | None = None
    ) -> bytes:
        model = self._ru_models["supertonic"]
        voice_name = voice or self.settings.tts_ru_supertonic_voice_style
        voice_style = self._supertonic_styles.get(voice_name)
        if voice_style is None:
            voice_style = model.get_voice_style(voice_name=voice_name)
            self._supertonic_styles[voice_name] = voice_style
        wav, _ = model.synthesize(
            text=text,
            lang="ru",
            voice_style=voice_style,
            total_steps=self.settings.tts_ru_supertonic_steps,
            speed=speed,
        )
        waveform = np.asarray(wav, dtype=np.float32).squeeze()
        return self._wav_bytes(
            waveform, self.settings.tts_ru_supertonic_sample_rate, speed=1.0
        )

    def _wav_bytes(self, waveform, sample_rate: int, speed: float = 1.0) -> bytes:
        import scipy.io.wavfile as wav_io

        waveform = np.asarray(waveform, dtype=np.float32)

        if speed != 1.0:
            from scipy.signal import resample

            new_length = max(1, int(len(waveform) / speed))
            waveform = resample(waveform, new_length).astype(np.float32)

        waveform_i16 = np.clip(waveform, -1.0, 1.0)
        waveform_i16 = (waveform_i16 * 32767).astype(np.int16)

        buf = io.BytesIO()
        wav_io.write(buf, sample_rate, waveform_i16)
        return buf.getvalue()


MmsTtsBackend = LocalTtsBackend
