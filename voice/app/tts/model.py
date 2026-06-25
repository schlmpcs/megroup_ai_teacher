import io
import re

import numpy as np
import scipy.io.wavfile as wav_io

from app.config import Settings

_LATIN_TO_CYR = {
    'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
    'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л',
    'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
    's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
    'y': 'й', 'z': 'з',
    'A': 'А', 'B': 'Б', 'C': 'К', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
    'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Й', 'K': 'К', 'L': 'Л',
    'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'Q': 'К', 'R': 'Р',
    'S': 'С', 'T': 'Т', 'U': 'У', 'V': 'В', 'W': 'В', 'X': 'КС',
    'Y': 'Й', 'Z': 'З',
}


def _transliterate_latin(text: str) -> str:
    """Replace Latin characters in text with Cyrillic equivalents."""
    def replace_word(m: re.Match) -> str:
        return "".join(_LATIN_TO_CYR.get(ch, ch) for ch in m.group())

    return re.sub(r"[A-Za-z]+", replace_word, text)


class MmsTtsBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded: list[str] = []
        self._models = {}
        self._tokenizers = {}
        self._providers = {}

    def load_models(self) -> None:
        if self.settings.tts_ru_backend not in {"mms", "supertonic"}:
            raise ValueError("TTS_RU_BACKEND must be one of: mms, supertonic")

        self._load_mms_model("kk", self.settings.tts_kk_model)
        self.loaded = ["tts_kk"]

        if self.settings.tts_ru_backend == "mms":
            self._load_mms_model("ru", self.settings.tts_ru_model)
        else:
            self._load_supertonic_ru_model()

        self.loaded.append("tts_ru")

    def _load_mms_model(self, language: str, model_id: str) -> None:
        from transformers import AutoTokenizer, VitsModel

        model = VitsModel.from_pretrained(model_id, cache_dir=self.settings.hf_cache).to(self.settings.device)
        model.eval()
        self._models[language] = model
        self._tokenizers[language] = AutoTokenizer.from_pretrained(model_id, cache_dir=self.settings.hf_cache)
        self._providers[language] = "mms"

    def _load_supertonic_ru_model(self) -> None:
        from supertonic import TTS

        model = TTS(auto_download=True)
        self._models["ru"] = model
        self._providers["ru"] = "supertonic"
        self._supertonic_style = model.get_voice_style(voice_name=self.settings.tts_ru_supertonic_voice_style)

    def synthesize(self, text: str, language: str = "ru", speed: float = 1.0) -> bytes:
        if language == "ru":
            text = _transliterate_latin(text)

        provider = self._providers.get(language)
        if provider == "supertonic":
            return self._synthesize_supertonic_ru(text, speed)
        if provider == "mms":
            return self._synthesize_mms(text, language, speed)
        raise ValueError(f"TTS model is not loaded for language: {language}")

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

    def _synthesize_supertonic_ru(self, text: str, speed: float = 1.0) -> bytes:
        model = self._models["ru"]
        wav, _ = model.synthesize(
            text=text,
            lang="ru",
            voice_style=self._supertonic_style,
            total_steps=self.settings.tts_ru_supertonic_steps,
            speed=speed,
        )
        waveform = np.asarray(wav, dtype=np.float32).squeeze()
        return self._wav_bytes(waveform, self.settings.tts_ru_supertonic_sample_rate, speed=1.0)

    def _wav_bytes(self, waveform, sample_rate: int, speed: float = 1.0) -> bytes:
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
