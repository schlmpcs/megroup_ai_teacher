import os
from dataclasses import dataclass

import torch


_requested_device = os.getenv("DEVICE", "cuda").strip().lower()
DEVICE = _requested_device if (_requested_device == "cpu" or torch.cuda.is_available()) else "cpu"


@dataclass(frozen=True)
class Settings:
    stt_kk_model: str = os.getenv("STT_KK_MODEL", "RakhatM/whisper-large-v3-turbo-kk-lora")
    stt_kk_base_model: str = os.getenv("STT_KK_BASE_MODEL", "openai/whisper-large-v3-turbo")
    stt_ru_model: str = os.getenv("STT_RU_MODEL", "openai/whisper-large-v3-turbo")
    tts_kk_model: str = os.getenv("TTS_KK_MODEL", "facebook/mms-tts-kaz")
    tts_ru_model: str = os.getenv("TTS_RU_MODEL", "facebook/mms-tts-rus")
    tts_ru_backend: str = os.getenv("TTS_RU_BACKEND", "supertonic").strip().lower()
    tts_ru_supertonic_voice_style: str = os.getenv("TTS_RU_SUPERTONIC_VOICE_STYLE", "M3")
    tts_ru_supertonic_steps: int = int(os.getenv("TTS_RU_SUPERTONIC_STEPS", "8"))
    tts_ru_supertonic_sample_rate: int = int(os.getenv("TTS_RU_SUPERTONIC_SAMPLE_RATE", "44100"))
    device: str = DEVICE
    hf_cache: str = os.getenv("HF_HOME", "/models/hf_cache")
    max_audio_duration_s: int = int(os.getenv("MAX_AUDIO_DURATION_S", "120"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


def get_settings() -> Settings:
    return Settings()
