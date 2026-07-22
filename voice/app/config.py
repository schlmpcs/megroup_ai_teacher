import os
from dataclasses import dataclass


def _device() -> str:
    requested = os.getenv("DEVICE", "cuda").strip().lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return requested if torch.cuda.is_available() else "cpu"


DEVICE = _device()


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip().lower()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    stt_kk_model: str = os.getenv(
        "STT_KK_MODEL", "RakhatM/whisper-large-v3-turbo-kk-lora"
    )
    stt_kk_base_model: str = os.getenv(
        "STT_KK_BASE_MODEL", "openai/whisper-large-v3-turbo"
    )
    stt_ru_model: str = os.getenv("STT_RU_MODEL", "openai/whisper-large-v3-turbo")
    tts_kk_model: str = os.getenv("TTS_KK_MODEL", "facebook/mms-tts-kaz")
    tts_ru_model: str = os.getenv("TTS_RU_MODEL", "facebook/mms-tts-rus")
    tts_ru_backend: str = os.getenv("TTS_RU_BACKEND", "supertonic").strip().lower()
    tts_en_backend: str = os.getenv("TTS_EN_BACKEND", "supertonic").strip().lower()
    tts_ru_backends: tuple[str, ...] = _csv_env("TTS_RU_BACKENDS", "supertonic,qwen")
    tts_ru_qwen_model: str = os.getenv(
        "TTS_RU_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    )
    tts_ru_qwen_speaker: str = os.getenv("TTS_RU_QWEN_SPEAKER", "Aiden")
    tts_ru_qwen_dtype: str = os.getenv("TTS_RU_QWEN_DTYPE", "bfloat16").strip().lower()
    tts_ru_qwen_attention: str = (
        os.getenv("TTS_RU_QWEN_ATTENTION", "sdpa").strip().lower()
    )
    tts_ru_qwen_max_new_tokens: int = int(
        os.getenv("TTS_RU_QWEN_MAX_NEW_TOKENS", "2048")
    )
    tts_ru_supertonic_voice_style: str = os.getenv(
        "TTS_RU_SUPERTONIC_VOICE_STYLE", "M3"
    )
    tts_ru_supertonic_steps: int = int(os.getenv("TTS_RU_SUPERTONIC_STEPS", "8"))
    tts_ru_supertonic_sample_rate: int = int(
        os.getenv("TTS_RU_SUPERTONIC_SAMPLE_RATE", "44100")
    )
    tts_supertonic_model_dir: str = os.getenv(
        "TTS_SUPERTONIC_MODEL_DIR",
        os.path.join(os.getenv("HF_HOME", "/models/hf_cache"), "supertonic3"),
    )
    tts_normalize_ru_numbers: bool = _bool_env("TTS_NORMALIZE_RU_NUMBERS", True)
    tts_normalize_en_numbers: bool = _bool_env("TTS_NORMALIZE_EN_NUMBERS", True)
    device: str = DEVICE
    hf_cache: str = os.getenv("HF_HOME", "/models/hf_cache")
    stt_keep_warm_enabled: bool = _bool_env("STT_KEEP_WARM_ENABLED", True)
    stt_keep_warm_real_idle_s: float = _positive_float_env(
        "STT_KEEP_WARM_REAL_IDLE_S", 180.0
    )
    stt_keep_warm_interval_s: float = _positive_float_env(
        "STT_KEEP_WARM_INTERVAL_S", 240.0
    )
    stt_keep_warm_poll_s: float = _positive_float_env(
        "STT_KEEP_WARM_POLL_S", 15.0
    )
    max_audio_duration_s: int = int(os.getenv("MAX_AUDIO_DURATION_S", "120"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


def get_settings() -> Settings:
    return Settings()
