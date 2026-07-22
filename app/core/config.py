from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.languages import normalize_language_code

_DEFAULT_SECRETS = {"change_me", "generate-with-python-secrets-token-urlsafe-32", ""}


class Settings(BaseSettings):
    """Environment-driven settings for the VR AI assistant proxy.

    The proxy is intentionally stateless: the knowledge base lives in a local
    Qdrant vector store (queried with a local bge-m3 embedder) and per-lab
    scenario context is loaded from local JSON files in ``SCENARIOS_DIR``.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "VR AI Assistant API"
    VERSION: str = "1.0.0"

    # ── Security (required) ──────────────────────────────────────────────────
    INTERNAL_API_KEY: str = ""
    ADMIN_API_KEY: str = ""

    @field_validator("INTERNAL_API_KEY", "ADMIN_API_KEY")
    @classmethod
    def _must_not_be_default(cls, v: str, info) -> str:
        if v in _DEFAULT_SECRETS:
            raise ValueError(
                f"{info.field_name} must be set to a strong secret "
                '(generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")'
            )
        return v

    # ── OpenAI (key required) ────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
    # "" omits the param; "priority" buys faster first-token at ~2x token cost.
    OPENAI_SERVICE_TIER: str = ""
    REQUEST_TIMEOUT_S: float = 60.0

    # ── In-process response caches (size or TTL <= 0 disables) ──────────────
    # Same teacher questions repeat across students; cached answers/audio
    # return in ~ms. Ephemeral per-process — cleared on every deploy.
    ANSWER_CACHE_SIZE: int = 512
    ANSWER_CACHE_TTL_S: float = 3600.0
    TTS_CACHE_SIZE: int = 128

    # ── Local retrieval: Qdrant vector store ────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "school_kb"

    # ── Local embeddings (bge-m3 served by the GPU "embedder" sidecar) ───────
    # Sidecar exposes POST /embed returning dense + sparse vectors.
    EMBEDDING_BASE_URL: str = "http://localhost:8080"
    EMBEDDING_DIM: int = 1024
    # Chunks are embedded in batches of this size (one HTTP call per batch) so a
    # single huge document can't blow REQUEST_TIMEOUT_S in one giant request.
    # <=0 sends everything in a single request (the old behaviour).
    EMBED_BATCH_SIZE: int = 64

    # ── Hybrid retrieval (dense + sparse, RRF fusion) ────────────────────────
    RETRIEVAL_TOP_K: int = 5  # chunks injected into the prompt
    RETRIEVAL_CANDIDATES: int = 20  # per-branch prefetch before fusion
    RETRIEVAL_SCORE_THRESHOLD: float = 0.0

    # ── Document chunking (we now parse + chunk locally) ─────────────────────
    CHUNK_SIZE: int = 800  # characters per chunk
    CHUNK_OVERLAP: int = 120  # characters of overlap between chunks

    # ── Corpus bulk ingest (offline; scripts/manage_corpus.py) ───────────────
    # Root of the school corpus tree and where the lab-completeness manifest is
    # written. Metadata (subject/grade/lang/lab_id) is derived from the paths.
    CORPUS_ROOT: str = "./Лабораторные физхимбио"
    LABS_MANIFEST: str = "./labs.json"
    INGESTION_DATA_DIR: str = "./data/ingestion"
    INGESTION_BATCH_MAX_FILES: int = 100
    INGESTION_BATCH_MAX_BYTES: int = 1_073_741_824
    INGESTION_API_BASE_URL: str = "http://localhost:8000"

    # ── OCR fallback (opt-in; ingest-time only, server-side) ─────────────────
    # Some textbooks are scanned page-images with no text layer. When OCR_ENABLED
    # and normal extraction yields too little educational text, ingestion renders
    # the pages and runs Tesseract (rus/kaz/eng)
    # instead of skipping. OFF by default so plain bulk-ingest never shells out
    # to Tesseract and the serving path is untouched. OCR_DPI is the PDF render
    # resolution; OCR_MAX_PAGES caps pages per document (0 = all).
    OCR_ENABLED: bool = False
    OCR_DPI: int = 200
    OCR_MAX_PAGES: int = 0

    # ── Voice (in-repo STT/TTS sidecar; see ./voice) ─────────────────────────
    # The GPU `voice` container (docker-compose service) serves STT (Whisper
    # ru/kk/en/auto) and TTS (shared Qwen3-TTS/Supertonic ru+en + MMS kaz) over HTTP.
    # Kazakh OmniVoice is isolated in its own service because it requires a
    # newer Transformers release than the existing Qwen TTS backend. In compose
    # the api reaches these services by their Docker DNS names; the defaults
    # below target host-mapped ports for local development. Omitted STT language
    # auto-detects RU/KK/EN; omitted standalone TTS language follows DEFAULT_LANGUAGE.
    # VOICE_VERIFY_SSL is irrelevant over internal HTTP but kept for the client.
    VOICE_BASE_URL: str = "http://localhost:8002"
    VOICE_VERIFY_SSL: bool = False
    VOICE_TIMEOUT_S: float = 120.0  # generous: covers GPU cold start + ≤120s audio
    VOICE_TTS_RU_DEFAULT_BACKEND: str = "supertonic"
    VOICE_TTS_EN_DEFAULT_BACKEND: str = "supertonic"
    VOICE_KK_OMNIVOICE_BASE_URL: str = "http://localhost:8003"
    VOICE_TTS_KK_DEFAULT_BACKEND: str = "omnivoice"

    @field_validator("VOICE_TTS_RU_DEFAULT_BACKEND")
    @classmethod
    def _valid_tts_backend(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in {"mms", "qwen", "supertonic"}:
            raise ValueError(
                "VOICE_TTS_RU_DEFAULT_BACKEND must be one of: mms, qwen, supertonic"
            )
        return normalized

    @field_validator("VOICE_TTS_EN_DEFAULT_BACKEND")
    @classmethod
    def _valid_english_tts_backend(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in {"qwen", "supertonic"}:
            raise ValueError(
                "VOICE_TTS_EN_DEFAULT_BACKEND must be one of: qwen, supertonic"
            )
        return normalized

    @field_validator("VOICE_TTS_KK_DEFAULT_BACKEND")
    @classmethod
    def _valid_kazakh_tts_backend(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in {"mms", "omnivoice"}:
            raise ValueError(
                "VOICE_TTS_KK_DEFAULT_BACKEND must be one of: mms, omnivoice"
            )
        return normalized

    # ── Behaviour ────────────────────────────────────────────────────────────
    DEFAULT_LANGUAGE: str = "ru"

    @field_validator("DEFAULT_LANGUAGE")
    @classmethod
    def _valid_default_language(cls, v: str) -> str:
        return normalize_language_code(v, field="DEFAULT_LANGUAGE")
    MAX_INPUT_CHARS: int = 4000
    LLM_MAX_TOKENS: int = 600
    LLM_TEMPERATURE: float = 0.2
    # Outside structured lab requests, when retrieval has no relevant usable
    # evidence and no scenario/procedure is authoritative, allow reliable general
    # science knowledge. Such answers are explicitly returned without citations.
    ALLOW_GENERAL_KNOWLEDGE_FALLBACK: bool = True
    SCENARIOS_DIR: str = "./scenarios"

    # Chat memory. Client-supplied /v1/chat/completions history and ephemeral
    # server-side VR conversations share the same per-conversation bounds.
    # The server store is TTL/LRU in-process and is cleared on deploy/restart.
    CHAT_MEMORY_MAX_CONVERSATIONS: int = 1024
    CHAT_MEMORY_TTL_S: float = 7200.0
    CHAT_MEMORY_MAX_MESSAGES: int = 16
    CHAT_MEMORY_HISTORY_CHARS: int = 6000
    CHAT_MEMORY_RETRIEVAL_CONTEXT_CHARS: int = 1200

    # ── CORS / limits ────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"
    RATE_LIMIT_PER_MINUTE: int = 60  # per client IP; <=0 disables rate limiting
    MAX_UPLOAD_BYTES: int = 26_214_400  # 25 MB — OpenAI per-file ceiling for STT.
    MAX_DOCUMENT_UPLOAD_BYTES: int = 104_857_600  # 100 MB for KB admin uploads.

    # ── Runtime toggles ──────────────────────────────────────────────────────
    ENABLE_DOCS: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_GENERATION: bool = True
    LOG_GENERATION_MAX_CHARS: int = 2000

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()


_REQUIRED_ENV_VARS = [
    "INTERNAL_API_KEY",
    "ADMIN_API_KEY",
    "OPENAI_API_KEY",
]


def missing_required_env_vars() -> List[str]:
    """Return required settings that are unset/blank (checked at startup)."""
    missing = []
    for var_name in _REQUIRED_ENV_VARS:
        value = getattr(settings, var_name, "")
        if not isinstance(value, str) or not value.strip():
            missing.append(var_name)
    return missing
