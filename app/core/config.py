from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    @field_validator("INTERNAL_API_KEY")
    @classmethod
    def _must_not_be_default(cls, v: str) -> str:
        if v in _DEFAULT_SECRETS:
            raise ValueError(
                "INTERNAL_API_KEY must be set to a strong secret "
                '(generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")'
            )
        return v

    # ── OpenAI (key required) ────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    REQUEST_TIMEOUT_S: float = 60.0

    # ── Local retrieval: Qdrant vector store ────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "school_kb"

    # ── Local embeddings (bge-m3 served by the GPU "embedder" sidecar) ───────
    # Sidecar exposes POST /embed returning dense + sparse vectors.
    EMBEDDING_BASE_URL: str = "http://localhost:8080"
    EMBEDDING_DIM: int = 1024

    # ── Hybrid retrieval (dense + sparse, RRF fusion) ────────────────────────
    RETRIEVAL_TOP_K: int = 5          # chunks injected into the prompt
    RETRIEVAL_CANDIDATES: int = 20    # per-branch prefetch before fusion
    RETRIEVAL_SCORE_THRESHOLD: float = 0.0

    # ── Document chunking (we now parse + chunk locally) ─────────────────────
    CHUNK_SIZE: int = 800             # characters per chunk
    CHUNK_OVERLAP: int = 120          # characters of overlap between chunks

    # ── Corpus bulk ingest (offline; scripts/manage_corpus.py) ───────────────
    # Root of the school corpus tree and where the lab-completeness manifest is
    # written. Metadata (subject/grade/lang/lab_id) is derived from the paths.
    CORPUS_ROOT: str = "./Лабораторные физхимбио"
    LABS_MANIFEST: str = "./labs.json"

    # ── Voice (in-repo STT/TTS sidecar; see ./voice) ─────────────────────────
    # The GPU `voice` container (docker-compose service) serves STT (Whisper
    # ru/kk/auto) and TTS (supertonic ru + MMS kaz) over plain HTTP. In compose
    # the api reaches it at http://voice:8001; the default below targets the
    # host-mapped port for local dev. STT/TTS language follows DEFAULT_LANGUAGE.
    # VOICE_VERIFY_SSL is irrelevant over internal HTTP but kept for the client.
    VOICE_BASE_URL: str = "http://localhost:8002"
    VOICE_VERIFY_SSL: bool = False
    VOICE_TIMEOUT_S: float = 120.0  # generous: covers GPU cold start + ≤120s audio

    # ── Behaviour ────────────────────────────────────────────────────────────
    DEFAULT_LANGUAGE: str = "ru"
    MAX_INPUT_CHARS: int = 4000
    LLM_MAX_TOKENS: int = 600
    LLM_TEMPERATURE: float = 0.2
    SCENARIOS_DIR: str = "./scenarios"

    # Chat memory — request-scoped, no persistence.
    CHAT_MEMORY_MAX_MESSAGES: int = 16
    CHAT_MEMORY_HISTORY_CHARS: int = 6000

    # ── CORS / limits ────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_UPLOAD_BYTES: int = 26_214_400  # 25 MB — OpenAI per-file ceiling for STT.

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
