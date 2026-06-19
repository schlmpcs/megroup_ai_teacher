"""Single shared AsyncOpenAI client.

Centralised so every service uses the same configured client and so tests can
monkeypatch ``app.services.openai_client.client`` with a fake.
"""

from openai import AsyncOpenAI

from app.core.config import settings


def _build_client() -> AsyncOpenAI:
    kwargs = {
        "api_key": settings.OPENAI_API_KEY or "EMPTY",
        "timeout": settings.REQUEST_TIMEOUT_S,
    }
    if settings.OPENAI_BASE_URL.strip():
        kwargs["base_url"] = settings.OPENAI_BASE_URL.strip()
    return AsyncOpenAI(**kwargs)


client: AsyncOpenAI = _build_client()
