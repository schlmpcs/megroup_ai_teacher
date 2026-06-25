"""Single shared AsyncOpenAI client.

Centralised so every service uses the same configured client and so tests can
monkeypatch ``app.services.openai_client.client`` with a fake.
"""

from openai import AsyncOpenAI

from app.core.config import settings


_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _build_client() -> AsyncOpenAI:
    # Always pass base_url explicitly. If we left it unset, the OpenAI SDK reads
    # the OPENAI_BASE_URL env var itself — and a *present-but-empty* value (e.g.
    # `OPENAI_BASE_URL=` in .env, the documented "leave blank for api.openai.com"
    # case) becomes a hostless base URL, so every request fails with
    # httpx.UnsupportedProtocol -> APIConnectionError. Passing an explicit
    # default makes "blank" mean the public endpoint, as documented.
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY or "EMPTY",
        timeout=settings.REQUEST_TIMEOUT_S,
        base_url=settings.OPENAI_BASE_URL.strip() or _DEFAULT_OPENAI_BASE_URL,
    )


client: AsyncOpenAI = _build_client()
