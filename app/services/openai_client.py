"""Single shared AsyncOpenAI client.

Centralised so every service uses the same configured client and so tests can
monkeypatch ``app.services.openai_client.client`` with a fake.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from openai import AsyncOpenAI

from app.core.config import settings


_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_KEEP_WARM_INTERVAL_S = 240.0
logger = logging.getLogger("assistant.openai")


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


async def _warm_once() -> bool:
    try:
        await client.responses.create(
            model=settings.OPENAI_MODEL,
            input="Reply with OK.",
            max_output_tokens=16,
            temperature=0,
            store=False,
            **(
                {"service_tier": settings.OPENAI_SERVICE_TIER}
                if settings.OPENAI_SERVICE_TIER
                else {}
            ),
        )
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        logger.warning("OpenAI warm-up failed: %s", exc)
        return False
    return True


async def _keep_warm_loop() -> None:
    while True:
        await asyncio.sleep(_KEEP_WARM_INTERVAL_S)
        await _warm_once()


@asynccontextmanager
async def keep_warm_lifespan(_app):
    if await _warm_once():
        logger.info("OpenAI generation path warmed")
    task = asyncio.create_task(_keep_warm_loop(), name="openai-keep-warm")
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
