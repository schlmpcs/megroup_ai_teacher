import asyncio
from types import SimpleNamespace

from app import main
from app.core.config import settings
from app.services import openai_client


async def test_app_lifespan_primes_and_keeps_openai_warm(monkeypatch):
    calls = []
    second_call = asyncio.Event()

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) >= 2:
            second_call.set()
        return SimpleNamespace()

    monkeypatch.setattr(
        openai_client,
        "client",
        SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        openai_client,
        "_KEEP_WARM_INTERVAL_S",
        0.001,
        raising=False,
    )
    monkeypatch.setattr(main, "missing_required_env_vars", lambda: [])
    monkeypatch.setattr(main.ingestion_jobs, "initialize", lambda: None)
    monkeypatch.setattr(main.ingestion_jobs, "cleanup_pending_deletions", lambda: None)
    monkeypatch.setattr(main.ingestion_jobs, "cleanup_stale_tmp", lambda: None)

    async with main.app.router.lifespan_context(main.app):
        assert len(calls) == 1
        await asyncio.wait_for(second_call.wait(), timeout=0.1)

    completed_calls = len(calls)
    await asyncio.sleep(0.01)

    assert len(calls) == completed_calls
    assert all(call["model"] == settings.OPENAI_MODEL for call in calls)
    assert all(call["max_output_tokens"] == 16 for call in calls)
    assert all(call["store"] is False for call in calls)
