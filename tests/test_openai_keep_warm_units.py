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


def test_generation_kwargs_reasoning_drops_temperature_and_adds_headroom(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setattr(settings, "OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setattr(settings, "OPENAI_REASONING_TOKEN_HEADROOM", 1024)
    monkeypatch.setattr(settings, "OPENAI_SERVICE_TIER", "")

    kwargs = openai_client.generation_kwargs(max_output_tokens=600, temperature=0.2)

    assert kwargs == {
        "model": "gpt-6-luna",
        "reasoning": {"effort": "low"},
        "max_output_tokens": 1624,
    }


def test_generation_kwargs_warmup_skips_headroom(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setattr(settings, "OPENAI_REASONING_TOKEN_HEADROOM", 1024)

    kwargs = openai_client.generation_kwargs(
        max_output_tokens=16, temperature=0, reasoning_headroom=False
    )

    assert kwargs["max_output_tokens"] == 16
    assert "temperature" not in kwargs


def test_generation_kwargs_effort_none_keeps_temperature(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_REASONING_EFFORT", "none")
    monkeypatch.setattr(settings, "OPENAI_SERVICE_TIER", "priority")

    kwargs = openai_client.generation_kwargs(max_output_tokens=600, temperature=0.2)

    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_output_tokens"] == 600
    assert kwargs["service_tier"] == "priority"


def test_generation_kwargs_blank_effort_omits_reasoning(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_REASONING_EFFORT", "")

    kwargs = openai_client.generation_kwargs(max_output_tokens=600, temperature=0.2)

    assert "reasoning" not in kwargs
    assert kwargs["temperature"] == 0.2


def test_reasoning_effort_setting_is_validated():
    import pytest

    from app.core.config import Settings

    with pytest.raises(ValueError):
        Settings.model_validate(
            {**settings.model_dump(), "OPENAI_REASONING_EFFORT": "turbo"}
        )
    assert (
        Settings.model_validate(
            {**settings.model_dump(), "OPENAI_REASONING_EFFORT": " LOW "}
        ).OPENAI_REASONING_EFFORT
        == "low"
    )
