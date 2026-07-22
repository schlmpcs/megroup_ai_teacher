import asyncio
import threading
import time
import wave
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest

from voice.app.config import Settings
from voice.app.main import (
    _load_stt_warmup_probes,
    _stt_keep_warm_loop,
    create_app,
)
from voice.app.stt.model import LocalWhisperSttBackend


def _stub_backend(monkeypatch):
    backend = LocalWhisperSttBackend(
        SimpleNamespace(max_audio_duration_s=120, device="cpu")
    )
    backend._processors = {"ru": object(), "kk": object()}
    calls = []

    def transcribe_reserved(audio_bytes, language):
        calls.append((language, audio_bytes))
        return {"text": "", "language": language, "duration_ms": 0}

    monkeypatch.setattr(backend, "_transcribe_reserved", transcribe_reserved)
    return backend, calls


def test_bundled_spoken_probes_are_short_16khz_mono_pcm():
    probes = _load_stt_warmup_probes()

    assert list(probes) == ["ru", "kk"]
    for audio_bytes in probes.values():
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16_000
            duration_s = wav_file.getnframes() / wav_file.getframerate()
            assert 0.2 < duration_s < 2.0
            assert any(wav_file.readframes(wav_file.getnframes()))


def test_warmup_runs_ru_then_kk_without_resetting_real_idle(monkeypatch):
    backend, calls = _stub_backend(monkeypatch)
    backend._last_real_inference_completed_at = 100.0
    backend._last_warmup_completed_at = 200.0
    monkeypatch.setattr("voice.app.stt.model.time.monotonic", lambda: 500.0)

    warmed = backend.warm_up(
        {"ru": b"ru", "kk": b"kk"},
        blocking=False,
        min_real_idle_s=180.0,
        min_warmup_interval_s=240.0,
    )

    assert warmed is True
    assert calls == [("ru", b"ru"), ("kk", b"kk")]
    assert backend._last_real_inference_completed_at == 100.0
    assert backend._last_warmup_completed_at == 500.0


def test_warmup_requires_real_idle_and_separate_warmup_interval(monkeypatch):
    backend, calls = _stub_backend(monkeypatch)
    monkeypatch.setattr("voice.app.stt.model.time.monotonic", lambda: 500.0)

    backend._last_real_inference_completed_at = 400.0
    backend._last_warmup_completed_at = 100.0
    assert not backend.warm_up(
        {"ru": b"ru"},
        blocking=False,
        min_real_idle_s=180.0,
        min_warmup_interval_s=240.0,
    )

    backend._last_real_inference_completed_at = 100.0
    backend._last_warmup_completed_at = 400.0
    assert not backend.warm_up(
        {"ru": b"ru"},
        blocking=False,
        min_real_idle_s=180.0,
        min_warmup_interval_s=240.0,
    )
    assert calls == []


def test_periodic_warmup_skips_queued_real_inference(monkeypatch):
    backend, calls = _stub_backend(monkeypatch)
    backend._last_real_inference_completed_at = 0.0
    backend._last_warmup_completed_at = 0.0
    monkeypatch.setattr("voice.app.stt.model.time.monotonic", lambda: 1_000.0)

    backend._inference_gate.acquire()
    real_thread = threading.Thread(
        target=backend.transcribe,
        args=(b"real", "ru"),
        daemon=True,
    )
    real_thread.start()
    deadline = time.monotonic() + 1.0
    while backend._pending_real_inferences != 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert backend._pending_real_inferences == 1

    started_at = time.monotonic()
    warmed = backend.warm_up(
        {"ru": b"ru", "kk": b"kk"},
        blocking=False,
        min_real_idle_s=180.0,
        min_warmup_interval_s=240.0,
    )

    assert warmed is False
    assert time.monotonic() - started_at < 0.1
    assert calls == []

    backend._inference_gate.release()
    real_thread.join(timeout=1.0)
    assert not real_thread.is_alive()
    assert calls == [("ru", b"real")]
    assert backend._pending_real_inferences == 0


def test_real_failure_updates_activity_and_releases_reservation(monkeypatch):
    backend, _ = _stub_backend(monkeypatch)

    def fail(_audio_bytes, _language):
        raise ValueError("bad audio")

    monkeypatch.setattr(backend, "_transcribe_reserved", fail)
    monkeypatch.setattr("voice.app.stt.model.time.monotonic", lambda: 321.0)

    with pytest.raises(ValueError, match="bad audio"):
        backend.transcribe(b"bad", "ru")

    assert backend._pending_real_inferences == 0
    assert backend._last_real_inference_completed_at == 321.0
    assert backend._inference_gate.acquire(blocking=False)
    backend._inference_gate.release()


def test_partial_warmup_failure_counts_as_attempt_and_releases_gate(monkeypatch):
    backend, _ = _stub_backend(monkeypatch)
    backend._last_real_inference_completed_at = 0.0
    backend._last_warmup_completed_at = 0.0
    monkeypatch.setattr("voice.app.stt.model.time.monotonic", lambda: 500.0)

    def fail_on_kazakh(_audio_bytes, language):
        if language == "kk":
            raise RuntimeError("kk failed")
        return {"text": "", "language": language, "duration_ms": 0}

    monkeypatch.setattr(backend, "_transcribe_reserved", fail_on_kazakh)

    with pytest.raises(RuntimeError, match="kk failed"):
        backend.warm_up(
            {"ru": b"ru", "kk": b"kk"},
            blocking=False,
            min_real_idle_s=180.0,
            min_warmup_interval_s=240.0,
        )

    assert backend._last_real_inference_completed_at == 0.0
    assert backend._last_warmup_completed_at == 500.0
    assert backend._inference_gate.acquire(blocking=False)
    backend._inference_gate.release()


class _FakeSttBackend:
    def __init__(self, events):
        self.events = events
        self.loaded = []

    def load_models(self):
        self.events.append("stt-load")
        self.loaded = ["stt_ru", "stt_kk"]

    def transcribe(self, _audio_bytes, language):
        return {"text": "ok", "language": language, "duration_ms": 0}

    def warm_up(self, probes, **kwargs):
        self.events.append(("warm", tuple(probes), kwargs))
        return True


class _FakeTtsBackend:
    loaded = []

    def __init__(self, events):
        self.events = events

    def load_models(self):
        self.events.append("tts-load")
        self.loaded = ["tts_ru", "tts_kk"]

    def select_backend(self, _language, _backend=None):
        return "fake"

    def synthesize(self, *_args, **_kwargs):
        return b"RIFF"


async def test_lifespan_warms_after_models_and_reports_language_capabilities():
    events = []
    settings = Settings(stt_keep_warm_enabled=False)
    app = create_app(
        stt_backend=_FakeSttBackend(events),
        tts_backend=_FakeTtsBackend(events),
        settings=settings,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/health")

    assert events[0:2] == ["stt-load", "tts-load"]
    assert events[2][0:2] == ("warm", ("ru", "kk"))
    assert events[2][2] == {
        "blocking": True,
        "min_real_idle_s": 0.0,
        "min_warmup_interval_s": 0.0,
    }
    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "supported_languages",
        "stt_models",
        "tts_models",
        "tts_backends",
        "tts_default_backends",
        "tts_number_normalization",
    }


async def test_startup_warmup_failure_fails_lifespan():
    events = []

    class FailingWarmupBackend(_FakeSttBackend):
        def warm_up(self, probes, **kwargs):
            super().warm_up(probes, **kwargs)
            return False

    app = create_app(
        stt_backend=FailingWarmupBackend(events),
        tts_backend=_FakeTtsBackend(events),
        settings=Settings(stt_keep_warm_enabled=False),
    )

    with pytest.raises(RuntimeError, match="Startup STT warm-up"):
        async with app.router.lifespan_context(app):
            pass


async def test_lifespan_cancels_and_awaits_sleeping_keep_warm_task():
    events = []
    settings = Settings(
        stt_keep_warm_enabled=True,
        stt_keep_warm_poll_s=60.0,
    )
    app = create_app(
        stt_backend=_FakeSttBackend(events),
        tts_backend=_FakeTtsBackend(events),
        settings=settings,
    )

    task = None
    started_at = time.monotonic()
    async with app.router.lifespan_context(app):
        task = app.state.stt_keep_warm_task
        assert task is not None
        assert not task.done()

    assert task.done()
    assert time.monotonic() - started_at < 1.0


async def test_cancellation_waits_for_inflight_warmup_worker():
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend:
        def warm_up(self, _probes, **_kwargs):
            started.set()
            release.wait(timeout=1.0)
            return True

    task = asyncio.create_task(
        _stt_keep_warm_loop(
            BlockingBackend(),
            {"ru": b"ru", "kk": b"kk"},
            poll_s=0.001,
            real_idle_s=180.0,
            interval_s=240.0,
        )
    )
    deadline = asyncio.get_running_loop().time() + 1.0
    while not started.is_set():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.001)

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_worker_failure_during_cancellation_preserves_cancellation():
    started = threading.Event()
    release = threading.Event()

    class FailingBackend:
        def warm_up(self, _probes, **_kwargs):
            started.set()
            release.wait(timeout=1.0)
            raise RuntimeError("probe failed during shutdown")

    task = asyncio.create_task(
        _stt_keep_warm_loop(
            FailingBackend(),
            {"ru": b"ru", "kk": b"kk"},
            poll_s=0.001,
            real_idle_s=180.0,
            interval_s=240.0,
        )
    )
    deadline = asyncio.get_running_loop().time() + 1.0
    while not started.is_set():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.001)

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_periodic_loop_logs_failure_and_continues(caplog):
    calls = 0
    completed = asyncio.Event()
    event_loop = asyncio.get_running_loop()

    class FlakyBackend:
        def warm_up(self, _probes, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary failure")
            event_loop.call_soon_threadsafe(completed.set)
            return True

    task = asyncio.create_task(
        _stt_keep_warm_loop(
            FlakyBackend(),
            {"ru": b"ru", "kk": b"kk"},
            poll_s=0.01,
            real_idle_s=180.0,
            interval_s=240.0,
        )
    )
    try:
        await asyncio.wait_for(completed.wait(), timeout=1.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls >= 2
    assert "Periodic STT warm-up failed" in caplog.text
