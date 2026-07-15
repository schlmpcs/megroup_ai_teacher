import app.services.llm as llm
import app.services.ttl_cache as ttl_cache_mod
import app.services.voice as voice
from app.services.llm import AnswerResult
from app.services.ttl_cache import TTLCache


# ── TTLCache ─────────────────────────────────────────────────────────────────


def test_ttl_cache_hit_and_miss():
    c = TTLCache(max_size=2, ttl_s=60)
    assert c.get("a") is None
    c.put("a", 1)
    assert c.get("a") == 1


def test_ttl_cache_expiry(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(ttl_cache_mod.time, "monotonic", lambda: now[0])
    c = TTLCache(max_size=2, ttl_s=10)
    c.put("a", 1)
    now[0] = 109.9
    assert c.get("a") == 1
    now[0] = 110.0
    assert c.get("a") is None


def test_ttl_cache_lru_eviction():
    c = TTLCache(max_size=2, ttl_s=60)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")  # refresh "a" so "b" is the LRU entry
    c.put("c", 3)
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_ttl_cache_disabled():
    c = TTLCache(max_size=0, ttl_s=60)
    c.put("a", 1)
    assert c.get("a") is None
    assert not c.enabled


def test_ttl_cache_clear_removes_all_entries():
    c = TTLCache(max_size=2, ttl_s=60)
    c.put("a", 1)
    c.put("b", 2)

    c.clear()

    assert c.get("a") is None
    assert c.get("b") is None


# ── Answer cache in generate_answer / stream_answer ─────────────────────────


class _FakeUsage:
    input_tokens = 5
    output_tokens = 7
    total_tokens = 12


class _FakeResponse:
    output_text = "Кипение — это парообразование."
    usage = _FakeUsage()


class _FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


async def _no_retrieve(query, **kwargs):
    return []


async def test_generate_answer_uses_cache(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(llm, "client", fake)
    monkeypatch.setattr(llm, "_retrieve", _no_retrieve)

    r1 = await llm.generate_answer("Что такое кипение?")
    r2 = await llm.generate_answer("что   такое кипение?")  # normalizes to same key
    assert len(fake.responses.calls) == 1
    assert r2 is r1

    # different scenario state -> different key -> new generation
    await llm.generate_answer("Что такое кипение?", scenario_state="шаг 2")
    assert len(fake.responses.calls) == 2


async def test_generate_answer_multiturn_not_cached(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(llm, "client", fake)
    monkeypatch.setattr(llm, "_retrieve", _no_retrieve)

    history = [
        {"role": "user", "content": "А почему?"},
        {"role": "assistant", "content": "Потому."},
        {"role": "user", "content": "А почему?"},
    ]
    await llm.generate_answer("А почему?", chat_history=history)
    await llm.generate_answer("А почему?", chat_history=history)
    assert len(fake.responses.calls) == 2


async def test_generate_answer_passes_service_tier(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(llm, "client", fake)
    monkeypatch.setattr(llm, "_retrieve", _no_retrieve)
    monkeypatch.setattr(llm.settings, "OPENAI_SERVICE_TIER", "priority")

    await llm.generate_answer("Что такое диффузия?")
    assert fake.responses.calls[0]["service_tier"] == "priority"


async def test_stream_answer_cache_hit_yields_full_answer():
    result = AnswerResult(
        answer="Готовый ответ.",
        citations=[{"filename": "a.pdf", "file_id": "f"}],
        usage={"total_tokens": 3},
    )
    key = llm._answer_cache_key("Вопрос?", None, None, None, None)
    llm._answer_cache.put(key, result)

    events = [e async for e in llm.stream_answer("Вопрос?")]
    assert events == [
        {"type": "delta", "text": "Готовый ответ."},
        {"type": "done", "citations": result.citations, "usage": result.usage},
    ]


# ── TTS cache in voice.synthesize ────────────────────────────────────────────


class _FakeTTSResponse:
    content = b"WAVBYTES"

    def raise_for_status(self):
        pass


class _FakeTTSClient:
    def __init__(self):
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        return _FakeTTSResponse()


async def test_synthesize_uses_cache(monkeypatch):
    fake = _FakeTTSClient()
    monkeypatch.setattr(voice, "_http", lambda: fake)

    a1 = await voice.synthesize("Привет", language="ru")
    a2 = await voice.synthesize("Привет", language="ru")
    assert fake.calls == 1
    assert a1 == a2 == (b"WAVBYTES", "audio/wav")

    await voice.synthesize("Привет", language="kk")  # different language -> miss
    assert fake.calls == 2

    await voice.synthesize("Привет", language="ru", backend="qwen")
    assert fake.calls == 3
