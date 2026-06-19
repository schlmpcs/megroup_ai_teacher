import json

import pytest

import app.api.routes as routes
from app.services.llm import AnswerResult, LLMTimeoutError


@pytest.fixture
def fake_answer(monkeypatch):
    async def _gen(
        query,
        scenario_context=None,
        chat_history=None,
        max_tokens=None,
        scenario_state=None,
        lab=None,
    ):
        # The scenario context should be threaded through when a scenario_id is given.
        suffix = " [scenario]" if scenario_context else ""
        if scenario_state:
            suffix += " [state]"
        if lab:
            suffix += f" [lab:{lab.get('lab_id')}]"
        return AnswerResult(
            answer=f"Ответ на: {query}{suffix}",
            citations=[{"filename": "physics_8.pdf", "file_id": "f1"}],
            usage={"total_tokens": 10},
        )

    monkeypatch.setattr(routes, "generate_answer", _gen)
    return _gen


# ── Auth ────────────────────────────────────────────────────────────────────


def test_missing_auth_rejected(client):
    r = client.post("/ask", json={"query": "привет"})
    assert r.status_code == 401


def test_bad_auth_rejected(client):
    r = client.post("/ask", json={"query": "привет"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 403


def test_health_no_auth(client):
    assert client.get("/health").status_code == 200


# ── /ask ──────────────────────────────────────────────────────────────────


def test_ask_returns_answer_and_citations(client, auth, fake_answer):
    r = client.post("/ask", json={"query": "Что такое кипение?"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Ответ на: Что такое кипение?")
    assert body["primary_source"]["filename"] == "physics_8.pdf"
    assert body["scenario_id"] is None


def test_ask_threads_scenario_context(client, auth, fake_answer):
    r = client.post(
        "/ask",
        json={"query": "Где термометр?", "scenario_id": "physics_lab_02_heating"},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["answer"].endswith("[scenario]")


def test_ask_threads_scenario_state(client, auth, fake_answer):
    r = client.post(
        "/ask",
        json={
            "query": "Что мне делать дальше?",
            "scenario_id": "physics_lab_02_heating",
            "scenario_state": {
                "current_step": "Зажечь спиртовку спичкой",
                "held_items": ["спички", "спиртовка"],
            },
        },
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["answer"].endswith("[scenario] [state]")


def test_ask_empty_scenario_state_is_noop(client, auth, fake_answer):
    # An empty state object must not inject a (blank) state block.
    r = client.post(
        "/ask",
        json={"query": "Привет", "scenario_state": {}},
        headers=auth,
    )
    assert r.status_code == 200
    assert "[state]" not in r.json()["answer"]


def test_ask_unknown_scenario_404(client, auth, fake_answer):
    r = client.post("/ask", json={"query": "x", "scenario_id": "nope"}, headers=auth)
    assert r.status_code == 404


def test_ask_blank_query_422(client, auth):
    r = client.post("/ask", json={"query": "   "}, headers=auth)
    assert r.status_code == 422


def test_ask_maps_timeout_to_504(client, auth, monkeypatch):
    async def _boom(*a, **k):
        raise LLMTimeoutError("slow")

    monkeypatch.setattr(routes, "generate_answer", _boom)
    r = client.post("/ask", json={"query": "x"}, headers=auth)
    assert r.status_code == 504


# ── /v1/chat/completions ─────────────────────────────────────────────────────


def test_chat_completions_nonstream(client, auth, fake_answer):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Привет"}]},
        headers=auth,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].startswith("Ответ на: Привет")
    assert body["metadata"]["primary_source"]["file_id"] == "f1"


def test_chat_completions_latest_must_be_user(client, auth):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "assistant", "content": "hi"}]},
        headers=auth,
    )
    assert r.status_code == 422


def test_chat_completions_stream(client, auth, monkeypatch):
    async def _stream(
        query,
        scenario_context=None,
        chat_history=None,
        max_tokens=None,
        scenario_state=None,
        lab=None,
    ):
        yield {"type": "delta", "text": "Ответ "}
        yield {"type": "delta", "text": "готов"}
        yield {"type": "done", "citations": [{"filename": "a.pdf", "file_id": "f"}], "usage": {}}

    monkeypatch.setattr(routes, "stream_answer", _stream)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "stream": True,
            "messages": [{"role": "user", "content": "Привет"}],
        },
        headers=auth,
    )
    assert r.status_code == 200
    text = r.text
    assert "Ответ " in text
    assert "готов" in text
    assert "[DONE]" in text
    # metadata frame carries citations
    assert "a.pdf" in text


# ── /hint ────────────────────────────────────────────────────────────────────


def test_hint_rephrases(client, auth, monkeypatch):
    async def _hint(hint_text, hint_level, scenario_context=None, scenario_state=None):
        return f"L{hint_level}: {hint_text}"

    monkeypatch.setattr(routes, "rephrase_hint", _hint)
    r = client.post(
        "/hint",
        json={"hint_text": "Подойди к трубке", "hint_level": 2},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["hint"] == "L2: Подойди к трубке"


def test_hint_level_out_of_range_422(client, auth):
    r = client.post("/hint", json={"hint_text": "x", "hint_level": 9}, headers=auth)
    assert r.status_code == 422


# ── Voice ────────────────────────────────────────────────────────────────────


def test_stt(client, auth, monkeypatch):
    async def _transcribe(audio_bytes, filename="audio.webm", language=None, prompt=None):
        return "распознанный текст"

    monkeypatch.setattr(routes, "transcribe", _transcribe)
    r = client.post(
        "/stt",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["text"] == "распознанный текст"


def test_stt_empty_file_400(client, auth):
    r = client.post("/stt", files={"file": ("q.webm", b"", "audio/webm")}, headers=auth)
    assert r.status_code == 400


def test_tts(client, auth, monkeypatch):
    async def _synth(text, voice=None, response_format=None, instructions=None, language=None):
        return b"AUDIOBYTES", "audio/wav"

    monkeypatch.setattr(routes, "synthesize", _synth)
    r = client.post("/tts", json={"text": "Привет, ученик"}, headers=auth)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content == b"AUDIOBYTES"


def test_voice_ask_full_pipeline(client, auth, monkeypatch, fake_answer):
    async def _transcribe(audio_bytes, filename="audio.webm", language=None, prompt=None):
        return "Зачем нагревать пробирку?"

    async def _synth(text, voice=None, response_format=None, instructions=None, language=None):
        return b"SPOKEN", "audio/wav"

    monkeypatch.setattr(routes, "transcribe", _transcribe)
    monkeypatch.setattr(routes, "synthesize", _synth)

    r = client.post(
        "/voice_ask",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        data={"scenario_id": "physics_lab_02_heating"},
        headers=auth,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "Зачем нагревать пробирку?"
    assert body["answer"].endswith("[scenario]")
    import base64

    assert base64.b64decode(body["audio_base64"]) == b"SPOKEN"
    assert "stt" in body["observability"]["latency_ms"]
    assert "tts" in body["observability"]["latency_ms"]


# ── Admin ────────────────────────────────────────────────────────────────────


def test_corpus_status(client, auth, monkeypatch):
    async def _status():
        return {"status": "ready", "file_counts": {"total": 3}}

    monkeypatch.setattr(routes.ingestion, "corpus_status", _status)
    r = client.get("/admin/corpus_status", headers=auth)
    assert r.status_code == 200
    assert r.json()["file_counts"]["total"] == 3


def test_list_scenarios_endpoint(client, auth):
    r = client.get("/admin/scenarios", headers=auth)
    assert r.status_code == 200
    ids = [s["scenario_id"] for s in r.json()["scenarios"]]
    assert "physics_lab_02_heating" in ids


def test_upload_unsupported_type_400(client, auth):
    r = client.post(
        "/admin/documents",
        files={"file": ("notes.xyz", b"data", "application/octet-stream")},
        headers=auth,
    )
    assert r.status_code == 400
