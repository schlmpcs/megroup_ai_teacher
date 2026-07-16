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
        _gen.calls.append(
            {
                "query": query,
                "scenario_context": scenario_context,
                "chat_history": chat_history,
                "scenario_state": scenario_state,
                "lab": lab,
            }
        )
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

    _gen.calls = []
    monkeypatch.setattr(routes, "generate_answer", _gen)
    return _gen


# ── Auth ────────────────────────────────────────────────────────────────────


def test_missing_auth_rejected(client):
    r = client.post("/ask", json={"query": "привет"})
    assert r.status_code == 401


def test_bad_auth_rejected(client):
    r = client.post(
        "/ask", json={"query": "привет"}, headers={"Authorization": "Bearer nope"}
    )
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
    assert body["conversation_id"]
    assert body["scenario_id"] is None


def test_ask_reuses_conversation_history_for_followup(client, auth, fake_answer):
    conversation_id = "vr-session-physics-1"
    first = client.post(
        "/ask",
        json={
            "query": "Что такое кипение?",
            "conversation_id": conversation_id,
        },
        headers=auth,
    )
    second = client.post(
        "/ask",
        json={
            "query": "Почему оно начинается?",
            "conversation_id": conversation_id,
        },
        headers=auth,
    )

    assert first.status_code == second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert fake_answer.calls[-1]["chat_history"] == [
        {"role": "user", "content": "Что такое кипение?"},
        {"role": "assistant", "content": "Ответ на: Что такое кипение?"},
        {"role": "user", "content": "Почему оно начинается?"},
    ]


def test_ask_rejects_invalid_conversation_id(client, auth):
    r = client.post(
        "/ask",
        json={"query": "Привет", "conversation_id": "invalid id with spaces"},
        headers=auth,
    )
    assert r.status_code == 422


def test_clear_conversation_forgets_history(client, auth, fake_answer):
    conversation_id = "vr-session-clear"
    client.post(
        "/ask",
        json={"query": "Первый вопрос", "conversation_id": conversation_id},
        headers=auth,
    )

    cleared = client.delete(f"/v1/conversations/{conversation_id}", headers=auth)
    second = client.post(
        "/ask",
        json={"query": "Новый вопрос", "conversation_id": conversation_id},
        headers=auth,
    )

    assert cleared.status_code == 200
    assert cleared.json() == {"conversation_id": conversation_id, "cleared": True}
    assert second.status_code == 200
    assert fake_answer.calls[-1]["chat_history"] == [
        {"role": "user", "content": "Новый вопрос"}
    ]


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
                "current_step_id": "ignite-burner",
                "current_step_index": 2,
                "current_step": "Зажечь спиртовку спичкой",
                "next_step_id": "heat-water",
                "next_step": "Начать нагрев воды",
                "completed_steps": ["prepare-workplace"],
                "held_items": ["спички", "спиртовка"],
                "visible_items": ["стакан", "термометр"],
                "allowed_actions": ["зажечь спиртовку"],
                "last_action": "Поднёс спичку к фитилю",
                "last_action_result": "Фитиль ещё не загорелся",
            },
        },
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["answer"].endswith("[scenario] [state]")
    state = fake_answer.calls[-1]["scenario_state"]
    assert "актуальный снимок сцены от симулятора" in state
    assert "ID текущего шага: ignite-burner" in state
    assert "Индекс текущего шага: 2" in state
    assert "Следующий шаг, назначенный симулятором: Начать нагрев воды" in state
    assert "Завершённые шаги: prepare-workplace" in state
    assert "Предметы, видимые ученику: стакан, термометр" in state
    assert "Разрешённые действия сейчас: зажечь спиртовку" in state
    assert "Результат последнего действия: Фитиль ещё не загорелся" in state


def test_ask_empty_scenario_state_is_noop(client, auth, fake_answer):
    # An empty state object must not inject a (blank) state block.
    r = client.post(
        "/ask",
        json={"query": "Привет", "scenario_state": {}},
        headers=auth,
    )
    assert r.status_code == 200
    assert "[state]" not in r.json()["answer"]


def test_ask_explicit_empty_held_items_is_authoritative(client, auth, fake_answer):
    r = client.post(
        "/ask",
        json={"query": "Что у меня в руках?", "scenario_state": {"held_items": []}},
        headers=auth,
    )
    assert r.status_code == 200
    assert "Предметы в руках у ученика: нет" in fake_answer.calls[-1]["scenario_state"]


def test_ask_rejects_oversized_scenario_state(client, auth):
    r = client.post(
        "/ask",
        json={
            "query": "Что дальше?",
            "scenario_state": {
                "current_step": "x" * routes.settings.MAX_INPUT_CHARS,
                "next_step": "y",
            },
        },
        headers=auth,
    )
    assert r.status_code == 422


def test_ask_unknown_scenario_404(client, auth, fake_answer):
    r = client.post("/ask", json={"query": "x", "scenario_id": "nope"}, headers=auth)
    assert r.status_code == 404


def test_ask_blank_query_422(client, auth):
    r = client.post("/ask", json={"query": "   "}, headers=auth)
    assert r.status_code == 422


def _sse_events(text):
    """Parse SSE body into a list of decoded JSON events (excluding [DONE])."""
    events = []
    for frame in text.split("\n\n"):
        if not frame.startswith("data: "):
            continue
        payload = frame[len("data: ") :]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_ask_stream(client, auth, monkeypatch):
    async def _stream(
        query,
        scenario_context=None,
        chat_history=None,
        max_tokens=None,
        scenario_state=None,
        lab=None,
    ):
        yield {"type": "delta", "text": "Кипение — "}
        yield {"type": "delta", "text": "это парообразование."}
        yield {
            "type": "done",
            "citations": [{"filename": "physics_8.pdf", "file_id": "f1"}],
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(routes, "stream_answer", _stream)
    r = client.post(
        "/ask", json={"query": "Что такое кипение?", "stream": True}, headers=auth
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" in r.text
    events = _sse_events(r.text)
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "Кипение — это парообразование."
    done = next(e for e in events if e["type"] == "done")
    assert done["primary_source"]["filename"] == "physics_8.pdf"


def test_ask_stream_remembers_answer_for_followup(
    client, auth, monkeypatch, fake_answer
):
    async def _stream(
        query,
        scenario_context=None,
        chat_history=None,
        max_tokens=None,
        scenario_state=None,
        lab=None,
    ):
        yield {"type": "delta", "text": "Первый "}
        yield {"type": "delta", "text": "ответ."}
        yield {"type": "done", "citations": [], "usage": {}}

    monkeypatch.setattr(routes, "stream_answer", _stream)
    conversation_id = "ask-stream-followup"
    first = client.post(
        "/ask",
        json={
            "query": "Первый вопрос",
            "conversation_id": conversation_id,
            "stream": True,
        },
        headers=auth,
    )
    second = client.post(
        "/ask",
        json={"query": "А почему?", "conversation_id": conversation_id},
        headers=auth,
    )

    assert first.status_code == second.status_code == 200
    assert fake_answer.calls[-1]["chat_history"] == [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ."},
        {"role": "user", "content": "А почему?"},
    ]


def test_split_ready():
    # incomplete sentence stays buffered
    assert routes._split_ready("Вода закипает при") == ([], "Вода закипает при")
    # complete sentence flushes, tail is kept
    ready, rest = routes._split_ready("Вода закипает при ста градусах. Это проц")
    assert ready == ["Вода закипает при ста градусах."]
    assert rest == "Это проц"
    # short fragments (list markers, decimals) are merged forward, not flushed
    assert routes._split_ready("1. Возьмите линейку")[0] == []
    assert routes._split_ready("Число пи равно 3.14 примерно") == (
        [],
        "Число пи равно 3.14 примерно",
    )


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
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Привет"}],
            "scenario_state": {
                "current_step_id": "observe",
                "next_step": "Записать наблюдение",
                "allowed_actions": ["открыть журнал"],
            },
        },
        headers=auth,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].startswith("Ответ на: Привет")
    assert body["metadata"]["primary_source"]["file_id"] == "f1"
    state = fake_answer.calls[-1]["scenario_state"]
    assert "ID текущего шага: observe" in state
    assert "Следующий шаг, назначенный симулятором: Записать наблюдение" in state
    assert "Разрешённые действия сейчас: открыть журнал" in state


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
        yield {
            "type": "done",
            "citations": [{"filename": "a.pdf", "file_id": "f"}],
            "usage": {},
        }

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
    seen = {}

    async def _hint(hint_text, hint_level, scenario_context=None, scenario_state=None):
        seen["scenario_state"] = scenario_state
        return f"L{hint_level}: {hint_text}"

    monkeypatch.setattr(routes, "rephrase_hint", _hint)
    r = client.post(
        "/hint",
        json={
            "hint_text": "Подойди к трубке",
            "hint_level": 2,
            "scenario_state": {
                "current_step_id": "connect-tube",
                "visible_items": ["трубка"],
                "last_action_result": "Трубка не подключена",
            },
        },
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["hint"] == "L2: Подойди к трубке"
    assert "ID текущего шага: connect-tube" in seen["scenario_state"]
    assert "Предметы, видимые ученику: трубка" in seen["scenario_state"]
    assert (
        "Результат последнего действия: Трубка не подключена" in seen["scenario_state"]
    )


def test_hint_level_out_of_range_422(client, auth):
    r = client.post("/hint", json={"hint_text": "x", "hint_level": 9}, headers=auth)
    assert r.status_code == 422


# ── Voice ────────────────────────────────────────────────────────────────────


def test_stt(client, auth, monkeypatch):
    calls = []

    async def _transcribe_with_language(
        audio_bytes, filename="audio.webm", language=None, prompt=None
    ):
        calls.append(language)
        return "танылған мәтін", "kk"

    monkeypatch.setattr(routes, "transcribe_with_language", _transcribe_with_language)
    r = client.post(
        "/stt",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json() == {"text": "танылған мәтін", "language": "kk"}
    assert calls == ["auto"]


def test_stt_empty_file_400(client, auth):
    r = client.post("/stt", files={"file": ("q.webm", b"", "audio/webm")}, headers=auth)
    assert r.status_code == 400


def test_tts(client, auth, monkeypatch):
    async def _synth(
        text,
        voice=None,
        response_format=None,
        instructions=None,
        language=None,
        backend=None,
    ):
        return b"AUDIOBYTES", "audio/wav"

    monkeypatch.setattr(routes, "synthesize", _synth)
    r = client.post("/tts", json={"text": "Привет, ученик"}, headers=auth)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["x-tts-backend"] == "supertonic"
    assert r.content == b"AUDIOBYTES"


def test_tts_selects_qwen(client, auth, monkeypatch):
    calls = []

    async def _synth(
        text,
        voice=None,
        response_format=None,
        instructions=None,
        language=None,
        backend=None,
    ):
        calls.append({"backend": backend, "voice": voice})
        return b"AUDIOBYTES", "audio/wav"

    monkeypatch.setattr(routes, "synthesize", _synth)
    r = client.post(
        "/tts",
        json={"text": "Привет, ученик", "backend": "qwen", "voice": "Aiden"},
        headers=auth,
    )

    assert r.status_code == 200
    assert r.headers["x-tts-backend"] == "qwen"
    assert calls == [{"backend": "qwen", "voice": "Aiden"}]


def test_voice_ask_full_pipeline(client, auth, monkeypatch, fake_answer):
    transcribe_languages = []
    synthesize_languages = []

    async def _transcribe_with_language(
        audio_bytes, filename="audio.webm", language=None, prompt=None
    ):
        transcribe_languages.append(language)
        return "Зачем нагревать пробирку?", "ru"

    async def _synth(
        text,
        voice=None,
        response_format=None,
        instructions=None,
        language=None,
        backend=None,
    ):
        synthesize_languages.append(language)
        return b"SPOKEN", "audio/wav"

    monkeypatch.setattr(routes, "transcribe_with_language", _transcribe_with_language)
    monkeypatch.setattr(routes, "synthesize", _synth)

    r = client.post(
        "/voice_ask",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        data={
            "scenario_id": "physics_lab_02_heating",
            "current_step_id": "heat-water",
            "current_step_index": "3",
            "current_step": "Нагреть воду",
            "next_step_id": "record-temperature",
            "next_step": "Записать температуру",
            "completed_steps": ["prepare-stand"],
            "held_items": [],
            "visible_items": ["термометр"],
            "allowed_actions": ["включить нагрев"],
            "last_action": "Поставил стакан",
            "last_action_result": "Успешно",
        },
        headers=auth,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "Зачем нагревать пробирку?"
    assert body["language"] == "ru"
    assert body["answer"].endswith("[scenario] [state]")
    assert transcribe_languages == ["auto"]
    assert synthesize_languages == ["ru"]
    state = fake_answer.calls[-1]["scenario_state"]
    assert "ID текущего шага: heat-water" in state
    assert "Индекс текущего шага: 3" in state
    assert "Следующий шаг, назначенный симулятором: Записать температуру" in state
    assert "Завершённые шаги: prepare-stand" in state
    assert "Предметы, видимые ученику: термометр" in state
    assert "Разрешённые действия сейчас: включить нагрев" in state
    assert "Результат последнего действия: Успешно" in state
    import base64

    assert base64.b64decode(body["audio_base64"]) == b"SPOKEN"
    assert "stt" in body["observability"]["latency_ms"]
    assert "tts" in body["observability"]["latency_ms"]


def test_voice_ask_reuses_conversation_history(client, auth, monkeypatch, fake_answer):
    questions = iter(["Что такое кипение?", "Почему оно начинается?"])

    async def _transcribe_with_language(
        audio_bytes, filename="audio.webm", language=None, prompt=None
    ):
        return next(questions), "ru"

    async def _synth(
        text,
        voice=None,
        response_format=None,
        instructions=None,
        language=None,
        backend=None,
    ):
        return b"SPOKEN", "audio/wav"

    monkeypatch.setattr(routes, "transcribe_with_language", _transcribe_with_language)
    monkeypatch.setattr(routes, "synthesize", _synth)
    request_args = {
        "files": {"file": ("q.webm", b"RIFFfake", "audio/webm")},
        "data": {"conversation_id": "voice-followup-session"},
        "headers": auth,
    }

    first = client.post("/voice_ask", **request_args)
    second = client.post("/voice_ask", **request_args)

    assert first.status_code == second.status_code == 200
    assert second.json()["conversation_id"] == "voice-followup-session"
    assert fake_answer.calls[-1]["chat_history"] == [
        {"role": "user", "content": "Что такое кипение?"},
        {"role": "assistant", "content": "Ответ на: Что такое кипение?"},
        {"role": "user", "content": "Почему оно начинается?"},
    ]


def test_voice_ask_uses_detected_language_for_tts_and_lab(
    client, auth, monkeypatch, fake_answer
):
    transcribe_languages = []
    synthesize_languages = []

    async def _transcribe_with_language(
        audio_bytes, filename="audio.webm", language=None, prompt=None
    ):
        transcribe_languages.append(language)
        return "Келесі қадам қандай?", "kk"

    async def _synth(
        text,
        voice=None,
        response_format=None,
        instructions=None,
        language=None,
        backend=None,
    ):
        synthesize_languages.append(language)
        return b"SPOKEN", "audio/wav"

    monkeypatch.setattr(routes, "transcribe_with_language", _transcribe_with_language)
    monkeypatch.setattr(routes, "synthesize", _synth)

    r = client.post(
        "/voice_ask",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        data={"subject": "physics", "grade": "10", "lab_number": "2"},
        headers=auth,
    )

    assert r.status_code == 200
    assert r.json()["language"] == "kk"
    assert transcribe_languages == ["auto"]
    assert synthesize_languages == ["kk"]
    lab = fake_answer.calls[-1]["lab"]
    assert lab["lang"] == "kk"
    assert lab["lab_id"] == "physics-10-kk-02"


def test_voice_ask_rejects_oversized_scene_field(client, auth):
    r = client.post(
        "/voice_ask",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        data={"current_step_id": "x" * 129},
        headers=auth,
    )
    assert r.status_code == 422


def test_voice_ask_stream(client, auth, monkeypatch):
    async def _transcribe_with_language(
        audio_bytes, filename="audio.webm", language=None, prompt=None
    ):
        return "Зачем нагревать пробирку?", "ru"

    async def _stream(
        query,
        scenario_context=None,
        chat_history=None,
        max_tokens=None,
        scenario_state=None,
        lab=None,
    ):
        yield {"type": "delta", "text": "Нагрев ускоряет реакцию. "}
        yield {"type": "delta", "text": "Молекулы движутся быстрее."}
        yield {
            "type": "done",
            "citations": [{"filename": "chem_8.pdf", "file_id": "f2"}],
            "usage": {"total_tokens": 12},
        }

    async def _synth(
        text,
        voice=None,
        response_format=None,
        instructions=None,
        language=None,
        backend=None,
    ):
        return f"WAV:{text}".encode(), "audio/wav"

    monkeypatch.setattr(routes, "transcribe_with_language", _transcribe_with_language)
    monkeypatch.setattr(routes, "stream_answer", _stream)
    monkeypatch.setattr(routes, "synthesize", _synth)

    r = client.post(
        "/voice_ask",
        files={"file": ("q.webm", b"RIFFfake", "audio/webm")},
        data={"stream": "true", "conversation_id": "voice-stream-session"},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)

    assert events[0] == {
        "type": "question",
        "text": "Зачем нагревать пробирку?",
        "language": "ru",
        "conversation_id": "voice-stream-session",
    }
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "Нагрев ускоряет реакцию. Молекулы движутся быстрее."
    # one audio frame per sentence, in order, carrying the spoken text
    import base64

    audio = [e for e in events if e["type"] == "audio"]
    assert [a["seq"] for a in audio] == [1, 2]
    assert audio[0]["text"] == "Нагрев ускоряет реакцию."
    assert audio[1]["text"] == "Молекулы движутся быстрее."
    assert (
        base64.b64decode(audio[0]["audio_base64"])
        == "WAV:Нагрев ускоряет реакцию.".encode()
    )
    done = next(e for e in events if e["type"] == "done")
    assert done["primary_source"]["filename"] == "chem_8.pdf"
    assert done["conversation_id"] == "voice-stream-session"
    assert "stt" in done["observability"]["latency_ms"]
    assert "[DONE]" in r.text


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


def test_upload_general_document_remains_compatible(client, auth, monkeypatch):
    call = {}
    cache_clears = []

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        call.update(
            filename=filename,
            raw=raw,
            metadata=metadata,
            doc_key=doc_key,
            ocr=ocr,
        )
        return {
            "file_id": "general-id",
            "filename": filename,
            "status": "ready",
            "chunks": 1,
        }

    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)
    monkeypatch.setattr(routes, "clear_answer_cache", lambda: cache_clears.append(True))
    r = client.post(
        "/admin/documents",
        files={"file": ("../notes.md", b"general notes", "text/markdown")},
        headers=auth,
    )

    assert r.status_code == 201
    assert r.json() == {
        "file_id": "general-id",
        "filename": "notes.md",
        "status": "ready",
        "chunks": 1,
    }
    assert call == {
        "filename": "notes.md",
        "raw": b"general notes",
        "metadata": None,
        "doc_key": None,
        "ocr": False,
    }
    assert cache_clears == [True]


def test_upload_textbook_with_structured_metadata(client, auth, monkeypatch):
    call = {}

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        call.update(metadata=metadata, doc_key=doc_key)
        return {
            "file_id": "textbook-id",
            "filename": filename,
            "status": "ready",
            "chunks": 4,
        }

    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)
    r = client.post(
        "/admin/documents",
        files={"file": ("Physics 8.pdf", b"pdf data", "application/pdf")},
        data={
            "doc_type": "textbook",
            "subject": "physics",
            "grade": "8",
            "lang": "ru",
        },
        headers=auth,
    )

    assert r.status_code == 201
    metadata = r.json()["metadata"]
    assert metadata == call["metadata"]
    assert metadata == {
        "doc_type": "textbook",
        "subject": "physics",
        "grade": 8,
        "lang": "ru",
        "source": "admin_uploads/textbook/physics/8/ru/Physics 8.pdf",
    }
    assert call["doc_key"] == metadata["source"]


def test_upload_lab_instruction_builds_lab_id(client, auth, monkeypatch):
    call = {}

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        call.update(metadata=metadata, doc_key=doc_key)
        return {
            "file_id": "lab-id",
            "filename": filename,
            "status": "ready",
            "chunks": 2,
        }

    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)
    r = client.post(
        "/admin/documents",
        files={"file": ("Lab 2.docx", b"docx data", "application/octet-stream")},
        data={
            "doc_type": "lab_instruction",
            "subject": "chemistry",
            "grade": "10",
            "lang": "kk",
            "lab_number": "2",
        },
        headers=auth,
    )

    assert r.status_code == 201
    metadata = r.json()["metadata"]
    assert metadata["lab_id"] == "chemistry-10-kk-02"
    assert metadata["lab_number"] == 2
    assert metadata["source"] == (
        "admin_uploads/lab_instruction/chemistry/10/kk/02/Lab 2.docx"
    )
    assert call["metadata"] == metadata
    assert call["doc_key"] == metadata["source"]


def test_upload_forwards_ocr_flag(client, auth, monkeypatch):
    call = {}

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        call["ocr"] = ocr
        return {
            "file_id": "ocr-id",
            "filename": filename,
            "status": "ready",
            "chunks": 1,
        }

    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)
    r = client.post(
        "/admin/documents",
        files={"file": ("scan.pdf", b"scanned pdf", "application/pdf")},
        data={"ocr": "true"},
        headers=auth,
    )

    assert r.status_code == 201
    assert call["ocr"] is True


@pytest.mark.parametrize(
    "metadata",
    [
        {"subject": "physics", "grade": "8", "lang": "ru"},
        {"doc_type": "textbook", "subject": "physics", "grade": "8"},
        {
            "doc_type": "lab_instruction",
            "subject": "chemistry",
            "grade": "10",
            "lang": "kk",
        },
        {
            "doc_type": "textbook",
            "subject": "physics",
            "grade": "8",
            "lang": "ru",
            "lab_number": "2",
        },
    ],
)
def test_upload_invalid_metadata_combination_400(client, auth, metadata):
    r = client.post(
        "/admin/documents",
        files={"file": ("document.pdf", b"pdf data", "application/pdf")},
        data=metadata,
        headers=auth,
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    "metadata",
    [
        {"doc_type": "notes"},
        {"doc_type": "textbook", "subject": "math"},
        {"doc_type": "textbook", "grade": "6"},
        {"doc_type": "textbook", "lang": "en"},
        {"doc_type": "lab_instruction", "lab_number": "100"},
    ],
)
def test_upload_enum_and_range_validation_422(client, auth, metadata):
    r = client.post(
        "/admin/documents",
        files={"file": ("document.pdf", b"pdf data", "application/pdf")},
        data=metadata,
        headers=auth,
    )
    assert r.status_code == 422


def test_upload_unsupported_type_400(client, auth):
    r = client.post(
        "/admin/documents",
        files={"file": ("notes.xyz", b"data", "application/octet-stream")},
        headers=auth,
    )
    assert r.status_code == 400


def test_delete_document_clears_answer_cache(client, auth, monkeypatch):
    cache_clears = []

    async def _delete(file_id):
        assert file_id == "chemistry-book"
        return True

    monkeypatch.setattr(routes.ingestion, "delete_document", _delete)
    monkeypatch.setattr(routes, "clear_answer_cache", lambda: cache_clears.append(True))

    r = client.delete("/admin/documents/chemistry-book", headers=auth)

    assert r.status_code == 200
    assert r.json() == {"deleted": True, "file_id": "chemistry-book"}
    assert cache_clears == [True]


def test_delete_missing_document_does_not_clear_answer_cache(client, auth, monkeypatch):
    cache_clears = []

    async def _delete(file_id):
        return False

    monkeypatch.setattr(routes.ingestion, "delete_document", _delete)
    monkeypatch.setattr(routes, "clear_answer_cache", lambda: cache_clears.append(True))

    r = client.delete("/admin/documents/missing", headers=auth)

    assert r.status_code == 404
    assert cache_clears == []
