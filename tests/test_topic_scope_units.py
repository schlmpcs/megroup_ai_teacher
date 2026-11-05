"""Off-topic refusal for requests with and without a structured lab."""

from types import SimpleNamespace

import pytest

from app.services import llm


def _chunk(doc_id, text):
    return {
        "score": 1.0,
        "payload": {
            "doc_id": doc_id,
            "filename": f"{doc_id}.pdf",
            "chunk_index": 0,
            "text": text,
            "doc_type": "textbook",
        },
    }


_PHYSICS_CHUNK = _chunk(
    "physics",
    "Электрическая цепь состоит из источника тока, проводников и потребителей.",
)


def _patch_retrieve(monkeypatch, chunks):
    async def _retrieve(query, **kwargs):
        return list(chunks)

    monkeypatch.setattr(llm, "_retrieve", _retrieve)


def _patch_create(monkeypatch, outputs):
    calls = []
    remaining = list(outputs)

    async def _create(**kwargs):
        calls.append(kwargs["instructions"])
        return SimpleNamespace(output_text=remaining.pop(0), usage=None)

    monkeypatch.setattr(llm.client.responses, "create", _create)
    return calls


class _FakeStream:
    def __init__(self, text):
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def _events():
            yield SimpleNamespace(type="response.output_text.delta", delta=self._text)

        return _events()

    async def get_final_response(self):
        return SimpleNamespace(output_text=self._text, usage=None)


# ── Prompt ────────────────────────────────────────────────────────────────────


def test_science_topic_scope_block_lists_sciences_and_exact_refusal():
    prompt = llm.build_system_prompt(answer_language="ru", topic_scope="science")

    assert "--- TOPIC SCOPE ---" in prompt
    assert "physics, chemistry, or biology" in prompt
    assert llm._SCIENCE_SCOPE_REFUSALS["ru"] in prompt
    assert "greeting, thanks, goodbye" in prompt


def test_profile_topic_scope_uses_role_and_generic_refusal():
    prompt = llm.build_system_prompt(answer_language="en", topic_scope="profile")

    assert "defined by your role" in prompt
    assert llm._PROFILE_SCOPE_REFUSALS["en"] in prompt
    assert llm._SCIENCE_SCOPE_REFUSALS["en"] not in prompt


def test_lab_scope_prompt_keeps_lab_block_and_allows_small_talk():
    prompt = llm.build_system_prompt(
        answer_language="ru", strict_lab_scope=True, topic_scope="science"
    )

    assert "CURRENT SUBJECT AND LABORATORY SCOPE" in prompt
    assert "--- TOPIC SCOPE ---" not in prompt
    assert "greeting, thanks, goodbye" in prompt


def test_fallback_mode_states_scope_is_not_overridden():
    prompt = llm.build_system_prompt(
        answer_language="ru", topic_scope="science", allow_general_knowledge=True
    )

    assert prompt.index("--- TOPIC SCOPE ---") < prompt.index(
        "GENERAL SCIENCE FALLBACK MODE"
    )
    assert "never overrides the topic scope" in prompt
    assert "topic scope above still applies" in (
        llm._force_general_knowledge_prompt(prompt)
    )


async def test_no_lab_default_profile_gets_science_scope(monkeypatch):
    _patch_retrieve(monkeypatch, [_PHYSICS_CHUNK])
    calls = _patch_create(monkeypatch, ["[[GROUNDED]] Цепь замкнута."])

    await llm.generate_answer("Как работает электрическая цепь?")

    assert "--- TOPIC SCOPE ---" in calls[0]
    assert llm._SCIENCE_SCOPE_REFUSALS["ru"] in calls[0]


async def test_no_lab_other_profile_gets_profile_scope(monkeypatch):
    _patch_retrieve(monkeypatch, [])
    calls = _patch_create(monkeypatch, ["Configured answer."])

    await llm.generate_answer("Explain this.", assistant_type="other_assistant")

    assert llm._PROFILE_SCOPE_REFUSALS["en"] in calls[0]
    assert llm._SCIENCE_SCOPE_REFUSALS["en"] not in calls[0]


async def test_lab_request_gets_no_topic_scope_block(monkeypatch):
    async def _lab_grounding(lab, collection_name=None):
        return ("Нагреть воду.", False, None, "ru", None, [])

    monkeypatch.setattr(llm, "_lab_grounding", _lab_grounding)
    _patch_retrieve(monkeypatch, [])
    calls = _patch_create(monkeypatch, ["Нагрейте воду."])

    await llm.generate_answer(
        "Что делать дальше?", lab={"subject": "physics", "grade": 8, "lang": "ru"}
    )

    assert "--- TOPIC SCOPE ---" not in calls[0]
    assert "CURRENT SUBJECT AND LABORATORY SCOPE" in calls[0]


# ── Enforcement of the model's refusal ───────────────────────────────────────


async def test_no_lab_refusal_has_no_citations_and_no_retry(monkeypatch):
    _patch_retrieve(monkeypatch, [_PHYSICS_CHUNK])
    calls = _patch_create(monkeypatch, [llm._SCIENCE_SCOPE_REFUSALS["ru"]])

    result = await llm.generate_answer("Кто выиграл чемпионат мира по футболу?")

    assert result.answer == llm._SCIENCE_SCOPE_REFUSALS["ru"]
    assert result.citations == []
    assert len(calls) == 1


async def test_refusal_with_appended_text_is_trimmed_to_sentence(monkeypatch):
    _patch_retrieve(monkeypatch, [_PHYSICS_CHUNK])
    _patch_create(
        monkeypatch,
        [
            "[[GENERAL_KNOWLEDGE]] "
            + llm._SCIENCE_SCOPE_REFUSALS["ru"]
            + " Но в 2022 году победила Аргентина."
        ],
    )

    result = await llm.generate_answer("Кто выиграл чемпионат мира по футболу?")

    assert result.answer == llm._SCIENCE_SCOPE_REFUSALS["ru"]
    assert result.citations == []


async def test_forced_general_retry_cannot_answer_off_topic(monkeypatch):
    _patch_retrieve(monkeypatch, [_PHYSICS_CHUNK])
    calls = _patch_create(
        monkeypatch,
        [
            "[[GROUNDED]] В материалах нет информации о футболе.",
            llm._SCIENCE_SCOPE_REFUSALS["ru"],
        ],
    )

    result = await llm.generate_answer("Кто выиграл чемпионат мира по футболу?")

    assert len(calls) == 2
    assert "topic scope above still applies" in calls[1]
    assert result.answer == llm._SCIENCE_SCOPE_REFUSALS["ru"]
    assert result.citations == []


async def test_in_scope_answer_keeps_citations(monkeypatch):
    _patch_retrieve(monkeypatch, [_PHYSICS_CHUNK])
    _patch_create(monkeypatch, ["[[GROUNDED]] Цепь состоит из источника и проводов."])

    result = await llm.generate_answer("Из чего состоит электрическая цепь?")

    assert result.answer == "Цепь состоит из источника и проводов."
    assert result.citations[0]["file_id"] == "physics"


async def test_streamed_refusal_drops_citations(monkeypatch):
    # A static scenario without a lab disables the fallback, so the direct
    # token stream is used rather than the checked completion.
    _patch_retrieve(monkeypatch, [_PHYSICS_CHUNK])
    captured = {}

    def _stream(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return _FakeStream(llm._SCIENCE_SCOPE_REFUSALS["ru"])

    monkeypatch.setattr(llm.client.responses, "stream", _stream)

    events = [
        event
        async for event in llm.stream_answer(
            "Кто выиграл чемпионат мира по футболу?",
            scenario_context="Сценарий нагревания воды.",
        )
    ]

    assert "--- TOPIC SCOPE ---" in captured["instructions"]
    assert events[0] == {"type": "delta", "text": llm._SCIENCE_SCOPE_REFUSALS["ru"]}
    assert events[-1]["type"] == "done"
    assert events[-1]["citations"] == []


def test_detected_scope_refusal_ignores_normal_answers():
    assert llm._detected_scope_refusal("Электрическая цепь замкнута.") is None
    assert llm._detected_scope_refusal("") is None
    assert (
        llm._detected_scope_refusal(llm._LAB_SCOPE_REFUSALS["kk"])
        == llm._LAB_SCOPE_REFUSALS["kk"]
    )


# ── Small talk ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "Привет!",
        "Здравствуйте",
        "Спасибо большое!",
        "Привет, кто ты?",
        "Сәлеметсіз бе!",
        "Рахмет",
        "Hello! What can you do?",
        "thank you",
    ],
)
def test_small_talk_detected(query):
    assert llm._is_small_talk(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Привет, кто выиграл чемпионат мира?",
        "Спасибо, а что такое сила тока?",
        "Кто написал роман Война и мир?",
        "hi there tell me a joke",
        "",
    ],
)
def test_small_talk_rejects_real_questions(query):
    assert llm._is_small_talk(query) is False


def test_lab_gate_lets_small_talk_through():
    refusal = llm._lab_scope_refusal(
        "Спасибо!",
        {"subject": "physics"},
        "Нагреть воду и записать температуру кипения.",
        [],
        "ru",
    )

    assert refusal is None
