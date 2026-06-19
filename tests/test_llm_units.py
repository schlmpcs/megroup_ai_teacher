from app.services import llm
from app.services.memory import (
    build_input_messages,
    latest_user_message,
    trim_history,
)


def test_build_system_prompt_without_scenario():
    prompt = llm.build_system_prompt(None)
    assert "VR-тренажёр" in prompt
    assert "ОПИСАНИЕ ТЕКУЩЕГО СЦЕНАРИЯ" not in prompt


def test_build_system_prompt_with_scenario():
    prompt = llm.build_system_prompt("Сценарий: тест")
    assert "ОПИСАНИЕ ТЕКУЩЕГО СЦЕНАРИЯ" in prompt
    assert "Сценарий: тест" in prompt


def test_build_system_prompt_with_scenario_state():
    prompt = llm.build_system_prompt(
        "Сценарий: тест", "Текущий шаг ученика: Зажечь спиртовку"
    )
    assert "ТЕКУЩЕЕ СОСТОЯНИЕ СЦЕНЫ" in prompt
    assert "Зажечь спиртовку" in prompt


def test_build_system_prompt_state_omitted_when_blank():
    prompt = llm.build_system_prompt("Сценарий: тест", "   ")
    assert "ТЕКУЩЕЕ СОСТОЯНИЕ СЦЕНЫ" not in prompt


# ── Citations from retrieved chunks (local hybrid RAG) ───────────────────────


def _chunk(doc_id, filename, text, chunk_index=0, score=1.0):
    return {
        "score": score,
        "payload": {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": chunk_index,
            "text": text,
        },
    }


def test_citations_from_chunks_dedupes_by_filename_first_seen():
    chunks = [
        _chunk("d1", "physics_8.pdf", "t1", chunk_index=0),
        _chunk("d1", "physics_8.pdf", "t2", chunk_index=1),  # dup filename
        _chunk("d2", "chem_9.pdf", "t3", chunk_index=0),
    ]
    citations = llm._citations_from_chunks(chunks)
    assert citations == [
        {"filename": "physics_8.pdf", "file_id": "d1"},
        {"filename": "chem_9.pdf", "file_id": "d2"},
    ]


def test_citations_from_chunks_empty():
    assert llm._citations_from_chunks([]) == []


def test_format_knowledge_empty():
    assert llm._format_knowledge([]) == ""


def test_format_knowledge_includes_text_and_filename():
    chunks = [
        _chunk("d1", "physics_8.pdf", "Кипение — это переход в пар."),
        _chunk("d2", "chem_9.pdf", "Реакция окисления."),
    ]
    block = llm._format_knowledge(chunks)
    assert "physics_8.pdf" in block
    assert "Кипение — это переход в пар." in block
    assert "chem_9.pdf" in block
    assert "Реакция окисления." in block


def test_build_system_prompt_injects_knowledge():
    prompt = llm.build_system_prompt(knowledge_context="[1] (a.pdf)\nфакт о воде")
    assert "БАЗА ЗНАНИЙ" in prompt
    assert "факт о воде" in prompt


def test_build_system_prompt_knowledge_omitted_when_blank():
    assert "БАЗА ЗНАНИЙ" not in llm.build_system_prompt(knowledge_context=None)
    assert "БАЗА ЗНАНИЙ" not in llm.build_system_prompt(knowledge_context="   ")


def test_answer_result_primary_source():
    result = llm.AnswerResult(answer="x", citations=[{"filename": "a", "file_id": "1"}])
    assert result.primary_source == {"filename": "a", "file_id": "1"}
    assert llm.AnswerResult(answer="x").primary_source is None


def test_trim_history_respects_message_cap():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    trimmed = trim_history(msgs, max_messages=5, max_chars=10_000)
    assert len(trimmed) == 5
    assert trimmed[-1]["content"] == "m19"


def test_trim_history_keeps_last_under_char_budget():
    msgs = [{"role": "user", "content": "x" * 100} for _ in range(10)]
    trimmed = trim_history(msgs, max_messages=10, max_chars=50)
    assert len(trimmed) == 1  # never drops the final message


def test_trim_history_strips_system_messages():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    trimmed = trim_history(msgs, max_messages=10, max_chars=1000)
    assert all(m["role"] != "system" for m in trimmed)


def test_build_input_messages_drops_empty_and_system():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": ""},
    ]
    out = build_input_messages(msgs)
    assert out == [{"role": "user", "content": "q"}]


def test_latest_user_message():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "second"},
    ]
    assert latest_user_message(msgs) == "second"
