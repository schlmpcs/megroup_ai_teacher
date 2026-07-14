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


def _chunk(doc_id, filename, text, chunk_index=0, score=1.0, **metadata):
    return {
        "score": score,
        "payload": {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": chunk_index,
            "text": text,
            **metadata,
        },
    }


def test_citations_from_chunks_groups_locators_by_document_first_seen():
    chunks = [
        _chunk(
            "d1",
            "physics_8.pdf",
            "t1",
            chunk_index=0,
            doc_type="textbook",
            source="Физика/physics_8.pdf",
            page_start=14,
            page_end=14,
            chapter="Глава 3",
        ),
        _chunk(
            "d1",
            "physics_8.pdf",
            "t2",
            chunk_index=1,
            doc_type="textbook",
            source="Физика/physics_8.pdf",
            pages=[14, 15],
            chapter="Глава 3",
        ),
        _chunk("d2", "chem_9.pdf", "t3", chunk_index=0),
    ]
    citations = llm._citations_from_chunks(chunks)
    assert citations[0] == {
        "filename": "physics_8.pdf",
        "file_id": "d1",
        "source_type": "textbook",
        "source_path": "Физика/physics_8.pdf",
        "chunk_indexes": [0, 1],
        "pages": [14, 15],
        "page_start": 14,
        "page_end": 15,
        "chapters": ["Глава 3"],
        "chapter": "Глава 3",
        "display_label": "physics_8, Глава 3, стр. 14-15",
    }
    assert citations[1]["filename"] == "chem_9.pdf"
    assert citations[1]["file_id"] == "d2"
    assert citations[1]["chunk_indexes"] == [0]


def test_citations_keep_same_filename_documents_distinct_and_skip_nulls():
    chunks = [
        _chunk("lab-a", "Лабораторная работа №2.docx", "a", source="7/a.docx"),
        _chunk("lab-b", "Лабораторная работа №2.docx", "b", source="8/b.docx"),
        {"payload": {"filename": None, "doc_id": None, "text": "noise"}},
    ]

    citations = llm._citations_from_chunks(chunks)

    assert [citation["file_id"] for citation in citations] == ["lab-a", "lab-b"]
    assert all(citation["filename"] for citation in citations)
    assert all(citation["file_id"] for citation in citations)


def test_lab_instruction_citation_has_human_readable_label():
    citation = llm._citations_from_chunks(
        [
            _chunk(
                "lab-2",
                "Лабораторная работа №2.docx",
                "ход работы",
                doc_type="lab_instruction",
                source="labs/Лабораторная работа №2.docx",
                lab_id="physics-8-ru-02",
                lab_number=2,
            )
        ]
    )[0]

    assert citation["source_type"] == "lab_instruction"
    assert citation["lab_id"] == "physics-8-ru-02"
    assert citation["lab_number"] == 2
    assert citation["display_label"] == "Инструкция к лабораторной работе №2"


def test_lab_procedure_query_intent_is_precision_biased_for_ru_and_kk():
    assert llm._is_lab_procedure_query("Что мне делать дальше?") is True
    assert llm._is_lab_procedure_query("Как выполнить лабораторную работу?") is True
    assert llm._is_lab_procedure_query("Опишите основные этапы выполнения работы") is True
    assert llm._is_lab_procedure_query("Что наблюдаем в ходе эксперимента?") is True
    assert llm._is_lab_procedure_query("Главный результат в конце работы?") is True
    assert llm._is_lab_procedure_query("Әрі қарай не істеу керек?") is True
    assert llm._is_lab_procedure_query("Келесі қадам қандай?") is True
    assert llm._is_lab_procedure_query("Почему вода кипит?") is False
    assert llm._is_lab_procedure_query("Как происходит кипение?") is False
    assert llm._is_lab_procedure_query("Кипение қалай жүреді?") is False


def test_answer_citations_order_theory_and_procedure_sources_by_query_intent():
    theory = [
        _chunk(
            "book",
            "Физика 8.pdf",
            "теория кипения",
            doc_type="textbook",
        )
    ]
    lab = [
        _chunk(
            "lab",
            "Лабораторная работа №2.docx",
            "ход работы",
            doc_type="lab_instruction",
            lab_number=2,
        )
    ]

    theory_answer = llm._answer_citations("Почему вода кипит?", theory, lab)
    procedure_answer = llm._answer_citations("Что делать дальше?", theory, lab)

    assert [c["source_type"] for c in theory_answer] == [
        "textbook",
        "lab_instruction",
    ]
    assert [c["source_type"] for c in procedure_answer] == [
        "lab_instruction",
        "textbook",
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
