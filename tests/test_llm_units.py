from types import SimpleNamespace

from app.services import llm
from app.services.memory import (
    build_input_messages,
    latest_user_message,
    trim_history,
)


def test_build_system_prompt_without_scenario():
    prompt = llm.build_system_prompt(None)
    assert "school VR laboratory" in prompt
    assert "CURRENT STATIC SCENARIO" not in prompt


def test_build_system_prompt_with_scenario():
    prompt = llm.build_system_prompt("Сценарий: тест")
    assert "CURRENT STATIC SCENARIO" in prompt
    assert "Сценарий: тест" in prompt


def test_build_system_prompt_with_scenario_state():
    prompt = llm.build_system_prompt(
        "Сценарий: тест", "Текущий шаг ученика: Зажечь спиртовку"
    )
    assert "LIVE SCENE STATE" in prompt
    assert "Зажечь спиртовку" in prompt


def test_build_system_prompt_strict_lab_scope_rejects_unrelated_topics():
    prompt = llm.build_system_prompt(
        "Сценарий: нагрев воды",
        strict_lab_scope=True,
    )

    assert "STRICT CURRENT LABORATORY SCOPE" in prompt
    assert "another topic" in prompt
    assert "matching textbook fragment" in prompt


def test_build_system_prompt_state_omitted_when_blank():
    prompt = llm.build_system_prompt("Сценарий: тест", "   ")
    assert "LIVE SCENE STATE" not in prompt


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


def test_lab_procedure_query_intent_is_precision_biased_for_ru_kk_and_en():
    assert llm._is_lab_procedure_query("Что мне делать дальше?") is True
    assert llm._is_lab_procedure_query("Как выполнить лабораторную работу?") is True
    assert llm._is_lab_procedure_query("Опишите основные этапы выполнения работы") is True
    assert llm._is_lab_procedure_query("Что наблюдаем в ходе эксперимента?") is True
    assert llm._is_lab_procedure_query("Главный результат в конце работы?") is True
    assert llm._is_lab_procedure_query("Әрі қарай не істеу керек?") is True
    assert llm._is_lab_procedure_query("Келесі қадам қандай?") is True
    assert llm._is_lab_procedure_query("What should I do next?") is True
    assert llm._is_lab_procedure_query("What is the current step?") is True
    assert llm._is_lab_procedure_query("How do I perform this experiment?") is True
    assert llm._is_lab_procedure_query("What is the purpose of this lab?") is True
    assert llm._is_lab_procedure_query("Почему вода кипит?") is False
    assert llm._is_lab_procedure_query("Как происходит кипение?") is False
    assert llm._is_lab_procedure_query("Кипение қалай жүреді?") is False


def test_lab_scope_rejects_explicitly_different_subject():
    refusal = llm._lab_scope_refusal(
        "Как происходит фотосинтез?",
        {"subject": "physics"},
        "Нагреть воду и записать показания термометра.",
        [],
        "ru",
    )

    assert refusal == llm._LAB_SCOPE_REFUSALS["ru"]


def test_lab_scope_rejects_unrelated_topic_from_same_subject():
    refusal = llm._lab_scope_refusal(
        "Как работает электрическая цепь?",
        {"subject": "physics"},
        "Нагреть воду и записать температуру кипения.",
        [
            _chunk(
                "electricity",
                "physics.pdf",
                "Электрическая цепь состоит из источника тока и проводников.",
            )
        ],
        "ru",
    )

    assert refusal == llm._LAB_SCOPE_REFUSALS["ru"]


def test_lab_scope_allows_theory_when_retrieval_bridges_to_lab_topic():
    refusal = llm._lab_scope_refusal(
        "Почему появляются пузырьки?",
        {"subject": "physics"},
        "Нагреть воду и наблюдать изменение температуры.",
        [
            _chunk(
                "boiling",
                "physics.pdf",
                "Пузырьки появляются в воде при нагревании перед кипением.",
            )
        ],
        "ru",
    )

    assert refusal is None


def test_structured_lab_disables_general_knowledge_fallback(monkeypatch):
    monkeypatch.setattr(
        llm.settings,
        "ALLOW_GENERAL_KNOWLEDGE_FALLBACK",
        True,
    )

    assert llm._general_fallback_allowed(
        "Когда кипит вода?",
        None,
        None,
        None,
        False,
        lab_active=True,
    ) is False


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
    assert "RETRIEVED KNOWLEDGE" in prompt
    assert "факт о воде" in prompt


def test_build_system_prompt_knowledge_omitted_when_blank():
    assert "RETRIEVED KNOWLEDGE" not in llm.build_system_prompt(knowledge_context=None)
    assert "RETRIEVED KNOWLEDGE" not in llm.build_system_prompt(knowledge_context="   ")


def test_build_system_prompt_general_fallback_is_explicit_and_language_locked():
    prompt = llm.build_system_prompt(
        answer_language="kk",
        allow_general_knowledge=True,
    )

    assert "ANSWER LANGUAGE: Kazakh" in prompt
    assert llm._GENERAL_KNOWLEDGE_MARKER in prompt
    assert llm._GROUNDED_MARKER in prompt
    assert "Do not refuse merely" in prompt


def test_query_inference_handles_ru_and_kk_across_school_subjects(monkeypatch):
    monkeypatch.setattr(llm.settings, "DEFAULT_LANGUAGE", "kk")
    assert llm._infer_query_context("қандай атақты химиктер бар?") == (
        "chemistry",
        "kk",
    )
    assert llm._infer_query_context("Какие известные физики изучали электричество?") == (
        "physics",
        "ru",
    )
    assert llm._infer_query_context("Жасуша туралы айтып бер") == ("biology", "kk")


def test_english_language_detection_ignores_non_prose_tokens():
    assert llm._detect_query_language("What should I do next?") == "en"
    assert llm._detect_query_language("How does photosynthesis work?") == "en"
    for ambiguous in ("H2O?", "README.md", "https://example.com", "NASA", "LAB-204"):
        assert llm._detect_query_language(ambiguous) is None


def test_answer_language_precedence(monkeypatch):
    monkeypatch.setattr(llm.settings, "DEFAULT_LANGUAGE", "ru")
    assert llm.resolve_answer_language("Что такое вода?", "en", "kk") == "en"
    assert llm.resolve_answer_language("Что такое вода?", None, "en") == "ru"
    assert llm.resolve_answer_language("H2O?", None, "en") == "en"
    assert llm.resolve_answer_language("H2O?", None, None) == "ru"


def test_english_subject_inference_is_precision_biased():
    assert llm._infer_query_subject("What force accelerates the cart?") == "physics"
    assert llm._infer_query_subject("Why does an acid react with an alkali?") == "chemistry"
    assert llm._infer_query_subject("How does photosynthesis work in a cell?") == "biology"
    assert llm._infer_query_subject("What is the mass?") is None


def test_english_scope_refusal_is_exact():
    refusal = llm._lab_scope_refusal(
        "How does an electric circuit work?",
        {"subject": "physics"},
        "Heat water and record its boiling temperature.",
        [
            _chunk(
                "electricity",
                "physics.pdf",
                "An electric circuit contains a source and conductors.",
            )
        ],
        "en",
    )
    assert refusal == (
        "I can only answer questions related to the current laboratory activity."
    )


def test_english_missing_evidence_refusal_is_detected():
    assert llm._is_missing_evidence_refusal(
        "The provided documents do not contain enough information to answer."
    )


def test_english_citation_label_is_localized():
    citation = llm._citations_from_chunks(
        [
            _chunk(
                "lab-en",
                "Lab work No. 2.docx",
                "procedure",
                doc_type="lab_instruction",
                lab_number=2,
                chapter="Chapter 3",
                page_start=10,
                page_end=12,
            )
        ],
        answer_language="en",
    )[0]
    assert citation["display_label"] == (
        "Lab instruction No. 2, Chapter 3, pp. 10-12"
    )


def test_answer_cache_key_isolated_by_language():
    ru_key = llm._answer_cache_key("H2O?", None, None, None, None, "ru")
    en_key = llm._answer_cache_key("H2O?", None, None, None, None, "en")
    assert ru_key != en_key


def test_usable_theory_chunks_reject_repetitive_textbook_watermark():
    watermark = "\n".join(
        ["OKULYK.KZ ОҚУЛЫҚ Учебники Казахстана"] * 12
    )
    clean = _chunk(
        "chemistry",
        "chemistry.pdf",
        "Дмитрий Менделеев создал периодическую систему химических элементов.",
    )

    chunks = llm._usable_theory_chunks(
        [clean, _chunk("noise", "bad.pdf", watermark)]
    )

    assert chunks == [clean]


def test_parse_answer_mode_strips_private_marker():
    answer, general = llm._parse_answer_mode(
        "[[GENERAL_KNOWLEDGE]] Дмитрий Менделеев.",
        allow_general_knowledge=True,
    )
    assert answer == "Дмитрий Менделеев."
    assert general is True


async def test_generate_answer_general_fallback_is_kazakh_and_uncited(monkeypatch):
    captured = {"filters": []}

    async def _embed_query(text):
        return SimpleNamespace(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(
        dense, sparse_indices, sparse_values, top_k, candidates, query_filter=None
    ):
        captured["filters"].append(query_filter)
        return [
            _chunk(
                "unrelated",
                "chemistry.pdf",
                "Күкірт қышқылының ерітіндісі зертханада сақтықпен қолданылады.",
            )
        ]

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(
            output_text=(
                "[[GENERAL_KNOWLEDGE]] Атақты химиктерге Дмитрий Менделеев, "
                "Мария Кюри, Антуан Лавуазье және Амедео Авогадро жатады."
            ),
            usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
        )

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer("қандай атақты химиктер бар?")

    first_filter = captured["filters"][0]
    conditions = {condition.key: condition.match.value for condition in first_filter.must}
    assert conditions == {"doc_type": "textbook", "subject": "chemistry", "lang": "kk"}
    assert "ANSWER LANGUAGE: Kazakh" in captured["instructions"]
    assert result.answer.startswith("Атақты химиктерге Дмитрий Менделеев")
    assert result.citations == []


async def test_generate_answer_grounded_mode_keeps_retrieved_citation(monkeypatch):
    chunk = _chunk(
        "chemists",
        "chemistry.pdf",
        "Дмитрий Менделеев создал периодическую систему химических элементов.",
    )

    async def _retrieve(query, **kwargs):
        return [chunk]

    async def _create(**kwargs):
        return SimpleNamespace(
            output_text="[[GROUNDED]] Дмитрий Менделеев создал периодическую систему.",
            usage=None,
        )

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer("Назови известного химика")

    assert result.answer == "Дмитрий Менделеев создал периодическую систему."
    assert result.citations[0]["file_id"] == "chemists"


async def test_cross_language_evidence_is_translated_to_english_with_citation(
    monkeypatch,
):
    chunk = _chunk(
        "physics-ru",
        "Физика 8.pdf",
        "Чистая вода кипит примерно при 100 °C при нормальном давлении.",
        doc_type="textbook",
        lang="ru",
        chapter="Глава 3",
        page_start=10,
        page_end=12,
    )
    captured = {}

    async def _retrieve(query, **kwargs):
        return [chunk]

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(
            output_text="[[GROUNDED]] Pure water boils at about 100 °C.",
            usage=None,
        )

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer(
        "What is the boiling point of pure water?", answer_language="en"
    )

    assert result.language == "en"
    assert result.answer == "Pure water boils at about 100 °C."
    assert result.citations[0]["file_id"] == "physics-ru"
    assert result.citations[0]["display_label"].endswith("pp. 10-12")
    assert "Translate it faithfully" in captured["instructions"]


async def test_english_answer_can_use_exact_russian_lab_instruction(monkeypatch):
    captured = {}

    async def _fetch(lab_id):
        assert lab_id == "physics-10-ru-02"
        return {
            "text": "Наденьте очки и закрепите пробирку.",
            "payloads": [
                {
                    "doc_id": "lab-ru",
                    "filename": "Лабораторная работа №2.docx",
                    "doc_type": "lab_instruction",
                    "lab_id": lab_id,
                    "lab_number": 2,
                    "chunk_index": 0,
                    "text": "Наденьте очки и закрепите пробирку.",
                }
            ],
        }

    async def _retrieve(query, **kwargs):
        captured["retrieval_lang"] = kwargs["lang"]
        return []

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(output_text="Put on the glasses and secure the tube.", usage=None)

    monkeypatch.setattr(llm.vectorstore, "fetch_lab_instruction_record", _fetch)
    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)
    lab = {
        "subject": "physics",
        "grade": 10,
        "lang": "ru",
        "lab_number": 2,
        "lab_id": "physics-10-ru-02",
    }

    result = await llm.generate_answer(
        "What should I do next?", lab=lab, answer_language="en"
    )

    assert captured["retrieval_lang"] == "ru"
    assert "ANSWER LANGUAGE: English" in captured["instructions"]
    assert result.language == "en"
    assert result.citations[0]["file_id"] == "lab-ru"


async def test_english_general_knowledge_fallback_is_uncited(monkeypatch):
    async def _retrieve(query, **kwargs):
        return []

    async def _create(**kwargs):
        return SimpleNamespace(
            output_text="[[GENERAL_KNOWLEDGE]] Gravity attracts masses.",
            usage=None,
        )

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer("What is gravity?", answer_language="en")

    assert result.answer == "Gravity attracts masses."
    assert result.language == "en"
    assert result.citations == []


async def test_grounded_missing_evidence_refusal_retries_as_uncited_general(monkeypatch):
    chunk = _chunk(
        "chemistry",
        "chemistry.pdf",
        "Күкірт қышқылы зертханада сақтықпен қолданылады.",
    )
    responses = iter(
        [
            SimpleNamespace(
                output_text=(
                    "[[GROUNDED]] Өкінішке орай, берілген материалдарда "
                    "атақты химиктер туралы ақпарат жоқ."
                ),
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=8, total_tokens=18
                ),
            ),
            SimpleNamespace(
                output_text=(
                    "[[GENERAL_KNOWLEDGE]] Атақты химиктерге Дмитрий Менделеев, "
                    "Мария Кюри және Антуан Лавуазье жатады."
                ),
                usage=SimpleNamespace(
                    input_tokens=12, output_tokens=10, total_tokens=22
                ),
            ),
        ]
    )
    calls = []

    async def _retrieve(query, **kwargs):
        return [chunk]

    async def _create(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer("қандай атақты химиктер бар?")

    assert len(calls) == 2
    assert "FORCED GENERAL SCIENCE ANSWER" in calls[1]["instructions"]
    assert result.answer.startswith("Атақты химиктерге Дмитрий Менделеев")
    assert result.citations == []
    assert result.usage == {
        "input_tokens": 22,
        "output_tokens": 18,
        "total_tokens": 40,
    }


async def test_authoritative_scenario_disables_general_fallback(monkeypatch):
    captured = {}

    async def _retrieve(query, **kwargs):
        return []

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(output_text="Нет данных в сценарии.", usage=None)

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer(
        "Что делать дальше?",
        scenario_context="Текущий сценарий: дождаться сигнала.",
    )

    assert llm._GENERAL_KNOWLEDGE_MARKER not in captured["instructions"]
    assert result.answer == "Нет данных в сценарии."


async def test_generate_answer_rejects_unrelated_lab_question_without_openai(
    monkeypatch,
):
    async def _lab_grounding(lab):
        return (
            "Нагреть воду и записать температуру кипения.",
            False,
            None,
            "ru",
            None,
            [],
        )

    async def _retrieve(query, **kwargs):
        return [
            _chunk(
                "electricity",
                "physics.pdf",
                "Электрическая цепь состоит из источника тока и проводников.",
            )
        ]

    async def _create(**kwargs):
        raise AssertionError("OpenAI must not be called for an unrelated lab question")

    monkeypatch.setattr(llm, "_lab_grounding", _lab_grounding)
    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer(
        "Как работает электрическая цепь?",
        scenario_context="Сценарий нагревания воды и измерения температуры.",
        lab={"subject": "physics", "grade": 8, "lang": "ru"},
    )

    assert result.answer == llm._LAB_SCOPE_REFUSALS["ru"]
    assert result.citations == []
    assert result.usage == {}


async def test_stream_answer_rejects_unrelated_lab_question_without_streaming_openai(
    monkeypatch,
):
    async def _lab_grounding(lab):
        return (
            "Нагреть воду и записать температуру кипения.",
            False,
            None,
            "ru",
            None,
            [],
        )

    async def _retrieve(query, **kwargs):
        return [
            _chunk(
                "electricity",
                "physics.pdf",
                "Электрическая цепь состоит из источника тока и проводников.",
            )
        ]

    def _stream(**kwargs):
        raise AssertionError("OpenAI stream must not run for an unrelated question")

    monkeypatch.setattr(llm, "_lab_grounding", _lab_grounding)
    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "stream", _stream)

    events = [
        event
        async for event in llm.stream_answer(
            "Что такое электрическое сопротивление?",
            scenario_context="Сценарий нагревания воды и измерения температуры.",
            lab={"subject": "physics", "grade": 8, "lang": "ru"},
        )
    ]

    assert events == [
        {"type": "delta", "text": llm._LAB_SCOPE_REFUSALS["ru"]},
        {"type": "done", "citations": [], "usage": {}, "language": "ru"},
    ]


async def test_stream_answer_uses_checked_completion_for_general_fallback(monkeypatch):
    async def _retrieve(query, **kwargs):
        return []

    async def _create(**kwargs):
        return SimpleNamespace(
            output_text="[[GENERAL_KNOWLEDGE]] Атақты химиктер бар.",
            usage=None,
        )

    def _stream_must_not_run(**kwargs):
        raise AssertionError("fallback-eligible streaming must use checked completion")

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)
    monkeypatch.setattr(llm.client.responses, "stream", _stream_must_not_run)

    events = [event async for event in llm.stream_answer("қандай атақты химиктер бар?")]

    assert events == [
        {"type": "delta", "text": "Атақты химиктер бар."},
        {"type": "done", "citations": [], "usage": {}, "language": "kk"},
    ]


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
