"""Grounded answer generation via the OpenAI Responses API + local hybrid RAG.

Retrieval is now EXPLICIT and local (no hosted ``file_search``): per request we
embed the query with the bge-m3 sidecar, run a hybrid (dense + sparse, RRF)
search against Qdrant, and inject the retrieved chunk text straight into the
system prompt. Generation itself still goes through the OpenAI Responses API.

Two grounding sources are combined per request:
  1. Subject theory (physics / chemistry / biology) — the local Qdrant
     knowledge base, retrieved here via ``_retrieve`` and rendered into the
     prompt by ``_format_knowledge``. Citations are derived from the retrieved
     chunks' payloads (see ``_citations_from_chunks``).
  2. Scenario context — the current VR lab/scene, injected into the system
     prompt by the caller (see app/services/scenarios.py).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.services import embeddings, vectorstore
from app.services.openai_client import client
from app.services.memory import build_input_messages, trim_history
from app.services.ttl_cache import TTLCache

# Service-layer exceptions live in app/services/errors.py so embeddings/
# vectorstore can raise them without importing this module. Re-exported here so
# existing importers (voice.py, tests) keep working unchanged.
from app.services.errors import (  # noqa: F401 - re-exported for backwards compat
    LLMError,
    LLMTimeoutError,
    LLMUpstreamError,
    LLMMalformedResponseError,
    _map_openai_error,
)

logger = logging.getLogger("assistant.llm")

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_KAZAKH_CHAR_RE = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)
_KAZAKH_WORD_RE = re.compile(
    r"\b(?:қандай|қалай|қайда|қайсы|қанша|неге|деген|туралы|атақты|ғалым|"
    r"болады|керек|қажет|үшін|және|немесе|бар|жоқ)\b",
    re.IGNORECASE,
)
_RUSSIAN_WORD_RE = re.compile(
    r"\b(?:что|какой|какая|какие|как|где|когда|почему|зачем|кто|есть|"
    r"известн\w*|учен\w*|назов\w*|расскаж\w*)\b",
    re.IGNORECASE,
)

# Precision-biased subject signals. Ambiguous terms such as "атом", "масса"
# and "диффузия" are deliberately omitted because they occur across subjects.
_SUBJECT_QUERY_RES: dict[str, re.Pattern[str]] = {
    "chemistry": re.compile(
        r"(?:хими\w*|химик\w*|реакци\w*|молекул\w*|элемент\w*|кислот\w*|"
        r"қышқыл\w*|щелоч\w*|сілті\w*|периодическ\w*|периодтық\w*|"
        r"менделе\w*|авогадро\w*|окислен\w*|тотығ\w*)",
        re.IGNORECASE,
    ),
    "physics": re.compile(
        r"(?:физик\w*|ньютон\w*|эйнштейн\w*|энштейн\w*|механик\w*|"
        r"электр\w*|напряж\w*|қысым\w*|давлен\w*|жылдамдық\w*|скорост\w*|"
        r"оптик\w*|жарық\w*|гравитац\w*)",
        re.IGNORECASE,
    ),
    "biology": re.compile(
        r"(?:биолог\w*|жасуш\w*|клетк\w*|организм\w*|өсімдік\w*|растени\w*|"
        r"животн\w*|генет\w*|эволюц\w*|анатом\w*|фотосинтез\w*|"
        r"экосистем\w*|днк\w*)",
        re.IGNORECASE,
    ),
}

_BOILERPLATE_RE = re.compile(
    r"(?:okulyk(?:\.kz)?|оқулық(?:тар)?|учебники\s+казахстана)",
    re.IGNORECASE,
)

_GENERAL_KNOWLEDGE_MARKER = "[[GENERAL_KNOWLEDGE]]"
_GROUNDED_MARKER = "[[GROUNDED]]"
_ANSWER_MODE_MARKERS = (_GENERAL_KNOWLEDGE_MARKER, _GROUNDED_MARKER)

# Precision-biased RU/KK intent signals for questions whose authoritative source
# is the current lab instruction. General theory wording such as "почему" or
# "как происходит" deliberately does not match, even when a lab is active.
_LAB_PROCEDURE_QUERY_RE = re.compile(
    r"(?:"
    r"(?:что|чего)\s+(?:мне\s+)?(?:делать|сделать)(?:\s+(?:дальше|сейчас|теперь))?"
    r"|(?:следующ\w*|текущ\w*)\s+шаг"
    r"|(?:порядок|последовательность)\s+(?:моих\s+)?действий"
    r"|ход\s+(?:этой\s+)?работы"
    r"|как\s+(?:мне\s+)?(?:выполнить|провести|начать|продолжить|завершить)"
    r"|куда\s+(?:мне\s+)?(?:положить|поставить|налить|переместить)"
    r"|(?:какова|какая|в\s+ч[её]м)\s+цель\s+(?:лабораторной\s+)?работы"
    r"|(?:опиши(?:те)?|перечисли(?:те)?|назови(?:те)?)\s+"
    r"(?:основн\w+\s+)?(?:этапы|шаги)\s+(?:выполнения|проведения)"
    r"|что\s+(?:мы\s+)?наблюда\w*\s+в\s+ходе\s+"
    r"(?:эксперимента|опыта|работы)"
    r"|(?:главн\w+\s+)?результат\s+(?:в\s+)?конце\s+"
    r"(?:эксперимента|опыта|работы)"
    r"|(?:енді|қазір|әрі\s+қарай)\s+не\s+істе(?:у(?:ім)?|ймін)"
    r"|не\s+істеу(?:ім)?\s+(?:керек|қажет)"
    r"|(?:келесі|қазіргі)\s+қадам"
    r"|әрекеттер?\s+(?:реті|тәртібі)"
    r"|жұмыс\s+(?:барысы|тәртібі)"
    r"|қалай\s+(?:орындау|жасау|бастау|жалғастыру|аяқтау)"
    r"|қайда\s+(?:қою|құю|орналастыру)"
    r"|(?:зертханалық\s+)?жұмыстың\s+мақсаты"
    r")",
    re.IGNORECASE,
)


# ── Prompt construction ──────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "Ты — дружелюбный ИИ-ассистент внутри школьного VR-тренажёра для "
    "лабораторных работ по физике, химии и биологии. Ты выступаешь в роли "
    "терпеливого учителя-помощника.\n\n"
    "Твои задачи:\n"
    "1. Подсказывать ученику по текущему сценарию: что делать дальше, как "
    "выполнить действие, зачем нужен этот этап.\n"
    "2. Объяснять теоретический материал по теме лабораторной работы.\n\n"
    "Правила:\n"
    "1. Отвечай на том же языке, на котором задан вопрос (русский или казахский).\n"
    "2. Материалы базы знаний, инструкция и описание сценария ниже являются "
    "приоритетными и авторитетными источниками. Не противоречь им.\n"
    "3. Общие научные знания разрешено использовать только когда ниже явно "
    "включён специальный режим. Без такого указания опирайся ТОЛЬКО на "
    "предоставленные материалы.\n"
    "4. Если ответа нет в разрешённых источниках, честно скажи об этом одной "
    "фразой и не выдумывай.\n"
    "5. Отвечай кратко и по существу: обычно 1–4 предложения. Говори тепло и "
    "понятно, как учитель школьнику.\n"
    "6. НЕ добавляй в конце строку «Источник: …», источники прикрепляются "
    "автоматически отдельно.\n"
    "7. Предыдущие реплики диалога используй только для понимания контекста, "
    "а не как источник фактов.\n"
)


def build_system_prompt(
    scenario_context: Optional[str] = None,
    scenario_state: Optional[str] = None,
    knowledge_context: Optional[str] = None,
    lab_instruction: Optional[str] = None,
    lab_incomplete: bool = False,
    answer_language: Optional[str] = None,
    allow_general_knowledge: bool = False,
) -> str:
    """Assemble the system prompt, appending the grounding blocks if present.

    ``scenario_context`` is the static lab description; ``scenario_state`` is the
    live per-request state from the simulator (current step, held items);
    ``knowledge_context`` is the formatted block of chunks retrieved from the
    local hybrid RAG store (see ``_format_knowledge``); ``lab_instruction`` is
    the authoritative procedure text for the current lab, injected verbatim.
    When ``lab_incomplete`` is set, the model is told the procedure is
    unavailable so it answers theory-only instead of inventing steps.
    """
    prompt = _BASE_SYSTEM_PROMPT
    if answer_language == "kk":
        prompt += (
            "\nЯЗЫК ОТВЕТА: казахский. Жауапты толық қазақ тілінде жаз. "
            "Орыс тіліне ауыспа.\n"
        )
    elif answer_language == "ru":
        prompt += "\nЯЗЫК ОТВЕТА: русский. Отвечай полностью на русском языке.\n"
    if lab_instruction and lab_instruction.strip():
        prompt += (
            "\n--- ИНСТРУКЦИЯ К ТЕКУЩЕЙ ЛАБОРАТОРНОЙ РАБОТЕ ---\n"
            f"{lab_instruction.strip()}\n"
            "--- КОНЕЦ ИНСТРУКЦИИ ---\n"
            "Это официальная методичка текущей лабораторной работы. На вопросы "
            "о шагах, цели и порядке действий отвечай строго по ней.\n"
        )
    elif lab_incomplete:
        prompt += (
            "\nВНИМАНИЕ: пошаговая инструкция для текущей лабораторной работы "
            "недоступна. Не выдумывай шаги — отвечай только на теоретические "
            "вопросы по базе знаний, а про порядок действий честно скажи, что "
            "точной инструкции у тебя нет.\n"
        )
    if knowledge_context and knowledge_context.strip():
        prompt += (
            "\n--- БАЗА ЗНАНИЙ (найденные документы) ---\n"
            f"{knowledge_context.strip()}\n"
            "--- КОНЕЦ БАЗЫ ЗНАНИЙ ---\n"
            "Отвечай на теоретические вопросы, опираясь на эти фрагменты. "
            "Если нужного нет — честно скажи.\n"
        )
    if scenario_context and scenario_context.strip():
        prompt += (
            "\n--- ОПИСАНИЕ ТЕКУЩЕГО СЦЕНАРИЯ ---\n"
            f"{scenario_context.strip()}\n"
            "--- КОНЕЦ ОПИСАНИЯ СЦЕНАРИЯ ---\n"
            "По вопросам про текущую сцену (где предмет, какой следующий шаг, "
            "зачем этот этап) отвечай строго по этому описанию.\n"
        )
    if scenario_state and scenario_state.strip():
        prompt += (
            "\n--- ТЕКУЩЕЕ СОСТОЯНИЕ СЦЕНЫ (актуально на этот запрос) ---\n"
            f"{scenario_state.strip()}\n"
            "--- КОНЕЦ СОСТОЯНИЯ СЦЕНЫ ---\n"
            "Это живое состояние от тренажёра. На вопросы «что дальше», «что "
            "сейчас делать», «что у меня в руках» отвечай с опорой на него.\n"
        )
    if allow_general_knowledge:
        prompt += (
            "\n--- РЕЖИМ РЕЗЕРВНОГО ОБЩЕНАУЧНОГО ОТВЕТА ---\n"
            "Сначала оцени, содержат ли найденные документы прямую и достаточную "
            "информацию для ответа на вопрос. Если да, отвечай строго по ним и "
            f"начни ответ с маркера {_GROUNDED_MARKER}. Если документов нет, они "
            "повреждены, состоят из служебного текста или не отвечают на вопрос, "
            "ты ОБЯЗАН использовать надёжные общеизвестные научные знания и "
            f"начать ответ с маркера {_GENERAL_KNOWLEDGE_MARKER}. В этом режиме "
            "запрещено отказывать только потому, что в найденных документах нет "
            "прямого ответа, и запрещено сообщать пользователю об отсутствии "
            "информации в материалах. Егер құжаттарда тікелей жауап болмаса, "
            "бас тартпай, жалпы ғылыми біліммен қазақ тілінде жауап бер. Не смешивай "
            "два режима. Маркер ставь первым, до любых других символов. После "
            "маркера сразу дай обычный ответ на языке вопроса.\n"
            "--- КОНЕЦ РЕЖИМА ---\n"
        )
    return prompt


# ── Local hybrid retrieval ────────────────────────────────────────────────────


def _infer_query_language(query: str) -> str:
    """Infer the supported answer language (Kazakh or Russian) from the query."""
    text = query or ""
    kazakh_score = len(_KAZAKH_CHAR_RE.findall(text)) * 2
    kazakh_score += len(_KAZAKH_WORD_RE.findall(text))
    russian_score = len(_RUSSIAN_WORD_RE.findall(text))
    if kazakh_score > russian_score:
        return "kk"
    if russian_score > kazakh_score:
        return "ru"
    return (
        settings.DEFAULT_LANGUAGE
        if settings.DEFAULT_LANGUAGE in {"ru", "kk"}
        else "ru"
    )


def _infer_query_subject(query: str) -> Optional[str]:
    """Infer a school subject only when the query contains a strong signal."""
    scores = {
        subject: len(pattern.findall(query or ""))
        for subject, pattern in _SUBJECT_QUERY_RES.items()
    }
    best_subject, best_score = max(scores.items(), key=lambda item: item[1])
    tied = sum(score == best_score for score in scores.values()) > 1
    return best_subject if best_score > 0 and not tied else None


def _infer_query_context(query: str) -> tuple[Optional[str], str]:
    """Return ``(subject, language)`` inferred from a standalone question."""
    return _infer_query_subject(query), _infer_query_language(query)


def _is_usable_knowledge_text(text: str) -> bool:
    """Reject empty, watermark-only and severely repetitive retrieval text."""
    normalized = _WS_RE.sub(" ", text or "").strip()
    words = [word.casefold() for word in _WORD_RE.findall(normalized)]
    if len(words) < 4 or sum(len(word) for word in words) < 20:
        return False
    if len(_BOILERPLATE_RE.findall(normalized)) >= 3:
        return False
    if len(words) >= 30 and len(set(words)) / len(words) < 0.12:
        return False

    lines = [_WS_RE.sub(" ", line).strip().casefold() for line in (text or "").splitlines()]
    substantial_lines = [line for line in lines if len(line) >= 12]
    if (
        len(substantial_lines) >= 4
        and len(set(substantial_lines)) / len(substantial_lines) < 0.35
    ):
        return False
    return True


def _usable_theory_chunks(chunks: list[dict]) -> list[dict]:
    """Keep only retrieved chunks containing plausible educational content."""
    return [
        chunk
        for chunk in chunks
        if _is_usable_knowledge_text((chunk.get("payload") or {}).get("text") or "")
    ]


async def _search(query_filter: Any, dense, sparse_indices, sparse_values) -> list[dict]:
    """One hybrid search + score-threshold filter for a given ``query_filter``."""
    kwargs = {}
    if query_filter is not None:
        kwargs["query_filter"] = query_filter
    chunks = await vectorstore.hybrid_search(
        dense,
        sparse_indices,
        sparse_values,
        top_k=settings.RETRIEVAL_TOP_K,
        candidates=settings.RETRIEVAL_CANDIDATES,
        **kwargs,
    )
    threshold = settings.RETRIEVAL_SCORE_THRESHOLD
    if threshold > 0:
        chunks = [c for c in chunks if c.get("score", 0.0) >= threshold]
    return chunks


def _dedup_key(chunk: dict) -> Any:
    """Stable identity for a retrieved chunk (document + position).

    Used to dedupe across retrieval tiers, which return freshly-built dicts (so
    ``id()`` can't match). ``(doc_id, chunk_index)`` is unique per chunk.
    """
    payload = chunk.get("payload") or {}
    return (payload.get("doc_id"), payload.get("chunk_index"))


async def _retrieve(
    query: str,
    query_filter: Any = None,
    lang: Optional[str] = None,
    fallback_filter: Any = None,
) -> list[dict]:
    """Embed the query and hybrid-search the local Qdrant knowledge base.

    Returns a list of scored chunks (``{"score", "payload"}``). Chunks scoring
    below ``RETRIEVAL_SCORE_THRESHOLD`` are dropped.

    Retrieval is tiered from the most- to the least-specific scope, deduped, and
    stops once ``RETRIEVAL_TOP_K`` chunks are gathered:

      * ``query_filter`` is the narrow scope (e.g. subject **and** grade) —
        right-grade chapters hold the worked examples with the actual
        numbers/formulas the answer needs.
      * ``fallback_filter`` (optional) is the broader scope (e.g. subject-only),
        used to backfill when the narrow scope comes back thin. Several
        grade-specific chapters are missing/poorly-OCR'd, so a hard narrow
        filter alone would turn those into refusals — the broader scope keeps
        them answerable.
      * within each scope, when ``lang`` is known we prefer same-language chunks
        first, then any language.

    This keeps retrieval grade- and language-targeted while degrading gracefully
    to the subject (and the other language) when the targeted KB is sparse.
    """
    emb = await embeddings.embed_query(query)

    # Ordered scopes, narrowest first. Only add the broader fallback when it is
    # genuinely broader than ``query_filter`` (i.e. a grade was supplied), so we
    # don't run a duplicate search when no grade narrowing is in play.
    bases = [query_filter]
    if fallback_filter is not None:
        bases.append(fallback_filter)

    tiers: list[Any] = []
    for base in bases:
        if lang:
            tiers.append(vectorstore.with_lang(base, lang))
        tiers.append(base)

    merged: list[dict] = []
    seen: set = set()
    for tier in tiers:
        chunks = await _search(tier, emb.dense, emb.sparse_indices, emb.sparse_values)
        for chunk in chunks:
            key = _dedup_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
        if len(merged) >= settings.RETRIEVAL_TOP_K:
            break
    return merged[: settings.RETRIEVAL_TOP_K]


async def _lab_grounding(
    lab: Optional[dict],
) -> tuple[Optional[str], bool, Any, Optional[str], Any, list[dict]]:
    """Resolve per-lab grounding from the structured ``lab`` context.

    Returns ``(lab_instruction, lab_incomplete, query_filter, lang,
    fallback_filter, lab_source_chunks)``:
      * ``lab_instruction`` — verbatim procedure text fetched from Qdrant by
        ``lab_id``, or None;
      * ``lab_incomplete`` — True when a specific lab was named (has ``lab_id``)
        but no instruction exists in the store;
      * ``query_filter`` — scopes theory retrieval to the lab's subject **and**
        grade, so a grade-7 question gets the grade-7 worked examples (with the
        real numbers) rather than generic high-school prose;
      * ``lang`` — the question's language ("ru"/"kk") from the lab context,
        used to prefer same-language chunks during retrieval;
      * ``fallback_filter`` — the broader subject-only scope ``_retrieve`` falls
        back to when the grade-scoped KB is thin (set only when a grade is
        known). Grade chapters are sometimes missing/poorly-OCR'd, so we never
        hard-filter on grade alone.
      * ``lab_source_chunks`` - stored payloads for the injected instruction,
        used to return a citation to the actual procedure document.
    """
    if not lab:
        return None, False, None, None, None, []

    subject = lab.get("subject")
    grade = lab.get("grade")
    query_filter = vectorstore.meta_filter(
        doc_type="textbook", subject=subject, grade=grade
    )
    # Subject-only backstop, used only when a grade actually narrowed the scope.
    fallback_filter = (
        vectorstore.meta_filter(doc_type="textbook", subject=subject)
        if grade is not None
        else None
    )
    lang = lab.get("lang")

    lab_id = lab.get("lab_id")
    if not lab_id:
        return None, False, query_filter, lang, fallback_filter, []

    record = await vectorstore.fetch_lab_instruction_record(lab_id)
    instruction = record["text"] if record else ""
    source_chunks = [
        {"payload": payload} for payload in (record or {}).get("payloads", [])
    ]
    return (
        (instruction or None),
        (not instruction),
        query_filter,
        lang,
        fallback_filter,
        source_chunks,
    )


def _format_knowledge(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt.

    Each chunk becomes ``[N] (filename)\\n<text>``. Returns "" when there are no
    chunks, so callers can simply pass the result through to the prompt builder.
    """
    if not chunks:
        return ""
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.get("payload") or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        filename = payload.get("filename") or "?"
        blocks.append(f"[{i}] ({filename})\n{text}")
    return "\n\n".join(blocks)


def _citations_from_chunks(chunks: list[dict]) -> list[dict]:
    """Group chunk metadata into stable, locator-rich document citations.

    ``filename`` and ``file_id`` remain present for existing clients. New fields
    describe the source type/path and aggregate chunk, page, chapter and section
    locators across all retrieved chunks from the same document. Payloads that
    cannot identify a real document are skipped, so no null citations leak into
    API responses.
    """
    grouped: dict[tuple[str, str], dict] = {}
    for chunk in chunks:
        payload = chunk.get("payload") or {}
        source_path = payload.get("source_path") or payload.get("source")
        filename = payload.get("filename")
        if not filename and source_path:
            filename = re.split(r"[/\\]", str(source_path))[-1]
        file_id = payload.get("doc_id") or payload.get("file_id")
        if not filename or not file_id:
            continue

        key = (str(file_id), str(source_path or filename))
        citation = grouped.get(key)
        if citation is None:
            citation = {"filename": filename, "file_id": file_id}
            for field_name, value in (
                ("source_type", payload.get("source_type") or payload.get("doc_type")),
                ("source_path", source_path),
                ("file_type", payload.get("file_type")),
                ("subject", payload.get("subject")),
                ("grade", payload.get("grade")),
                ("lang", payload.get("lang")),
                ("lab_id", payload.get("lab_id")),
                ("lab_number", payload.get("lab_number")),
            ):
                if value is not None and value != "":
                    citation[field_name] = value
            citation["_chunk_indexes"] = set()
            citation["_pages"] = set()
            citation["_chapters"] = []
            citation["_sections"] = []
            grouped[key] = citation

        chunk_indexes = payload.get("chunk_indexes")
        if not isinstance(chunk_indexes, (list, tuple, set)):
            chunk_indexes = [payload.get("chunk_index")]
        citation["_chunk_indexes"].update(
            value for value in chunk_indexes if isinstance(value, int)
        )

        pages = payload.get("pages")
        if not isinstance(pages, (list, tuple, set)):
            pages = [pages]
        pages = list(pages) + [payload.get("page_start"), payload.get("page_end")]
        citation["_pages"].update(value for value in pages if isinstance(value, int))

        for field_name, accumulator in (
            ("chapter", "_chapters"),
            ("section", "_sections"),
        ):
            value = payload.get(field_name)
            if value and value not in citation[accumulator]:
                citation[accumulator].append(value)

    citations: list[dict] = []
    for citation in grouped.values():
        chunk_indexes = sorted(citation.pop("_chunk_indexes"))
        pages = sorted(citation.pop("_pages"))
        chapters = citation.pop("_chapters")
        sections = citation.pop("_sections")
        if chunk_indexes:
            citation["chunk_indexes"] = chunk_indexes
        if pages:
            citation["pages"] = pages
            citation["page_start"] = pages[0]
            citation["page_end"] = pages[-1]
        if chapters:
            citation["chapters"] = chapters
            if len(chapters) == 1:
                citation["chapter"] = chapters[0]
        if sections:
            citation["sections"] = sections
            if len(sections) == 1:
                citation["section"] = sections[0]
        citation["display_label"] = _citation_display_label(citation)
        citations.append(citation)
    return citations


def _citation_display_label(citation: dict) -> str:
    """Build a concise Russian display label from structured citation data."""
    if citation.get("source_type") == "lab_instruction":
        number = citation.get("lab_number")
        label = "Инструкция к лабораторной работе"
        if number is not None:
            label += f" №{number}"
    else:
        filename = str(citation.get("filename") or "Источник")
        label = re.sub(r"\.[^.]+$", "", filename)

    if citation.get("chapter"):
        label += f", {citation['chapter']}"
    elif citation.get("section"):
        label += f", {citation['section']}"

    page_start = citation.get("page_start")
    page_end = citation.get("page_end")
    if page_start is not None:
        page_label = str(page_start)
        if page_end is not None and page_end != page_start:
            page_label += f"-{page_end}"
        label += f", стр. {page_label}"
    return label


def _is_lab_procedure_query(query: str) -> bool:
    """Whether RU/KK wording clearly asks for current-lab procedure details."""
    normalized = _WS_RE.sub(" ", query or "").strip()
    return bool(normalized and _LAB_PROCEDURE_QUERY_RE.search(normalized))


def _answer_citations(
    query: str, theory_chunks: list[dict], lab_source_chunks: list[dict]
) -> list[dict]:
    """Build citations with primary-source ordering matched to query intent.

    Lab procedure/current-step questions prefer the lab instruction. Theory
    questions prefer retrieved textbooks, while still retaining the injected
    lab instruction after them for transparency.
    """
    if _is_lab_procedure_query(query):
        ordered_chunks = [*lab_source_chunks, *theory_chunks]
    else:
        ordered_chunks = [*theory_chunks, *lab_source_chunks]
    return _citations_from_chunks(ordered_chunks)


def _general_fallback_allowed(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab_instruction: Optional[str],
    lab_incomplete: bool,
) -> bool:
    """Whether this request may fall back to reliable general science facts."""
    if not settings.ALLOW_GENERAL_KNOWLEDGE_FALLBACK:
        return False
    if any(
        value and value.strip()
        for value in (scenario_context, scenario_state, lab_instruction)
    ):
        return False
    # A missing lab procedure must never be replaced by invented generic steps.
    if lab_incomplete and _is_lab_procedure_query(query):
        return False
    return True


def _parse_answer_mode(
    text: str,
    *,
    allow_general_knowledge: bool,
    default_general: bool = False,
) -> tuple[str, bool]:
    """Strip the private answer-mode marker and report general-knowledge use."""
    answer = (text or "").strip()
    if not allow_general_knowledge:
        return answer, False
    if answer.startswith(_GENERAL_KNOWLEDGE_MARKER):
        return answer[len(_GENERAL_KNOWLEDGE_MARKER) :].lstrip(), True
    if answer.startswith(_GROUNDED_MARKER):
        return answer[len(_GROUNDED_MARKER) :].lstrip(), False
    # Be conservative when the model omitted the requested marker: empty
    # retrieval cannot produce a grounded answer, while non-empty retrieval
    # keeps its citations unless the model explicitly selected general mode.
    return answer, default_general


class _StreamingAnswerModeParser:
    """Suppress a possibly split private answer-mode marker from SSE deltas."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.buffer = ""
        self.resolved = not enabled

    def feed(self, delta: str) -> str:
        if self.resolved:
            return delta
        self.buffer += delta
        candidate = self.buffer.lstrip()
        for marker in _ANSWER_MODE_MARKERS:
            if candidate.startswith(marker):
                self.resolved = True
                remainder = candidate[len(marker) :].lstrip()
                self.buffer = ""
                return remainder
        if any(marker.startswith(candidate) for marker in _ANSWER_MODE_MARKERS):
            return ""
        self.resolved = True
        buffered = self.buffer
        self.buffer = ""
        return buffered

    def finish(self) -> str:
        if self.resolved:
            return ""
        self.resolved = True
        buffered = self.buffer
        self.buffer = ""
        cleaned, _ = _parse_answer_mode(
            buffered,
            allow_general_knowledge=self.enabled,
        )
        return cleaned


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    # Fallback: walk the output items.
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if t:
                parts.append(t)
    return "".join(parts).strip()


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def primary_source(self) -> Optional[dict]:
        return self.citations[0] if self.citations else None


def _usage_dict(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": getattr(usage, "total_tokens", input_tokens + output_tokens),
    }


def _flat(text: str, limit: int) -> str:
    flat = _WS_RE.sub(" ", text or "").strip()
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _log_generation(tag: str, query: str, result: AnswerResult) -> None:
    if not settings.LOG_GENERATION:
        return
    limit = settings.LOG_GENERATION_MAX_CHARS
    sources = "; ".join(c.get("filename", "?") for c in result.citations) or "(none)"
    logger.info(
        "[%s]\n  QUERY: %s\n  SOURCES: %s\n  ANSWER: %s",
        tag,
        _flat(query, limit),
        sources,
        _flat(result.answer, limit),
    )


# ── Answer cache + generation params ────────────────────────────────────────

_answer_cache = TTLCache(settings.ANSWER_CACHE_SIZE, settings.ANSWER_CACHE_TTL_S)


def clear_answer_cache() -> None:
    """Drop cached answers after the knowledge corpus changes."""
    _answer_cache.clear()


def _answer_cache_key(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab: Optional[dict],
    max_tokens: Optional[int],
) -> tuple:
    """Everything that changes the generated answer, minus chat history.

    Only single-turn requests are cached (multi-turn answers depend on the
    dialogue); the caller passes a key only in that case.
    """
    return (
        _WS_RE.sub(" ", query).strip().casefold(),
        scenario_context or "",
        scenario_state or "",
        tuple(sorted((k, str(v)) for k, v in (lab or {}).items())),
        max_tokens or 0,
    )


def _tier_kwargs() -> dict:
    """Optional OpenAI service tier (e.g. "priority" for faster first-token)."""
    tier = settings.OPENAI_SERVICE_TIER
    return {"service_tier": tier} if tier else {}


@dataclass
class _AnswerGrounding:
    system_prompt: str
    theory_chunks: list[dict]
    lab_source_chunks: list[dict]
    allow_general_knowledge: bool


async def _prepare_answer_grounding(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab: Optional[dict],
) -> _AnswerGrounding:
    """Build identical retrieval and fallback state for both generation paths."""
    inferred_subject, answer_language = _infer_query_context(query)
    (
        lab_instruction,
        lab_incomplete,
        query_filter,
        retrieval_lang,
        fallback_filter,
        lab_source_chunks,
    ) = await _lab_grounding(lab)

    if not lab and inferred_subject:
        query_filter = vectorstore.meta_filter(
            doc_type="textbook", subject=inferred_subject
        )
    if not lab:
        # Apply language preference together with an inferred subject. For
        # subject-ambiguous questions, a global language-only filter can hide
        # the best cross-language semantic hit; the answer prompt is still
        # locked to the detected query language.
        retrieval_lang = answer_language if inferred_subject else None
    elif not retrieval_lang:
        retrieval_lang = answer_language

    retrieved = await _retrieve(
        query,
        query_filter=query_filter,
        lang=retrieval_lang,
        fallback_filter=fallback_filter,
    )
    theory_chunks = _usable_theory_chunks(retrieved)
    allow_general_knowledge = _general_fallback_allowed(
        query,
        scenario_context,
        scenario_state,
        lab_instruction,
        lab_incomplete,
    )
    system_prompt = build_system_prompt(
        scenario_context,
        scenario_state,
        knowledge_context=_format_knowledge(theory_chunks),
        lab_instruction=lab_instruction,
        lab_incomplete=lab_incomplete,
        answer_language=answer_language,
        allow_general_knowledge=allow_general_knowledge,
    )
    return _AnswerGrounding(
        system_prompt=system_prompt,
        theory_chunks=theory_chunks,
        lab_source_chunks=lab_source_chunks,
        allow_general_knowledge=allow_general_knowledge,
    )


# ── Core completion ────────────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type((LLMTimeoutError, LLMUpstreamError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def generate_answer(
    query: str,
    scenario_context: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
    max_tokens: Optional[int] = None,
    scenario_state: Optional[str] = None,
    lab: Optional[dict] = None,
) -> AnswerResult:
    """Generate a grounded answer (non-streaming).

    ``lab`` is the structured lab context from the simulator (subject/grade/
    lang/lab_number + composed ``lab_id``); it scopes retrieval to the subject
    and injects the lab's procedure verbatim. Retries up to 3 times on
    transient timeout / upstream (5xx) errors. Single-turn requests are served
    from the in-process answer cache when an identical question (same scenario/
    state/lab) was answered within ``ANSWER_CACHE_TTL_S``.
    """
    cache_key = (
        _answer_cache_key(query, scenario_context, scenario_state, lab, max_tokens)
        if chat_history is None or len(chat_history) <= 1
        else None
    )
    if cache_key is not None:
        cached = _answer_cache.get(cache_key)
        if cached is not None:
            _log_generation("answer_cached", query, cached)
            return cached

    grounding = await _prepare_answer_grounding(
        query,
        scenario_context,
        scenario_state,
        lab,
    )
    raw_history = chat_history if chat_history is not None else [{"role": "user", "content": query}]
    history = trim_history(
        raw_history,
        max_messages=settings.CHAT_MEMORY_MAX_MESSAGES,
        max_chars=settings.CHAT_MEMORY_HISTORY_CHARS,
    )
    input_messages = build_input_messages(history)

    try:
        response = await client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=grounding.system_prompt,
            input=input_messages,
            max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            **_tier_kwargs(),
        )
    except openai.APIError as exc:
        raise _map_openai_error(exc) from exc

    answer, used_general_knowledge = _parse_answer_mode(
        _response_text(response),
        allow_general_knowledge=grounding.allow_general_knowledge,
        default_general=not grounding.theory_chunks,
    )
    if not answer:
        raise LLMMalformedResponseError("OpenAI returned an empty answer")

    result = AnswerResult(
        answer=answer,
        citations=(
            []
            if used_general_knowledge
            else _answer_citations(
                query, grounding.theory_chunks, grounding.lab_source_chunks
            )
        ),
        usage=_usage_dict(response),
    )
    _log_generation("answer", query, result)
    if cache_key is not None:
        _answer_cache.put(cache_key, result)
    return result


async def stream_answer(
    query: str,
    scenario_context: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
    max_tokens: Optional[int] = None,
    scenario_state: Optional[str] = None,
    lab: Optional[dict] = None,
) -> AsyncIterator[dict]:
    """Stream a grounded answer as a sequence of events:

      {"type": "delta", "text": "..."}        # incremental answer text
      {"type": "done", "citations": [...], "usage": {...}}
      {"type": "error", "message": "..."}     # on upstream failure

    The route layer turns these into SSE frames. ``lab`` carries the same
    structured lab context as :func:`generate_answer`. Shares the answer cache
    with :func:`generate_answer` — a hit yields the whole answer as one delta.
    """
    cache_key = (
        _answer_cache_key(query, scenario_context, scenario_state, lab, max_tokens)
        if chat_history is None or len(chat_history) <= 1
        else None
    )
    if cache_key is not None:
        cached = _answer_cache.get(cache_key)
        if cached is not None:
            _log_generation("answer_stream_cached", query, cached)
            yield {"type": "delta", "text": cached.answer}
            yield {"type": "done", "citations": cached.citations, "usage": cached.usage}
            return

    grounding = await _prepare_answer_grounding(
        query,
        scenario_context,
        scenario_state,
        lab,
    )
    raw_history = chat_history if chat_history is not None else [{"role": "user", "content": query}]
    history = trim_history(
        raw_history,
        max_messages=settings.CHAT_MEMORY_MAX_MESSAGES,
        max_chars=settings.CHAT_MEMORY_HISTORY_CHARS,
    )
    input_messages = build_input_messages(history)

    marker_parser = _StreamingAnswerModeParser(grounding.allow_general_knowledge)
    try:
        async with client.responses.stream(
            model=settings.OPENAI_MODEL,
            instructions=grounding.system_prompt,
            input=input_messages,
            max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            **_tier_kwargs(),
        ) as stream:
            async for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        visible_delta = marker_parser.feed(delta)
                        if visible_delta:
                            yield {"type": "delta", "text": visible_delta}
            final = await stream.get_final_response()
            trailing = marker_parser.finish()
            if trailing:
                yield {"type": "delta", "text": trailing}
    except openai.APIError as exc:
        mapped = _map_openai_error(exc)
        logger.error("Streaming LLM error: %s", mapped, exc_info=True)
        yield {"type": "error", "message": str(mapped)}
        return

    answer, used_general_knowledge = _parse_answer_mode(
        _response_text(final),
        allow_general_knowledge=grounding.allow_general_knowledge,
        default_general=not grounding.theory_chunks,
    )
    result = AnswerResult(
        answer=answer,
        citations=(
            []
            if used_general_knowledge
            else _answer_citations(
                query, grounding.theory_chunks, grounding.lab_source_chunks
            )
        ),
        usage=_usage_dict(final),
    )
    _log_generation("answer_stream", query, result)
    if cache_key is not None and result.answer:
        _answer_cache.put(cache_key, result)
    yield {"type": "done", "citations": result.citations, "usage": result.usage}


# ── Hint rephrasing (Task 2 of the spec) ─────────────────────────────────────

_HINT_SYSTEM_PROMPT = (
    "Ты — ИИ-ассистент школьного VR-тренажёра. Тренажёр сам решает, КОГДА и "
    "КАКУЮ подсказку показать. Твоя единственная задача — перефразировать "
    "готовый текст подсказки так, чтобы он звучал естественно и по-учительски "
    "в контексте сцены.\n\n"
    "Правила перефразирования:\n"
    "- Полностью сохраняй смысл исходной подсказки.\n"
    "- Используй контекст сцены, чтобы звучать естественнее.\n"
    "- Уровень 1: краткая фраза (1 предложение).\n"
    "- Уровень 2: конкретнее, с упоминанием объектов из сцены (1–2 предложения).\n"
    "- Уровень 3: подробно, с пошаговым действием (2–3 предложения).\n"
    "- Не добавляй информацию, которой нет в описании сценария или подсказке.\n"
    "- Отвечай ТОЛЬКО перефразированным текстом, без пояснений и кавычек.\n"
)


@retry(
    retry=retry_if_exception_type((LLMTimeoutError, LLMUpstreamError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def rephrase_hint(
    hint_text: str,
    hint_level: int,
    scenario_context: Optional[str] = None,
    scenario_state: Optional[str] = None,
) -> str:
    """Rephrase a simulator-provided hint at the given verbosity level.

    No file search — the hint and scenario context are all the model needs.
    """
    instructions = _HINT_SYSTEM_PROMPT
    if scenario_context and scenario_context.strip():
        instructions += (
            "\n--- ОПИСАНИЕ СЦЕНАРИЯ ---\n"
            f"{scenario_context.strip()}\n"
            "--- КОНЕЦ ---\n"
        )
    if scenario_state and scenario_state.strip():
        instructions += (
            "\n--- ТЕКУЩЕЕ СОСТОЯНИЕ СЦЕНЫ ---\n"
            f"{scenario_state.strip()}\n"
            "--- КОНЕЦ ---\n"
        )
    user_msg = f"Уровень подсказки: {hint_level}\nТекст подсказки: {hint_text}"

    try:
        response = await client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=instructions,
            input=[{"role": "user", "content": user_msg}],
            max_output_tokens=256,
            **_tier_kwargs(),
            temperature=0.4,
        )
    except openai.APIError as exc:
        raise _map_openai_error(exc) from exc

    text = _response_text(response)
    if not text:
        raise LLMMalformedResponseError("OpenAI returned an empty hint")
    return text
