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
    "2. Опирайся ТОЛЬКО на материалы из базы знаний (найденные документы) и на "
    "описание текущего сценария ниже. Не придумывай факты.\n"
    "3. Если ответа нет ни в документах, ни в сценарии — честно скажи об этом "
    "одной фразой и не выдумывай.\n"
    "4. Отвечай кратко и по существу: обычно 1–4 предложения. Говори тепло и "
    "понятно, как учитель школьнику.\n"
    "5. НЕ добавляй в конце строку «Источник: …» — источники прикрепляются "
    "автоматически отдельно.\n"
    "6. Предыдущие реплики диалога используй только для понимания контекста, "
    "а не как источник фактов.\n"
)


def build_system_prompt(
    scenario_context: Optional[str] = None,
    scenario_state: Optional[str] = None,
    knowledge_context: Optional[str] = None,
    lab_instruction: Optional[str] = None,
    lab_incomplete: bool = False,
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
    return prompt


# ── Local hybrid retrieval ────────────────────────────────────────────────────


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
) -> tuple[Optional[str], bool, Any, Optional[str], Any]:
    """Resolve per-lab grounding from the structured ``lab`` context.

    Returns ``(lab_instruction, lab_incomplete, query_filter, lang,
    fallback_filter)``:
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
    """
    if not lab:
        return None, False, None, None, None

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
        return None, False, query_filter, lang, fallback_filter

    instruction = await vectorstore.fetch_lab_instruction(lab_id)
    return (
        (instruction or None),
        (not instruction),
        query_filter,
        lang,
        fallback_filter,
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
    """Derive citations from retrieved chunks, deduped by filename.

    Returns ``{"filename", "file_id"}`` per distinct filename, in first-seen
    order. Tolerant of missing payload keys.
    """
    citations: list[dict] = []
    seen: set = set()
    for chunk in chunks:
        payload = chunk.get("payload") or {}
        filename = payload.get("filename")
        if filename in seen:
            continue
        seen.add(filename)
        citations.append(
            {"filename": filename, "file_id": payload.get("doc_id")}
        )
    return citations


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

    lab_instruction, lab_incomplete, query_filter, lang, fallback_filter = (
        await _lab_grounding(lab)
    )
    chunks = await _retrieve(
        query, query_filter=query_filter, lang=lang, fallback_filter=fallback_filter
    )
    knowledge = _format_knowledge(chunks)
    system_prompt = build_system_prompt(
        scenario_context,
        scenario_state,
        knowledge_context=knowledge,
        lab_instruction=lab_instruction,
        lab_incomplete=lab_incomplete,
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
            instructions=system_prompt,
            input=input_messages,
            max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            **_tier_kwargs(),
        )
    except openai.APIError as exc:
        raise _map_openai_error(exc) from exc

    answer = _response_text(response)
    if not answer:
        raise LLMMalformedResponseError("OpenAI returned an empty answer")

    result = AnswerResult(
        answer=answer,
        citations=_citations_from_chunks(chunks),
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

    lab_instruction, lab_incomplete, query_filter, lang, fallback_filter = (
        await _lab_grounding(lab)
    )
    chunks = await _retrieve(
        query, query_filter=query_filter, lang=lang, fallback_filter=fallback_filter
    )
    knowledge = _format_knowledge(chunks)
    system_prompt = build_system_prompt(
        scenario_context,
        scenario_state,
        knowledge_context=knowledge,
        lab_instruction=lab_instruction,
        lab_incomplete=lab_incomplete,
    )
    raw_history = chat_history if chat_history is not None else [{"role": "user", "content": query}]
    history = trim_history(
        raw_history,
        max_messages=settings.CHAT_MEMORY_MAX_MESSAGES,
        max_chars=settings.CHAT_MEMORY_HISTORY_CHARS,
    )
    input_messages = build_input_messages(history)

    try:
        async with client.responses.stream(
            model=settings.OPENAI_MODEL,
            instructions=system_prompt,
            input=input_messages,
            max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            **_tier_kwargs(),
        ) as stream:
            async for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield {"type": "delta", "text": delta}
            final = await stream.get_final_response()
    except openai.APIError as exc:
        mapped = _map_openai_error(exc)
        logger.error("Streaming LLM error: %s", mapped, exc_info=True)
        yield {"type": "error", "message": str(mapped)}
        return

    result = AnswerResult(
        answer=_response_text(final),
        citations=_citations_from_chunks(chunks),
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
