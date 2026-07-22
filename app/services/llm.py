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
from app.core.languages import (
    LANGUAGE_NAMES,
    LanguageCode,
    is_language_code,
    normalize_language_code,
)
from app.services import embeddings, vectorstore
from app.services.openai_client import client
from app.services.memory import (
    build_input_messages,
    build_retrieval_query,
    trim_history,
)
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
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_ENGLISH_SIGNAL_RE = re.compile(
    r"\b(?:what|which|where|when|why|who|how|is|are|was|were|do|does|did|"
    r"should|can|could|would|will|i|you|we|they|my|your|please|need|help|"
    r"show|tell|explain|calculate|measure|heat|the|this|that|these|those|a|an|of|to|in|"
    r"on|for|with|from|into|next|current|purpose|perform|experiment|lab|"
    r"laboratory|physics|chemistry|biology|chemical|reaction|molecule|"
    r"temperature|boiling|force|energy|motion|acceleration|gravity|voltage|"
    r"pressure|acid|cell|organism|photosynthesis)\b",
    re.IGNORECASE,
)
_ENGLISH_SINGLE_WORD_SIGNALS = {
    "biology",
    "boiling",
    "chemistry",
    "density",
    "electricity",
    "energy",
    "evaporation",
    "force",
    "gravity",
    "molecule",
    "photosynthesis",
    "physics",
    "temperature",
}
_LATIN_NON_PROSE_RE = re.compile(
    r"(?i:\b(?:https?://|www\.)\S+)|(?i:\b\S+@\S+\.\S+\b)|"
    r"(?<!\w)(?:[^\s/\\]+[/\\])*[^\s/\\]+\.[A-Za-z0-9]{1,12}(?!\w)|"
    r"(?<!\w)[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_.-]*(?!\w)|"
    r"(?<!\w)\d+[A-Za-z][A-Za-z0-9_.-]*(?!\w)|"
    r"(?<!\w)[A-Z]{2,}(?!\w)"
)

# Precision-biased subject signals. Ambiguous terms such as "атом", "масса"
# and "диффузия" are deliberately omitted because they occur across subjects.
_SUBJECT_QUERY_RES: dict[str, re.Pattern[str]] = {
    "chemistry": re.compile(
        r"(?:хими\w*|химик\w*|реакци\w*|молекул\w*|элемент\w*|кислот\w*|"
        r"қышқыл\w*|щелоч\w*|сілті\w*|периодическ\w*|периодтық\w*|"
        r"менделе\w*|авогадро\w*|окислен\w*|тотығ\w*|"
        r"\bchem(?:istry|ical|ist)s?\b|\breaction\w*\b|\bmolecule\w*\b|"
        r"\belement\w*\b|\bacid\w*\b|\balkali\w*\b|\bperiodic\w*\b|"
        r"\bmendeleev\w*\b|\bavogadro\w*\b|\boxidation\w*\b)",
        re.IGNORECASE,
    ),
    "physics": re.compile(
        r"(?:физик\w*|ньютон\w*|эйнштейн\w*|энштейн\w*|механик\w*|"
        r"электр\w*|напряж\w*|қысым\w*|давлен\w*|жылдамдық\w*|скорост\w*|"
        r"оптик\w*|жарық\w*|гравитац\w*|\bphysics?\b|\bnewton\w*\b|"
        r"\bmechanic\w*\b|\belectric\w*\b|\bvoltage\w*\b|\bpressure\w*\b|"
        r"\bvelocity\w*\b|\bacceleration\w*\b|\bmomentum\w*\b|\bforce\w*\b|"
        r"\benergy\b|\bmotion\b|\boptics?\b|\bgravity\w*\b)",
        re.IGNORECASE,
    ),
    "biology": re.compile(
        r"(?:биолог\w*|жасуш\w*|клетк\w*|организм\w*|өсімдік\w*|растени\w*|"
        r"животн\w*|генет\w*|эволюц\w*|анатом\w*|фотосинтез\w*|"
        r"экосистем\w*|днк\w*|\bbiology\b|\bbiological\w*\b|\bcell(?:s|ular)?\b|"
        r"\borganism\w*\b|\bplant\w*\b|\banimal\w*\b|\bgenetic\w*\b|"
        r"\bevolution\w*\b|\banatomy\b|\bphotosynthesis\b|\becosystem\w*\b|"
        r"\bdna\b)",
        re.IGNORECASE,
    ),
}

_BOILERPLATE_RE = re.compile(
    r"(?:okulyk(?:\.kz)?|оқулық(?:тар)?|учебники\s+казахстана)",
    re.IGNORECASE,
)

_GENERAL_KNOWLEDGE_MARKER = "[[GENERAL_KNOWLEDGE]]"
_GROUNDED_MARKER = "[[GROUNDED]]"
_LAB_SCOPE_REFUSALS = {
    "ru": (
        "Я могу отвечать только на вопросы, связанные с текущим предметом "
        "или лабораторной работой."
    ),
    "kk": (
        "Мен тек осы пәнге немесе ағымдағы зертханалық жұмысқа қатысты "
        "сұрақтарға жауап бере аламын."
    ),
    "en": (
        "I can only answer questions related to the current subject or "
        "laboratory activity."
    ),
}
# Generic prompt/scenario words must not make an unrelated question look related
# merely because both texts mention a lab, a question, or a current step.
_LAB_SCOPE_GENERIC_PREFIXES = (
    "авторитет",
    "актуаль",
    "вопрос",
    "жауап",
    "зертхан",
    "лаборатор",
    "описан",
    "ответ",
    "предмет",
    "работ",
    "сценар",
    "сұрақ",
    "текущ",
    "теор",
    "ученик",
    "эксперимент",
    "жұмыс",
    "қазір",
    "қадам",
    "мақсат",
    "опыт",
    "цель",
    "шаг",
    "activity",
    "answer",
    "current",
    "experiment",
    "lab",
    "laborator",
    "purpose",
    "question",
    "scenario",
    "step",
    "student",
    "theor",
    "work",
)
_MISSING_EVIDENCE_REFUSAL_RE = re.compile(
    r"(?:"
    r"(?:материал|құжат|баз)[^\n]{0,220}(?:ақпарат|мәлімет|тізім)[^\n]{0,80}жоқ"
    r"|(?:в\s+)?(?:материал|документ|баз)[^\n]{0,220}нет\s+"
    r"(?:информац|сведен|данн)"
    r"|не\s+могу\s+(?:ответить|помочь)[^\n]{0,120}(?:материал|документ|баз)"
    r"|(?:materials?|documents?|sources?|evidence)[^\n]{0,160}"
    r"(?:do(?:es)?\s+not|don't|doesn't|cannot|can't)[^\n]{0,80}"
    r"(?:contain|provide|include|have)[^\n]{0,80}(?:information|answer|evidence)"
    r"|(?:i\s+)?(?:cannot|can't|am\s+unable\s+to)\s+(?:answer|help)"
    r"[^\n]{0,120}(?:materials?|documents?|sources?|evidence)"
    r")",
    re.IGNORECASE,
)

# Precision-biased RU/KK/EN intent signals for questions whose authoritative source
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
    r"|what\s+(?:should|do)\s+i\s+do\s+(?:next|now)"
    r"|what\s+is\s+(?:the\s+)?(?:current|next)\s+step"
    r"|how\s+(?:do|should|can)\s+i\s+(?:perform|conduct|start|continue|finish)\s+"
    r"(?:this\s+)?(?:experiment|lab(?:oratory)?(?:\s+activity|\s+work)?)"
    r"|what\s+is\s+the\s+purpose\s+of\s+(?:this\s+)?(?:experiment|lab(?:oratory)?(?:\s+activity|\s+work)?)"
    r"|(?:describe|list)\s+the\s+(?:main\s+)?(?:steps|procedure)"
    r"|where\s+(?:should|do)\s+i\s+(?:put|place|pour|move)"
    r")",
    re.IGNORECASE,
)


# ── Prompt construction ──────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "You are a friendly teaching assistant inside a school VR laboratory for "
    "physics, chemistry, and biology. Help the student understand the current "
    "activity, its safe next actions, and the theory directly related to it.\n\n"
    "Rules:\n"
    "1. The requested answer language is stated explicitly below and must be "
    "followed. Supported answer languages are Russian, Kazakh, and English.\n"
    "2. The retrieved knowledge, laboratory instruction, static scenario, and "
    "live scene state are authoritative. Do not contradict them. Evidence may "
    "be written in another supported language. Translate it faithfully into the "
    "requested answer language without changing facts, measurements, or steps.\n"
    "3. General scientific knowledge is allowed only when a special fallback "
    "mode below explicitly enables it. Otherwise use only the supplied evidence.\n"
    "4. If the permitted evidence does not support an answer, say so briefly and "
    "do not invent details.\n"
    "5. Usually answer in 1 to 4 concise sentences, in a warm teacher-like tone.\n"
    "6. Do not append a source line. Citations are attached separately.\n"
    "7. Use earlier conversation turns only to resolve context, never as factual "
    "evidence.\n"
)


def build_system_prompt(
    scenario_context: Optional[str] = None,
    scenario_state: Optional[str] = None,
    knowledge_context: Optional[str] = None,
    lab_instruction: Optional[str] = None,
    lab_incomplete: bool = False,
    answer_language: Optional[LanguageCode] = None,
    allow_general_knowledge: bool = False,
    strict_lab_scope: bool = False,
) -> str:
    """Assemble the system prompt, appending the grounding blocks if present.

    ``scenario_context`` is the static lab description; ``scenario_state`` is the
    live per-request state from the simulator (current step, held items);
    ``knowledge_context`` is the formatted block of chunks retrieved from the
    local hybrid RAG store (see ``_format_knowledge``); ``lab_instruction`` is
    the authoritative procedure text for the current lab, injected verbatim.
    When ``lab_incomplete`` is set, the model is told the procedure is
    unavailable so it answers theory-only instead of inventing steps. Structured
    lab requests set ``strict_lab_scope`` so questions may cover either the
    current school subject or the current lab, while unrelated topics remain out
    of scope.
    """
    language = answer_language or normalize_language_code(
        settings.DEFAULT_LANGUAGE, field="DEFAULT_LANGUAGE"
    )
    prompt = _BASE_SYSTEM_PROMPT
    language_rules = {
        "ru": "Write the entire answer in Russian. Do not switch languages.",
        "kk": "Write the entire answer in Kazakh. Do not switch languages.",
        "en": "Write the entire answer in English. Do not switch languages.",
    }
    prompt += (
        f"\nANSWER LANGUAGE: {LANGUAGE_NAMES[language]}. "
        f"{language_rules[language]}\n"
    )
    if strict_lab_scope:
        refusal = _LAB_SCOPE_REFUSALS[language]
        prompt += (
            "\n--- CURRENT SUBJECT AND LABORATORY SCOPE ---\n"
            "Answer questions related to the current school subject or the current "
            "laboratory activity. A question about the current subject may cover "
            "any topic in that subject and does not need to be directly related to "
            "this exact activity. For procedure, equipment, safety, and live scene "
            "state, use only evidence from the current activity and scenario. Do "
            "not substantively answer questions about another school subject or "
            "unrelated matters. For an out-of-scope question, "
            f"return exactly this one sentence: {refusal}\n"
            "--- END SUBJECT AND LABORATORY SCOPE ---\n"
        )
    if lab_instruction and lab_instruction.strip():
        prompt += (
            "\n--- CURRENT LABORATORY INSTRUCTION ---\n"
            f"{lab_instruction.strip()}\n"
            "--- END LABORATORY INSTRUCTION ---\n"
            "This is the official procedure for the current activity. For steps, "
            "purpose, and action order, follow it exactly. It may be translated "
            "faithfully into the requested answer language.\n"
        )
    elif lab_incomplete:
        prompt += (
            "\nIMPORTANT: the exact procedure for the current laboratory activity "
            "is unavailable. Do not invent or substitute steps from another "
            "language or activity. Answer supported theory questions only, and say "
            "briefly that the exact instruction is unavailable when asked for the "
            "procedure.\n"
        )
    if knowledge_context and knowledge_context.strip():
        prompt += (
            "\n--- RETRIEVED KNOWLEDGE ---\n"
            f"{knowledge_context.strip()}\n"
            "--- END RETRIEVED KNOWLEDGE ---\n"
            "Ground theoretical answers in these excerpts. They may be in Russian, "
            "Kazakh, or English and may be translated faithfully.\n"
        )
    if scenario_context and scenario_context.strip():
        prompt += (
            "\n--- CURRENT STATIC SCENARIO ---\n"
            f"{scenario_context.strip()}\n"
            "--- END STATIC SCENARIO ---\n"
            "For questions about scene objects, steps, or purpose, use this "
            "description unless the live state below is more current.\n"
        )
    if scenario_state and scenario_state.strip():
        prompt += (
            "\n--- LIVE SCENE STATE FOR THIS REQUEST ---\n"
            f"{scenario_state.strip()}\n"
            "--- END LIVE SCENE STATE ---\n"
            "This state is authoritative for current and next steps, visible or "
            "held objects, allowed actions, and the last action result.\n"
        )
    if allow_general_knowledge:
        prompt += (
            "\n--- GENERAL SCIENCE FALLBACK MODE ---\n"
            "First decide whether the retrieved excerpts directly and sufficiently "
            "support the answer. If they do, answer only from them and begin with "
            f"{_GROUNDED_MARKER}. If they do not, use reliable, widely accepted "
            "scientific knowledge and begin with "
            f"{_GENERAL_KNOWLEDGE_MARKER}. Do not refuse merely because retrieved "
            "evidence is absent, thin, corrupt, or irrelevant. Do not mention the "
            "evidence gap to the user. Use exactly one private marker as the first "
            "characters, then give the normal answer in the requested language.\n"
            "--- END GENERAL SCIENCE FALLBACK MODE ---\n"
        )
    return prompt


# ── Local hybrid retrieval ────────────────────────────────────────────────────


def _detect_query_language(query: str) -> Optional[LanguageCode]:
    """Detect supported natural-language prose without guessing from identifiers.

    Latin formulas, URLs, filenames, acronyms, and mixed alphanumeric identifiers
    are removed before English scoring. Short ambiguous inputs therefore return
    ``None`` and are resolved by explicit language, lab language, or the configured
    default instead of being mislabeled as English.
    """
    text = query or ""
    if not text.strip():
        return None

    kazakh_score = len(_KAZAKH_CHAR_RE.findall(text)) * 3
    kazakh_score += len(_KAZAKH_WORD_RE.findall(text)) * 2
    russian_score = len(_RUSSIAN_WORD_RE.findall(text)) * 2
    cyrillic_words = re.findall(r"[Ѐ-ӿ]{2,}", text)
    if cyrillic_words:
        if kazakh_score:
            kazakh_score += len(cyrillic_words)
        else:
            russian_score += len(cyrillic_words)

    prose = _LATIN_NON_PROSE_RE.sub(" ", text)
    english_words = [word.casefold() for word in _ENGLISH_WORD_RE.findall(prose)]
    english_signals = len(_ENGLISH_SIGNAL_RE.findall(prose))
    english_score = english_signals * 2
    if len(english_words) >= 4 and english_signals:
        english_score += len(english_words)
    elif len(english_words) >= 2 and english_signals >= 2:
        english_score += len(english_words)
    elif len(english_words) == 1 and english_words[0] in _ENGLISH_SINGLE_WORD_SIGNALS:
        english_score += 3

    scores: dict[LanguageCode, int] = {
        "ru": russian_score,
        "kk": kazakh_score,
        "en": english_score,
    }
    best_language, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return None
    if sum(score == best_score for score in scores.values()) > 1:
        return None
    return best_language


def resolve_answer_language(
    query: str,
    explicit_language: Optional[LanguageCode] = None,
    lab_language: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> LanguageCode:
    """Resolve language using request, query, recent chat, lab, then default."""
    if explicit_language is not None:
        return normalize_language_code(explicit_language)
    detected = _detect_query_language(query)
    if detected is not None:
        return detected
    recent_messages = build_input_messages(history or [])
    if recent_messages and recent_messages[-1].get("role") == "user":
        latest = recent_messages[-1].get("content", "").strip()
        if latest == query.strip():
            recent_messages = recent_messages[:-1]
    for message in reversed(recent_messages):
        detected = _detect_query_language(message.get("content", ""))
        if detected is not None:
            return detected
    if lab_language and is_language_code(lab_language):
        return lab_language
    return normalize_language_code(settings.DEFAULT_LANGUAGE, field="DEFAULT_LANGUAGE")


def _infer_query_language(
    query: str,
    explicit_language: Optional[LanguageCode] = None,
    lab_language: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> LanguageCode:
    """Backward-compatible alias for the canonical answer-language resolver."""
    return resolve_answer_language(query, explicit_language, lab_language, history)


def _infer_query_subject(query: str) -> Optional[str]:
    """Infer a school subject only when the query contains a strong signal."""
    scores = {
        subject: len(pattern.findall(query or ""))
        for subject, pattern in _SUBJECT_QUERY_RES.items()
    }
    best_subject, best_score = max(scores.items(), key=lambda item: item[1])
    tied = sum(score == best_score for score in scores.values()) > 1
    return best_subject if best_score > 0 and not tied else None


def _infer_query_context(
    query: str,
    explicit_language: Optional[LanguageCode] = None,
    lab_language: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> tuple[Optional[str], LanguageCode]:
    """Return ``(subject, language)`` inferred from a standalone question."""
    return (
        _infer_query_subject(query),
        resolve_answer_language(query, explicit_language, lab_language, history),
    )


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


def _citations_from_chunks(
    chunks: list[dict], answer_language: LanguageCode = "ru"
) -> list[dict]:
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
        citation["display_label"] = _citation_display_label(
            citation, answer_language=answer_language
        )
        citations.append(citation)
    return citations


def _citation_display_label(
    citation: dict, answer_language: LanguageCode = "ru"
) -> str:
    """Build a concise display label localized to the answer language."""
    if citation.get("source_type") == "lab_instruction":
        number = citation.get("lab_number")
        if answer_language == "en":
            label = "Lab instruction"
            if number is not None:
                label += f" No. {number}"
        else:
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
        if answer_language == "en":
            page_prefix = "pp." if page_end is not None and page_end != page_start else "p."
            label += f", {page_prefix} {page_label}"
        else:
            label += f", стр. {page_label}"
    return label


def _is_lab_procedure_query(query: str) -> bool:
    """Whether RU/KK/EN wording clearly asks for current-lab procedure details."""
    normalized = _WS_RE.sub(" ", query or "").strip()
    return bool(normalized and _LAB_PROCEDURE_QUERY_RE.search(normalized))


def _lab_scope_terms(text: Optional[str]) -> set[str]:
    """Return small morphology-tolerant topic keys for RU/KK/EN lab text."""
    terms: set[str] = set()
    for raw_word in _WORD_RE.findall((text or "").casefold().replace("ё", "е")):
        if len(raw_word) < 4:
            continue
        if any(raw_word.startswith(prefix) for prefix in _LAB_SCOPE_GENERIC_PREFIXES):
            continue
        # Three Cyrillic characters cover common RU/KK inflection changes such as
        # вода/воды and кипит/кипение. Latin terms use four characters to avoid
        # excessively broad matches.
        is_cyrillic = any(
            "а" <= char <= "я" or char in "әғқңөұүһі" for char in raw_word
        )
        width = 3 if is_cyrillic else 4
        terms.add(raw_word[:width])
    return terms


def _lab_scope_refusal(
    query: str,
    lab: Optional[dict],
    scope_text: str,
    theory_chunks: list[dict],
    answer_language: LanguageCode,
) -> Optional[str]:
    """Reject questions unrelated to both the current subject and current lab.

    The gate is deliberately conservative: procedure questions and ambiguous
    short follow-ups remain answerable. Any clearly identified question from the
    current subject is accepted, even when it is not about this exact lab. A
    question is also accepted when it overlaps the authoritative lab text or a
    retrieved textbook chunk from the subject-scoped search. Explicit questions
    about a different school subject are always rejected.
    """
    if not lab or _is_lab_procedure_query(query):
        return None

    lab_subject = lab.get("subject")
    query_subject = _infer_query_subject(query)
    if query_subject and lab_subject:
        if query_subject != lab_subject:
            return _LAB_SCOPE_REFUSALS.get(
                answer_language, _LAB_SCOPE_REFUSALS["ru"]
            )
        return None

    scope_terms = _lab_scope_terms(scope_text)
    query_terms = _lab_scope_terms(query)
    if not scope_terms or not query_terms:
        return None
    if scope_terms & query_terms:
        return None

    for chunk in theory_chunks:
        chunk_text = (chunk.get("payload") or {}).get("text") or ""
        chunk_terms = _lab_scope_terms(chunk_text)
        if query_terms & chunk_terms:
            return None

    return _LAB_SCOPE_REFUSALS.get(answer_language, _LAB_SCOPE_REFUSALS["ru"])


def _answer_citations(
    query: str,
    theory_chunks: list[dict],
    lab_source_chunks: list[dict],
    answer_language: LanguageCode = "ru",
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
    return _citations_from_chunks(ordered_chunks, answer_language=answer_language)


def _general_fallback_allowed(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab_instruction: Optional[str],
    lab_incomplete: bool,
    lab_active: bool = False,
) -> bool:
    """Whether this request may fall back to reliable general science facts."""
    if not settings.ALLOW_GENERAL_KNOWLEDGE_FALLBACK:
        return False
    # Structured lab mode is intentionally closed-world. Missing context must
    # produce a refusal, not a broad answer from the model's general knowledge.
    if lab_active:
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


def _is_missing_evidence_refusal(answer: str) -> bool:
    """Whether an answer refuses specifically because retrieved evidence is absent."""
    normalized = _WS_RE.sub(" ", answer or "").strip()
    return bool(normalized and _MISSING_EVIDENCE_REFUSAL_RE.search(normalized))


def _force_general_knowledge_prompt(system_prompt: str) -> str:
    """Override a mistaken grounded refusal for a fallback-eligible request."""
    return (
        system_prompt
        + "\n--- FORCED GENERAL SCIENCE ANSWER ---\n"
        "The previous attempt incorrectly refused because the documents did not "
        "contain the fact. Do not evaluate or mention the documents now. Answer "
        "using reliable, widely accepted scientific knowledge in the already "
        "specified answer language. Begin with the private marker "
        f"{_GENERAL_KNOWLEDGE_MARKER}.\n"
        "--- END FORCED GENERAL SCIENCE ANSWER ---\n"
    )


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
    language: LanguageCode = "ru"

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
    answer_language: LanguageCode = "ru",
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
        answer_language,
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
    answer_language: LanguageCode
    scope_refusal: Optional[str] = None


async def _prepare_answer_grounding(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab: Optional[dict],
    retrieval_query: Optional[str] = None,
    answer_language: Optional[LanguageCode] = None,
) -> _AnswerGrounding:
    """Build identical retrieval and fallback state for both generation paths."""
    effective_retrieval_query = retrieval_query or query
    resolved_answer_language = answer_language or resolve_answer_language(
        query, lab_language=(lab or {}).get("lang")
    )
    inferred_subject = _infer_query_subject(query)
    if not inferred_subject and effective_retrieval_query != query:
        inferred_subject = _infer_query_subject(effective_retrieval_query)
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
        retrieval_lang = resolved_answer_language if inferred_subject else None
    elif not retrieval_lang:
        retrieval_lang = resolved_answer_language

    retrieved = await _retrieve(
        effective_retrieval_query,
        query_filter=query_filter,
        lang=retrieval_lang,
        fallback_filter=fallback_filter,
    )
    theory_chunks = _usable_theory_chunks(retrieved)
    lab_active = bool(lab)
    scope_text = "\n".join(
        value.strip()
        for value in (scenario_context, scenario_state, lab_instruction)
        if value and value.strip()
    )
    scope_refusal = _lab_scope_refusal(
        query,
        lab,
        scope_text,
        theory_chunks,
        resolved_answer_language,
    )
    allow_general_knowledge = _general_fallback_allowed(
        query,
        scenario_context,
        scenario_state,
        lab_instruction,
        lab_incomplete,
        lab_active=lab_active,
    )
    system_prompt = build_system_prompt(
        scenario_context,
        scenario_state,
        knowledge_context=_format_knowledge(theory_chunks),
        lab_instruction=lab_instruction,
        lab_incomplete=lab_incomplete,
        answer_language=resolved_answer_language,
        allow_general_knowledge=allow_general_knowledge,
        strict_lab_scope=lab_active,
    )
    return _AnswerGrounding(
        system_prompt=system_prompt,
        theory_chunks=theory_chunks,
        lab_source_chunks=lab_source_chunks,
        allow_general_knowledge=allow_general_knowledge,
        answer_language=resolved_answer_language,
        scope_refusal=scope_refusal,
    )


def _combined_usage(*responses: Any) -> dict:
    """Sum token usage across an initial completion and an optional retry."""
    usages = [_usage_dict(response) for response in responses]
    usages = [usage for usage in usages if usage]
    if not usages:
        return {}
    input_tokens = sum(usage.get("input_tokens", 0) for usage in usages)
    output_tokens = sum(usage.get("output_tokens", 0) for usage in usages)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


async def _complete_answer(
    query: str,
    grounding: _AnswerGrounding,
    input_messages: list[dict],
    max_tokens: Optional[int],
) -> AnswerResult:
    """Create a checked completion and retry grounded refusals as general science."""

    if grounding.scope_refusal:
        return AnswerResult(
            answer=grounding.scope_refusal,
            citations=[],
            usage={},
            language=grounding.answer_language,
        )

    async def _create(instructions: str):
        try:
            return await client.responses.create(
                model=settings.OPENAI_MODEL,
                instructions=instructions,
                input=input_messages,
                max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                **_tier_kwargs(),
            )
        except openai.APIError as exc:
            raise _map_openai_error(exc) from exc

    responses = [await _create(grounding.system_prompt)]
    answer, used_general_knowledge = _parse_answer_mode(
        _response_text(responses[-1]),
        allow_general_knowledge=grounding.allow_general_knowledge,
        default_general=not grounding.theory_chunks,
    )
    if (
        grounding.allow_general_knowledge
        and not used_general_knowledge
        and _is_missing_evidence_refusal(answer)
    ):
        responses.append(
            await _create(_force_general_knowledge_prompt(grounding.system_prompt))
        )
        answer, used_general_knowledge = _parse_answer_mode(
            _response_text(responses[-1]),
            allow_general_knowledge=True,
            default_general=True,
        )

    if not answer:
        raise LLMMalformedResponseError("OpenAI returned an empty answer")

    return AnswerResult(
        answer=answer,
        citations=(
            []
            if used_general_knowledge
            else _answer_citations(
                query,
                grounding.theory_chunks,
                grounding.lab_source_chunks,
                grounding.answer_language,
            )
        ),
        usage=_combined_usage(*responses),
        language=grounding.answer_language,
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
    answer_language: Optional[LanguageCode] = None,
) -> AnswerResult:
    """Generate a grounded answer (non-streaming).

    ``lab`` is the structured lab context from the simulator (subject/grade/
    lang/lab_number + composed ``lab_id``); it scopes retrieval to the subject
    and injects the lab's procedure verbatim. Retries up to 3 times on
    transient timeout / upstream (5xx) errors. Single-turn requests are served
    from the in-process answer cache when an identical question (same scenario/
    state/lab) was answered within ``ANSWER_CACHE_TTL_S``.
    """
    resolved_answer_language = resolve_answer_language(
        query,
        explicit_language=answer_language,
        lab_language=(lab or {}).get("lang"),
        history=chat_history,
    )
    cache_key = (
        _answer_cache_key(
            query,
            scenario_context,
            scenario_state,
            lab,
            max_tokens,
            resolved_answer_language,
        )
        if chat_history is None or len(chat_history) <= 1
        else None
    )
    if cache_key is not None:
        cached = _answer_cache.get(cache_key)
        if cached is not None:
            _log_generation("answer_cached", query, cached)
            return cached

    raw_history = (
        chat_history
        if chat_history is not None
        else [{"role": "user", "content": query}]
    )
    history = trim_history(
        raw_history,
        max_messages=settings.CHAT_MEMORY_MAX_MESSAGES,
        max_chars=settings.CHAT_MEMORY_HISTORY_CHARS,
    )
    retrieval_query = build_retrieval_query(
        query,
        history,
        max_context_chars=settings.CHAT_MEMORY_RETRIEVAL_CONTEXT_CHARS,
    )
    grounding = await _prepare_answer_grounding(
        query,
        scenario_context,
        scenario_state,
        lab,
        retrieval_query=retrieval_query,
        answer_language=resolved_answer_language,
    )
    input_messages = build_input_messages(history)

    result = await _complete_answer(query, grounding, input_messages, max_tokens)
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
    answer_language: Optional[LanguageCode] = None,
) -> AsyncIterator[dict]:
    """Stream a grounded answer as a sequence of events:

      {"type": "delta", "text": "..."}        # incremental answer text
      {"type": "done", "citations": [...], "usage": {...}}
      {"type": "error", "message": "..."}     # on upstream failure

    The route layer turns these into SSE frames. ``lab`` carries the same
    structured lab context as :func:`generate_answer`. Shares the answer cache
    with :func:`generate_answer` — a hit yields the whole answer as one delta.
    """
    resolved_answer_language = resolve_answer_language(
        query,
        explicit_language=answer_language,
        lab_language=(lab or {}).get("lang"),
        history=chat_history,
    )
    cache_key = (
        _answer_cache_key(
            query,
            scenario_context,
            scenario_state,
            lab,
            max_tokens,
            resolved_answer_language,
        )
        if chat_history is None or len(chat_history) <= 1
        else None
    )
    if cache_key is not None:
        cached = _answer_cache.get(cache_key)
        if cached is not None:
            _log_generation("answer_stream_cached", query, cached)
            yield {"type": "delta", "text": cached.answer}
            yield {
                "type": "done",
                "citations": cached.citations,
                "usage": cached.usage,
                "language": cached.language,
            }
            return

    raw_history = (
        chat_history
        if chat_history is not None
        else [{"role": "user", "content": query}]
    )
    history = trim_history(
        raw_history,
        max_messages=settings.CHAT_MEMORY_MAX_MESSAGES,
        max_chars=settings.CHAT_MEMORY_HISTORY_CHARS,
    )
    retrieval_query = build_retrieval_query(
        query,
        history,
        max_context_chars=settings.CHAT_MEMORY_RETRIEVAL_CONTEXT_CHARS,
    )
    grounding = await _prepare_answer_grounding(
        query,
        scenario_context,
        scenario_state,
        lab,
        retrieval_query=retrieval_query,
        answer_language=resolved_answer_language,
    )
    input_messages = build_input_messages(history)

    if grounding.scope_refusal or grounding.allow_general_knowledge:
        try:
            result = await _complete_answer(
                query, grounding, input_messages, max_tokens
            )
        except LLMError as exc:
            logger.error("Fallback completion error: %s", exc, exc_info=True)
            yield {"type": "error", "message": str(exc)}
            return
        _log_generation("answer_stream_checked", query, result)
        if cache_key is not None and result.answer:
            _answer_cache.put(cache_key, result)
        yield {"type": "delta", "text": result.answer}
        yield {
            "type": "done",
            "citations": result.citations,
            "usage": result.usage,
            "language": result.language,
        }
        return

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
                        yield {"type": "delta", "text": delta}
            final = await stream.get_final_response()
    except openai.APIError as exc:
        mapped = _map_openai_error(exc)
        logger.error("Streaming LLM error: %s", mapped, exc_info=True)
        yield {"type": "error", "message": str(mapped)}
        return

    result = AnswerResult(
        answer=_response_text(final),
        citations=_answer_citations(
            query,
            grounding.theory_chunks,
            grounding.lab_source_chunks,
            grounding.answer_language,
        ),
        usage=_usage_dict(final),
        language=grounding.answer_language,
    )
    _log_generation("answer_stream", query, result)
    if cache_key is not None and result.answer:
        _answer_cache.put(cache_key, result)
    yield {
        "type": "done",
        "citations": result.citations,
        "usage": result.usage,
        "language": result.language,
    }


# ── Hint rephrasing (Task 2 of the spec) ─────────────────────────────────────

_HINT_SYSTEM_PROMPT = (
    "You rephrase a simulator-provided hint for a student in a school VR lab. "
    "The simulator already decided when to show it and what it means.\n\n"
    "Rules:\n"
    "- Preserve the hint's meaning exactly.\n"
    "- Use supplied scene context only to make the wording natural.\n"
    "- Level 1: one short sentence.\n"
    "- Level 2: one or two concrete sentences that may mention scene objects.\n"
    "- Level 3: two or three sentences with the provided action expressed clearly.\n"
    "- Do not add facts or actions absent from the hint or scenario.\n"
    "- Return only the rephrased hint, with no explanation or quotation marks.\n"
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
    answer_language: Optional[LanguageCode] = None,
) -> str:
    """Rephrase a simulator-provided hint at the given verbosity level.

    No file search — the hint and scenario context are all the model needs.
    """
    language = resolve_answer_language(hint_text, explicit_language=answer_language)
    instructions = (
        _HINT_SYSTEM_PROMPT
        + f"\nOUTPUT LANGUAGE: {LANGUAGE_NAMES[language]}. Preserve the input "
        "meaning and write the entire result in this language.\n"
    )
    if scenario_context and scenario_context.strip():
        instructions += (
            "\n--- SCENARIO ---\n"
            f"{scenario_context.strip()}\n"
            "--- END SCENARIO ---\n"
        )
    if scenario_state and scenario_state.strip():
        instructions += (
            "\n--- LIVE SCENE STATE ---\n"
            f"{scenario_state.strip()}\n"
            "--- END LIVE SCENE STATE ---\n"
        )
    user_msg = f"Hint level: {hint_level}\nHint text: {hint_text}"

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
