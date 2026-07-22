"""Per-lab scenario context (Task 2 of the spec — "Контекст ПО").

Each VR lab/scenario is described by a small JSON document. The simulator sends
a ``scenario_id`` with every request; we load that document and inject it into
the system prompt so the assistant "knows" the current scene: what step the
user is on, which objects exist, the correct action sequence, etc.

Scenario documents are tiny and need exact grounding, so they go straight into
the prompt rather than into the (fuzzy, chunked) vector store. Subject theory
— the big textbook PDFs/EPUBs — lives in the local Qdrant collection instead,
and the per-lab procedure text is injected verbatim by ``llm`` (fetched from
Qdrant by ``lab_id``). This JSON describes scene logic (objects, risks, common
mistakes) that complements that procedure.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Optional

from app.core.config import settings
from app.core.languages import LanguageCode, is_language_code

logger = logging.getLogger("assistant.scenarios")

# Fields rendered into the system prompt, in display order. Mirrors the table
# in the spec (ts_scenarios.md). Missing fields are simply skipped.
_FIELD_KEYS = (
    "scenario_name",
    "language",
    "subject",
    "topic",
    "lab_number",
    "environment_description",
    "objects",
    "action_sequence",
    "current_step_hint",
    "risks",
    "common_mistakes",
    "correct_behavior",
    "regulations",
)
_FIELD_LABELS: dict[str, dict[str, str]] = {
    "ru": {
        "scenario_name": "Сценарий",
        "language": "Язык",
        "subject": "Предмет",
        "topic": "Тема",
        "lab_number": "Номер лабораторной работы",
        "environment_description": "Виртуальное окружение",
        "objects": "Ключевые объекты",
        "action_sequence": "Последовательность действий",
        "current_step_hint": "Подсказка по текущему шагу",
        "risks": "Возможные риски",
        "common_mistakes": "Типовые ошибки",
        "correct_behavior": "Правильная логика поведения",
        "regulations": "Связанные материалы и инструкции",
    },
    "en": {
        "scenario_name": "Scenario",
        "language": "Language",
        "subject": "Subject",
        "topic": "Topic",
        "lab_number": "Laboratory activity number",
        "environment_description": "Virtual environment",
        "objects": "Key objects",
        "action_sequence": "Action sequence",
        "current_step_hint": "Current step hint",
        "risks": "Risks",
        "common_mistakes": "Common mistakes",
        "correct_behavior": "Correct behavior",
        "regulations": "Related materials and instructions",
    },
}


class ScenarioNotFoundError(Exception):
    """Raised when a requested scenario_id has no document on disk."""


def _scenario_path(scenario_id: str) -> str:
    # Guard against path traversal: scenario_id must be a bare filename stem.
    safe = os.path.basename(scenario_id)
    return os.path.join(settings.SCENARIOS_DIR, f"{safe}.json")


@lru_cache(maxsize=256)
def _load_cached(scenario_id: str, mtime: float) -> dict[str, Any]:
    """Load + parse a scenario file. Keyed on mtime so edits bust the cache."""
    path = _scenario_path(scenario_id)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_scenario(scenario_id: str) -> dict[str, Any]:
    """Return the scenario document for ``scenario_id``.

    Raises ScenarioNotFoundError if the file does not exist, ValueError if it
    is not valid JSON.
    """
    path = _scenario_path(scenario_id)
    if not os.path.isfile(path):
        raise ScenarioNotFoundError(f"Unknown scenario_id '{scenario_id}'")
    try:
        return _load_cached(scenario_id, os.path.getmtime(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scenario '{scenario_id}' is not valid JSON: {exc}") from exc


def list_scenarios() -> list[dict[str, Any]]:
    """List available scenarios with additive declared-language metadata."""
    directory = settings.SCENARIOS_DIR
    if not os.path.isdir(directory):
        return []
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        scenario_id = name[: -len(".json")]
        try:
            doc = load_scenario(scenario_id)
        except (ValueError, ScenarioNotFoundError):
            logger.warning("Skipping unreadable scenario file: %s", name)
            continue
        out.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": doc.get("scenario_name"),
                "subject": doc.get("subject"),
                "language": doc.get("language"),
            }
        )
    return out


def _render_value(value: Any) -> str:
    """Render a scenario field value (str | list | dict) as readable text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"- {str(item).strip()}" for item in value if str(item).strip())
    if isinstance(value, dict):
        return "\n".join(f"- {k}: {v}" for k, v in value.items())
    return str(value)


def format_scenario_context(
    doc: dict[str, Any], language: Optional[LanguageCode] = None
) -> str:
    """Render a scenario document into the labelled block injected into the
    system prompt. Only non-empty known fields are included."""
    declared = doc.get("language")
    selected = language or (declared if is_language_code(declared) else "ru")
    labels = _FIELD_LABELS.get(selected, _FIELD_LABELS["en"])
    lines: list[str] = []
    for key in _FIELD_KEYS:
        label = labels.get(key, key)
        value = doc.get(key)
        if value in (None, "", [], {}):
            continue
        rendered = _render_value(value)
        if not rendered:
            continue
        if "\n" in rendered:
            lines.append(f"{label}:\n{rendered}")
        else:
            lines.append(f"{label}: {rendered}")
    return "\n".join(lines)


def format_scenario_state(
    current_step: Optional[str] = None,
    held_items: Optional[list[str]] = None,
    *,
    current_step_id: Optional[str] = None,
    current_step_index: Optional[int] = None,
    next_step_id: Optional[str] = None,
    next_step: Optional[str] = None,
    completed_steps: Optional[list[str]] = None,
    visible_items: Optional[list[str]] = None,
    allowed_actions: Optional[list[str]] = None,
    last_action: Optional[str] = None,
    last_action_result: Optional[str] = None,
    language: LanguageCode = "ru",
) -> str:
    """Render the LIVE per-request scene state (ТЗ §3.2) into a labelled block.

    Unlike the static scenario document, this is an authoritative snapshot of
    what the simulator reports *right now*. Explicit empty lists render as
    ``нет`` so the model can distinguish "none" from an omitted/unknown field.
    Returns "" when no usable live state was supplied.
    """
    labels = {
        "ru": {
            "current_step_id": "ID текущего шага",
            "current_step_index": "Индекс текущего шага",
            "current_step": "Текущий шаг ученика",
            "next_step_id": "ID следующего шага",
            "next_step": "Следующий шаг, назначенный симулятором",
            "completed_steps": "Завершённые шаги",
            "held_items": "Предметы в руках у ученика",
            "visible_items": "Предметы, видимые ученику",
            "allowed_actions": "Разрешённые действия сейчас",
            "last_action": "Последнее действие ученика",
            "last_action_result": "Результат последнего действия",
            "none": "нет",
            "authority": (
                "Авторитетность: это актуальный снимок сцены от симулятора для "
                "текущего запроса. При расхождении со статическим описанием "
                "сценария эти данные имеют приоритет."
            ),
        },
        "en": {
            "current_step_id": "current_step_id",
            "current_step_index": "current_step_index",
            "current_step": "current_step",
            "next_step_id": "next_step_id",
            "next_step": "next_step",
            "completed_steps": "completed_steps",
            "held_items": "held_items",
            "visible_items": "visible_items",
            "allowed_actions": "allowed_actions",
            "last_action": "last_action",
            "last_action_result": "last_action_result",
            "none": "none",
            "authority": (
                "Authority: this is the simulator's live state for the current "
                "request. It takes priority over conflicting static scenario data."
            ),
        },
    }.get(language, {})
    if not labels:
        labels = {
            **_FIELD_LABELS["en"],
            "current_step_id": "current_step_id",
            "current_step_index": "current_step_index",
            "current_step": "current_step",
            "next_step_id": "next_step_id",
            "next_step": "next_step",
            "completed_steps": "completed_steps",
            "held_items": "held_items",
            "visible_items": "visible_items",
            "allowed_actions": "allowed_actions",
            "last_action": "last_action",
            "last_action_result": "last_action_result",
            "none": "none",
            "authority": "Authority: live simulator state for this request.",
        }
    lines: list[str] = []

    def add_text(label: str, value: Optional[str]) -> None:
        if value and value.strip():
            lines.append(f"{label}: {value.strip()}")

    def add_list(label: str, values: Optional[list[str]]) -> None:
        if values is None:
            return
        items = [str(item).strip() for item in values if str(item).strip()]
        lines.append(f"{label}: {', '.join(items) if items else labels['none']}")

    add_text(labels["current_step_id"], current_step_id)
    if current_step_index is not None:
        lines.append(f"{labels['current_step_index']}: {current_step_index}")
    if current_step and current_step.strip():
        lines.append(f"{labels['current_step']}: {current_step.strip()}")
    add_text(labels["next_step_id"], next_step_id)
    add_text(labels["next_step"], next_step)
    add_list(labels["completed_steps"], completed_steps)
    add_list(labels["held_items"], held_items)
    add_list(labels["visible_items"], visible_items)
    add_list(labels["allowed_actions"], allowed_actions)
    add_text(labels["last_action"], last_action)
    add_text(labels["last_action_result"], last_action_result)

    if not lines:
        return ""
    return "\n".join([labels["authority"], *lines])


def get_scenario_context(scenario_id: Optional[str]) -> Optional[str]:
    """Convenience: load + format a scenario, or None if no id was supplied.

    Raises ScenarioNotFoundError / ValueError on a bad id so the caller can map
    it to an HTTP 404 / 400.
    """
    if not scenario_id:
        return None
    return format_scenario_context(load_scenario(scenario_id))
