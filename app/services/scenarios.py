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

logger = logging.getLogger("assistant.scenarios")

# Fields rendered into the system prompt, in display order. Mirrors the table
# in the spec (ts_scenarios.md). Missing fields are simply skipped.
_FIELD_LABELS: list[tuple[str, str]] = [
    ("scenario_name", "Сценарий"),
    ("subject", "Предмет"),
    ("topic", "Тема"),
    ("lab_number", "Номер лабораторной работы"),
    ("environment_description", "Виртуальное окружение"),
    ("objects", "Ключевые объекты"),
    ("action_sequence", "Последовательность действий"),
    ("current_step_hint", "Подсказка по текущему шагу"),
    ("risks", "Возможные риски"),
    ("common_mistakes", "Типовые ошибки"),
    ("correct_behavior", "Правильная логика поведения"),
    ("regulations", "Связанные материалы и инструкции"),
]


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
    """List available scenarios as {scenario_id, scenario_name, subject}."""
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


def format_scenario_context(doc: dict[str, Any]) -> str:
    """Render a scenario document into the labelled block injected into the
    system prompt. Only non-empty known fields are included."""
    lines: list[str] = []
    for key, label in _FIELD_LABELS:
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
) -> str:
    """Render the LIVE per-request scene state (ТЗ §3.2) into a labelled block.

    Unlike the static scenario document, this reflects what the simulator
    reports *right now*: which step the user is performing and which objects
    they are currently holding. Returns "" when no live state was supplied.
    """
    lines: list[str] = []
    if current_step and current_step.strip():
        lines.append(f"Текущий шаг ученика: {current_step.strip()}")
    items = [str(i).strip() for i in (held_items or []) if str(i).strip()]
    if items:
        lines.append("Предметы в руках у ученика: " + ", ".join(items))
    return "\n".join(lines)


def get_scenario_context(scenario_id: Optional[str]) -> Optional[str]:
    """Convenience: load + format a scenario, or None if no id was supplied.

    Raises ScenarioNotFoundError / ValueError on a bad id so the caller can map
    it to an HTTP 404 / 400.
    """
    if not scenario_id:
        return None
    return format_scenario_context(load_scenario(scenario_id))
