"""Request-scoped chat memory helpers (no persistence).

The Responses API takes the conversation via ``input`` (a list of role/content
messages) and the system prompt via ``instructions``. These helpers trim the
client-supplied history so a long VR session can't blow the context budget,
and strip any system messages the client sent (we build the system prompt
ourselves).
"""

from typing import Optional


def strip_system_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]


def trim_history(messages: list[dict], max_messages: int, max_chars: int) -> list[dict]:
    """Keep the most recent turns within both a message-count and char budget.

    Always preserves at least the final (latest user) message.
    """
    clean = strip_system_messages(messages)
    trimmed = list(clean[-max_messages:]) if len(clean) > max_messages else list(clean)
    while len(trimmed) > 1 and sum(len(m.get("content", "")) for m in trimmed) > max_chars:
        trimmed.pop(0)
    return trimmed


def build_input_messages(history: list[dict]) -> list[dict]:
    """Normalise trimmed history into Responses API ``input`` messages."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in strip_system_messages(history)
        if m.get("content")
    ]


def latest_user_message(messages: list[dict]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None
