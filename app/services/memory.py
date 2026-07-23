"""Bounded chat-history helpers and ephemeral server-side conversations.

The Responses API takes the conversation via ``input`` (a list of role/content
messages) and the system prompt via ``instructions``. These helpers trim the
client-supplied history so a long VR session can't blow the context budget,
and strip any system messages the client sent (we build the system prompt
ourselves).

``ConversationMemory`` adds optional follow-up support for the VR ``/ask`` and
``/voice_ask`` endpoints. It is deliberately in-process and TTL-bounded, like
the answer and TTS caches: a restart/deploy clears it, and no conversation data
is written to disk.
"""

from typing import Optional

from app.core.config import settings
from app.services.assistant_profiles import DEFAULT_ASSISTANT_TYPE
from app.services.ttl_cache import TTLCache


def strip_system_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]


def trim_history(messages: list[dict], max_messages: int, max_chars: int) -> list[dict]:
    """Keep the most recent turns within both a message-count and char budget.

    Always preserves at least the final (latest user) message.
    """
    clean = strip_system_messages(messages)
    trimmed = list(clean[-max_messages:]) if len(clean) > max_messages else list(clean)
    while (
        len(trimmed) > 1
        and sum(len(m.get("content", "")) for m in trimmed) > max_chars
    ):
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


def build_retrieval_query(
    query: str,
    history: Optional[list[dict]],
    max_context_chars: int,
) -> str:
    """Add recent dialogue text to an ambiguous follow-up retrieval query.

    The original query remains the final line. Single-turn requests are left
    unchanged, while follow-ups such as "why?" gain enough subject vocabulary
    from the preceding exchange for hybrid retrieval to find relevant chunks.
    """
    if not history or max_context_chars <= 0:
        return query

    clean = build_input_messages(history)
    if clean and clean[-1].get("role") == "user":
        latest = clean[-1].get("content", "").strip()
        if latest == query.strip():
            clean = clean[:-1]
    if not clean:
        return query

    selected: list[str] = []
    used = 0
    for message in reversed(clean):
        content = message.get("content", "").strip()
        if not content:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        selected.append(content[-remaining:])
        used += min(len(content), remaining)

    selected.reverse()
    return "\n".join([*selected, query]) if selected else query


class ConversationMemory:
    """TTL/LRU conversation store used by the VR text and voice endpoints."""

    def __init__(
        self,
        max_conversations: int,
        ttl_s: float,
        max_messages: int,
        max_chars: int,
    ):
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._cache = TTLCache(max_conversations, ttl_s)

    @property
    def enabled(self) -> bool:
        return self._cache.enabled

    def history_for(
        self,
        conversation_id: str,
        query: str,
        *,
        namespace: str = DEFAULT_ASSISTANT_TYPE,
    ) -> list[dict]:
        """Return stored history plus the current user turn, already trimmed."""
        stored = self._cache.get((namespace, conversation_id)) or []
        return trim_history(
            [*stored, {"role": "user", "content": query}],
            max_messages=self.max_messages,
            max_chars=self.max_chars,
        )

    def remember(
        self,
        conversation_id: str,
        history: list[dict],
        answer: str,
        *,
        namespace: str = DEFAULT_ASSISTANT_TYPE,
    ) -> None:
        """Commit a successful assistant answer to a conversation."""
        if not answer or not self.enabled:
            return
        updated = trim_history(
            [*history, {"role": "assistant", "content": answer}],
            max_messages=self.max_messages,
            max_chars=self.max_chars,
        )
        self._cache.put((namespace, conversation_id), updated)

    def clear(
        self,
        conversation_id: str,
        *,
        namespace: str = DEFAULT_ASSISTANT_TYPE,
    ) -> bool:
        return self._cache.delete((namespace, conversation_id))

    def clear_all(self) -> None:
        self._cache.clear()


conversation_memory = ConversationMemory(
    max_conversations=settings.CHAT_MEMORY_MAX_CONVERSATIONS,
    ttl_s=settings.CHAT_MEMORY_TTL_S,
    max_messages=settings.CHAT_MEMORY_MAX_MESSAGES,
    max_chars=settings.CHAT_MEMORY_HISTORY_CHARS,
)
