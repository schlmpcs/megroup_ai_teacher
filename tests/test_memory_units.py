import app.services.ttl_cache as ttl_cache_mod
from app.services.memory import ConversationMemory, build_retrieval_query


def _store(ttl_s=60):
    return ConversationMemory(
        max_conversations=2,
        ttl_s=ttl_s,
        max_messages=6,
        max_chars=1_000,
    )


def test_conversation_memory_builds_followup_history():
    store = _store()
    first = store.history_for("vr-session", "Что такое кипение?")
    store.remember(
        "vr-session", first, "Это переход жидкости в пар."
    )

    followup = store.history_for(
        "vr-session", "При какой температуре?"
    )

    assert followup == [
        {"role": "user", "content": "Что такое кипение?"},
        {"role": "assistant", "content": "Это переход жидкости в пар."},
        {"role": "user", "content": "При какой температуре?"},
    ]


def test_conversation_memory_expires(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(ttl_cache_mod.time, "monotonic", lambda: now[0])
    store = _store(ttl_s=10)
    history = store.history_for("vr-session", "Первый вопрос")
    store.remember("vr-session", history, "Первый ответ")

    now[0] = 110.0
    followup = store.history_for("vr-session", "Второй вопрос")

    assert followup == [{"role": "user", "content": "Второй вопрос"}]


def test_conversation_memory_clear():
    store = _store()
    history = store.history_for("vr-session", "Первый вопрос")
    store.remember("vr-session", history, "Первый ответ")

    assert store.clear("vr-session") is True
    assert store.clear("vr-session") is False
    assert store.history_for("vr-session", "Новый вопрос") == [
        {"role": "user", "content": "Новый вопрос"}
    ]


def test_build_retrieval_query_adds_previous_exchange_for_followup():
    history = [
        {"role": "user", "content": "Что такое кипение?"},
        {"role": "assistant", "content": "Это переход жидкости в пар."},
        {"role": "user", "content": "Почему оно начинается?"},
    ]

    retrieval_query = build_retrieval_query(
        "Почему оно начинается?", history, max_context_chars=500
    )

    assert "Что такое кипение?" in retrieval_query
    assert "переход жидкости в пар" in retrieval_query
    assert retrieval_query.endswith("Почему оно начинается?")


def test_build_retrieval_query_leaves_single_turn_unchanged():
    query = "Что такое диффузия?"
    assert (
        build_retrieval_query(
            query,
            [{"role": "user", "content": query}],
            max_context_chars=500,
        )
        == query
    )
