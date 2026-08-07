import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

import app.api.admin_routes as admin_routes
import app.api.routes as routes
from app.services import assistant_profiles, ingestion_jobs, ingestion_worker
from app.services.errors import LLMUpstreamError
from app.services.llm import AnswerResult
from scripts import manage_corpus


@pytest.fixture
def profile_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        assistant_profiles.settings,
        "INGESTION_DATA_DIR",
        str(tmp_path),
    )
    return tmp_path


def _insert_profile(
    profile_store: Path,
    *,
    assistant_type: str,
    system_prompt: str,
    qdrant_collection: str,
    corpus_root: str,
) -> None:
    assistant_profiles.initialize()
    with sqlite3.connect(profile_store / "assistant_profiles.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO assistant_profiles (
                assistant_type, system_prompt, qdrant_collection, corpus_root
            ) VALUES (?, ?, ?, ?)
            """,
            (assistant_type, system_prompt, qdrant_collection, corpus_root),
        )


def test_profile_store_initialization_is_idempotent(profile_store):
    assistant_profiles.initialize()
    assistant_profiles.initialize()

    database = profile_store / "assistant_profiles.sqlite3"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(assistant_profiles)")
        }

    assert columns == {
        "assistant_type",
        "system_prompt",
        "qdrant_collection",
        "corpus_root",
    }


async def test_dynamic_profile_persists_and_survives_new_connection(
    profile_store, monkeypatch
):
    collections = []

    async def ensure_collection(*, collection_name=None):
        collections.append(collection_name)

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )

    profile = await assistant_profiles.create_assistant_profile(
        "math_tutor",
        "  You are a mathematics tutor.  ",
    )

    expected_root = profile_store / "assistant_corpora" / "math_tutor"
    assert profile == assistant_profiles.AssistantProfile(
        assistant_type="math_tutor",
        system_prompt="You are a mathematics tutor.",
        qdrant_collection="math_tutor_kb",
        corpus_root=str(expected_root),
    )
    assert collections == ["math_tutor_kb"]
    assert expected_root.is_dir()

    with sqlite3.connect(
        profile_store / "assistant_profiles.sqlite3"
    ) as connection:
        row = connection.execute(
            "SELECT * FROM assistant_profiles WHERE assistant_type = ?",
            ("math_tutor",),
        ).fetchone()

    assert row == (
        "math_tutor",
        "You are a mathematics tutor.",
        "math_tutor_kb",
        str(expected_root),
    )
    assert assistant_profiles.get_assistant_profile("math_tutor") == profile


def test_get_profile_prefers_builtin_then_resolves_dynamic(profile_store):
    _insert_profile(
        profile_store,
        assistant_type=assistant_profiles.DEFAULT_ASSISTANT_TYPE,
        system_prompt="Untrusted replacement.",
        qdrant_collection="replacement_kb",
        corpus_root=str(profile_store / "replacement"),
    )
    _insert_profile(
        profile_store,
        assistant_type="math_tutor",
        system_prompt="You are a mathematics tutor.",
        qdrant_collection="math_tutor_kb",
        corpus_root=str(profile_store / "assistant_corpora" / "math_tutor"),
    )

    assert assistant_profiles.get_assistant_profile(
        assistant_profiles.DEFAULT_ASSISTANT_TYPE
    ) == assistant_profiles.ASSISTANT_PROFILES[
        assistant_profiles.DEFAULT_ASSISTANT_TYPE
    ]
    assert (
        assistant_profiles.get_assistant_profile("math_tutor").qdrant_collection
        == "math_tutor_kb"
    )


def test_available_types_merge_builtins_and_sorted_dynamic_profiles(profile_store):
    for assistant_type in ("zeta_tutor", "alpha_tutor"):
        _insert_profile(
            profile_store,
            assistant_type=assistant_type,
            system_prompt=f"Prompt for {assistant_type}.",
            qdrant_collection=f"{assistant_type}_kb",
            corpus_root=str(
                profile_store / "assistant_corpora" / assistant_type
            ),
        )

    assert assistant_profiles.available_assistant_types() == (
        assistant_profiles.DEFAULT_ASSISTANT_TYPE,
        "other_assistant",
        "alpha_tutor",
        "zeta_tutor",
    )


def test_malformed_dynamic_profile_fails_explicitly(profile_store):
    _insert_profile(
        profile_store,
        assistant_type="math_tutor",
        system_prompt="You are a mathematics tutor.",
        qdrant_collection="",
        corpus_root=str(profile_store / "assistant_corpora" / "math_tutor"),
    )

    with pytest.raises(ValueError, match="Malformed stored assistant profile"):
        assistant_profiles.get_assistant_profile("math_tutor")


async def test_create_rejects_builtin_and_dynamic_identifiers(
    profile_store, monkeypatch
):
    async def ensure_collection(*, collection_name=None):
        return None

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )

    with pytest.raises(assistant_profiles.AssistantProfileConflictError):
        await assistant_profiles.create_assistant_profile(
            assistant_profiles.DEFAULT_ASSISTANT_TYPE,
            "Replacement prompt.",
        )

    await assistant_profiles.create_assistant_profile(
        "math_tutor",
        "You are a mathematics tutor.",
    )
    with pytest.raises(assistant_profiles.AssistantProfileConflictError):
        await assistant_profiles.create_assistant_profile(
            "math_tutor",
            "Another prompt.",
        )


async def test_create_rejects_collection_and_corpus_root_collisions(
    profile_store, monkeypatch
):
    async def unexpected_ensure_collection(*, collection_name=None):
        raise AssertionError("Qdrant must not be called for a profile collision")

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        unexpected_ensure_collection,
    )
    _insert_profile(
        profile_store,
        assistant_type="legacy_collection",
        system_prompt="Legacy collection owner.",
        qdrant_collection="math_tutor_kb",
        corpus_root=str(profile_store / "assistant_corpora" / "legacy_collection"),
    )

    with pytest.raises(
        assistant_profiles.AssistantProfileConflictError,
        match="qdrant_collection",
    ):
        await assistant_profiles.create_assistant_profile(
            "math_tutor",
            "You are a mathematics tutor.",
        )

    _insert_profile(
        profile_store,
        assistant_type="legacy_root",
        system_prompt="Legacy corpus owner.",
        qdrant_collection="legacy_root_kb",
        corpus_root=str(profile_store / "assistant_corpora" / "science_tutor"),
    )

    with pytest.raises(
        assistant_profiles.AssistantProfileConflictError,
        match="corpus_root",
    ):
        await assistant_profiles.create_assistant_profile(
            "science_tutor",
            "You are a science tutor.",
        )


async def test_concurrent_create_has_one_winner(profile_store, monkeypatch):
    async def ensure_collection(*, collection_name=None):
        await asyncio.sleep(0)

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )

    results = await asyncio.gather(
        assistant_profiles.create_assistant_profile("math_tutor", "First prompt."),
        assistant_profiles.create_assistant_profile("math_tutor", "Second prompt."),
        return_exceptions=True,
    )

    assert sum(
        isinstance(result, assistant_profiles.AssistantProfile)
        for result in results
    ) == 1
    assert sum(
        isinstance(result, assistant_profiles.AssistantProfileConflictError)
        for result in results
    ) == 1


def test_create_profile_requires_admin_authentication(client, profile_store):
    response = client.post(
        "/admin/assistant-profiles",
        json={
            "assistant_type": "math_tutor",
            "system_prompt": "You are a mathematics tutor.",
        },
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {
            "assistant_type": "MathTutor",
            "system_prompt": "You are a mathematics tutor.",
        },
        {
            "assistant_type": " math_tutor",
            "system_prompt": "You are a mathematics tutor.",
        },
        {
            "assistant_type": "a" * 65,
            "system_prompt": "You are a mathematics tutor.",
        },
        {"assistant_type": "math_tutor", "system_prompt": "   "},
        {
            "assistant_type": "math_tutor",
            "system_prompt": "x" * (assistant_profiles.settings.MAX_INPUT_CHARS + 1),
        },
        {
            "assistant_type": "math_tutor",
            "system_prompt": "You are a mathematics tutor.",
            "qdrant_collection": "attacker_chosen",
        },
    ],
)
def test_create_profile_rejects_invalid_or_extra_fields(
    client, admin_auth, profile_store, payload
):
    response = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json=payload,
    )

    assert response.status_code == 422


def test_create_profile_rejects_builtin_and_dynamic_duplicates(
    client, admin_auth, profile_store, monkeypatch
):
    async def ensure_collection(*, collection_name=None):
        return None

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )

    built_in = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json={
            "assistant_type": assistant_profiles.DEFAULT_ASSISTANT_TYPE,
            "system_prompt": "Replacement prompt.",
        },
    )
    first = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json={
            "assistant_type": "math_tutor",
            "system_prompt": "You are a mathematics tutor.",
        },
    )
    duplicate = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json={
            "assistant_type": "math_tutor",
            "system_prompt": "Another prompt.",
        },
    )

    assert built_in.status_code == 409
    assert first.status_code == 201
    assert duplicate.status_code == 409


def test_create_profile_derives_configuration_and_persists(
    client, admin_auth, profile_store, monkeypatch
):
    collections = []

    async def ensure_collection(*, collection_name=None):
        collections.append(collection_name)

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )

    response = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json={
            "assistant_type": "math_tutor",
            "system_prompt": "  Вы математический наставник.  ",
        },
    )

    expected_root = profile_store / "assistant_corpora" / "math_tutor"
    assert response.status_code == 201
    assert response.json() == {
        "assistant_type": "math_tutor",
        "system_prompt": "Вы математический наставник.",
        "qdrant_collection": "math_tutor_kb",
        "corpus_root": str(expected_root),
        "built_in": False,
    }
    assert collections == ["math_tutor_kb"]
    assert expected_root.is_dir()
    assert assistant_profiles.get_assistant_profile("math_tutor").system_prompt == (
        "Вы математический наставник."
    )


def test_qdrant_failure_does_not_persist_profile(
    client, admin_auth, profile_store, monkeypatch
):
    async def fail_collection(*, collection_name=None):
        raise LLMUpstreamError("Qdrant unavailable")

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        fail_collection,
    )

    response = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json={
            "assistant_type": "math_tutor",
            "system_prompt": "You are a mathematics tutor.",
        },
    )

    assert response.status_code == 502
    with pytest.raises(ValueError, match="Unknown assistant_type"):
        assistant_profiles.get_assistant_profile("math_tutor")
    assert not (profile_store / "assistant_corpora" / "math_tutor").exists()


async def test_list_profiles_returns_builtins_then_sorted_dynamic(
    client, admin_auth, profile_store, monkeypatch
):
    async def ensure_collection(*, collection_name=None):
        return None

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )
    await assistant_profiles.create_assistant_profile("zeta_tutor", "Zeta prompt.")
    await assistant_profiles.create_assistant_profile("alpha_tutor", "Alpha prompt.")

    response = client.get("/admin/assistant-profiles", headers=admin_auth)

    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert [profile["assistant_type"] for profile in profiles] == [
        assistant_profiles.DEFAULT_ASSISTANT_TYPE,
        "other_assistant",
        "alpha_tutor",
        "zeta_tutor",
    ]
    assert [profile["built_in"] for profile in profiles] == [True, True, False, False]


def test_new_profile_is_accepted_by_ask(
    client, auth, admin_auth, profile_store, monkeypatch
):
    async def ensure_collection(*, collection_name=None):
        return None

    calls = []

    async def generate_answer(query, **kwargs):
        calls.append(kwargs["assistant_type"])
        return AnswerResult(answer="Answer", language="en")

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )
    monkeypatch.setattr(routes, "generate_answer", generate_answer)
    created = client.post(
        "/admin/assistant-profiles",
        headers=admin_auth,
        json={
            "assistant_type": "math_tutor",
            "system_prompt": "You are a mathematics tutor.",
        },
    )

    response = client.post(
        "/ask",
        headers=auth,
        json={"assistant_type": "math_tutor", "query": "Explain algebra."},
    )

    assert created.status_code == 201
    assert response.status_code == 200
    assert calls == ["math_tutor"]


async def test_admin_corpus_endpoint_accepts_dynamic_profile(
    client, admin_auth, profile_store, monkeypatch
):
    async def ensure_collection(*, collection_name=None):
        return None

    collections = []

    async def corpus_status(*, collection_name=None):
        collections.append(collection_name)
        return {"status": "ready", "documents": 0, "points": 0}

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )
    monkeypatch.setattr(admin_routes.ingestion, "corpus_status", corpus_status)
    await assistant_profiles.create_assistant_profile(
        "math_tutor",
        "You are a mathematics tutor.",
    )

    response = client.get(
        "/admin/corpus_status",
        headers=admin_auth,
        params={"assistant_type": "math_tutor"},
    )

    assert response.status_code == 200
    assert collections == ["math_tutor_kb"]


async def test_worker_resolves_new_profile_from_sqlite(
    profile_store, monkeypatch
):
    async def ensure_collection(*, collection_name=None):
        return None

    collections = []

    async def upload_document(filename, content, **kwargs):
        collections.append(kwargs["collection_name"])
        await kwargs["progress"]("indexing")
        return {
            "file_id": "doc-1",
            "filename": filename,
            "status": "ready",
            "chunks": 1,
        }

    async def invalidate_answer_cache():
        return True

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )
    monkeypatch.setattr(
        ingestion_worker.ingestion,
        "upload_document",
        upload_document,
    )
    monkeypatch.setattr(
        ingestion_worker,
        "invalidate_answer_cache",
        invalidate_answer_cache,
    )
    ingestion_jobs.initialize()
    await assistant_profiles.create_assistant_profile(
        "math_tutor",
        "You are a mathematics tutor.",
    )
    upload = profile_store / "uploads" / "job-1" / "item-1"
    upload.parent.mkdir(parents=True)
    upload.write_text("algebra", encoding="utf-8")
    ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "notes.md",
                "relative_path": "notes.md",
                "stored_path": "uploads/job-1/item-1",
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
        options={"assistant_type": "math_tutor"},
    )

    assert await ingestion_worker.run_once("worker-1") is True
    assert collections == ["math_tutor_kb"]
    assert ingestion_jobs.get_job("job-1")["status"] == "completed"


def test_manage_corpus_cli_resolves_dynamic_profile(profile_store, monkeypatch):
    async def ensure_collection(*, collection_name=None):
        return None

    collections = []

    async def status(collection_name):
        collections.append(collection_name)

    monkeypatch.setattr(
        assistant_profiles.vectorstore,
        "ensure_collection",
        ensure_collection,
    )
    asyncio.run(
        assistant_profiles.create_assistant_profile(
            "math_tutor",
            "You are a mathematics tutor.",
        )
    )
    monkeypatch.setattr(manage_corpus, "_status", status)
    monkeypatch.setattr(
        sys,
        "argv",
        ["manage_corpus", "status", "--assistant-type", "math_tutor"],
    )

    manage_corpus.main()

    assert collections == ["math_tutor_kb"]
