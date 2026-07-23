import pytest
from pydantic import ValidationError

import app.api.admin_routes as admin_routes
from app.core.config import Settings, settings


def test_internal_key_cannot_access_admin_routes(client, auth):
    response = client.get("/admin/corpus_status", headers=auth)
    assert response.status_code == 403


def test_admin_key_can_access_admin_routes(client, admin_auth, monkeypatch):
    async def fake_status(**_):
        return {"status": "ready", "documents": 0, "points": 0}

    monkeypatch.setattr(admin_routes.ingestion, "corpus_status", fake_status)
    response = client.get("/admin/corpus_status", headers=admin_auth)
    assert response.status_code == 200


def test_admin_corpus_status_uses_selected_profile_collection(
    client, admin_auth, monkeypatch
):
    captured = {}

    async def fake_status(*, collection_name=None, **_):
        captured["collection_name"] = collection_name
        return {"status": "ready", "documents": 0, "points": 0}

    monkeypatch.setattr(admin_routes.ingestion, "corpus_status", fake_status)

    response = client.get(
        "/admin/corpus_status",
        headers=admin_auth,
        params={"assistant_type": "other_assistant"},
    )

    assert response.status_code == 200
    assert captured["collection_name"] == "other_assistant_kb"


def test_admin_corpus_status_rejects_unknown_assistant_type_before_qdrant(
    client, admin_auth, monkeypatch
):
    called = []

    async def fake_status(**_):
        called.append(True)
        return {"status": "ready"}

    monkeypatch.setattr(admin_routes.ingestion, "corpus_status", fake_status)

    response = client.get(
        "/admin/corpus_status",
        headers=admin_auth,
        params={"assistant_type": "missing"},
    )

    assert response.status_code == 422
    assert "Unknown assistant_type" in response.json()["detail"]
    assert called == []


def test_admin_list_and_delete_use_selected_profile_collection(
    client, admin_auth, monkeypatch
):
    captured = {}

    async def _list_documents(*, collection_name=None, **_):
        captured["list"] = collection_name
        return []

    async def _delete_document(file_id, *, collection_name=None, **_):
        captured["delete"] = (file_id, collection_name)
        return True

    monkeypatch.setattr(admin_routes.ingestion, "list_documents", _list_documents)
    monkeypatch.setattr(admin_routes.ingestion, "delete_document", _delete_document)

    listed = client.get(
        "/admin/documents",
        headers=admin_auth,
        params={"assistant_type": "other_assistant"},
    )
    deleted = client.delete(
        "/admin/documents/doc-1",
        headers=admin_auth,
        params={"assistant_type": "other_assistant"},
    )

    assert listed.status_code == deleted.status_code == 200
    assert captured == {
        "list": "other_assistant_kb",
        "delete": ("doc-1", "other_assistant_kb"),
    }


def test_admin_key_cannot_access_consumer_routes(client, admin_auth):
    response = client.post("/ask", json={"query": "hello"}, headers=admin_auth)
    assert response.status_code == 403


def test_admin_key_is_required_configuration():
    assert settings.ADMIN_API_KEY == "test-admin-key-1234567890"


def test_internal_and_admin_keys_cannot_match():
    shared = "same-strong-secret-value-1234567890"
    with pytest.raises(ValidationError):
        Settings(
            INTERNAL_API_KEY=shared,
            ADMIN_API_KEY=shared,
            OPENAI_API_KEY="test-openai-key-1234567890",
        )
