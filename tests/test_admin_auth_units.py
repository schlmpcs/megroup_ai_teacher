import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings


def test_internal_key_cannot_access_admin_routes(client, auth):
    response = client.get("/admin/corpus_status", headers=auth)
    assert response.status_code == 403


def test_admin_key_can_access_admin_routes(client, admin_auth, monkeypatch):
    async def fake_status():
        return {"status": "ready", "documents": 0, "points": 0}

    import app.api.admin_routes as admin_routes

    monkeypatch.setattr(admin_routes.ingestion, "corpus_status", fake_status)
    response = client.get("/admin/corpus_status", headers=admin_auth)
    assert response.status_code == 200


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
