import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from admin_ui.auth import hash_password
from admin_ui.config import AdminSettings, get_settings
from admin_ui.main import app, login_limiter


def _configure(monkeypatch):
    monkeypatch.setenv("ADMIN_UI_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_UI_PASSWORD_HASH", hash_password("secret"))
    monkeypatch.setenv("ADMIN_UI_SESSION_SECRET", "session-secret-1234567890-abcdef")
    monkeypatch.setenv("ADMIN_UI_COOKIE_SECURE", "false")
    monkeypatch.setenv("BACKEND_BASE_URL", "http://backend")
    monkeypatch.setenv("BACKEND_ADMIN_API_KEY", "backend-admin-key")
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "field",
    ["ADMIN_UI_SESSION_SECRET", "BACKEND_ADMIN_API_KEY"],
)
def test_settings_reject_known_secret_placeholder(field):
    values = {
        "ADMIN_UI_USERNAME": "admin",
        "ADMIN_UI_PASSWORD_HASH": hash_password("secret"),
        "ADMIN_UI_SESSION_SECRET": "session-secret-1234567890-abcdef",
        "BACKEND_ADMIN_API_KEY": "backend-admin-key",
    }
    values[field] = "generate-with-python-secrets-token-urlsafe-32"

    with pytest.raises(ValueError, match=field):
        AdminSettings(**values)


@pytest.fixture(autouse=True)
def reset_admin_state():
    get_settings.cache_clear()
    login_limiter.failures.clear()
    if hasattr(app.state, "backend_transport"):
        delattr(app.state, "backend_transport")
    yield
    get_settings.cache_clear()
    login_limiter.failures.clear()
    if hasattr(app.state, "backend_transport"):
        delattr(app.state, "backend_transport")


def test_login_sets_http_only_cookie_and_returns_csrf(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    assert response.status_code == 200
    assert response.json()["csrf_token"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]


def test_session_endpoints_and_admin_proxy_disable_caching(monkeypatch):
    _configure(monkeypatch)
    app.state.backend_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"status": "ready"})
    )
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        csrf = login.json()["csrf_token"]
        session = client.get("/api/session")
        proxy = client.get("/api/admin/corpus_status")
        logout = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})

    for response in (login, session, proxy, logout):
        assert response.headers["cache-control"] == "no-store"


def test_index_serves_operator_shell_without_backend_secrets(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "VR AI Assistant Admin" in response.text
    assert "backend-admin-key" not in response.text
    assert "BACKEND_BASE_URL" not in response.text


def test_static_javascript_never_uses_browser_storage(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "localStorage" not in response.text
    assert "sessionStorage" not in response.text


def test_static_javascript_general_identity_uses_relative_path(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "return `general:${row.relative_path}`;" in response.text
    assert "return `general:${row.filename}`;" not in response.text


def test_static_javascript_tracks_and_refreshes_selected_job(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "selectedJobId: null" in response.text
    assert "state.selectedJobId = job.id;" in response.text
    assert "await refreshSelectedJob();" in response.text
    assert "quietMissing" in response.text
    assert "clearJobDetails();" in response.text


def test_static_javascript_login_reset_and_logout_clear_state(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    show_login = re.search(r"function showLogin\(\) \{(?P<body>.*?)\n\}", response.text, re.S)
    assert show_login
    for reset in [
        'state.csrf = "";',
        "state.files = [];",
        "state.corpusPreview = null;",
        "state.jobs = [];",
        "state.documents = [];",
        "state.selectedJobId = null;",
        "clearJobDetails();",
        "renderServiceStatus(null, null);",
    ]:
        assert reset in show_login.group("body")
    assert "try {" in response.text
    assert "finally {" in response.text
    assert "showLogin();" in response.text


def test_static_javascript_resets_form_controls_and_requires_corpus_preview(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    for reset in [
        'state.activeSource = "upload";',
        '$("fileInput").value = "";',
        '$("folderInput").value = "";',
        '$("corpusSubtree").value = "";',
        '$("corpusOcr").checked = false;',
        '$("corpusPrune").checked = false;',
        '$("corpusPrune").disabled = false;',
        '$("queueCorpus").disabled = true;',
        '$("documentSearch").value = "";',
        '$("documentType").value = "";',
        '$("documentSubject").value = "";',
        '$("documentGrade").value = "";',
        '$("documentLanguage").value = "";',
        'selectSource("upload");',
        "if (!state.corpusPreview) {",
        "if (!state.corpusPreview) return;",
    ]:
        assert reset in response.text


def test_static_javascript_initializes_ocr_default_once_per_session(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "ocrDefaultInitialized: false" in response.text
    assert "if (state.ocrDefaultInitialized) return;" in response.text
    assert "if (!row.ocrOverridden) row.ocr = state.ocrDefault;" in response.text
    assert 'if (field === "ocr") row.ocrOverridden = true;' in response.text
    assert "ocr: state.ocrDefault" in response.text
    assert 'initializeOcrDefault(status.ocr_default);' in response.text


def test_static_javascript_polling_respects_hidden_app(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    assert 'if (!$("appView").hidden) {' in response.text
    assert "state.pollTimer = window.setTimeout(tick, hasActiveJobs() ? 2000 : 10000);" in response.text


def test_static_css_file_button_has_keyboard_focus_style(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert ".file-button:focus-within" in response.text
    assert "outline: 3px solid rgba(15, 118, 110, .22);" in response.text
    assert "outline-offset: 1px;" in response.text


def test_mutating_proxy_requires_csrf(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        response = client.post("/api/admin/ingestion/jobs/corpus", json={})
    assert login.status_code == 200
    assert response.status_code == 403


def test_proxy_injects_backend_key_and_strips_browser_authorization(monkeypatch):
    _configure(monkeypatch)
    captured = {}

    def handler(request: httpx.Request):
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"status": "ready"})

    app.state.backend_transport = httpx.MockTransport(handler)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        csrf = login.json()["csrf_token"]
        response = client.get(
            "/api/admin/corpus_status",
            headers={"Authorization": "Bearer browser-key", "X-CSRF-Token": csrf},
        )
    assert response.status_code == 200
    assert captured["authorization"] == "Bearer backend-admin-key"


def test_invalid_credentials_are_generic(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        response = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_sixth_failed_login_is_rate_limited(monkeypatch):
    _configure(monkeypatch)
    login_limiter.clear("testclient")
    with TestClient(app) as client:
        for _ in range(5):
            assert client.post("/auth/login", json={"username": "bad", "password": "bad"}).status_code == 401
        assert client.post("/auth/login", json={"username": "bad", "password": "bad"}).status_code == 429


def test_logout_requires_csrf_and_expires_cookie(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        csrf = login.json()["csrf_token"]
        assert client.post("/auth/logout").status_code == 403
        response = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200
    assert "admin_session=" in response.headers["set-cookie"]


def test_proxy_rejects_missing_session_and_disallowed_path(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        assert client.get("/api/admin/corpus_status").status_code == 401
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        assert login.status_code == 200
        assert client.get("/api/admin/not-allowed").status_code == 404


def test_session_is_invalidated_when_configured_username_changes(monkeypatch):
    _configure(monkeypatch)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        assert login.status_code == 200
        assert client.get("/api/session").status_code == 200
        monkeypatch.setenv("ADMIN_UI_USERNAME", "other-admin")
        get_settings.cache_clear()
        assert client.get("/api/session").status_code == 401


def test_proxy_forwards_body_query_and_maps_transport_failure(monkeypatch):
    _configure(monkeypatch)
    captured = {}

    async def handler(request: httpx.Request):
        query = request.url.query
        captured["query"] = query.decode() if isinstance(query, bytes) else str(query)
        captured["body"] = await request.aread()
        return httpx.Response(202, json={"ok": True})

    app.state.backend_transport = httpx.MockTransport(handler)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        csrf = login.json()["csrf_token"]
        response = client.post(
            "/api/admin/ingestion/jobs/corpus?source=ui",
            headers={"X-CSRF-Token": csrf},
            json={"subtree": "", "ocr": False, "prune": False},
        )
    assert response.status_code == 202
    assert captured["query"] == "source=ui"
    assert json.loads(captured["body"]) == {"subtree": "", "ocr": False, "prune": False}

    async def broken_handler(request: httpx.Request):
        raise httpx.ConnectError("offline", request=request)

    app.state.backend_transport = httpx.MockTransport(broken_handler)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "secret"})
        assert client.get("/api/admin/corpus_status").status_code == 502
