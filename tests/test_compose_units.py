import re
from pathlib import Path


def test_admin_ui_receives_only_explicit_required_environment():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    match = re.search(r"^  admin-ui:\n(?P<body>.*?)(?=^  \S)", compose, re.M | re.S)
    assert match
    service = match.group("body")
    environment = set(re.findall(r"^      - ([A-Z][A-Z0-9_]*)=", service, re.M))

    assert "env_file:" not in service
    assert environment == {
        "ADMIN_UI_USERNAME",
        "ADMIN_UI_PASSWORD_HASH",
        "ADMIN_UI_SESSION_SECRET",
        "ADMIN_UI_SESSION_TTL_S",
        "ADMIN_UI_COOKIE_SECURE",
        "BACKEND_BASE_URL",
        "BACKEND_ADMIN_API_KEY",
    }
    assert "INTERNAL_API_KEY=" not in service
    assert "OPENAI_API_KEY=" not in service


def test_gateway_owns_public_api_port_and_routes_admin_shell():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    api = re.search(r"^  api:\n(?P<body>.*?)(?=^  \S)", compose, re.M | re.S)
    gateway = re.search(r"^  gateway:\n(?P<body>.*?)(?=^  \S)", compose, re.M | re.S)
    config = Path("gateway/nginx.conf").read_text(encoding="utf-8")

    assert api and '"8001:8000"' not in api.group("body")
    assert gateway and '"8001:8080"' in gateway.group("body")
    assert "http://127.0.0.1:8080/health" in gateway.group("body")
    assert "http://127.0.0.1:8080/" in gateway.group("body")
    for route in ["location = /", "location /static/", "location /auth/", "location = /api/session", "location /api/admin/"]:
        assert route in config
    assert "proxy_pass http://admin-ui:8000" in config
    assert "location / {" in config
    assert "proxy_pass http://api:8000" in config
    assert "proxy_buffering off" in config
    assert "proxy_request_buffering off" in config
