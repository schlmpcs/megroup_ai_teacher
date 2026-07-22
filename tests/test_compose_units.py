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
