"""Test fixtures. Sets required env BEFORE the app is imported anywhere."""

import os

os.environ.setdefault("INTERNAL_API_KEY", "test-internal-key-1234567890")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("SCENARIOS_DIR", "./scenarios")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import _answer_cache
from app.services.memory import conversation_memory
from app.services.voice import _tts_cache

AUTH = {"Authorization": "Bearer test-internal-key-1234567890"}


@pytest.fixture(autouse=True)
def _clear_response_caches():
    """Process-global caches are isolated between tests."""
    _answer_cache._data.clear()
    _tts_cache._data.clear()
    conversation_memory.clear_all()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth():
    return dict(AUTH)
