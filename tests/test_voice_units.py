"""Unit tests for the self-hosted STT/TTS sidecar client (app/services/voice).

The sidecar is mocked at the httpx boundary (``voice._http``); there is no live
GPU service in this environment.
"""

import httpx
import pytest

from app.core.config import settings
from app.services import voice
from app.services.errors import (
    LLMError,
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)


class _FakeResponse:
    def __init__(self, *, json_body=None, content=b"", status_code=200):
        self._json = json_body
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://voice/x")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("err", request=request, response=response)


class _FakeHTTP:
    """Minimal stand-in for httpx.AsyncClient capturing the last POST."""

    def __init__(self, response):
        self._response = response
        self.calls: list[tuple] = []

    async def post(self, url, files=None, data=None, params=None, json=None):
        self.calls.append(
            (url, {"files": files, "data": data, "params": params, "json": json})
        )
        return self._response


async def test_transcribe_posts_multipart_and_returns_text(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(json_body={"text": "  привет  ", "language": "ru"}))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    text = await voice.transcribe(b"RIFFfake", filename="q.wav", language="ru")

    assert text == "привет"
    url, kw = fake.calls[0]
    assert url == "/stt/recognize"
    assert kw["data"] == {"language": "ru"}
    assert kw["files"]["audio"][0] == "q.wav"
    assert kw["files"]["audio"][1] == b"RIFFfake"


async def test_transcribe_defaults_language(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(json_body={"text": "ok"}))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    await voice.transcribe(b"x")

    assert fake.calls[0][1]["data"] == {"language": settings.DEFAULT_LANGUAGE}


async def test_transcribe_missing_text_raises(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(json_body={"language": "ru"}))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    with pytest.raises(LLMMalformedResponseError):
        await voice.transcribe(b"x")


async def test_synthesize_posts_json_and_returns_wav(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(content=b"WAVDATA"))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    audio, media_type = await voice.synthesize("Привет", language="kk")

    assert audio == b"WAVDATA"
    assert media_type == "audio/wav"
    url, kw = fake.calls[0]
    assert url == "/tts/synthesize"
    assert kw["params"] == {"format": "wav"}
    assert kw["json"] == {
        "text": "Привет",
        "language": "kk",
        "speed": 1.0,
        "backend": "mms",
    }


async def test_synthesize_defaults_russian_to_supertonic(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(content=b"WAVDATA"))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    await voice.synthesize("Здравствуйте", language="ru")

    assert fake.calls[0][1]["json"] == {
        "text": "Здравствуйте",
        "language": "ru",
        "speed": 1.0,
        "backend": "supertonic",
    }


async def test_synthesize_can_select_qwen_and_forward_voice(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(content=b"WAVDATA"))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    await voice.synthesize("Здравствуйте", language="ru", backend="qwen", voice="Aiden")

    assert fake.calls[0][1]["json"]["backend"] == "qwen"
    assert fake.calls[0][1]["json"]["voice"] == "Aiden"


async def test_synthesize_empty_audio_raises(monkeypatch):
    fake = _FakeHTTP(_FakeResponse(content=b""))
    monkeypatch.setattr(voice, "_http", lambda: fake)

    with pytest.raises(LLMError):
        await voice.synthesize("hi")


def test_map_http_error_connect_to_timeout():
    assert isinstance(
        voice._map_http_error(httpx.ConnectError("refused")), LLMTimeoutError
    )


def test_map_http_error_5xx_to_upstream():
    request = httpx.Request("POST", "https://voice/tts/synthesize")
    response = httpx.Response(503, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)
    assert isinstance(voice._map_http_error(exc), LLMUpstreamError)


def test_map_http_error_4xx_to_malformed():
    request = httpx.Request("POST", "https://voice/stt/recognize")
    response = httpx.Response(422, request=request)
    exc = httpx.HTTPStatusError("bad", request=request, response=response)
    assert isinstance(voice._map_http_error(exc), LLMMalformedResponseError)
