"""Voice I/O via the in-repo STT/TTS sidecar (the ``voice`` service, ./voice).

Both speech-to-text and text-to-speech are served by a GPU container vendored
into this repo (``./voice``, formerly the standalone ``../vrrag_ttsstt``) that
exposes a plain HTTP API:

  STT: POST {VOICE_BASE_URL}/stt/recognize   (multipart: audio file + language)
       -> {"text": "...", "language": "ru", "confidence": null, "duration_ms": N}
  TTS: POST {VOICE_BASE_URL}/tts/synthesize?format=wav
       (json: text/language/speed/backend/voice)
       -> audio/wav bytes

STT runs a multilingual Whisper (ru/kk/auto). Russian TTS defaults to Supertonic
and can explicitly select Qwen3-TTS 0.6B. Kazakh defaults to the fixed young
male OmniVoice profile, which is served by a separate container because its
Transformers requirement conflicts with Qwen TTS; MMS remains available as a
fallback. The local ``voice`` control selects the Qwen speaker or Supertonic
style. The cloud-era ``instructions`` / ``response_format`` knobs remain
compatibility-only.

This module mirrors ``embeddings.py``: a lazy shared ``httpx.AsyncClient`` and
upstream failures mapped onto the shared ``LLMError`` family so routes translate
them to HTTP status codes uniformly (see app/services/errors.py). Under docker
compose the sidecar is reached over plain HTTP on the internal network
(``http://voice:8001``), so ``VOICE_VERIFY_SSL`` is moot but kept for the client
(and for any external HTTPS deployment).
"""

import logging
from typing import Optional

import httpx

from app.core.config import settings
from app.services.errors import (
    LLMError,
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.services.ttl_cache import TTLCache

logger = logging.getLogger("assistant.voice")

# The sidecar only ever returns WAV.
_TTS_MEDIA_TYPE = "audio/wav"

# Answers repeat across students (and across the sentence-chunked streaming
# path), so cache synthesized WAVs by (text, language). ~0.5MB per entry.
_tts_cache = TTLCache(settings.TTS_CACHE_SIZE, settings.ANSWER_CACHE_TTL_S)


# ── Lazy shared HTTP client ──────────────────────────────────────────────────

# Module-level singleton, created on first use. Kept referenceable (rather than
# hidden in a closure) so tests can reset/monkeypatch it; note that the primary
# patch points for other modules' tests are the ``transcribe`` / ``synthesize``
# functions below (patched on ``app.api.routes``).
_client: Optional[httpx.AsyncClient] = None
_omnivoice_client: Optional[httpx.AsyncClient] = None


def _http() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.VOICE_BASE_URL,
            timeout=settings.VOICE_TIMEOUT_S,
            verify=settings.VOICE_VERIFY_SSL,
        )
    return _client


def _omnivoice_http() -> httpx.AsyncClient:
    """Return the Kazakh OmniVoice client, creating it on first call."""
    global _omnivoice_client
    if _omnivoice_client is None:
        _omnivoice_client = httpx.AsyncClient(
            base_url=settings.VOICE_KK_OMNIVOICE_BASE_URL,
            timeout=settings.VOICE_TIMEOUT_S,
            verify=settings.VOICE_VERIFY_SSL,
        )
    return _omnivoice_client


# ── Error mapping ────────────────────────────────────────────────────────────


def _map_http_error(exc: Exception) -> Exception:
    """Map an httpx failure onto the shared service-layer exception family."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return LLMTimeoutError(f"Voice sidecar request timed out / unreachable: {exc}")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status >= 500:
            return LLMUpstreamError(f"Voice sidecar upstream error {status}: {exc}")
        return LLMMalformedResponseError(f"Voice sidecar returned {status}: {exc}")
    if isinstance(exc, httpx.HTTPError):
        return LLMUpstreamError(f"Voice sidecar HTTP error: {exc}")
    return LLMMalformedResponseError(f"Unexpected voice sidecar error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────


async def transcribe_with_language(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = None,
    prompt: Optional[str] = None,  # noqa: ARG001, accepted for call-site compat
) -> tuple[str, str]:
    """Transcribe audio and return ``(text, resolved_language)``.

    ``language`` is ``"ru"``, ``"kk"`` or ``"auto"`` (Whisper language
    detection); omission defaults to ``"auto"``. ``prompt`` is ignored because the
    local Whisper backend does not take a decoding prompt, but it is kept in the
    signature so callers need not special-case the backend.
    """
    lang = language or "auto"
    files = {"audio": (filename, audio_bytes, "application/octet-stream")}

    try:
        response = await _http().post(
            "/stt/recognize", files=files, data={"language": lang}
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise _map_http_error(exc) from exc

    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise LLMMalformedResponseError("Voice sidecar transcription returned no text")

    resolved_language = payload.get("language")
    if resolved_language not in {"ru", "kk"}:
        if lang in {"ru", "kk"}:
            resolved_language = lang
        else:
            raise LLMMalformedResponseError(
                "Voice sidecar transcription returned no resolved language"
            )
    return text.strip(), resolved_language


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Transcribe recorded mic audio, auto-detecting language when omitted."""
    text, _ = await transcribe_with_language(
        audio_bytes,
        filename=filename,
        language=language,
        prompt=prompt,
    )
    return text


async def synthesize(
    text: str,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,  # noqa: ARG001, sidecar always WAV
    instructions: Optional[str] = None,  # noqa: ARG001, no instruction control
    language: Optional[str] = None,
    backend: Optional[str] = None,
) -> tuple[bytes, str]:
    """Synthesize ``text`` to speech via the sidecar. Returns (audio_bytes, media_type).

    Russian supports ``qwen`` and ``supertonic`` backends. Supertonic is selected
    by default and ``voice`` chooses its style (or the Qwen speaker). Kazakh
    defaults to the fixed young-male ``omnivoice`` backend; ``mms`` remains a
    fallback. Output is always WAV.
    """
    lang = language or settings.DEFAULT_LANGUAGE
    selected_backend = backend
    if selected_backend is None:
        if lang == "ru":
            selected_backend = settings.VOICE_TTS_RU_DEFAULT_BACKEND
        elif lang == "kk":
            selected_backend = settings.VOICE_TTS_KK_DEFAULT_BACKEND

    cache_key = (text, lang, selected_backend, voice)
    cached = _tts_cache.get(cache_key)
    if cached is not None:
        return cached
    body = {"text": text, "language": lang, "speed": 1.0}
    if selected_backend is not None:
        body["backend"] = selected_backend
    if voice is not None:
        body["voice"] = voice

    try:
        client = _omnivoice_http() if selected_backend == "omnivoice" else _http()
        response = await client.post(
            "/tts/synthesize", params={"format": "wav"}, json=body
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise _map_http_error(exc) from exc

    audio = response.content
    if not audio:
        raise LLMError("Voice sidecar synthesis returned no audio")
    result = (audio, _TTS_MEDIA_TYPE)
    _tts_cache.put(cache_key, result)
    return result
