"""HTTP client for the local bge-m3 embedder sidecar.

A separate GPU container (service name ``embedder``) serves the bge-m3 model
over plain HTTP. This module is the thin proxy-side client used by the hybrid
RAG retrieval layer: it turns text into bge-m3's hybrid representation — a dense
vector plus a sparse (lexical) term map — by POSTing to the sidecar.

Sidecar contract:
  POST {EMBEDDING_BASE_URL}/embed   body: {"inputs": ["t1", "t2", ...]}
  ->  {"embeddings": [
        {"dense": [float, ... 1024], "sparse": {"indices": [int...], "values": [float...]}},
        ...
      ]}                            # one entry per input, in order

Upstream failures are mapped onto the shared service-layer ``LLMError`` family
so routes can translate them to HTTP status codes uniformly (see
app/services/errors.py).
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.services.errors import (
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)

logger = logging.getLogger("assistant.embeddings")


@dataclass
class Embedding:
    """A single bge-m3 hybrid embedding: dense vector + sparse term map."""

    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


# ── Lazy shared HTTP client ──────────────────────────────────────────────────

# Module-level singleton, created on first use. Kept referenceable (rather than
# hidden inside a closure) so tests can reset/monkeypatch it; note, however,
# that the primary patch points for other modules' tests are the module-level
# ``embed_texts`` / ``embed_query`` functions below.
_client: Optional[httpx.AsyncClient] = None


def _http() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=settings.REQUEST_TIMEOUT_S,
        )
    return _client


# ── Error mapping ────────────────────────────────────────────────────────────


def _map_http_error(exc: Exception) -> Exception:
    """Map an httpx failure onto the shared service-layer exception family."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return LLMTimeoutError(f"Embedder request timed out / unreachable: {exc}")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status >= 500:
            return LLMUpstreamError(f"Embedder upstream error {status}: {exc}")
        return LLMMalformedResponseError(f"Embedder returned {status}: {exc}")
    if isinstance(exc, httpx.HTTPError):
        return LLMUpstreamError(f"Embedder HTTP error: {exc}")
    return LLMMalformedResponseError(f"Unexpected embedder error: {exc}")


# ── Response parsing ──────────────────────────────────────────────────────────


def _parse_embedding(item: Any) -> Embedding:
    """Turn one response entry into an ``Embedding``.

    Tolerant of a missing/empty ``sparse`` object (yields empty indices/values),
    which bge-m3 can produce for inputs with no salient lexical terms.
    """
    dense = item.get("dense")
    if not isinstance(dense, list):
        raise LLMMalformedResponseError("Embedder entry missing 'dense' vector")

    sparse = item.get("sparse") or {}
    indices = sparse.get("indices") or []
    values = sparse.get("values") or []
    return Embedding(dense=dense, sparse_indices=indices, sparse_values=values)


# ── Public API ────────────────────────────────────────────────────────────────


async def embed_texts(texts: list[str]) -> list[Embedding]:
    """Embed a batch of texts via the sidecar, preserving input order.

    Returns one ``Embedding`` per input. An empty input list short-circuits
    without an HTTP call.
    """
    if not texts:
        return []

    try:
        response = await _http().post("/embed", json={"inputs": texts})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise _map_http_error(exc) from exc

    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(embeddings, list):
        raise LLMMalformedResponseError("Embedder response missing 'embeddings' list")
    if len(embeddings) != len(texts):
        raise LLMMalformedResponseError(
            f"Embedder returned {len(embeddings)} embeddings for {len(texts)} inputs"
        )

    return [_parse_embedding(item) for item in embeddings]


async def embed_query(text: str) -> Embedding:
    """Embed a single query string (convenience wrapper around ``embed_texts``)."""
    return (await embed_texts([text]))[0]
