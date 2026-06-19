"""Unit tests for the local hybrid-RAG layer (embeddings / ingestion / llm).

Everything below the proxy is mocked: there is no live Qdrant or embedder
sidecar in this environment. We patch the network boundaries (the httpx client
for embeddings; the ``embeddings``/``vectorstore``/``client`` module attributes
elsewhere) and assert the wiring around them.

``asyncio_mode = auto`` (pytest.ini) lets ``async def test_*`` run without a
per-test marker.
"""

from types import SimpleNamespace

import httpx
import pytest

from app.services import embeddings, ingestion
from app.services import llm
from app.services.embeddings import Embedding
from app.services.errors import (
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)


# ── embeddings ───────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHTTP:
    """Minimal stand-in for httpx.AsyncClient capturing the last POST."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def post(self, url, json=None):
        self.calls.append((url, json))
        return _FakeResponse(self._payload)


async def test_embed_texts_parses_response(monkeypatch):
    payload = {
        "embeddings": [
            {"dense": [0.1, 0.2], "sparse": {"indices": [3], "values": [0.9]}},
            {"dense": [0.3, 0.4], "sparse": {"indices": [], "values": []}},
        ]
    }
    fake = _FakeHTTP(payload)
    monkeypatch.setattr(embeddings, "_http", lambda: fake)

    out = await embeddings.embed_texts(["a", "b"])
    assert len(out) == 2
    assert isinstance(out[0], Embedding)
    assert out[0].dense == [0.1, 0.2]
    assert out[0].sparse_indices == [3]
    assert out[0].sparse_values == [0.9]
    # second one had an empty sparse map
    assert out[1].sparse_indices == []
    assert fake.calls == [("/embed", {"inputs": ["a", "b"]})]


async def test_embed_texts_empty_no_http(monkeypatch):
    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("_http() should not be called for empty input")

    monkeypatch.setattr(embeddings, "_http", _boom)
    assert await embeddings.embed_texts([]) == []


async def test_embed_query_returns_single(monkeypatch):
    payload = {"embeddings": [{"dense": [1.0], "sparse": {"indices": [], "values": []}}]}
    monkeypatch.setattr(embeddings, "_http", lambda: _FakeHTTP(payload))
    emb = await embeddings.embed_query("вопрос")
    assert isinstance(emb, Embedding)
    assert emb.dense == [1.0]


def test_map_http_error_connect_to_timeout():
    mapped = embeddings._map_http_error(httpx.ConnectError("refused"))
    assert isinstance(mapped, LLMTimeoutError)


def test_map_http_error_5xx_to_upstream():
    request = httpx.Request("POST", "http://embedder/embed")
    response = httpx.Response(500, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)
    mapped = embeddings._map_http_error(exc)
    assert isinstance(mapped, LLMUpstreamError)


def test_map_http_error_4xx_to_malformed():
    request = httpx.Request("POST", "http://embedder/embed")
    response = httpx.Response(400, request=request)
    exc = httpx.HTTPStatusError("bad", request=request, response=response)
    assert isinstance(embeddings._map_http_error(exc), LLMMalformedResponseError)


# ── ingestion ────────────────────────────────────────────────────────────────


class _RecordingVectorstore:
    def __init__(self):
        self.upserted = None
        self.ensured = False
        self.deleted = []

    async def ensure_collection(self):
        self.ensured = True

    async def delete_document(self, doc_id):
        self.deleted.append(doc_id)
        return False

    async def upsert_points(self, points):
        self.upserted = points
        return len(points)


def _patch_ingestion(monkeypatch, vs):
    async def _embed_texts(texts):
        return [
            Embedding(dense=[float(i)], sparse_indices=[i], sparse_values=[1.0])
            for i, _ in enumerate(texts)
        ]

    monkeypatch.setattr(ingestion.embeddings, "embed_texts", _embed_texts)
    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", vs.ensure_collection)
    monkeypatch.setattr(ingestion.vectorstore, "delete_document", vs.delete_document)
    monkeypatch.setattr(ingestion.vectorstore, "upsert_points", vs.upsert_points)


async def test_upload_document_txt_chunks_and_upserts(monkeypatch):
    vs = _RecordingVectorstore()
    _patch_ingestion(monkeypatch, vs)
    # Force a tiny chunk size so the payload splits into several chunks.
    monkeypatch.setattr(ingestion.settings, "CHUNK_SIZE", 20, raising=False)
    monkeypatch.setattr(ingestion.settings, "CHUNK_OVERLAP", 5, raising=False)

    content = ("слово " * 30).encode("utf-8")
    result = await ingestion.upload_document("notes.txt", content)

    assert vs.ensured is True
    assert result["status"] == "ready"
    assert result["filename"] == "notes.txt"
    assert result["file_id"] == ingestion._doc_id("notes.txt")
    assert result["chunks"] == len(vs.upserted)
    assert len(vs.upserted) > 1

    point = vs.upserted[0]
    assert set(point.keys()) >= {"id", "dense", "sparse_indices", "sparse_values", "payload"}
    payload = point["payload"]
    assert payload["doc_id"] == result["file_id"]
    assert payload["filename"] == "notes.txt"
    assert payload["chunk_index"] == 0
    assert isinstance(payload["text"], str) and payload["text"]


async def test_upload_document_unsupported_extension_raises(monkeypatch):
    vs = _RecordingVectorstore()
    _patch_ingestion(monkeypatch, vs)
    with pytest.raises(ValueError):
        await ingestion.upload_document("notes.xyz", b"data")


def test_chunk_text_overlap(monkeypatch):
    monkeypatch.setattr(ingestion.settings, "CHUNK_SIZE", 10, raising=False)
    monkeypatch.setattr(ingestion.settings, "CHUNK_OVERLAP", 4, raising=False)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = ingestion._chunk_text(text)
    # step = size - overlap = 6
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ghijklmnop"  # starts 6 in, overlaps last 4 of chunk 0
    # reassembling by step should recover the source
    assert chunks[0][:6] + chunks[1][:6] + chunks[2][:6] == text[:18]


def test_chunk_text_empty():
    assert ingestion._chunk_text("   \n\t  ") == []


def test_doc_id_stable():
    assert ingestion._doc_id("a.pdf") == ingestion._doc_id("a.pdf")
    assert ingestion._doc_id("a.pdf") != ingestion._doc_id("b.pdf")


# ── llm.generate_answer wiring ───────────────────────────────────────────────


async def test_generate_answer_uses_retrieved_citations(monkeypatch):
    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, sparse_indices, sparse_values, top_k, candidates):
        return [
            {
                "score": 0.9,
                "payload": {
                    "doc_id": "d1",
                    "filename": "physics_8.pdf",
                    "chunk_index": 0,
                    "text": "Кипение происходит при 100°C.",
                },
            }
        ]

    async def _create(**kwargs):
        return SimpleNamespace(
            output_text="Вода кипит при 100 градусах.",
            usage=SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20),
        )

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    result = await llm.generate_answer("Когда кипит вода?")
    assert result.answer == "Вода кипит при 100 градусах."
    assert result.citations == [{"filename": "physics_8.pdf", "file_id": "d1"}]
    assert result.usage == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
