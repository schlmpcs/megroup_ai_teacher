"""Unit tests for batched embedding in ``app.services.embeddings``.

A single huge document (e.g. ~800 chunks) posted in one request times out / OOMs
the embedder, so ``embed_texts`` splits inputs into ``EMBED_BATCH_SIZE`` chunks,
issues one request per batch, and concatenates the results in input order. The
sidecar HTTP client is replaced with a fake that records each batch — no network.
"""

import asyncio

from app.services import embeddings


class _FakeResponse:
    def __init__(self, inputs: list[str]):
        self._inputs = inputs

    def raise_for_status(self):  # noqa: D401 - mimic httpx.Response
        return None

    def json(self):
        # Encode each input's length into dense[0] so order is verifiable.
        return {
            "embeddings": [
                {"dense": [float(len(t))], "sparse": {"indices": [], "values": []}}
                for t in self._inputs
            ]
        }


class _FakeClient:
    def __init__(self):
        self.batches: list[list[str]] = []

    async def post(self, url, json):
        inputs = list(json["inputs"])
        self.batches.append(inputs)
        return _FakeResponse(inputs)


def _run(coro):
    return asyncio.run(coro)


def test_embed_texts_batches_and_preserves_order(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_http", lambda: fake)
    monkeypatch.setattr(embeddings.settings, "EMBED_BATCH_SIZE", 2)

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = _run(embeddings.embed_texts(texts))

    # 5 inputs / batch size 2 -> 3 requests of sizes 2, 2, 1.
    assert [len(b) for b in fake.batches] == [2, 2, 1]
    # Order preserved across the concatenated batches (dense[0] == len(text)).
    assert [e.dense[0] for e in result] == [float(len(t)) for t in texts]


def test_embed_texts_single_batch_when_within_size(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_http", lambda: fake)
    monkeypatch.setattr(embeddings.settings, "EMBED_BATCH_SIZE", 64)

    result = _run(embeddings.embed_texts(["x", "y", "z"]))
    assert len(fake.batches) == 1
    assert len(result) == 3


def test_embed_texts_batch_size_zero_means_one_request(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_http", lambda: fake)
    monkeypatch.setattr(embeddings.settings, "EMBED_BATCH_SIZE", 0)

    result = _run(embeddings.embed_texts(["a", "b", "c", "d"]))
    assert len(fake.batches) == 1
    assert len(result) == 4


def test_embed_texts_empty_short_circuits(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_http", lambda: fake)
    assert _run(embeddings.embed_texts([])) == []
    assert fake.batches == []
