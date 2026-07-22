"""Regression tests for non-destructive Qdrant document replacement."""

from types import SimpleNamespace

import pytest

from app.services import vectorstore
from app.services.errors import LLMUpstreamError


def _point(point_id: str) -> dict:
    return {
        "id": point_id,
        "dense": [0.1],
        "sparse_indices": [],
        "sparse_values": [],
        "payload": {"doc_id": "doc-1", "text": point_id},
    }


class _Client:
    def __init__(self, *, fail_upsert: bool = False, old_ids=None):
        self.fail_upsert = fail_upsert
        self.old_ids = old_ids or ["keep", "stale"]
        self.calls = []

    async def scroll(self, **kwargs):
        self.calls.append("scroll")
        return [SimpleNamespace(id=point_id) for point_id in self.old_ids], None

    async def upsert(self, **kwargs):
        self.calls.append("upsert")
        if self.fail_upsert:
            raise RuntimeError("write failed")

    async def delete(self, **kwargs):
        self.calls.append(("delete", kwargs["points_selector"].points))


async def test_upsert_failure_keeps_existing_document_points(monkeypatch):
    client = _Client(fail_upsert=True)
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    with pytest.raises(LLMUpstreamError, match="write failed"):
        await vectorstore.upsert_points([_point("keep")])

    assert client.calls == ["scroll", "upsert"]


async def test_upsert_deletes_only_stale_points_after_success(monkeypatch):
    client = _Client()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    assert await vectorstore.upsert_points([_point("keep")]) == 1

    assert client.calls == ["scroll", "upsert", ("delete", ["stale"])]


async def test_upsert_treats_hyphenated_and_hex_uuid_ids_as_the_same_point(monkeypatch):
    hyphenated = "550e8400-e29b-41d4-a716-446655440000"
    compact = hyphenated.replace("-", "")
    client = _Client(old_ids=[hyphenated])
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    assert await vectorstore.upsert_points([_point(compact)]) == 1

    assert client.calls == ["scroll", "upsert"]
