"""Regression tests for non-destructive Qdrant document replacement."""

import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.services import vectorstore
from app.services.errors import LLMTimeoutError, LLMUpstreamError


def _point(point_id: str) -> dict:
    return {
        "id": point_id,
        "dense": [0.1],
        "sparse_indices": [],
        "sparse_values": [],
        "payload": {"doc_id": "doc-1", "text": point_id},
    }


class _Client:
    def __init__(
        self,
        *,
        fail_upsert: bool = False,
        old_ids=None,
        upsert_status=models.UpdateStatus.COMPLETED,
        delete_status=models.UpdateStatus.COMPLETED,
    ):
        self.fail_upsert = fail_upsert
        self.old_ids = old_ids or ["keep", "stale"]
        self.upsert_status = upsert_status
        self.delete_status = delete_status
        self.calls = []

    async def scroll(self, **kwargs):
        self.calls.append("scroll")
        return [SimpleNamespace(id=point_id) for point_id in self.old_ids], None

    async def upsert(self, **kwargs):
        self.calls.append("upsert")
        if self.fail_upsert:
            raise RuntimeError("write failed")
        return SimpleNamespace(status=self.upsert_status)

    async def delete(self, **kwargs):
        self.calls.append(("delete", kwargs["points_selector"].points))
        return SimpleNamespace(status=self.delete_status)


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


async def test_wait_timeout_does_not_delete_stale_points(monkeypatch):
    client = _Client(upsert_status=models.UpdateStatus.WAIT_TIMEOUT)
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    with pytest.raises(Exception, match="wait_timeout"):
        await vectorstore.upsert_points([_point("keep")])

    assert client.calls == ["scroll", "upsert"]


async def test_stale_delete_wait_timeout_is_reported(monkeypatch):
    client = _Client(delete_status=models.UpdateStatus.WAIT_TIMEOUT)
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    with pytest.raises(Exception, match="wait_timeout"):
        await vectorstore.upsert_points([_point("keep")])

    assert client.calls == ["scroll", "upsert", ("delete", ["stale"])]


async def test_upsert_treats_hyphenated_and_hex_uuid_ids_as_the_same_point(monkeypatch):
    hyphenated = "550e8400-e29b-41d4-a716-446655440000"
    compact = hyphenated.replace("-", "")
    client = _Client(old_ids=[hyphenated])
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    assert await vectorstore.upsert_points([_point(compact)]) == 1

    assert client.calls == ["scroll", "upsert"]


async def test_concurrent_upserts_are_serialized(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    class _ConcurrentClient(_Client):
        def __init__(self):
            super().__init__(old_ids=[])
            self.active = 0
            self.max_active = 0

        async def upsert(self, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if not entered.is_set():
                entered.set()
                await release.wait()
            self.active -= 1
            return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

    client = _ConcurrentClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    first = asyncio.create_task(vectorstore.upsert_points([_point("first")]))
    await entered.wait()
    second = asyncio.create_task(vectorstore.upsert_points([_point("second")]))
    await asyncio.sleep(0)

    assert client.max_active == 1
    release.set()
    await asyncio.gather(first, second)
    assert client.max_active == 1


@pytest.mark.parametrize(
    ("activation_status", "delete_status", "expected_texts", "expected_lab", "expected_generation"),
    [
        (
            models.UpdateStatus.COMPLETED,
            models.UpdateStatus.WAIT_TIMEOUT,
            ["new complete"],
            "new complete",
            "0002",
        ),
        (
            models.UpdateStatus.WAIT_TIMEOUT,
            models.UpdateStatus.COMPLETED,
            ["old head", "old tail"],
            "old head\nold tail",
            "0001",
        ),
    ],
)
async def test_replacement_failure_keeps_one_complete_generation_visible(
    monkeypatch,
    activation_status,
    delete_status,
    expected_texts,
    expected_lab,
    expected_generation,
):
    def payload(generation, index, count, text):
        return {
            "doc_id": "doc-1",
            "lab_id": "physics-8-ru-02",
            "generation": generation,
            "active_generation": generation,
            "chunk_index": index,
            "chunk_count": count,
            "text": text,
            "status": "ready",
        }

    class _GenerationClient:
        def __init__(self):
            self.records = {
                "old-0": SimpleNamespace(
                    id="old-0", payload=payload("0001", 0, 2, "old head")
                ),
                "old-1": SimpleNamespace(
                    id="old-1", payload=payload("0001", 1, 2, "old tail")
                ),
            }

        async def collection_exists(self, name):
            return True

        async def scroll(self, **kwargs):
            return list(self.records.values()), None

        async def upsert(self, **kwargs):
            for point in kwargs["points"]:
                self.records[str(point.id)] = SimpleNamespace(
                    id=point.id, payload=dict(point.payload or {})
                )
            return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

        async def set_payload(self, **kwargs):
            if activation_status == models.UpdateStatus.COMPLETED:
                if kwargs["payload"].get("status") == "superseded":
                    active = next(iter(self.records.values())).payload[
                        "active_generation"
                    ]
                    for record in self.records.values():
                        if record.payload.get("generation") != active:
                            record.payload.update(kwargs["payload"])
                else:
                    for record in self.records.values():
                        record.payload.update(kwargs["payload"])
            return SimpleNamespace(status=activation_status)

        async def delete(self, **kwargs):
            return SimpleNamespace(status=delete_status)

        async def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(score=1.0, payload=record.payload)
                    for record in self.records.values()
                ]
            )

    client = _GenerationClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    new = _point("new-0")
    new["payload"] = payload("0002", 0, 1, "new complete")

    if activation_status == models.UpdateStatus.WAIT_TIMEOUT:
        with pytest.raises(LLMTimeoutError, match="wait_timeout"):
            await vectorstore.upsert_points([new])
    else:
        assert await vectorstore.upsert_points([new]) == 1

    results = await vectorstore.hybrid_search([0.1], [], [], 5, 10)
    lab = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert [result["payload"]["text"] for result in results] == expected_texts
    assert lab["text"] == expected_lab
    assert {item["generation"] for item in lab["payloads"]} == {
        expected_generation
    }
    if activation_status == models.UpdateStatus.COMPLETED:
        assert {
            record.payload["status"]
            for key, record in client.records.items()
            if key.startswith("old-")
        } == {"superseded"}


async def test_hybrid_search_does_not_scan_candidate_documents(monkeypatch):
    class _QueryClient:
        async def query_points(self, **kwargs):
            inactive = kwargs["prefetch"][0].filter.must_not[-1].match.any
            assert "superseded" in inactive
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=1.0,
                        payload={"doc_id": "legacy", "text": "result"},
                    )
                ]
            )

        async def scroll(self, **kwargs):
            raise AssertionError("search must not scan document payloads")

    monkeypatch.setattr(vectorstore, "get_client", lambda: _QueryClient())

    results = await vectorstore.hybrid_search([0.1], [], [], 5, 10)

    assert [result["payload"]["text"] for result in results] == ["result"]


@pytest.mark.parametrize("legacy", [False, True])
async def test_hybrid_search_retries_past_hidden_generation(monkeypatch, legacy):
    stale = {
        "doc_id": "doc-1",
        "active_generation": "new",
        "text": "stale",
        "status": "ready",
    }
    if not legacy:
        stale["generation"] = "old"
    active = {
        "doc_id": "doc-1",
        "generation": "new",
        "active_generation": "new",
        "text": "active",
        "status": "ready",
    }

    class _QueryClient:
        def __init__(self):
            self.calls = 0

        async def query_points(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    points=[SimpleNamespace(id="stale", score=1.0, payload=stale)]
                )
            exclusions = kwargs["prefetch"][0].filter.must_not
            if legacy:
                assert any(isinstance(condition, models.Filter) for condition in exclusions)
            else:
                assert any(
                    isinstance(condition, models.FieldCondition)
                    and condition.key == "generation"
                    for condition in exclusions
                )
            return SimpleNamespace(
                points=[SimpleNamespace(id="active", score=0.9, payload=active)]
            )

    client = _QueryClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    results = await vectorstore.hybrid_search([0.1], [], [], 1, 1)

    assert [result["payload"]["text"] for result in results] == ["active"]
    assert client.calls == 2


async def test_hybrid_search_retries_past_stacked_hidden_generations(monkeypatch):
    generations = ["old-1", "old-2", "old-3", "old-4"]

    class _QueryClient:
        def __init__(self):
            self.calls = 0

        async def query_points(self, **kwargs):
            self.calls += 1
            if generations:
                generation = generations.pop(0)
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id=generation,
                            score=1.0,
                            payload={
                                "doc_id": "doc-1",
                                "generation": generation,
                                "active_generation": "new",
                                "text": generation,
                                "status": "ready",
                            },
                        )
                    ]
                )
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="active",
                        score=0.9,
                        payload={
                            "doc_id": "doc-1",
                            "generation": "new",
                            "active_generation": "new",
                            "text": "active",
                            "status": "ready",
                        },
                    )
                ]
            )

    client = _QueryClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    results = await vectorstore.hybrid_search([0.1], [], [], 1, 1)

    assert [result["payload"]["text"] for result in results] == ["active"]
    assert client.calls == 5


async def test_replacement_does_not_delete_foreign_inflight_generation(monkeypatch):
    def payload(generation, text, status, active=None):
        return {
            "doc_id": "doc-1",
            "generation": generation,
            "active_generation": active,
            "chunk_index": 0,
            "chunk_count": 1,
            "text": text,
            "status": status,
        }

    class _ConcurrentClient:
        def __init__(self):
            self.records = {
                "old": SimpleNamespace(
                    id="old", payload=payload("old", "old", "ready", "old")
                ),
                "foreign": SimpleNamespace(
                    id="foreign",
                    payload=payload("foreign", "foreign", "staging"),
                ),
            }

        async def scroll(self, **kwargs):
            return list(self.records.values()), None

        async def upsert(self, **kwargs):
            for point in kwargs["points"]:
                self.records[str(point.id)] = SimpleNamespace(
                    id=point.id, payload=dict(point.payload or {})
                )
            return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

        async def set_payload(self, **kwargs):
            if kwargs["payload"].get("status") == "superseded":
                active = next(iter(self.records.values())).payload[
                    "active_generation"
                ]
                for record in self.records.values():
                    if record.payload.get("generation") != active:
                        record.payload.update(kwargs["payload"])
            else:
                for record in self.records.values():
                    record.payload.update(kwargs["payload"])
            return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

        async def delete(self, **kwargs):
            ids = set(kwargs["points_selector"].filter.must[0].has_id)
            assert ids == {"old"}
            for point_id in ids:
                self.records.pop(point_id, None)
            return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

        async def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(score=1.0, payload=record.payload)
                    for record in self.records.values()
                ]
            )

    client = _ConcurrentClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    ours = _point("ours")
    ours["payload"] = payload("ours", "ours", "ready")

    assert await vectorstore.upsert_points([ours]) == 1
    assert "foreign" in client.records

    for record in client.records.values():
        record.payload.update(active_generation="foreign", status="ready")
    client.records["ours"].payload["status"] = "superseded"

    results = await vectorstore.hybrid_search([0.1], [], [], 5, 10)

    assert [result["payload"]["text"] for result in results] == ["foreign"]


async def test_delete_document_requires_completed_status(monkeypatch):
    class _DeleteClient:
        async def count(self, **kwargs):
            return SimpleNamespace(count=1)

        async def delete(self, **kwargs):
            return SimpleNamespace(status=models.UpdateStatus.WAIT_TIMEOUT)

    monkeypatch.setattr(vectorstore, "get_client", lambda: _DeleteClient())

    with pytest.raises(LLMTimeoutError, match="wait_timeout"):
        await vectorstore.delete_document("doc-1")


async def test_concurrent_ensure_collection_creates_once(monkeypatch):
    entered_create = asyncio.Event()
    release_create = asyncio.Event()

    class _CollectionClient:
        def __init__(self):
            self.created = False
            self.create_calls = 0

        async def collection_exists(self, name):
            return self.created

        async def create_collection(self, **kwargs):
            self.create_calls += 1
            entered_create.set()
            if self.create_calls == 1:
                await release_create.wait()
            self.created = True

    client = _CollectionClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    first = asyncio.create_task(vectorstore.ensure_collection())
    await entered_create.wait()
    second = asyncio.create_task(vectorstore.ensure_collection())
    await asyncio.sleep(0)

    assert client.create_calls == 1
    release_create.set()
    await asyncio.gather(first, second)
    assert client.create_calls == 1


async def test_ensure_collection_accepts_cross_process_create_race(monkeypatch):
    class _CollectionClient:
        def __init__(self):
            self.exists_checks = 0

        async def collection_exists(self, name):
            self.exists_checks += 1
            return self.exists_checks > 1

        async def create_collection(self, **kwargs):
            raise RuntimeError("collection already exists")

    monkeypatch.setattr(vectorstore, "get_client", lambda: _CollectionClient())

    await vectorstore.ensure_collection()
