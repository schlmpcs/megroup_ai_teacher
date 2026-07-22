from types import SimpleNamespace

from app.services import ingestion
from app.services import vectorstore


class _Client:
    def __init__(self, records):
        self._records = records

    async def collection_exists(self, name):
        return True

    async def scroll(self, **kwargs):
        return [SimpleNamespace(payload=payload) for payload in self._records], None


async def test_fetch_lab_instruction_record_skips_pending_and_staging(monkeypatch):
    ready_1 = {
        "doc_id": "lab-ready",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 1,
        "text": "Шаг второй.",
        "status": "ready",
    }
    ready_0 = {**ready_1, "chunk_index": 0, "text": "Шаг первый."}
    pending = {
        **ready_0,
        "doc_id": "lab-pending",
        "text": "Черновик.",
        "status": "pending",
    }
    staging = {
        **ready_0,
        "doc_id": "lab-staging",
        "text": "Промежуточная версия.",
        "status": "staging",
    }
    monkeypatch.setattr(
        vectorstore, "get_client", lambda: _Client([ready_1, pending, staging, ready_0])
    )

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record == {
        "text": "Шаг первый.\nШаг второй.",
        "payloads": [ready_0, ready_1],
    }


async def test_fetch_lab_instruction_record_returns_none_for_ambiguous_lab(monkeypatch):
    first = {
        "doc_id": "lab-a",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "text": "Первая инструкция.",
        "status": "ready",
    }
    second = {
        "doc_id": "lab-b",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "text": "Вторая инструкция.",
        "status": "ready",
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client([first, second]))

    assert await vectorstore.fetch_lab_instruction_record("physics-8-ru-02") is None


async def test_fetch_lab_instruction_record_counts_empty_duplicate_as_ambiguous(
    monkeypatch,
):
    empty = {
        "doc_id": "lab-empty",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "text": "",
        "status": "ready",
    }
    ready = {
        "doc_id": "lab-ready",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "text": "Инструкция.",
        "status": "ready",
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client([empty, ready]))

    assert await vectorstore.fetch_lab_instruction_record("physics-8-ru-02") is None


async def test_fetch_lab_instruction_record_removes_overlaps_using_char_offsets(
    monkeypatch,
):
    first = {
        "doc_id": "lab-doc",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 6,
        "text": "abcdef",
        "status": "ready",
    }
    second = {
        "doc_id": "lab-doc",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 1,
        "char_start": 3,
        "char_end": 9,
        "text": "defghi",
        "status": "ready",
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client([second, first]))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record == {
        "text": "abcdefghi",
        "payloads": [first, second],
    }


async def test_fetch_lab_instruction_record_uses_chunk_index_for_legacy_chunks(monkeypatch):
    second = {
        "doc_id": "lab-doc",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 1,
        "text": "Шаг второй.",
        "status": "ready",
    }
    first = {
        **second,
        "chunk_index": 0,
        "text": "Шаг первый.",
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client([second, first]))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record == {
        "text": "Шаг первый.\nШаг второй.",
        "payloads": [first, second],
    }


async def test_fetch_lab_instruction_record_round_trips_real_trimmed_chunks(
    monkeypatch,
):
    monkeypatch.setattr(ingestion.settings, "CHUNK_SIZE", 7)
    monkeypatch.setattr(ingestion.settings, "CHUNK_OVERLAP", 2)
    source = "abc def ghi jkl mn"
    payloads = [
        {
            **record,
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": index,
            "status": "ready",
        }
        for index, record in enumerate(ingestion._chunk_records(source))
    ]
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client(payloads[::-1]))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record["text"] == source


async def test_fetch_lab_instruction_record_preserves_repeated_legacy_text(
    monkeypatch,
):
    first = {
        "doc_id": "lab-doc",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 7,
        "text": "aaaa a",
        "status": "ready",
    }
    second = {
        **first,
        "chunk_index": 1,
        "char_start": 5,
        "char_end": 8,
        "text": "a a",
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client([second, first]))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record["text"] == "aaaa a a"


async def test_fetch_lab_instruction_record_marks_ambiguous_final_boundary(
    monkeypatch,
):
    first = {
        "doc_id": "lab-doc",
        "lab_id": "physics-8-ru-02",
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 2,
        "text": "ab",
        "status": "ready",
    }
    second = {
        **first,
        "chunk_index": 1,
        "char_start": 2,
        "char_end": 4,
        "text": "c",
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client([second, first]))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record["text"] == "ab\nc"


async def test_fetch_lab_instruction_record_marks_ambiguous_internal_boundary(
    monkeypatch,
):
    payloads = [
        {
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 2,
            "text": "ab",
            "status": "ready",
        },
        {
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": 1,
            "char_start": 2,
            "char_end": 4,
            "text": "c",
            "status": "ready",
        },
        {
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": 2,
            "char_start": 4,
            "char_end": 5,
            "text": "d",
            "status": "ready",
        },
    ]
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client(payloads))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record["text"] == "ab\nc\nd"


async def test_fetch_lab_instruction_record_does_not_move_ambiguous_space(
    monkeypatch,
):
    payloads = [
        {
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "text": "abcd",
            "status": "ready",
        },
        {
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": 1,
            "char_start": 4,
            "char_end": 8,
            "text": "efg",
            "status": "ready",
        },
        {
            "doc_id": "lab-doc",
            "lab_id": "physics-8-ru-02",
            "chunk_index": 2,
            "char_start": 8,
            "char_end": 9,
            "text": "h",
            "status": "ready",
        },
    ]
    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client(payloads))

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record["text"] == "abcd\nefg\nh"
