"""Regression tests for document replacement and bulk-ingest safety."""

import asyncio
from pathlib import Path

import pytest

from app.services import corpus_meta, ingestion
from app.services.embeddings import Embedding


async def _embeddings(texts):
    return [
        Embedding(dense=[float(index)], sparse_indices=[index], sparse_values=[1.0])
        for index, _ in enumerate(texts)
    ]


async def test_replacement_does_not_delete_old_document_before_failed_upsert(
    monkeypatch,
):
    calls = []

    async def ensure_collection():
        calls.append("ensure")

    async def delete_document(_doc_id):
        calls.append("delete")
        return True

    async def upsert_points(_points):
        calls.append("upsert")
        raise RuntimeError("qdrant write failed")

    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", ensure_collection)
    monkeypatch.setattr(ingestion.vectorstore, "delete_document", delete_document)
    monkeypatch.setattr(ingestion.vectorstore, "upsert_points", upsert_points)
    monkeypatch.setattr(ingestion.embeddings, "embed_texts", _embeddings)

    with pytest.raises(RuntimeError, match="qdrant write failed"):
        await ingestion.upload_document("notes.md", b"replacement text")

    assert calls == ["ensure", "upsert"]


async def test_invalid_document_is_rejected_before_qdrant(monkeypatch):
    async def unexpected_ensure():
        raise AssertionError("Qdrant must not be contacted for invalid input")

    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", unexpected_ensure)

    with pytest.raises(ValueError, match="Invalid PDF"):
        await ingestion.upload_document("bad.pdf", b"not a pdf")


async def test_extraction_runs_off_the_event_loop(monkeypatch):
    calls = []
    real_to_thread = asyncio.to_thread

    async def tracked_to_thread(function, /, *args, **kwargs):
        calls.append(function)
        return await real_to_thread(function, *args, **kwargs)

    async def ensure_collection():
        return None

    async def upsert_points(points):
        return len(points)

    monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)
    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", ensure_collection)
    monkeypatch.setattr(ingestion.vectorstore, "upsert_points", upsert_points)
    monkeypatch.setattr(ingestion.embeddings, "embed_texts", _embeddings)

    await ingestion.upload_document("notes.md", b"threaded extraction")

    assert ingestion.to_markdown in calls


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("bad.pdf", b"not a pdf"),
        ("bad.docx", b"not a docx"),
        ("bad.epub", b"not an epub"),
    ],
)
def test_to_markdown_rejects_mislabeled_binary_documents(filename, content):
    with pytest.raises(ValueError, match="Invalid"):
        ingestion.to_markdown(filename, content)


def test_lab_upload_identity_is_lab_id_not_filename():
    first, first_key = corpus_meta.build_upload_metadata(
        "First name.docx",
        doc_type="lab_instruction",
        subject="physics",
        grade=10,
        lang="ru",
        lab_number=2,
    )
    second, second_key = corpus_meta.build_upload_metadata(
        "Renamed.docx",
        doc_type="lab_instruction",
        subject="physics",
        grade=10,
        lang="ru",
        lab_number=2,
    )

    assert first_key == second_key == "admin_uploads/lab_instruction/physics-10-ru-02"
    assert first["source"] != second["source"]


def _lab_dir(root: Path) -> Path:
    path = root / "Laboratory works" / "Physics" / "Physics Grade 10" / "en"
    path.mkdir(parents=True)
    return path


def _book_dir(root: Path) -> Path:
    path = root / "School materials" / "Biology" / "en"
    path.mkdir(parents=True)
    return path


async def test_bulk_ingest_rejects_missing_root():
    with pytest.raises(ValueError, match="Corpus root"):
        await ingestion.bulk_ingest_tree("/definitely/not/a/corpus")


async def test_bulk_ingest_skips_incomplete_metadata(tmp_path, monkeypatch):
    path = _lab_dir(tmp_path) / "Electrolyte current conditions.md"
    path.write_text("procedure", encoding="utf-8")
    uploads = []

    async def upload(*args, **kwargs):
        uploads.append((args, kwargs))
        return {"status": "ready", "chunks": 1, "file_id": "unexpected"}

    async def list_documents():
        return []

    monkeypatch.setattr(ingestion, "upload_document", upload)
    monkeypatch.setattr(ingestion.vectorstore, "list_documents", list_documents)

    summary = await ingestion.bulk_ingest_tree(str(tmp_path))

    assert uploads == []
    assert summary["skipped"] == 1
    assert summary["errors"][0]["source"].endswith(path.name)
    assert "lab_number" in summary["errors"][0]["error"]


async def test_bulk_ingest_rejects_duplicate_lab_ids(tmp_path, monkeypatch):
    lab_dir = _lab_dir(tmp_path)
    first = lab_dir / "Lab work 1.md"
    second = lab_dir / "Lab work No. 1.md"
    first.write_text("first procedure", encoding="utf-8")
    second.write_text("second procedure", encoding="utf-8")
    parsed = [
        corpus_meta.parse_path(str(path), corpus_root=str(tmp_path))
        for path in (first, second)
    ]
    uploads = []
    deleted = []

    async def upload(*args, **kwargs):
        uploads.append((args, kwargs))
        return {"status": "ready", "chunks": 1, "file_id": "unexpected"}

    async def delete_document(doc_id):
        deleted.append(doc_id)
        return True

    async def list_documents():
        return [
            {
                "file_id": ingestion._doc_id(meta["source"]),
                "source_path": meta["source"],
            }
            for meta in parsed
        ]

    monkeypatch.setattr(ingestion, "upload_document", upload)
    monkeypatch.setattr(ingestion.vectorstore, "delete_document", delete_document)
    monkeypatch.setattr(ingestion.vectorstore, "list_documents", list_documents)

    summary = await ingestion.bulk_ingest_tree(str(tmp_path))

    expected_ids = {ingestion._doc_id(meta["source"]) for meta in parsed}
    assert uploads == []
    assert set(deleted) == expected_ids
    assert summary["skipped"] == 2
    assert len(summary["errors"]) == 2
    assert all("Duplicate lab_id" in error["error"] for error in summary["errors"])


async def test_full_bulk_ingest_prunes_removed_corpus_documents(tmp_path, monkeypatch):
    current = _book_dir(tmp_path) / "Biology Grade 9.md"
    current.write_text("cell theory", encoding="utf-8")
    deleted = []

    async def upload_document(filename, content, metadata=None, doc_key=None, **kwargs):
        return {
            "status": "ready",
            "chunks": 1,
            "file_id": ingestion._doc_id(doc_key),
            "filename": filename,
        }

    async def list_documents():
        return [
            {
                "file_id": "stale-corpus-id",
                "source_path": "School materials/Biology/en/Old Biology Grade 9.md",
            },
            {
                "file_id": "admin-id",
                "source_path": "admin_uploads/textbook/biology/9/en/Admin.pdf",
            },
        ]

    async def delete_document(doc_id):
        deleted.append(doc_id)
        return True

    monkeypatch.setattr(ingestion, "upload_document", upload_document)
    monkeypatch.setattr(ingestion.vectorstore, "list_documents", list_documents)
    monkeypatch.setattr(ingestion.vectorstore, "delete_document", delete_document)

    summary = await ingestion.bulk_ingest_tree(str(tmp_path))

    assert deleted == ["stale-corpus-id"]
    assert summary["pruned"] == 1


async def test_bulk_ingest_uses_ocr_setting_when_unspecified(tmp_path, monkeypatch):
    book = _book_dir(tmp_path) / "Biology Grade 9.md"
    book.write_text("cell theory", encoding="utf-8")
    seen = []

    async def upload_document(filename, content, metadata=None, doc_key=None, ocr=False):
        seen.append(ocr)
        return {
            "status": "ready",
            "chunks": 1,
            "file_id": ingestion._doc_id(doc_key),
            "filename": filename,
        }

    async def list_documents():
        return []

    monkeypatch.setattr(ingestion.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(ingestion, "upload_document", upload_document)
    monkeypatch.setattr(ingestion.vectorstore, "list_documents", list_documents)

    await ingestion.bulk_ingest_tree(str(tmp_path), ocr=None)

    assert seen == [True]


def test_manifest_omits_ambiguous_duplicate_lab_ids(tmp_path):
    lab_dir = _lab_dir(tmp_path)
    first = lab_dir / "Lab work 1.md"
    second = lab_dir / "Lab work No. 1.md"
    first.write_text("first procedure " * 30, encoding="utf-8")
    second.write_text("second procedure " * 30, encoding="utf-8")

    manifest = ingestion.build_manifest(str(tmp_path))

    assert "physics-10-en-01" not in manifest["labs"]
    assert set(manifest["missing_metadata"]) == {
        "Laboratory works/Physics/Physics Grade 10/en/Lab work 1.md",
        "Laboratory works/Physics/Physics Grade 10/en/Lab work No. 1.md",
    }
