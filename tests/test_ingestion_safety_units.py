"""Regression tests for document replacement and bulk-ingest safety."""

import asyncio
import io
import zipfile
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


async def test_fake_pdf_magic_is_parsed_before_qdrant(monkeypatch):
    async def unexpected_ensure():
        raise AssertionError("Qdrant must not be contacted for invalid input")

    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "text")
    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", unexpected_ensure)

    with pytest.raises(ValueError, match="Invalid PDF"):
        await ingestion.upload_document("fake.pdf", b"%PDF-fake")


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


@pytest.mark.parametrize("filename", ["fake.docx", "fake.epub"])
def test_to_markdown_rejects_wrong_zip_container(filename):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not the requested document format")

    with pytest.raises(ValueError, match="Invalid"):
        ingestion.to_markdown(filename, buffer.getvalue())


@pytest.mark.parametrize("marker", ["mimetype", "container", "opf"])
def test_to_markdown_rejects_incomplete_epub_container(marker):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if marker == "mimetype":
            archive.writestr("mimetype", "application/epub+zip")
        elif marker == "container":
            archive.writestr("META-INF/container.xml", "<container/>")
        else:
            archive.writestr("content.opf", "<package/>")

    with pytest.raises(ValueError, match="Invalid EPUB"):
        ingestion.to_markdown("fake.epub", buffer.getvalue())


@pytest.mark.parametrize("filename", ["broken.docx", "broken.epub"])
async def test_malformed_office_container_is_value_error_before_qdrant(
    filename, monkeypatch
):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if filename.endswith(".docx"):
            archive.writestr("[Content_Types].xml", "not xml")
            archive.writestr("word/document.xml", "not xml")
        else:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container><rootfiles><rootfile '
                'full-path="content.opf"/></rootfiles></container>',
            )
            archive.writestr("content.opf", "not xml")

    async def unexpected_ensure():
        raise AssertionError("Qdrant must not be contacted for invalid input")

    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: None)
    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", unexpected_ensure)

    with pytest.raises(ValueError, match="Invalid"):
        await ingestion.upload_document(filename, buffer.getvalue())


@pytest.mark.parametrize(
    ("container_root", "opf_root"),
    [("not-container", "package"), ("container", "not-package")],
)
def test_epub_rejects_wrong_xml_roots(container_root, opf_root):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            f"<{container_root}><rootfile full-path='content.opf'/></{container_root}>",
        )
        archive.writestr("content.opf", f"<{opf_root}/>")

    with pytest.raises(ValueError, match="Invalid EPUB"):
        ingestion.to_markdown("fake.epub", buffer.getvalue())


async def test_reingest_uses_distinct_complete_generations(monkeypatch):
    writes = []

    async def ensure_collection():
        return None

    async def upsert_points(points):
        writes.append(points)
        return len(points)

    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", ensure_collection)
    monkeypatch.setattr(ingestion.vectorstore, "upsert_points", upsert_points)
    monkeypatch.setattr(ingestion.embeddings, "embed_texts", _embeddings)

    await ingestion.upload_document("notes.md", b"same text")
    await ingestion.upload_document("notes.md", b"same text")

    assert all("generation" in point["payload"] for batch in writes for point in batch)
    assert len({point["payload"]["generation"] for point in writes[0]}) == 1
    assert len({point["payload"]["generation"] for point in writes[1]}) == 1
    assert writes[0][0]["payload"]["generation"] != writes[1][0]["payload"]["generation"]
    assert {point["id"] for point in writes[0]}.isdisjoint(
        point["id"] for point in writes[1]
    )


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


async def test_resolve_corpus_scope_rejects_traversal_and_escaping_symlink(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="inside CORPUS_ROOT"):
        ingestion.resolve_corpus_scope(str(root), "../outside")
    with pytest.raises(ValueError, match="inside CORPUS_ROOT"):
        ingestion.resolve_corpus_scope(str(root), "escape")


def test_scan_corpus_tree_reports_same_candidates_as_bulk_validation(tmp_path):
    path = _book_dir(tmp_path) / "Biology Grade 9.md"
    path.write_text("cell theory", encoding="utf-8")

    scan = ingestion.scan_corpus_tree(str(tmp_path), subtree="School materials")

    assert [item["metadata"]["source"] for item in scan["candidates"]] == [
        "School materials/Biology/en/Biology Grade 9.md"
    ]
    assert scan["skipped"] == []
    assert scan["counts_by_type"] == {"textbook": 1}
    assert scan["counts_by_language"] == {"en": 1}
    assert scan["duplicate_lab_ids"] == []


def test_scan_corpus_tree_skips_supported_file_symlink_that_escapes_root(tmp_path):
    root = tmp_path / "corpus"
    book_dir = _book_dir(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "Biology Grade 9.md"
    target.write_text("cell theory", encoding="utf-8")
    (book_dir / "Biology Grade 9.md").symlink_to(target)

    scan = ingestion.scan_corpus_tree(str(root))

    assert scan["candidates"] == []
    assert scan["counts_by_type"] == {}
    assert scan["counts_by_language"] == {}
    assert scan["skipped"] == [
        {
            "source": "School materials/Biology/en/Biology Grade 9.md",
            "error": "Corpus file must remain inside CORPUS_ROOT",
        }
    ]
    assert scan["errors"] == scan["skipped"]


def test_scan_corpus_tree_reports_duplicate_lab_ids(tmp_path):
    lab_dir = _lab_dir(tmp_path)
    (lab_dir / "Lab work 1.md").write_text("first", encoding="utf-8")
    (lab_dir / "Lab work No. 1.md").write_text("second", encoding="utf-8")

    scan = ingestion.scan_corpus_tree(str(tmp_path))

    assert scan["candidates"] == []
    assert scan["duplicate_lab_ids"] == ["physics-10-en-01"]
    assert len(scan["errors"]) == 2


async def test_upload_document_reports_stages_and_stops_before_indexing(monkeypatch):
    stages = []
    indexed = []

    async def progress(stage):
        stages.append(stage)

    async def should_cancel():
        return stages == ["extracting", "embedding"]

    async def fake_embed(chunks):
        from types import SimpleNamespace

        return [
            SimpleNamespace(dense=[0.1], sparse_indices=[], sparse_values=[])
            for _ in chunks
        ]

    async def fake_upsert(points):
        indexed.extend(points)
        return len(points)

    monkeypatch.setattr(
        ingestion, "to_markdown", lambda *args, **kwargs: "usable educational text"
    )
    monkeypatch.setattr(ingestion.embeddings, "embed_texts", fake_embed)
    monkeypatch.setattr(ingestion.vectorstore, "upsert_points", fake_upsert)

    with pytest.raises(ingestion.IngestionCancelled):
        await ingestion.upload_document(
            "notes.md",
            b"content",
            progress=progress,
            should_cancel=should_cancel,
        )

    assert stages == ["extracting", "embedding"]
    assert indexed == []


async def test_bulk_ingest_rejects_missing_root():
    with pytest.raises(ValueError, match="Corpus root"):
        await ingestion.bulk_ingest_tree("/definitely/not/a/corpus")


@pytest.mark.parametrize("with_unrecognised_file", [False, True])
async def test_full_bulk_ingest_does_not_prune_without_valid_candidates(
    tmp_path, monkeypatch, with_unrecognised_file
):
    if with_unrecognised_file:
        (tmp_path / "misc.md").write_text("not a corpus path", encoding="utf-8")

    async def unexpected_list():
        raise AssertionError("invalid corpus snapshots must not trigger pruning")

    monkeypatch.setattr(ingestion.vectorstore, "list_documents", unexpected_list)

    summary = await ingestion.bulk_ingest_tree(str(tmp_path))

    assert summary["pruned"] == 0


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

    assert uploads == []
    assert deleted == []
    assert summary["skipped"] == 2
    assert summary["pruned"] == 0
    assert len(summary["errors"]) == 2
    assert all("Duplicate lab_id" in error["error"] for error in summary["errors"])


async def test_full_bulk_ingest_does_not_prune_skipped_existing_files(
    tmp_path, monkeypatch
):
    path = _lab_dir(tmp_path) / "Electrolyte current conditions.md"
    path.write_text("procedure", encoding="utf-8")
    meta = corpus_meta.parse_path(str(path), corpus_root=str(tmp_path))
    deleted = []

    async def list_documents():
        return [
            {
                "file_id": ingestion._doc_id(meta["source"]),
                "source_path": meta["source"],
            }
        ]

    async def delete_document(doc_id):
        deleted.append(doc_id)
        return True

    monkeypatch.setattr(ingestion.vectorstore, "list_documents", list_documents)
    monkeypatch.setattr(ingestion.vectorstore, "delete_document", delete_document)

    summary = await ingestion.bulk_ingest_tree(str(tmp_path))

    assert deleted == []
    assert summary["pruned"] == 0


async def test_full_bulk_ingest_prunes_removed_documents_only_when_requested(
    tmp_path, monkeypatch
):
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

    assert deleted == []
    assert summary["pruned"] == 0

    summary = await ingestion.bulk_ingest_tree(str(tmp_path), prune=True)

    assert deleted == ["stale-corpus-id"]
    assert summary["pruned"] == 1


async def test_bulk_ingest_rejects_prune_with_only(tmp_path):
    with pytest.raises(ValueError, match="prune cannot be combined"):
        await ingestion.bulk_ingest_tree(str(tmp_path), only="Biology", prune=True)


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
