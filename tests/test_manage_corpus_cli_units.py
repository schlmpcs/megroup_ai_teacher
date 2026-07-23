"""Unit tests for the corpus management CLI safeguards."""

import sys

import pytest

import scripts.manage_corpus as manage_corpus
from app.services.assistant_profiles import ASSISTANT_PROFILES


async def _summary_with_errors(*, ocr, only=None, root=None, collection_name=None):
    return {
        "root": root,
        "ready": 1,
        "empty": 0,
        "skipped": 0,
        "filtered": 0,
        "errors": [{"source": "bad.docx", "error": "boom"}],
        "total": 1,
    }


async def _summary_without_errors(*, ocr, only=None, root=None, collection_name=None):
    return {
        "root": root,
        "ready": 2,
        "empty": 0,
        "skipped": 0,
        "filtered": 0,
        "errors": [],
        "total": 2,
    }


def test_bulk_ingest_uses_configured_ocr_when_flag_is_omitted(monkeypatch, capsys):
    called = {}

    async def _bulk_ingest_tree(root, *, ocr, only=None, prune=False, collection_name=None):
        called["args"] = {
            "root": root,
            "ocr": ocr,
            "only": only,
            "prune": prune,
            "collection_name": collection_name,
        }
        return await _summary_without_errors(
            ocr=ocr,
            only=only,
            root=root,
            collection_name=collection_name,
        )

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", "corpus-root"])

    manage_corpus.main()

    assert called["args"] == {
        "root": "corpus-root",
        "ocr": True,
        "only": None,
        "prune": False,
        "collection_name": "school_kb",
    }
    assert "Bulk ingest of corpus-root: 2 ready, 0 empty, 0 skipped, 0 filtered, 0 errors (of 2 files)" in capsys.readouterr().out


def test_bulk_ingest_accepts_no_ocr_override(monkeypatch):
    called = {}

    async def _bulk_ingest_tree(root, *, ocr, only=None, prune=False, collection_name=None):
        called["args"] = {
            "root": root,
            "ocr": ocr,
            "only": only,
            "prune": prune,
            "collection_name": collection_name,
        }
        return await _summary_without_errors(
            ocr=ocr,
            only=only,
            root=root,
            collection_name=collection_name,
        )

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", "--no-ocr"])

    manage_corpus.main()

    assert called["args"] == {
        "root": manage_corpus.settings.CORPUS_ROOT,
        "ocr": False,
        "only": None,
        "prune": False,
        "collection_name": "school_kb",
    }


def test_bulk_ingest_accepts_ocr_override(monkeypatch):
    called = {}

    async def _bulk_ingest_tree(root, *, ocr, only=None, prune=False, collection_name=None):
        called["args"] = {
            "root": root,
            "ocr": ocr,
            "only": only,
            "prune": prune,
            "collection_name": collection_name,
        }
        return await _summary_without_errors(
            ocr=ocr,
            only=only,
            root=root,
            collection_name=collection_name,
        )

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", False)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", "--ocr"])

    manage_corpus.main()

    assert called["args"] == {
        "root": manage_corpus.settings.CORPUS_ROOT,
        "ocr": True,
        "only": None,
        "prune": False,
        "collection_name": "school_kb",
    }


def test_bulk_ingest_accepts_explicit_prune(monkeypatch):
    called = {}

    async def _bulk_ingest_tree(root, *, ocr, only=None, prune=False, collection_name=None):
        called["prune"] = prune
        return await _summary_without_errors(
            ocr=ocr,
            only=only,
            root=root,
            collection_name=collection_name,
        )

    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", "--prune"])

    manage_corpus.main()

    assert called["prune"] is True


def test_bulk_ingest_rejects_prune_with_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["manage_corpus", "bulk-ingest", "--prune", "--only", "Biology"],
    )

    with pytest.raises(SystemExit) as excinfo:
        manage_corpus.main()

    assert excinfo.value.code == 2


def test_bulk_ingest_exits_nonzero_when_summary_contains_errors(monkeypatch, capsys):
    async def _bulk_ingest_tree(root, *, ocr, only=None, prune=False, collection_name=None):
        return await _summary_with_errors(
            ocr=ocr,
            only=only,
            root=root,
            collection_name=collection_name,
        )

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", False)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest"])

    with pytest.raises(SystemExit) as excinfo:
        manage_corpus.main()

    assert excinfo.value.code == 1
    out = capsys.readouterr()
    assert "Bulk ingest of" in out.out
    assert "ERROR  bad.docx: boom" in out.err


def test_create_collection_uses_selected_profile_collection(monkeypatch, capsys):
    captured = {}

    async def _ensure_collection(*, collection_name=None):
        captured["collection_name"] = collection_name

    monkeypatch.setattr(manage_corpus.vectorstore, "ensure_collection", _ensure_collection)
    monkeypatch.setattr(
        sys,
        "argv",
        ["manage_corpus", "create-collection", "--assistant-type", "other_assistant"],
    )

    manage_corpus.main()

    assert captured["collection_name"] == "other_assistant_kb"
    assert "Collection ready: other_assistant_kb" in capsys.readouterr().out


def test_upload_list_status_and_delete_use_selected_profile_collection(monkeypatch, capsys, tmp_path):
    calls = {}
    path = tmp_path / "notes.md"
    path.write_text("notes", encoding="utf-8")

    async def _upload_document(filename, content, **kwargs):
        calls["upload"] = kwargs["collection_name"]
        return {"status": "ready", "chunks": 1, "file_id": "doc-1", "filename": filename}

    async def _list_documents(**kwargs):
        calls["list"] = kwargs["collection_name"]
        return [{"status": "ready", "file_id": "doc-1", "filename": "notes.md", "chunks": 1}]

    async def _corpus_status(**kwargs):
        calls["status"] = kwargs["collection_name"]
        return {"status": "ready"}

    async def _delete_document(doc_id, **kwargs):
        calls["delete"] = (doc_id, kwargs["collection_name"])
        return True

    monkeypatch.setattr(manage_corpus.ingestion, "upload_document", _upload_document)
    monkeypatch.setattr(manage_corpus.ingestion, "list_documents", _list_documents)
    monkeypatch.setattr(manage_corpus.ingestion, "corpus_status", _corpus_status)
    monkeypatch.setattr(manage_corpus.ingestion, "delete_document", _delete_document)

    for argv in (
        ["manage_corpus", "upload", str(path), "--assistant-type", "other_assistant"],
        ["manage_corpus", "list", "--assistant-type", "other_assistant"],
        ["manage_corpus", "status", "--assistant-type", "other_assistant"],
        ["manage_corpus", "delete", "doc-1", "--assistant-type", "other_assistant"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        manage_corpus.main()

    assert calls == {
        "upload": "other_assistant_kb",
        "list": "other_assistant_kb",
        "status": "other_assistant_kb",
        "delete": ("doc-1", "other_assistant_kb"),
    }


def test_bulk_ingest_and_manifest_default_to_profile_corpus_root(monkeypatch):
    calls = {}
    profile = ASSISTANT_PROFILES["other_assistant"]

    async def _bulk_ingest_tree(root, *, ocr, only=None, prune=False, collection_name=None):
        calls["bulk"] = {
            "root": root,
            "ocr": ocr,
            "only": only,
            "prune": prune,
            "collection_name": collection_name,
        }
        return await _summary_without_errors(ocr=ocr, only=only, root=root)

    def _write_manifest(root, out):
        calls["manifest"] = {"root": root, "out": out}
        return {
            "labs": {},
            "textbooks": 0,
            "textbooks_by_language": {},
            "missing_metadata": [],
        }

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", False)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(manage_corpus.ingestion, "write_manifest", _write_manifest)

    monkeypatch.setattr(
        sys,
        "argv",
        ["manage_corpus", "bulk-ingest", "--assistant-type", profile.assistant_type],
    )
    manage_corpus.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_corpus",
            "gen-manifest",
            "--assistant-type",
            profile.assistant_type,
            "--out",
            "labs.json",
        ],
    )
    manage_corpus.main()

    assert calls == {
        "bulk": {
            "root": profile.corpus_root,
            "ocr": False,
            "only": None,
            "prune": False,
            "collection_name": profile.qdrant_collection,
        },
        "manifest": {"root": profile.corpus_root, "out": "labs.json"},
    }


def test_explicit_empty_root_is_not_replaced_by_profile_root(monkeypatch):
    calls = {}

    async def _bulk_ingest_tree(
        root, *, ocr, only=None, prune=False, collection_name=None
    ):
        calls["bulk"] = root
        return await _summary_without_errors(ocr=ocr, only=only, root=root)

    def _write_manifest(root, out):
        calls["manifest"] = root
        return {
            "labs": {},
            "textbooks": 0,
            "textbooks_by_language": {},
            "missing_metadata": [],
        }

    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(manage_corpus.ingestion, "write_manifest", _write_manifest)

    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", ""])
    manage_corpus.main()

    monkeypatch.setattr(sys, "argv", ["manage_corpus", "gen-manifest", ""])
    manage_corpus.main()

    assert calls == {"bulk": "", "manifest": ""}


def test_cli_rejects_unknown_assistant_type_before_work(monkeypatch):
    called = []

    def _get_profile(name=None):
        called.append(name)
        raise ValueError("Unknown assistant_type: 'missing'")

    monkeypatch.setattr(manage_corpus, "get_assistant_profile", _get_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        ["manage_corpus", "status", "--assistant-type", "missing"],
    )

    with pytest.raises(SystemExit) as excinfo:
        manage_corpus.main()

    assert excinfo.value.code == 2
    assert called == ["missing"]
