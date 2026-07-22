"""Unit tests for the corpus management CLI safeguards."""

import sys

import pytest

import scripts.manage_corpus as manage_corpus


async def _summary_with_errors(*, ocr, only=None, root=None):
    return {
        "root": root,
        "ready": 1,
        "empty": 0,
        "skipped": 0,
        "filtered": 0,
        "errors": [{"source": "bad.docx", "error": "boom"}],
        "total": 1,
    }


async def _summary_without_errors(*, ocr, only=None, root=None):
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

    async def _bulk_ingest_tree(root, *, ocr, only=None):
        called["args"] = {"root": root, "ocr": ocr, "only": only}
        return await _summary_without_errors(ocr=ocr, only=only, root=root)

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", "corpus-root"])

    manage_corpus.main()

    assert called["args"] == {"root": "corpus-root", "ocr": True, "only": None}
    assert "Bulk ingest of corpus-root: 2 ready, 0 empty, 0 skipped, 0 filtered, 0 errors (of 2 files)" in capsys.readouterr().out


def test_bulk_ingest_accepts_no_ocr_override(monkeypatch):
    called = {}

    async def _bulk_ingest_tree(root, *, ocr, only=None):
        called["args"] = {"root": root, "ocr": ocr, "only": only}
        return await _summary_without_errors(ocr=ocr, only=only, root=root)

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest", "--no-ocr"])

    manage_corpus.main()

    assert called["args"] == {"root": manage_corpus.settings.CORPUS_ROOT, "ocr": False, "only": None}


def test_bulk_ingest_exits_nonzero_when_summary_contains_errors(monkeypatch, capsys):
    async def _bulk_ingest_tree(root, *, ocr, only=None):
        return await _summary_with_errors(ocr=ocr, only=only, root=root)

    monkeypatch.setattr(manage_corpus.settings, "OCR_ENABLED", False)
    monkeypatch.setattr(manage_corpus.ingestion, "bulk_ingest_tree", _bulk_ingest_tree)
    monkeypatch.setattr(sys, "argv", ["manage_corpus", "bulk-ingest"])

    with pytest.raises(SystemExit) as excinfo:
        manage_corpus.main()

    assert excinfo.value.code == 1
    out = capsys.readouterr()
    assert "Bulk ingest of" in out.out
    assert "ERROR  bad.docx: boom" in out.err
