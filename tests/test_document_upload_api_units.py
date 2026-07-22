import pytest
from fastapi import HTTPException

import app.api.routes as routes
from app.services.errors import LLMTimeoutError


class _ChunkedUpload:
    def __init__(self, *chunks: bytes):
        self._chunks = list(chunks)
        self.read_sizes = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size == -1:
            remaining = b"".join(self._chunks)
            self._chunks.clear()
            return remaining
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_read_upload_rejects_oversize_before_consuming_all_chunks():
    file = _ChunkedUpload(b"ab", b"cd", b"ef", b"gh")

    with pytest.raises(HTTPException) as exc_info:
        await routes._read_upload(file, max_bytes=5, chunk_size=2)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "File exceeds maximum size of 5 bytes"
    assert file.read_sizes == [2, 2, 2]


@pytest.mark.asyncio
async def test_read_upload_keeps_empty_file_response():
    file = _ChunkedUpload()

    with pytest.raises(HTTPException) as exc_info:
        await routes._read_upload(file, max_bytes=5, chunk_size=2)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Uploaded file is empty"


def test_admin_documents_uses_document_upload_limit(client, auth, monkeypatch):
    payload = b"x" * 20
    captured = {}

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        captured["filename"] = filename
        captured["raw"] = raw
        return {
            "file_id": "doc-id",
            "filename": filename,
            "status": "ready",
            "chunks": 1,
        }

    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)
    monkeypatch.setattr(routes.settings, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(routes.settings, "MAX_DOCUMENT_UPLOAD_BYTES", 100)

    response = client.post(
        "/admin/documents",
        files={"file": ("notes.md", payload, "text/markdown")},
        headers=auth,
    )

    assert response.status_code == 201
    assert response.json() == {
        "file_id": "doc-id",
        "filename": "notes.md",
        "status": "ready",
        "chunks": 1,
    }
    assert captured == {"filename": "notes.md", "raw": payload}


def test_admin_documents_uses_configured_ocr_when_omitted(client, auth, monkeypatch):
    captured = {}

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        captured["ocr"] = ocr
        return {
            "file_id": "doc-id",
            "filename": filename,
            "status": "ready",
            "chunks": 1,
        }

    monkeypatch.setattr(routes.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)

    response = client.post(
        "/admin/documents",
        files={"file": ("notes.md", b"notes", "text/markdown")},
        headers=auth,
    )

    assert response.status_code == 201
    assert captured["ocr"] is True


@pytest.mark.parametrize(
    ("configured", "requested", "expected"),
    [(True, "false", False), (False, "true", True)],
)
def test_admin_documents_explicit_ocr_overrides_setting(
    client, auth, monkeypatch, configured, requested, expected
):
    captured = {}

    async def _upload(filename, raw, metadata=None, doc_key=None, ocr=False, **_):
        captured["ocr"] = ocr
        return {
            "file_id": "doc-id",
            "filename": filename,
            "status": "ready",
            "chunks": 1,
        }

    monkeypatch.setattr(routes.settings, "OCR_ENABLED", configured)
    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)

    response = client.post(
        "/admin/documents",
        files={"file": ("notes.md", b"notes", "text/markdown")},
        data={"ocr": requested},
        headers=auth,
    )

    assert response.status_code == 201
    assert captured["ocr"] is expected


def test_admin_documents_clears_cache_after_ambiguous_write(
    client, auth, monkeypatch
):
    clears = []

    async def _upload(*args, **kwargs):
        raise LLMTimeoutError("write outcome unknown")

    monkeypatch.setattr(routes.ingestion, "upload_document", _upload)
    monkeypatch.setattr(routes, "clear_answer_cache", lambda: clears.append(True))

    response = client.post(
        "/admin/documents",
        files={"file": ("notes.md", b"notes", "text/markdown")},
        headers=auth,
    )

    assert response.status_code == 504
    assert clears == [True]
