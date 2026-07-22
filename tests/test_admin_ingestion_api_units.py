import json
from pathlib import Path

import pytest

import app.api.admin_routes as admin_routes
from app.services import ingestion_jobs


@pytest.fixture(autouse=True)
def isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_jobs.settings, "INGESTION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(admin_routes.settings, "INGESTION_DATA_DIR", str(tmp_path))
    ingestion_jobs.initialize()


def test_preview_suggests_folder_metadata(client, admin_auth):
    response = client.post(
        "/admin/ingestion/preview",
        headers=admin_auth,
        json={
            "paths": [
                "Laboratory works/Physics/Physics Grade 10/en/Lab work 2.docx"
            ]
        },
    )
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["metadata"]["lab_id"] == "physics-10-en-02"
    assert item["doc_key"] == "admin_uploads/lab_instruction/physics-10-en-02"
    assert item["errors"] == []


def test_preview_marks_duplicate_suggested_identity(client, admin_auth):
    path = "Laboratory works/Physics/Physics Grade 10/en/Lab work 2.docx"
    response = client.post(
        "/admin/ingestion/preview",
        headers=admin_auth,
        json={"paths": [path, path]},
    )
    assert response.status_code == 200
    assert all(
        "duplicate document identity" in item["errors"][0].lower()
        for item in response.json()["items"]
    )


def test_preview_uses_relative_path_identity_for_general_uploads(client, admin_auth):
    response = client.post(
        "/admin/ingestion/preview",
        headers=admin_auth,
        json={"paths": ["folder-a/notes.md", "folder-b/notes.md", "folder-a/notes.md"]},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["doc_key"] == "admin_uploads/general/folder-a/notes.md"
    assert items[1]["doc_key"] == "admin_uploads/general/folder-b/notes.md"
    assert "duplicate document identity" in items[0]["errors"][0].lower()
    assert "duplicate document identity" in items[2]["errors"][0].lower()
    assert items[1]["errors"] == []


def test_preview_preserves_legacy_identity_for_root_level_general_uploads(client, admin_auth):
    response = client.post(
        "/admin/ingestion/preview",
        headers=admin_auth,
        json={"paths": ["notes.md", "folder/notes.md", "notes.md"]},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["doc_key"] is None
    assert items[1]["doc_key"] == "admin_uploads/general/folder/notes.md"
    assert "duplicate document identity" in items[0]["errors"][0].lower()
    assert "duplicate document identity" in items[2]["errors"][0].lower()
    assert items[1]["errors"] == []


def test_upload_job_streams_files_and_returns_202(client, admin_auth):
    manifest = [
        {
            "filename": "Physics 8.md",
            "relative_path": "Physics 8.md",
            "doc_type": "textbook",
            "subject": "physics",
            "grade": 8,
            "lang": "ru",
            "lab_number": None,
            "ocr": False,
        }
    ]
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("Physics 8.md", b"theory", "text/markdown"))],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "upload"
    assert job["status"] == "queued"
    assert job["items"][0]["doc_key"] == "admin_uploads/textbook/physics/8/ru/Physics 8.md"
    stored = Path(ingestion_jobs.settings.INGESTION_DATA_DIR) / job["items"][0]["stored_path"]
    assert stored.read_bytes() == b"theory"


def test_upload_job_preserves_legacy_identity_for_root_level_general_upload(client, admin_auth):
    manifest = [{"filename": "notes.md", "relative_path": "notes.md", "ocr": False}]
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 202
    assert response.json()["items"][0]["doc_key"] is None


def test_upload_job_accepts_same_filename_from_different_relative_paths(client, admin_auth):
    manifest = [
        {"filename": "notes.md", "relative_path": relative_path, "ocr": False}
        for relative_path in ("folder-a/notes.md", "folder-b/notes.md")
    ]
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[
            ("files", ("notes.md", b"one", "text/markdown")),
            ("files", ("notes.md", b"two", "text/markdown")),
        ],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 202
    items = response.json()["items"]
    assert [item["doc_key"] for item in items] == [
        "admin_uploads/general/folder-a/notes.md",
        "admin_uploads/general/folder-b/notes.md",
    ]


@pytest.mark.parametrize("relative_path", ["/notes.md", "../notes.md", "folder/../notes.md"])
def test_upload_job_rejects_invalid_relative_path(client, admin_auth, relative_path):
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={
            "manifest": json.dumps(
                [{"filename": "notes.md", "relative_path": relative_path, "ocr": False}]
            )
        },
    )
    assert response.status_code == 400
    assert "relative_path" in response.json()["detail"]


def test_upload_job_rejects_relative_path_leaf_mismatch(client, admin_auth):
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={
            "manifest": json.dumps(
                [{"filename": "notes.md", "relative_path": "folder/other.md", "ocr": False}]
            )
        },
    )
    assert response.status_code == 400
    assert "relative_path" in response.json()["detail"]


def test_upload_job_rejects_duplicate_identity_and_cleans_tmp(client, admin_auth):
    manifest = [
        {
            "filename": name,
            "relative_path": name,
            "doc_type": "lab_instruction",
            "subject": "physics",
            "grade": 10,
            "lang": "ru",
            "lab_number": 2,
            "ocr": False,
        }
        for name in ("first.docx", "renamed.docx")
    ]
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[
            ("files", ("first.docx", b"one", "application/octet-stream")),
            ("files", ("renamed.docx", b"two", "application/octet-stream")),
        ],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 400
    assert "duplicate document identity" in response.json()["detail"].lower()
    assert list((Path(ingestion_jobs.settings.INGESTION_DATA_DIR) / "tmp").iterdir()) == []


def test_corpus_preview_rejects_escape(client, admin_auth, tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setattr(admin_routes.settings, "CORPUS_ROOT", str(corpus))
    response = client.post(
        "/admin/ingestion/corpus/preview",
        headers=admin_auth,
        json={"subtree": "../outside", "ocr": False, "prune": False},
    )
    assert response.status_code == 400


def test_corpus_preview_returns_scan_summaries(client, admin_auth, monkeypatch):
    monkeypatch.setattr(
        admin_routes.ingestion,
        "scan_corpus_tree",
        lambda *args, **kwargs: {
            "root": "/corpus",
            "subtree": "",
            "total": 3,
            "candidates": [{"metadata": {"doc_type": "textbook", "lang": "en"}}],
            "skipped": [],
            "present_doc_ids": {"doc-1"},
            "duplicate_lab_ids": ["physics-10-en-02"],
            "counts_by_type": {"textbook": 1},
            "counts_by_language": {"en": 1},
        },
    )

    async def no_documents():
        return []

    monkeypatch.setattr(admin_routes.ingestion, "list_documents", no_documents)
    response = client.post(
        "/admin/ingestion/corpus/preview",
        headers=admin_auth,
        json={"subtree": "", "ocr": False, "prune": False},
    )
    assert response.status_code == 200
    assert response.json()["counts_by_type"] == {"textbook": 1}
    assert response.json()["counts_by_language"] == {"en": 1}
    assert response.json()["duplicate_lab_ids"] == ["physics-10-en-02"]


def test_prune_is_rejected_with_subtree(client, admin_auth):
    response = client.post(
        "/admin/ingestion/jobs/corpus",
        headers=admin_auth,
        json={"subtree": "Biology", "ocr": False, "prune": True},
    )
    assert response.status_code == 400


def test_job_lifecycle_endpoints(client, admin_auth):
    created = client.post(
        "/admin/ingestion/jobs/corpus",
        headers=admin_auth,
        json={"subtree": "", "ocr": False, "prune": False},
    ).json()
    assert client.get("/admin/ingestion/jobs", headers=admin_auth).json()["jobs"][0]["id"] == created["id"]
    cancelled = client.post(
        f"/admin/ingestion/jobs/{created['id']}/cancel", headers=admin_auth
    )
    assert cancelled.json()["status"] == "cancelled"
    deleted = client.delete(
        f"/admin/ingestion/jobs/{created['id']}", headers=admin_auth
    )
    assert deleted.json() == {"deleted": True, "job_id": created["id"]}


def test_job_lifecycle_conflicts(client, admin_auth):
    queued = client.post(
        "/admin/ingestion/jobs/corpus",
        headers=admin_auth,
        json={"subtree": "", "ocr": False, "prune": False},
    ).json()
    completed = client.post(
        "/admin/ingestion/jobs/corpus",
        headers=admin_auth,
        json={"subtree": "", "ocr": False, "prune": False},
    ).json()
    ingestion_jobs.finish_job(completed["id"], status="completed")

    retry_queued = client.post(
        f"/admin/ingestion/jobs/{queued['id']}/retry", headers=admin_auth
    )
    cancel_completed = client.post(
        f"/admin/ingestion/jobs/{completed['id']}/cancel", headers=admin_auth
    )

    assert retry_queued.status_code == 409
    assert cancel_completed.status_code == 409


def test_upload_job_rejects_malformed_manifest(client, admin_auth):
    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={"manifest": "{"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid upload manifest"


def test_upload_job_rejects_count_and_filename_mismatch(client, admin_auth):
    empty = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={"manifest": "[]"},
    )
    assert empty.status_code == 400
    manifest = [{"filename": "other.md", "relative_path": "other.md", "ocr": False}]
    mismatched = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={"manifest": json.dumps(manifest)},
    )
    assert mismatched.status_code == 400


def test_upload_job_enforces_file_and_batch_limits(client, admin_auth, monkeypatch):
    manifest = [{"filename": "one.md", "relative_path": "one.md", "ocr": False}]
    monkeypatch.setattr(admin_routes.settings, "MAX_DOCUMENT_UPLOAD_BYTES", 3)
    too_large = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("one.md", b"four", "text/markdown"))],
        data={"manifest": json.dumps(manifest)},
    )
    assert too_large.status_code == 413

    monkeypatch.setattr(admin_routes.settings, "MAX_DOCUMENT_UPLOAD_BYTES", 10)
    monkeypatch.setattr(admin_routes.settings, "INGESTION_BATCH_MAX_BYTES", 5)
    two_items = [
        {"filename": name, "relative_path": name, "ocr": False}
        for name in ("one.md", "two.md")
    ]
    too_large_batch = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[
            ("files", ("one.md", b"one", "text/markdown")),
            ("files", ("two.md", b"two", "text/markdown")),
        ],
        data={"manifest": json.dumps(two_items)},
    )
    assert too_large_batch.status_code == 413


def test_job_queries_validate_pagination_and_missing_ids(client, admin_auth):
    assert client.get("/admin/ingestion/jobs?limit=201", headers=admin_auth).status_code == 422
    missing = "0" * 32
    assert client.get(f"/admin/ingestion/jobs/{missing}", headers=admin_auth).status_code == 404
    assert client.post(f"/admin/ingestion/jobs/{missing}/retry", headers=admin_auth).status_code == 404


def test_ingestion_status_reports_offline_worker(client, admin_auth):
    response = client.get("/admin/ingestion/status", headers=admin_auth)
    assert response.status_code == 200
    assert response.json()["worker"]["online"] is False
