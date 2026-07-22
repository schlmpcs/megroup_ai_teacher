from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.services import ingestion_jobs


@pytest.fixture
def job_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_jobs.settings, "INGESTION_DATA_DIR", str(tmp_path))
    ingestion_jobs.initialize()
    return tmp_path


def test_initialize_creates_versioned_schema(job_dir):
    assert (job_dir / "jobs.sqlite3").is_file()
    with ingestion_jobs.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_claim_next_job_is_fifo_and_atomic(job_dir):
    first = ingestion_jobs.enqueue_corpus_job({"subtree": "A", "ocr": False, "prune": False})
    second = ingestion_jobs.enqueue_corpus_job({"subtree": "B", "ocr": False, "prune": False})

    claimed = ingestion_jobs.claim_next_job("worker-1")

    assert claimed["id"] == first["id"]
    assert claimed["status"] == "running"
    assert ingestion_jobs.get_job(second["id"])["status"] == "queued"


def test_recover_interrupted_jobs_marks_nonterminal_items_failed(job_dir):
    job = ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": "uploads/job-1/item-1",
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    ingestion_jobs.claim_next_job("worker-1")
    ingestion_jobs.update_item("item-1", status="running", stage="embedding")

    assert ingestion_jobs.recover_interrupted_jobs() == 1
    recovered = ingestion_jobs.get_job(job["id"])
    assert recovered["status"] == "failed"
    assert recovered["items"][0]["status"] == "failed"
    assert "interrupted" in recovered["items"][0]["error"].lower()


def test_cancel_queued_upload_marks_pending_items_cancelled(job_dir):
    job = ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": "uploads/job-1/item-1",
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )

    cancelled = ingestion_jobs.request_cancel(job["id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert cancelled["items"][0]["status"] == "cancelled"
    assert cancelled["items"][0]["stage"] == "done"


def test_retry_upload_job_copies_only_failed_and_cancelled_items(job_dir):
    upload_dir = job_dir / "uploads" / "job-1"
    upload_dir.mkdir(parents=True)
    items = []
    for position, item_id in enumerate(("done", "failed", "cancelled")):
        stored_path = f"uploads/job-1/{item_id}"
        (job_dir / stored_path).write_text(item_id, encoding="utf-8")
        items.append(
            {
                "id": item_id,
                "position": position,
                "filename": f"{item_id}.md",
                "relative_path": f"{item_id}.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        )
    ingestion_jobs.enqueue_upload_job("job-1", items)
    ingestion_jobs.update_item("done", status="completed", stage="done")
    ingestion_jobs.update_item("failed", status="failed", stage="done", error="boom")
    ingestion_jobs.update_item("cancelled", status="cancelled", stage="done", error="cancelled")
    ingestion_jobs.finish_job("job-1", status="partial")

    original = ingestion_jobs.get_job("job-1")
    retry = ingestion_jobs.retry_job(original["id"])

    assert retry["retry_of"] == original["id"]
    assert [item["filename"] for item in retry["items"]] == ["failed.md", "cancelled.md"]
    assert [item["status"] for item in retry["items"]] == ["pending", "pending"]
    copied = [job_dir / item["stored_path"] for item in retry["items"]]
    assert [path.read_text(encoding="utf-8") for path in copied] == ["failed", "cancelled"]


def test_cancel_running_job_is_idempotent_when_already_requested(job_dir):
    job = ingestion_jobs.enqueue_corpus_job({"subtree": "A", "ocr": False, "prune": False})
    ingestion_jobs.claim_next_job("worker-1")

    first = ingestion_jobs.request_cancel(job["id"])
    second = ingestion_jobs.request_cancel(job["id"])

    assert first["status"] == "running"
    assert first["cancel_requested"] is True
    assert second["status"] == "running"
    assert second["cancel_requested"] is True


def test_cancel_rejects_terminal_job(job_dir):
    job = ingestion_jobs.enqueue_corpus_job({"subtree": "A", "ocr": False, "prune": False})
    ingestion_jobs.finish_job(job["id"], status="completed")

    with pytest.raises(ValueError, match="queued or running"):
        ingestion_jobs.request_cancel(job["id"])


def test_retry_rejects_non_retryable_statuses(job_dir):
    queued = ingestion_jobs.enqueue_corpus_job({"subtree": "A", "ocr": False, "prune": False})
    running = ingestion_jobs.enqueue_corpus_job({"subtree": "B", "ocr": False, "prune": False})
    ingestion_jobs.claim_next_job("worker-1")
    completed = ingestion_jobs.enqueue_corpus_job({"subtree": "C", "ocr": False, "prune": False})
    ingestion_jobs.finish_job(completed["id"], status="completed")

    for job_id in (queued["id"], running["id"], completed["id"]):
        with pytest.raises(ValueError, match="failed, partial, or cancelled"):
            ingestion_jobs.retry_job(job_id)


def test_retry_upload_job_rejects_without_failed_or_cancelled_items(job_dir):
    upload_dir = job_dir / "uploads" / "job-1"
    upload_dir.mkdir(parents=True)
    stored_path = "uploads/job-1/item-1"
    (job_dir / stored_path).write_text("content", encoding="utf-8")
    ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    ingestion_jobs.update_item("item-1", status="completed", stage="done")
    ingestion_jobs.finish_job("job-1", status="partial")

    with pytest.raises(ValueError, match="no failed or cancelled items"):
        ingestion_jobs.retry_job("job-1")


def test_retry_upload_job_cleans_copied_files_when_enqueue_fails(job_dir, monkeypatch):
    upload_dir = job_dir / "uploads" / "job-1"
    upload_dir.mkdir(parents=True)
    stored_path = "uploads/job-1/item-1"
    (job_dir / stored_path).write_text("content", encoding="utf-8")
    ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    ingestion_jobs.update_item("item-1", status="failed", stage="done", error="boom")
    ingestion_jobs.finish_job("job-1", status="failed", error="boom")

    def boom(*args, **kwargs):
        raise RuntimeError("db write failed")

    monkeypatch.setattr(ingestion_jobs, "enqueue_upload_job", boom)

    with pytest.raises(RuntimeError, match="db write failed"):
        ingestion_jobs.retry_job("job-1")
    assert [path.name for path in (job_dir / "uploads").iterdir()] == ["job-1"]


def test_delete_rejects_active_job_and_removes_terminal_artifacts(job_dir):
    upload_dir = job_dir / "uploads" / "job-1"
    upload_dir.mkdir(parents=True)
    stored_path = "uploads/job-1/item-1"
    (job_dir / stored_path).write_text("content", encoding="utf-8")
    job = ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    with pytest.raises(ValueError, match="terminal"):
        ingestion_jobs.delete_job(job["id"])
    ingestion_jobs.finish_job(job["id"], status="failed", error="boom")
    assert ingestion_jobs.delete_job(job["id"]) is True
    assert ingestion_jobs.get_job(job["id"]) is None
    assert not upload_dir.exists()


def test_list_jobs_filters_and_paginates(job_dir):
    first = ingestion_jobs.enqueue_corpus_job({"subtree": "A", "ocr": False, "prune": False})
    second = ingestion_jobs.enqueue_corpus_job({"subtree": "B", "ocr": False, "prune": False})
    ingestion_jobs.finish_job(first["id"], status="failed", error="boom")

    assert [job["id"] for job in ingestion_jobs.list_jobs(status="queued")] == [second["id"]]
    page_one = ingestion_jobs.list_jobs(limit=1)
    page_two = ingestion_jobs.list_jobs(limit=1, offset=1)
    assert len(page_one) == len(page_two) == 1
    assert page_one[0]["id"] != page_two[0]["id"]


def test_worker_status_reports_fresh_and_stale_heartbeats(job_dir):
    ingestion_jobs.write_heartbeat("worker-1")
    fresh = ingestion_jobs.worker_status()
    assert fresh["online"] is True
    assert fresh["worker_id"] == "worker-1"
    stale = (datetime.now(UTC) - timedelta(seconds=16)).isoformat()
    with ingestion_jobs.connect() as connection:
        connection.execute("UPDATE worker_state SET heartbeat_at = ?", (stale,))
    stale_status = ingestion_jobs.worker_status()
    assert stale_status["online"] is False
    assert stale_status["worker_id"] == "worker-1"


def test_cleanup_removes_only_orphaned_old_tmp_directories(job_dir):
    old = job_dir / "tmp" / "old-job"
    new = job_dir / "tmp" / "new-job"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    cutoff = datetime.now(UTC) - timedelta(hours=25)
    old_timestamp = cutoff.timestamp()
    import os

    os.utime(old, (old_timestamp, old_timestamp))
    assert ingestion_jobs.cleanup_stale_tmp() == 1
    assert not old.exists()
    assert new.exists()


@pytest.mark.parametrize(
    ("stored_path", "expected"),
    [
        ("/tmp/escape", "absolute"),
        ("uploads/other-job/item-1", "job"),
        ("uploads/job-1/other-item", "item"),
        ("uploads/job-1/item-1/../../escape", "stored_path"),
    ],
)
def test_enqueue_upload_job_rejects_invalid_stored_paths(job_dir, stored_path, expected):
    with pytest.raises(ValueError, match=expected):
        ingestion_jobs.enqueue_upload_job(
            "job-1",
            [
                {
                    "id": "item-1",
                    "position": 0,
                    "filename": "one.md",
                    "relative_path": "one.md",
                    "stored_path": stored_path,
                    "source_path": None,
                    "metadata": None,
                    "doc_key": None,
                    "ocr": False,
                }
            ],
        )


def test_retry_upload_job_rejects_tampered_stored_path(job_dir):
    stored_path = "uploads/job-1/item-1"
    (job_dir / stored_path).parent.mkdir(parents=True, exist_ok=True)
    (job_dir / stored_path).write_text("content", encoding="utf-8")
    ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    ingestion_jobs.update_item("item-1", status="failed", stage="done", error="boom")
    ingestion_jobs.finish_job("job-1", status="failed", error="boom")
    with ingestion_jobs.connect() as connection:
        connection.execute(
            "UPDATE job_items SET stored_path = ? WHERE id = ?",
            ("../outside/item-1", "item-1"),
        )

    with pytest.raises(ValueError, match="stored_path"):
        ingestion_jobs.retry_job("job-1")
    assert [job["id"] for job in ingestion_jobs.list_jobs()] == ["job-1"]


def test_delete_job_keeps_row_when_artifact_cleanup_fails(job_dir, monkeypatch):
    upload_dir = job_dir / "uploads" / "job-1"
    upload_dir.mkdir(parents=True)
    stored_path = "uploads/job-1/item-1"
    (job_dir / stored_path).write_text("content", encoding="utf-8")
    ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    ingestion_jobs.finish_job("job-1", status="failed", error="boom")

    def _boom(path, *args, **kwargs):
        if Path(path).name.startswith("job-1.delete-"):
            raise OSError("disk error")
        return None

    monkeypatch.setattr(ingestion_jobs.shutil, "rmtree", _boom)

    with pytest.raises(OSError, match="disk error"):
        ingestion_jobs.delete_job("job-1")
    assert ingestion_jobs.get_job("job-1") is None
    assert any(path.name.startswith("job-1.delete-") for path in (job_dir / "uploads").iterdir())


def test_delete_job_restores_artifacts_when_sqlite_delete_fails(job_dir):
    upload_dir = job_dir / "uploads" / "job-1"
    tmp_dir = job_dir / "tmp" / "job-1"
    upload_dir.mkdir(parents=True)
    tmp_dir.mkdir(parents=True)
    stored_path = "uploads/job-1/item-1"
    (job_dir / stored_path).write_text("content", encoding="utf-8")
    (tmp_dir / "scratch.txt").write_text("tmp", encoding="utf-8")
    ingestion_jobs.enqueue_upload_job(
        "job-1",
        [
            {
                "id": "item-1",
                "position": 0,
                "filename": "one.md",
                "relative_path": "one.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
    )
    ingestion_jobs.finish_job("job-1", status="failed", error="boom")
    with ingestion_jobs.connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER jobs_abort_delete
            BEFORE DELETE ON jobs
            BEGIN
                SELECT RAISE(ABORT, 'db delete blocked');
            END;
            """
        )

    with pytest.raises(Exception, match="db delete blocked"):
        ingestion_jobs.delete_job("job-1")

    assert ingestion_jobs.get_job("job-1") is not None
    assert upload_dir.is_dir()
    assert tmp_dir.is_dir()
    assert (upload_dir / "item-1").read_text(encoding="utf-8") == "content"
    assert (tmp_dir / "scratch.txt").read_text(encoding="utf-8") == "tmp"


def test_cleanup_stale_tmp_preserves_terminal_job_directory(job_dir):
    job = ingestion_jobs.enqueue_corpus_job({"subtree": "A", "ocr": False, "prune": False})
    ingestion_jobs.finish_job(job["id"], status="failed", error="boom")
    tmp_dir = job_dir / "tmp" / job["id"]
    tmp_dir.mkdir(parents=True)
    cutoff = datetime.now(UTC) - timedelta(hours=25)
    import os

    os.utime(tmp_dir, (cutoff.timestamp(), cutoff.timestamp()))

    assert ingestion_jobs.cleanup_stale_tmp() == 0
    assert tmp_dir.exists()


def test_cleanup_stale_tmp_surfaces_removal_failure(job_dir, monkeypatch):
    old = job_dir / "tmp" / "old-job"
    old.mkdir(parents=True)
    cutoff = datetime.now(UTC) - timedelta(hours=25)
    import os

    os.utime(old, (cutoff.timestamp(), cutoff.timestamp()))

    def _boom(path, *args, **kwargs):
        if Path(path) == old:
            raise OSError("cannot remove")
        return None

    monkeypatch.setattr(ingestion_jobs.shutil, "rmtree", _boom)

    with pytest.raises(OSError, match="cannot remove"):
        ingestion_jobs.cleanup_stale_tmp()
    assert old.exists()


def test_missing_job_actions_raise_key_error(job_dir):
    with pytest.raises(KeyError):
        ingestion_jobs.request_cancel("missing")
    with pytest.raises(KeyError):
        ingestion_jobs.retry_job("missing")
