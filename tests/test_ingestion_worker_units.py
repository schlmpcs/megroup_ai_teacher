from pathlib import Path

import pytest

from app.services import ingestion_jobs, ingestion_worker


@pytest.fixture
def worker_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_jobs.settings, "INGESTION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ingestion_worker.settings, "INGESTION_DATA_DIR", str(tmp_path))
    ingestion_jobs.initialize()
    return tmp_path


def queue_upload(worker_store: Path, job_id: str, filenames: list[str]) -> dict:
    upload_dir = worker_store / "uploads" / job_id
    upload_dir.mkdir(parents=True)
    items = []
    for position, filename in enumerate(filenames):
        item_id = f"item-{position}"
        stored_path = f"uploads/{job_id}/{item_id}"
        (worker_store / stored_path).write_text(filename, encoding="utf-8")
        items.append(
            {
                "id": item_id,
                "position": position,
                "filename": filename,
                "relative_path": filename,
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        )
    return ingestion_jobs.enqueue_upload_job(job_id, items)


async def successful_invalidation():
    return True


def test_worker_lock_is_exclusive(worker_store):
    with ingestion_worker.worker_lock():
        with pytest.raises(RuntimeError, match="already running"):
            with ingestion_worker.worker_lock():
                pass


def test_worker_lock_is_released(worker_store):
    with ingestion_worker.worker_lock():
        pass
    with ingestion_worker.worker_lock():
        pass


async def test_second_worker_fails_before_recovery(worker_store, monkeypatch):
    events = []
    monkeypatch.setattr(
        ingestion_worker.ingestion_jobs,
        "recover_interrupted_jobs",
        lambda: events.append("recover"),
    )

    with ingestion_worker.worker_lock():
        with pytest.raises(RuntimeError, match="already running"):
            await ingestion_worker.run_forever("worker-2")

    assert events == []


async def test_run_once_processes_one_upload_job_and_updates_stages(worker_store, monkeypatch):
    queue_upload(worker_store, "job-1", ["notes.md"])
    stages = []

    async def fake_upload(filename, content, **kwargs):
        await kwargs["progress"]("extracting")
        stages.append("extracting")
        await kwargs["progress"]("embedding")
        stages.append("embedding")
        await kwargs["progress"]("indexing")
        stages.append("indexing")
        stages.extend([filename, content.decode()])
        return {"file_id": "doc-1", "filename": filename, "status": "ready", "chunks": 3}

    monkeypatch.setattr(ingestion_worker.ingestion, "upload_document", fake_upload)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", successful_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    job = ingestion_jobs.get_job("job-1")
    assert job["status"] == "completed"
    assert job["items"][0]["stage"] == "done"
    assert job["items"][0]["chunks"] == 3
    assert stages == ["extracting", "embedding", "indexing", "notes.md", "notes.md"]


async def test_run_once_keeps_batch_running_after_one_file_fails(worker_store, monkeypatch):
    queue_upload(worker_store, "job-1", ["bad.md", "good.md"])

    async def fake_upload(filename, content, **kwargs):
        await kwargs["progress"]("extracting")
        if filename == "bad.md":
            raise RuntimeError("broken file")
        await kwargs["progress"]("embedding")
        await kwargs["progress"]("indexing")
        return {"file_id": "doc-good", "filename": filename, "status": "ready", "chunks": 1}

    monkeypatch.setattr(ingestion_worker.ingestion, "upload_document", fake_upload)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", successful_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    job = ingestion_jobs.get_job("job-1")
    assert job["status"] == "partial"
    assert [item["status"] for item in job["items"]] == ["failed", "completed"]


async def test_running_cancel_stops_before_next_safe_stage(worker_store, monkeypatch):
    queue_upload(worker_store, "job-1", ["cancel.md"])
    stages = []

    async def fake_upload(filename, content, **kwargs):
        await kwargs["progress"]("extracting")
        stages.append("extracting")
        ingestion_jobs.request_cancel("job-1")
        if await kwargs["should_cancel"]():
            raise ingestion_worker.ingestion.IngestionCancelled("cancelled")
        stages.append("indexing")
        raise AssertionError("cancel check should stop the item")

    monkeypatch.setattr(ingestion_worker.ingestion, "upload_document", fake_upload)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", successful_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    assert ingestion_jobs.get_job("job-1")["status"] == "cancelled"
    assert stages == ["extracting"]


async def test_corpus_job_scans_items_and_prunes_only_after_clean_run(worker_store, monkeypatch):
    corpus = worker_store / "corpus"
    source = corpus / "School materials" / "Biology" / "en" / "Biology Grade 9.md"
    source.parent.mkdir(parents=True)
    source.write_text("cell theory", encoding="utf-8")
    monkeypatch.setattr(ingestion_worker.settings, "CORPUS_ROOT", str(corpus))
    ingestion_jobs.enqueue_corpus_job({"subtree": "", "ocr": False, "prune": True})
    claimed_id = ingestion_jobs.list_jobs()[0]["id"]
    pruned_with = []
    scan_errors = []

    def fake_scan(root, *, subtree="", only=None):
        return {
            "root": str(corpus),
            "subtree": "",
            "total": 2,
            "filtered": 0,
            "candidates": [
                {
                    "path": str(source),
                    "metadata": {
                        "source": "School materials/Biology/en/Biology Grade 9.md",
                        "doc_type": "textbook",
                        "subject": "biology",
                        "grade": 9,
                        "lang": "en",
                    },
                    "doc_id": "doc-bio",
                }
            ],
            "skipped": [{"source": "misc.md", "error": "Unrecognised corpus path"}],
            "errors": list(scan_errors),
            "present_doc_ids": {"doc-bio", "doc-misc"},
        }

    async def fake_upload(filename, content, **kwargs):
        await kwargs["progress"]("extracting")
        await kwargs["progress"]("embedding")
        await kwargs["progress"]("indexing")
        return {"file_id": "doc-bio", "filename": filename, "status": "ready", "chunks": 2}

    async def fake_prune(present_doc_ids):
        pruned_with.append(present_doc_ids)
        return 1

    monkeypatch.setattr(ingestion_worker.ingestion, "scan_corpus_tree", fake_scan)
    monkeypatch.setattr(ingestion_worker.ingestion, "upload_document", fake_upload)
    monkeypatch.setattr(ingestion_worker.ingestion, "prune_missing_corpus_documents", fake_prune)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", successful_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    job = ingestion_jobs.get_job(claimed_id)
    assert job["status"] == "completed"
    assert job["skipped_items"] == 1
    assert [item["status"] for item in job["items"]] == ["completed", "skipped"]
    assert pruned_with == [{"doc-bio", "doc-misc"}]

    scan_errors.append({"source": "bad.md", "error": "Incomplete metadata"})
    ingestion_jobs.enqueue_corpus_job({"subtree": "", "ocr": False, "prune": True})
    assert await ingestion_worker.run_once("worker-1") is True
    assert pruned_with == [{"doc-bio", "doc-misc"}]


async def test_cache_invalidation_failure_marks_successful_job_partial(worker_store, monkeypatch):
    queue_upload(worker_store, "job-1", ["notes.md"])

    async def fake_upload(filename, content, **kwargs):
        await kwargs["progress"]("indexing")
        return {"file_id": "doc-1", "filename": filename, "status": "ready", "chunks": 1}

    async def failed_invalidation():
        return False

    monkeypatch.setattr(ingestion_worker.ingestion, "upload_document", fake_upload)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", failed_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    job = ingestion_jobs.get_job("job-1")
    assert job["status"] == "partial"
    assert job["items"][0]["status"] == "completed"
    assert "cache" in job["warning"].lower()


async def test_run_forever_invalidates_after_recovering_interrupted_jobs(worker_store, monkeypatch):
    events = []

    async def fake_invalidate():
        events.append("invalidate")
        return True

    async def stop_after_startup(worker_id):
        events.append("poll")
        raise RuntimeError("stop")

    monkeypatch.setattr(ingestion_worker.ingestion_jobs, "recover_interrupted_jobs", lambda: 1)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", fake_invalidate)
    monkeypatch.setattr(ingestion_worker, "run_once", stop_after_startup)

    with pytest.raises(RuntimeError, match="stop"):
        await ingestion_worker.run_forever("worker-1")

    assert events == ["invalidate", "poll"]


async def test_run_once_fatal_exit_invalidates_and_cancels_pending_items(
    worker_store, monkeypatch
):
    queue_upload(worker_store, "job-1", ["started.md", "pending.md"])
    invalidations = []

    async def fatal_runner(job):
        ingestion_jobs.update_item(
            "item-0",
            status="running",
            stage="indexing",
            started_at=ingestion_jobs.now(),
        )
        raise RuntimeError("qdrant write crashed")

    async def failed_invalidation():
        invalidations.append(True)
        return False

    monkeypatch.setattr(ingestion_worker, "_run_upload_job", fatal_runner)
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", failed_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    job = ingestion_jobs.get_job("job-1")
    assert job["status"] == "failed"
    assert job["error"] == "qdrant write crashed"
    assert "cache" in job["warning"].lower()
    assert invalidations == [True]
    assert [item["status"] for item in job["items"]] == ["failed", "cancelled"]


async def test_prune_failure_invalidates_even_when_items_only_skipped(
    worker_store, monkeypatch
):
    corpus = worker_store / "corpus"
    source = corpus / "School materials" / "Biology" / "en" / "Empty Grade 9.md"
    source.parent.mkdir(parents=True)
    source.write_text("thin", encoding="utf-8")
    monkeypatch.setattr(ingestion_worker.settings, "CORPUS_ROOT", str(corpus))
    ingestion_jobs.enqueue_corpus_job({"subtree": "", "ocr": False, "prune": True})
    job_id = ingestion_jobs.list_jobs()[0]["id"]
    pruned = []
    invalidations = []

    def fake_scan(root, *, subtree="", only=None):
        return {
            "root": str(corpus),
            "subtree": "",
            "total": 1,
            "filtered": 0,
            "candidates": [
                {
                    "path": str(source),
                    "metadata": {
                        "source": "School materials/Biology/en/Empty Grade 9.md",
                        "doc_type": "textbook",
                        "subject": "biology",
                        "grade": 9,
                        "lang": "en",
                    },
                    "doc_id": "doc-empty",
                }
            ],
            "skipped": [],
            "errors": [],
            "present_doc_ids": {"doc-empty"},
        }

    async def empty_upload(filename, content, **kwargs):
        await kwargs["progress"]("extracting")
        return {"file_id": "doc-empty", "filename": filename, "status": "empty", "chunks": 0}

    async def failing_prune(present_doc_ids):
        pruned.append("deleted-one")
        raise RuntimeError("prune crashed")

    async def failed_invalidation():
        invalidations.append(True)
        return False

    monkeypatch.setattr(ingestion_worker.ingestion, "scan_corpus_tree", fake_scan)
    monkeypatch.setattr(ingestion_worker.ingestion, "upload_document", empty_upload)
    monkeypatch.setattr(
        ingestion_worker.ingestion,
        "prune_missing_corpus_documents",
        failing_prune,
    )
    monkeypatch.setattr(ingestion_worker, "invalidate_answer_cache", failed_invalidation)

    assert await ingestion_worker.run_once("worker-1") is True
    job = ingestion_jobs.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "prune crashed"
    assert job["completed_items"] == 0
    assert job["skipped_items"] == 1
    assert "cache" in job["warning"].lower()
    assert pruned == ["deleted-one"]
    assert invalidations == [True]
