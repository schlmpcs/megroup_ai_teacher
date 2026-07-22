import asyncio
import fcntl
import logging
import os
import socket
from contextlib import contextmanager, suppress
from pathlib import Path

import httpx

from app.core.config import settings
from app.services import ingestion, ingestion_jobs

logger = logging.getLogger("assistant.ingestion_worker")
_HEARTBEAT_S = 5.0
_POLL_S = 1.0


@contextmanager
def worker_lock():
    lock_path = ingestion_jobs.data_dir() / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An ingestion worker is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        lock_file.close()


async def invalidate_answer_cache() -> bool:
    headers = {"Authorization": f"Bearer {settings.ADMIN_API_KEY}"}
    delays = (0.0, 0.5, 1.0)
    async with httpx.AsyncClient(
        base_url=settings.INGESTION_API_BASE_URL,
        timeout=10.0,
    ) as client:
        for delay in delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await client.post("/admin/cache/answers/clear", headers=headers)
                response.raise_for_status()
                return True
            except httpx.HTTPError:
                continue
    return False


def _item_file(item: dict) -> Path:
    if item.get("source_path"):
        root = Path(settings.CORPUS_ROOT).resolve(strict=True)
        path = Path(item["source_path"]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Corpus source file is missing or outside CORPUS_ROOT")
        return path
    root = ingestion_jobs.data_dir().resolve(strict=True)
    path = (root / item["stored_path"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Stored upload file is missing or outside ingestion data")
    return path


async def _run_item(job: dict, item: dict) -> tuple[bool, bool]:
    ingestion_jobs.update_item(
        item["id"],
        status="running",
        stage="extracting",
        started_at=ingestion_jobs.now(),
    )

    async def progress(stage: str) -> None:
        ingestion_jobs.update_item(item["id"], stage=stage)
        ingestion_jobs.update_job(
            job["id"],
            current_item=item["filename"],
            current_stage=stage,
        )

    async def should_cancel() -> bool:
        return ingestion_jobs.cancellation_requested(job["id"])

    reached_indexing = False
    try:

        async def tracked_progress(stage: str) -> None:
            nonlocal reached_indexing
            reached_indexing = reached_indexing or stage == "indexing"
            await progress(stage)

        content = await asyncio.to_thread(_item_file(item).read_bytes)
        result = await ingestion.upload_document(
            item["filename"],
            content,
            metadata=item["metadata"],
            doc_key=item["doc_key"],
            ocr=item["ocr"],
            progress=tracked_progress,
            should_cancel=should_cancel,
        )
    except ingestion.IngestionCancelled:
        ingestion_jobs.update_item(
            item["id"],
            status="cancelled",
            stage="done",
            error="Cancelled by administrator",
            finished_at=ingestion_jobs.now(),
        )
        return False, reached_indexing
    except Exception as exc:
        ingestion_jobs.update_item(
            item["id"],
            status="failed",
            stage="done",
            error=str(exc)[:2000],
            finished_at=ingestion_jobs.now(),
        )
        return False, reached_indexing

    if result["status"] == "empty":
        ingestion_jobs.update_item(
            item["id"],
            status="skipped",
            stage="done",
            file_id=result["file_id"],
            chunks=0,
            error="No usable text extracted",
            finished_at=ingestion_jobs.now(),
        )
        return False, False
    ingestion_jobs.update_item(
        item["id"],
        status="completed",
        stage="done",
        file_id=result["file_id"],
        chunks=result["chunks"],
        finished_at=ingestion_jobs.now(),
    )
    return True, True


def _terminal_status(job: dict) -> str:
    if job["cancel_requested"]:
        return "cancelled"
    if job["completed_items"] and job["failed_items"]:
        return "partial"
    if job["completed_items"]:
        return "completed"
    return "failed"


def _cache_warning(warning: str | None) -> str:
    message = "Corpus changed but answer-cache invalidation failed"
    if not warning:
        return message
    if "cache" in warning.lower():
        return warning
    return f"{warning}; {message}"


def _needs_cache_invalidation(job: dict) -> bool:
    return any(
        item["status"] == "completed" or item["stage"] == "indexing"
        for item in job["items"]
    )


def _fail_active_items(job: dict, error: str) -> None:
    for item in job["items"]:
        if item["status"] == "running":
            ingestion_jobs.update_item(
                item["id"],
                status="failed",
                stage="done",
                error=error,
                finished_at=ingestion_jobs.now(),
            )
    ingestion_jobs.cancel_pending_items(job["id"])


async def _run_upload_job(job: dict) -> tuple[bool, bool]:
    mutated = False
    ambiguous = False
    for item in job["items"]:
        if ingestion_jobs.cancellation_requested(job["id"]):
            ingestion_jobs.cancel_pending_items(job["id"])
            break
        item_mutated, item_ambiguous = await _run_item(job, item)
        mutated = mutated or item_mutated
        ambiguous = ambiguous or item_ambiguous
    return mutated, ambiguous


async def _run_corpus_job(job: dict) -> tuple[bool, bool]:
    options = job["options"]
    scan = await asyncio.to_thread(
        ingestion.scan_corpus_tree,
        settings.CORPUS_ROOT,
        subtree=options["subtree"],
    )
    items = []
    for position, candidate in enumerate(scan["candidates"]):
        items.append(
            {
                "id": ingestion_jobs.new_id(),
                "position": position,
                "filename": Path(candidate["path"]).name,
                "relative_path": candidate["metadata"]["source"],
                "stored_path": None,
                "source_path": candidate["path"],
                "metadata": candidate["metadata"],
                "doc_key": candidate["metadata"]["source"],
                "ocr": options["ocr"],
                "status": "pending",
            }
        )
    for skipped in scan["skipped"]:
        items.append(
            {
                "id": ingestion_jobs.new_id(),
                "position": len(items),
                "filename": Path(skipped["source"]).name,
                "relative_path": skipped["source"],
                "stored_path": None,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": options["ocr"],
                "status": "skipped",
                "error": skipped["error"],
            }
        )
    ingestion_jobs.replace_job_items(job["id"], items)
    refreshed = ingestion_jobs.get_job(job["id"])
    mutated = False
    ambiguous = False
    for item in refreshed["items"]:
        if item["status"] == "skipped":
            continue
        if ingestion_jobs.cancellation_requested(job["id"]):
            ingestion_jobs.cancel_pending_items(job["id"])
            break
        item_mutated, item_ambiguous = await _run_item(job, item)
        mutated = mutated or item_mutated
        ambiguous = ambiguous or item_ambiguous
    if (
        options["prune"]
        and scan["candidates"]
        and not scan["errors"]
        and not ingestion_jobs.cancellation_requested(job["id"])
    ):
        current = ingestion_jobs.get_job(job["id"])
        if current["failed_items"] == 0:
            try:
                pruned = await ingestion.prune_missing_corpus_documents(
                    scan["present_doc_ids"]
                )
            except Exception:
                if not await invalidate_answer_cache():
                    ingestion_jobs.update_job(
                        job["id"],
                        warning=_cache_warning(current.get("warning")),
                    )
                raise
            mutated = mutated or pruned > 0
    return mutated, ambiguous


async def run_once(worker_id: str) -> bool:
    ingestion_jobs.write_heartbeat(worker_id)
    job = ingestion_jobs.claim_next_job(worker_id)
    if job is None:
        return False
    try:
        if job["kind"] == "upload":
            mutated, ambiguous = await _run_upload_job(job)
        else:
            mutated, ambiguous = await _run_corpus_job(job)
    except Exception as exc:
        error = str(exc)[:2000]
        failed = ingestion_jobs.get_job(job["id"])
        warning = failed.get("warning") if failed else None
        if failed and _needs_cache_invalidation(failed):
            if not await invalidate_answer_cache():
                warning = _cache_warning(warning)
        if failed:
            _fail_active_items(failed, error)
        ingestion_jobs.finish_job(job["id"], status="failed", error=error, warning=warning)
        return True

    final = ingestion_jobs.get_job(job["id"])
    status_value = _terminal_status(final)
    warning = final.get("warning")
    if mutated or ambiguous:
        if not await invalidate_answer_cache():
            warning = _cache_warning(warning)
            if status_value == "completed":
                status_value = "partial"
    ingestion_jobs.finish_job(
        job["id"],
        status=status_value,
        error=final.get("error"),
        warning=warning,
    )
    return True


async def _heartbeat_loop(worker_id: str) -> None:
    while True:
        ingestion_jobs.write_heartbeat(worker_id)
        await asyncio.sleep(_HEARTBEAT_S)


async def run_forever(worker_id: str | None = None) -> None:
    ingestion_jobs.initialize()
    with worker_lock():
        recovered = ingestion_jobs.recover_interrupted_jobs()
        if recovered:
            if not await invalidate_answer_cache():
                logger.warning("Answer-cache invalidation failed after recovering jobs")
        worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        heartbeat = asyncio.create_task(_heartbeat_loop(worker_id))
        try:
            while True:
                worked = await run_once(worker_id)
                if not worked:
                    await asyncio.sleep(_POLL_S)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
