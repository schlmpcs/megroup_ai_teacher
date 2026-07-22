import json
import os
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath

from app.core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('upload', 'corpus')),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'completed', 'partial', 'failed', 'cancelled')
    ),
    options_json TEXT NOT NULL DEFAULT '{}',
    retry_of TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    total_items INTEGER NOT NULL DEFAULT 0,
    completed_items INTEGER NOT NULL DEFAULT 0,
    failed_items INTEGER NOT NULL DEFAULT 0,
    skipped_items INTEGER NOT NULL DEFAULT 0,
    current_item TEXT,
    current_stage TEXT,
    error TEXT,
    warning TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS job_items (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    filename TEXT NOT NULL,
    relative_path TEXT,
    stored_path TEXT,
    source_path TEXT,
    metadata_json TEXT,
    doc_key TEXT,
    ocr INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled')
    ),
    stage TEXT NOT NULL CHECK (
        stage IN ('queued', 'extracting', 'embedding', 'indexing', 'done')
    ),
    file_id TEXT,
    chunks INTEGER,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE(job_id, position)
);
CREATE INDEX IF NOT EXISTS job_items_job_idx ON job_items(job_id, position);
CREATE TABLE IF NOT EXISTS worker_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    worker_id TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);
PRAGMA user_version=1;
"""

TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})
_TMP_MAX_AGE_S = 24 * 60 * 60
_ITEM_TERMINAL_STATUSES = frozenset({"completed", "failed", "skipped", "cancelled"})
_HEARTBEAT_TTL_S = 15


def data_dir() -> Path:
    return Path(settings.INGESTION_DATA_DIR)


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(data_dir() / "jobs.sqlite3", timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def initialize() -> None:
    root = data_dir()
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def now() -> str:
    return _now()


def new_id() -> str:
    return uuid.uuid4().hex


def _json_dump(value):
    return json.dumps(value, ensure_ascii=False)


def _json_load(value):
    if value is None:
        return None
    return json.loads(value)


def _path_value(value):
    return None if value is None else str(value)


def _validated_upload_stored_path(stored_path: str | None, *, job_id: str, item_id: str) -> str | None:
    if stored_path is None:
        return None
    path = PurePosixPath(str(stored_path))
    if path.is_absolute():
        raise ValueError("stored_path must not be absolute")
    if ".." in path.parts:
        raise ValueError("stored_path must stay inside uploads")
    expected = ("uploads", job_id, item_id)
    if path.parts[:1] != ("uploads",):
        raise ValueError("stored_path must stay inside uploads")
    if len(path.parts) != 3:
        raise ValueError("stored_path must match uploads/<job_id>/<item_id>")
    if path.parts[1] != job_id:
        raise ValueError("stored_path job segment must match job id")
    if path.parts[2] != item_id:
        raise ValueError("stored_path item segment must match item id")
    return path.as_posix()


def _artifact_dir(kind: str, job_id: str) -> Path:
    return data_dir() / kind / job_id


def _remove_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _quarantine_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.delete-{new_id()}")


def _quarantine_artifacts(job_id: str) -> list[tuple[Path, Path]]:
    moved = []
    for kind in ("uploads", "tmp"):
        original = _artifact_dir(kind, job_id)
        if not original.exists():
            continue
        quarantine = _quarantine_path(original)
        try:
            os.replace(original, quarantine)
        except Exception:
            for restore_original, restore_quarantine in reversed(moved):
                os.replace(restore_quarantine, restore_original)
            raise
        moved.append((original, quarantine))
    return moved


def _restore_quarantined_artifacts(moved: list[tuple[Path, Path]]) -> None:
    for original, quarantine in reversed(moved):
        if quarantine.exists():
            os.replace(quarantine, original)


def _item_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "position": row["position"],
        "filename": row["filename"],
        "relative_path": row["relative_path"],
        "stored_path": row["stored_path"],
        "source_path": row["source_path"],
        "metadata": _json_load(row["metadata_json"]),
        "doc_key": row["doc_key"],
        "ocr": bool(row["ocr"]),
        "status": row["status"],
        "stage": row["stage"],
        "file_id": row["file_id"],
        "chunks": row["chunks"],
        "error": row["error"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


def _job_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
    item_rows = connection.execute(
        """
        SELECT *
        FROM job_items
        WHERE job_id = ?
        ORDER BY position ASC, rowid ASC
        """,
        (row["id"],),
    ).fetchall()
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "options": _json_load(row["options_json"]) or {},
        "retry_of": row["retry_of"],
        "cancel_requested": bool(row["cancel_requested"]),
        "total_items": row["total_items"],
        "completed_items": row["completed_items"],
        "failed_items": row["failed_items"],
        "skipped_items": row["skipped_items"],
        "current_item": row["current_item"],
        "current_stage": row["current_stage"],
        "error": row["error"],
        "warning": row["warning"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "items": [_item_from_row(item_row) for item_row in item_rows],
    }


def _get_job(connection: sqlite3.Connection, job_id: str) -> dict | None:
    row = connection.execute(
        "SELECT rowid, * FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    return _job_from_row(connection, row)


def _normalize_job_fields(fields: dict) -> dict:
    normalized = {}
    for key, value in fields.items():
        if key == "options":
            normalized["options_json"] = _json_dump(value or {})
        elif key == "cancel_requested":
            normalized[key] = int(bool(value))
        else:
            normalized[key] = value
    return normalized


def _normalize_item_fields(fields: dict) -> dict:
    normalized = {}
    for key, value in fields.items():
        if key == "metadata":
            normalized["metadata_json"] = None if value is None else _json_dump(value)
        elif key == "ocr":
            normalized[key] = int(bool(value))
        elif key in {"relative_path", "stored_path", "source_path"}:
            normalized[key] = _path_value(value)
        else:
            normalized[key] = value
    return normalized


def _update_row(
    connection: sqlite3.Connection,
    table: str,
    key_column: str,
    key_value: str,
    fields: dict,
) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    params = tuple(fields.values()) + (key_value,)
    connection.execute(
        f"UPDATE {table} SET {assignments} WHERE {key_column} = ?",
        params,
    )


def _insert_items(connection: sqlite3.Connection, job_id: str, items: list[dict]) -> None:
    rows = []
    for item in items:
        stored_path = _validated_upload_stored_path(
            item.get("stored_path"),
            job_id=job_id,
            item_id=item["id"],
        )
        rows.append(
            (
                item["id"],
                job_id,
                item["position"],
                item["filename"],
                _path_value(item.get("relative_path")),
                stored_path,
                _path_value(item.get("source_path")),
                None if item.get("metadata") is None else _json_dump(item.get("metadata")),
                item.get("doc_key"),
                int(bool(item.get("ocr", False))),
                item.get("status", "pending"),
                item.get("stage", "queued"),
                item.get("file_id"),
                item.get("chunks"),
                item.get("error"),
                item.get("started_at"),
                item.get("finished_at"),
            )
        )
    if rows:
        connection.executemany(
            """
            INSERT INTO job_items (
                id, job_id, position, filename, relative_path, stored_path, source_path,
                metadata_json, doc_key, ocr, status, stage, file_id, chunks, error,
                started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _refresh_job_counts(connection: sqlite3.Connection, job_id: str) -> None:
    counts = dict(
        connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM job_items
            WHERE job_id = ?
            GROUP BY status
            """,
            (job_id,),
        ).fetchall()
    )
    running_item = connection.execute(
        """
        SELECT id, stage
        FROM job_items
        WHERE job_id = ? AND status = 'running'
        ORDER BY position ASC, rowid ASC
        LIMIT 1
        """,
        (job_id,),
    ).fetchone()
    total = sum(int(value) for value in counts.values())
    connection.execute(
        """
        UPDATE jobs
        SET total_items = ?,
            completed_items = ?,
            failed_items = ?,
            skipped_items = ?,
            current_item = ?,
            current_stage = ?
        WHERE id = ?
        """,
        (
            total,
            int(counts.get("completed", 0)),
            int(counts.get("failed", 0)),
            int(counts.get("skipped", 0)),
            None if running_item is None else running_item["id"],
            None if running_item is None else running_item["stage"],
            job_id,
        ),
    )


def enqueue_upload_job(job_id: str, items: list[dict], *, retry_of: str | None = None) -> dict:
    created_at = _now()
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO jobs (
                id, kind, status, options_json, retry_of, cancel_requested,
                total_items, completed_items, failed_items, skipped_items,
                current_item, current_stage, error, warning, created_at, started_at, finished_at
            ) VALUES (?, 'upload', 'queued', '{}', ?, 0, ?, 0, 0, 0, NULL, NULL, NULL, NULL, ?, NULL, NULL)
            """,
            (job_id, retry_of, len(items), created_at),
        )
        _insert_items(connection, job_id, items)
        _refresh_job_counts(connection, job_id)
        return _get_job(connection, job_id)


def enqueue_corpus_job(options: dict, *, retry_of: str | None = None) -> dict:
    job_id = new_id()
    created_at = _now()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, kind, status, options_json, retry_of, cancel_requested,
                total_items, completed_items, failed_items, skipped_items,
                current_item, current_stage, error, warning, created_at, started_at, finished_at
            ) VALUES (?, 'corpus', 'queued', ?, ?, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, ?, NULL, NULL)
            """,
            (job_id, _json_dump(options or {}), retry_of, created_at),
        )
        return _get_job(connection, job_id)


def claim_next_job(worker_id: str) -> dict | None:
    claimed_id = None
    started_at = _now()
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT rowid, id
            FROM jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC, rowid ASC
            LIMIT 1
            """,
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        claimed_id = row["id"]
        connection.execute(
            """
            UPDATE jobs
            SET status = 'running', started_at = ?, current_item = NULL, current_stage = NULL
            WHERE id = ? AND status = 'queued'
            """,
            (started_at, claimed_id),
        )
        connection.commit()
    return get_job(claimed_id)


def get_job(job_id: str) -> dict | None:
    with connect() as connection:
        return _get_job(connection, job_id)


def list_jobs(
    *,
    status: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    where = []
    params = []
    if status is not None:
        where.append("status = ?")
        params.append(status)
    if kind is not None:
        where.append("kind = ?")
        params.append(kind)
    query = """
        SELECT rowid, *
        FROM jobs
    """
    if where:
        query += " WHERE " + " AND ".join(where)
    query += """
        ORDER BY
            CASE
                WHEN status = 'running' THEN 0
                WHEN status = 'queued' THEN 1
                ELSE 2
            END ASC,
            CASE WHEN status IN ('running', 'queued') THEN created_at END ASC,
            CASE WHEN status IN ('running', 'queued') THEN rowid END ASC,
            CASE WHEN status NOT IN ('running', 'queued') THEN created_at END DESC,
            CASE WHEN status NOT IN ('running', 'queued') THEN rowid END DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    with connect() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
        return [_job_from_row(connection, row) for row in rows]


def queue_status() -> dict:
    with connect() as connection:
        counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        }
    return {
        "total": sum(counts.values()),
        "queued": int(counts.get("queued", 0)),
        "running": int(counts.get("running", 0)),
        "completed": int(counts.get("completed", 0)),
        "partial": int(counts.get("partial", 0)),
        "failed": int(counts.get("failed", 0)),
        "cancelled": int(counts.get("cancelled", 0)),
        "worker": worker_status(),
    }


def replace_job_items(job_id: str, items: list[dict]) -> None:
    with connect() as connection:
        connection.execute("DELETE FROM job_items WHERE job_id = ?", (job_id,))
        _insert_items(connection, job_id, items)
        _refresh_job_counts(connection, job_id)


def update_job(job_id: str, **fields) -> None:
    with connect() as connection:
        _update_row(connection, "jobs", "id", job_id, _normalize_job_fields(fields))


def update_item(item_id: str, **fields) -> None:
    with connect() as connection:
        job_row = connection.execute(
            "SELECT job_id FROM job_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if job_row is None:
            raise ValueError(f"Unknown job item: {item_id}")
        _update_row(connection, "job_items", "id", item_id, _normalize_item_fields(fields))
        _refresh_job_counts(connection, job_row["job_id"])


def cancellation_requested(job_id: str) -> bool:
    with connect() as connection:
        row = connection.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return False if row is None else bool(row["cancel_requested"])


def _cancel_pending_items(connection: sqlite3.Connection, job_id: str) -> int:
    finished_at = _now()
    cursor = connection.execute(
        """
        UPDATE job_items
        SET status = 'cancelled',
            stage = 'done',
            error = COALESCE(error, 'Cancelled before processing'),
            finished_at = COALESCE(finished_at, ?)
        WHERE job_id = ? AND status = 'pending'
        """,
        (finished_at, job_id),
    )
    _refresh_job_counts(connection, job_id)
    return cursor.rowcount


def cancel_pending_items(job_id: str) -> int:
    with connect() as connection:
        return _cancel_pending_items(connection, job_id)


def request_cancel(job_id: str) -> dict:
    with connect() as connection:
        job = _get_job(connection, job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in {"queued", "running"}:
            raise ValueError("Cancel is allowed only for queued or running jobs")
        connection.execute(
            "UPDATE jobs SET cancel_requested = 1 WHERE id = ?",
            (job_id,),
        )
        if job["status"] == "queued":
            _cancel_pending_items(connection, job_id)
            _finish_job(
                connection,
                job_id,
                status="cancelled",
                error=job["error"],
                warning=job["warning"],
            )
        return _get_job(connection, job_id)


def _finish_job(
    connection: sqlite3.Connection,
    job_id: str,
    *,
    status: str,
    error: str | None = None,
    warning: str | None = None,
) -> dict:
    _refresh_job_counts(connection, job_id)
    connection.execute(
        """
        UPDATE jobs
        SET status = ?,
            error = ?,
            warning = ?,
            current_item = NULL,
            current_stage = NULL,
            finished_at = ?
        WHERE id = ?
        """,
        (status, error, warning, _now(), job_id),
    )
    return _get_job(connection, job_id)


def finish_job(
    job_id: str,
    *,
    status: str,
    error: str | None = None,
    warning: str | None = None,
) -> dict:
    with connect() as connection:
        return _finish_job(connection, job_id, status=status, error=error, warning=warning)


def recover_interrupted_jobs() -> int:
    recovered_at = _now()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id
            FROM jobs
            WHERE status = 'running'
            ORDER BY created_at ASC, rowid ASC
            """
        ).fetchall()
        for row in rows:
            connection.execute(
                """
                UPDATE job_items
                SET status = 'failed',
                    stage = 'done',
                    error = COALESCE(error, 'Interrupted while processing'),
                    started_at = COALESCE(started_at, ?),
                    finished_at = COALESCE(finished_at, ?)
                WHERE job_id = ? AND status NOT IN ('completed', 'failed', 'skipped', 'cancelled')
                """,
                (recovered_at, recovered_at, row["id"]),
            )
            _finish_job(
                connection,
                row["id"],
                status="failed",
                error="Interrupted while processing",
            )
        return len(rows)


def write_heartbeat(worker_id: str) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO worker_state (singleton, worker_id, heartbeat_at)
            VALUES (1, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                worker_id = excluded.worker_id,
                heartbeat_at = excluded.heartbeat_at
            """,
            (worker_id, _now()),
        )


def worker_status() -> dict:
    with connect() as connection:
        row = connection.execute(
            "SELECT worker_id, heartbeat_at FROM worker_state WHERE singleton = 1"
        ).fetchone()
    if row is None:
        return {"online": False, "worker_id": None, "heartbeat_at": None}
    heartbeat_at = row["heartbeat_at"]
    online = False
    if heartbeat_at:
        age_s = (datetime.now(UTC) - datetime.fromisoformat(heartbeat_at)).total_seconds()
        online = age_s <= _HEARTBEAT_TTL_S
    return {
        "online": online,
        "worker_id": row["worker_id"],
        "heartbeat_at": heartbeat_at,
    }


def retry_job(job_id: str) -> dict:
    original = get_job(job_id)
    if original is None:
        raise KeyError(job_id)
    if original["status"] not in {"failed", "partial", "cancelled"}:
        raise ValueError("Retry is allowed only for failed, partial, or cancelled jobs")
    if original["kind"] == "corpus":
        return enqueue_corpus_job(original["options"], retry_of=original["id"])

    eligible = [
        item for item in original["items"] if item["status"] in {"failed", "cancelled"}
    ]
    if not eligible:
        raise ValueError("Upload retry has no failed or cancelled items")
    retry_id = new_id()
    copied_paths = []
    retry_items = []
    try:
        for position, item in enumerate(eligible):
            item_id = new_id()
            stored_path = f"uploads/{retry_id}/{item_id}"
            source_stored_path = _validated_upload_stored_path(
                item["stored_path"],
                job_id=original["id"],
                item_id=item["id"],
            )
            source = data_dir() / source_stored_path
            target = data_dir() / stored_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_paths.append(target)
            retry_items.append(
                {
                    "id": item_id,
                    "position": position,
                    "filename": item["filename"],
                    "relative_path": item["relative_path"],
                    "stored_path": stored_path,
                    "source_path": None,
                    "metadata": item["metadata"],
                    "doc_key": item["doc_key"],
                    "ocr": item["ocr"],
                }
            )
        try:
            return enqueue_upload_job(retry_id, retry_items, retry_of=original["id"])
        except Exception:
            for copied_path in reversed(copied_paths):
                copied_path.unlink(missing_ok=True)
            retry_dir = _artifact_dir("uploads", retry_id)
            if retry_dir.exists():
                shutil.rmtree(retry_dir)
            raise
    except Exception:
        for copied_path in reversed(copied_paths):
            copied_path.unlink(missing_ok=True)
        retry_dir = _artifact_dir("uploads", retry_id)
        if retry_dir.exists():
            shutil.rmtree(retry_dir)
        raise


def delete_job(job_id: str) -> bool:
    with connect() as connection:
        job = _get_job(connection, job_id)
        if job is None:
            return False
        if job["status"] not in TERMINAL_STATUSES:
            raise ValueError("Only terminal jobs can be deleted")
    moved = _quarantine_artifacts(job_id)
    with connect() as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            connection.commit()
        except Exception:
            connection.rollback()
            _restore_quarantined_artifacts(moved)
            raise
    for _, quarantine in moved:
        _remove_tree_if_exists(quarantine)
    return True


def cleanup_stale_tmp() -> int:
    tmp_root = data_dir() / "tmp"
    if not tmp_root.exists():
        return 0
    with connect() as connection:
        known_ids = {
            row["id"]
            for row in connection.execute("SELECT id FROM jobs").fetchall()
        }
    now_ts = datetime.now(UTC).timestamp()
    removed = 0
    for path in tmp_root.iterdir():
        if not path.is_dir():
            continue
        if path.name in known_ids:
            continue
        age_s = now_ts - path.stat().st_mtime
        if age_s <= _TMP_MAX_AGE_S:
            continue
        shutil.rmtree(path)
        removed += 1
    return removed
