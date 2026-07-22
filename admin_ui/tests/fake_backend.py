import json
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

app = FastAPI()
jobs = {}
documents = [
    {
        "file_id": "doc-1",
        "filename": "Physics 8.md",
        "doc_type": "textbook",
        "subject": "physics",
        "grade": 8,
        "lang": "ru",
        "chunks": 3,
        "status": "ready",
    }
]


def authorize(authorization: str = Header("")):
    if authorization != "Bearer smoke-backend-key":
        raise HTTPException(status_code=403)


@app.get("/admin/corpus_status", dependencies=[Depends(authorize)])
async def corpus_status():
    return {"status": "ready", "documents": len(documents), "points": 3}


@app.get("/admin/ingestion/status", dependencies=[Depends(authorize)])
async def ingestion_status():
    return {
        "queue": {
            "queued": sum(job["status"] == "queued" for job in jobs.values()),
            "running": sum(job["status"] == "running" for job in jobs.values()),
        },
        "worker": {"online": True},
    }


@app.post("/admin/ingestion/preview", dependencies=[Depends(authorize)])
async def preview(payload: dict):
    return {
        "items": [
            {
                "path": path,
                "filename": path.rsplit("/", 1)[-1],
                "metadata": None,
                "doc_key": None,
                "errors": [],
            }
            for path in payload["paths"]
        ]
    }


@app.post("/admin/ingestion/corpus/preview", dependencies=[Depends(authorize)])
async def corpus_preview(payload: dict):
    return {
        "root": "/corpus",
        "subtree": payload.get("subtree", ""),
        "total": 2,
        "recognized": 1,
        "skipped": [{"source": "misc.md", "error": "Unrecognised corpus path"}],
        "items": [
            {
                "source": "School materials/Biology/en/Biology Grade 9.md",
                "doc_type": "textbook",
                "subject": "biology",
                "grade": 9,
                "lang": "en",
            }
        ],
        "prunable": 0,
        "duplicate_lab_ids": [],
        "counts_by_type": {"textbook": 1},
        "counts_by_language": {"en": 1},
    }


def make_job(kind: str, status: str, items: list[dict], *, retry_of: str | None = None) -> dict:
    job_id = uuid.uuid4().hex
    completed = sum(item["status"] == "completed" for item in items)
    failed = sum(item["status"] == "failed" for item in items)
    skipped = sum(item["status"] == "skipped" for item in items)
    job = {
        "id": job_id,
        "kind": kind,
        "status": status,
        "retry_of": retry_of,
        "cancel_requested": False,
        "total_items": len(items),
        "completed_items": completed,
        "failed_items": failed,
        "skipped_items": skipped,
        "current_item": None,
        "current_stage": None,
        "created_at": "2026-07-22T10:00:00+00:00",
        "started_at": "2026-07-22T10:00:01+00:00",
        "finished_at": "2026-07-22T10:00:02+00:00",
        "warning": None,
        "error": None,
        "items": items,
    }
    jobs[job_id] = job
    return job


@app.post("/admin/ingestion/jobs/upload", dependencies=[Depends(authorize)], status_code=202)
async def create_upload_job(files: list[UploadFile] = File(...), manifest: str = Form(...)):
    entries = json.loads(manifest)
    items = []
    for position, (file, entry) in enumerate(zip(files, entries, strict=True)):
        await file.read()
        document = {
            "file_id": uuid.uuid4().hex,
            "filename": file.filename,
            "doc_type": entry.get("doc_type"),
            "subject": entry.get("subject"),
            "grade": entry.get("grade"),
            "lang": entry.get("lang"),
            "chunks": 1,
            "status": "ready",
        }
        documents.append(document)
        items.append(
            {
                "id": uuid.uuid4().hex,
                "position": position,
                "filename": file.filename,
                "relative_path": entry.get("relative_path"),
                "status": "completed",
                "stage": "done",
                "chunks": 1,
                "file_id": document["file_id"],
                "error": None,
            }
        )
    return make_job("upload", "completed", items)


@app.post("/admin/ingestion/jobs/corpus", dependencies=[Depends(authorize)], status_code=202)
async def create_corpus_job(payload: dict):
    item = {
        "id": uuid.uuid4().hex,
        "position": 0,
        "filename": "Biology Grade 9.md",
        "relative_path": "School materials/Biology/en/Biology Grade 9.md",
        "status": "failed",
        "stage": "done",
        "chunks": None,
        "file_id": None,
        "error": "Synthetic smoke-test failure",
    }
    return make_job("corpus", "failed", [item])


@app.get("/admin/ingestion/jobs", dependencies=[Depends(authorize)])
async def list_jobs():
    return {"jobs": list(reversed(list(jobs.values())))}


@app.get("/admin/ingestion/jobs/{job_id}", dependencies=[Depends(authorize)])
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404)
    return jobs[job_id]


@app.post("/admin/ingestion/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404)
    jobs[job_id]["status"] = "cancelled"
    jobs[job_id]["cancel_requested"] = True
    return jobs[job_id]


@app.post("/admin/ingestion/jobs/{job_id}/retry", dependencies=[Depends(authorize)], status_code=202)
async def retry_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404)
    retried_items = [
        {**item, "id": uuid.uuid4().hex, "status": "completed", "error": None}
        for item in jobs[job_id]["items"]
    ]
    return make_job(jobs[job_id]["kind"], "completed", retried_items, retry_of=job_id)


@app.delete("/admin/ingestion/jobs/{job_id}", dependencies=[Depends(authorize)])
async def delete_job(job_id: str):
    if jobs.pop(job_id, None) is None:
        raise HTTPException(status_code=404)
    return {"deleted": True, "job_id": job_id}


@app.get("/admin/documents", dependencies=[Depends(authorize)])
async def list_documents():
    return {"documents": documents}


@app.delete("/admin/documents/{file_id}", dependencies=[Depends(authorize)])
async def delete_document(file_id: str):
    for index, document in enumerate(documents):
        if document["file_id"] == file_id:
            documents.pop(index)
            return {"deleted": True, "file_id": file_id}
    raise HTTPException(status_code=404)
