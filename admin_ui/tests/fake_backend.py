import asyncio
import base64
import json
import os
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

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


def authorize_internal(authorization: str = Header("")):
    if authorization != "Bearer smoke-internal-key":
        raise HTTPException(status_code=403)


@app.get("/admin/corpus_status", dependencies=[Depends(authorize)])
async def corpus_status():
    if os.environ.get("FAKE_CORPUS_STATUS_FAIL", "false").lower() == "true":
        raise HTTPException(status_code=503, detail="Synthetic corpus status failure")
    return {"status": "ready", "documents": len(documents), "points": 3}


@app.get("/admin/scenarios", dependencies=[Depends(authorize)])
async def scenarios():
    return {
        "scenarios": [
            {
                "scenario_id": "physics_lab_02_heating",
                "scenario_name": "Нагревание воды",
                "language": "ru",
            }
        ]
    }


@app.get("/health", dependencies=[Depends(authorize_internal)])
async def health():
    return {"status": "ok"}


@app.get("/ready", dependencies=[Depends(authorize_internal)])
async def ready():
    return {"status": "ready", "checks": {"qdrant": "ok", "voice": "ok"}}


@app.post("/ask", dependencies=[Depends(authorize_internal)])
async def ask(payload: dict):
    return {
        "answer": "Вода кипит, когда давление насыщенного пара сравнивается с внешним давлением.",
        "citations": [{"filename": "Physics 8.md", "display_label": "Кипение"}],
        "primary_source": {"filename": "Physics 8.md"},
        "usage": {"input_tokens": 10, "output_tokens": 12},
        "language": payload.get("language", "ru"),
    }


@app.post("/v1/chat/completions", dependencies=[Depends(authorize_internal)])
async def chat(payload: dict):
    if not payload.get("stream"):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Теплопроводность переносит энергию внутри вещества."}}],
            "metadata": {"citations": [], "primary_source": None},
        }

    async def frames():
        yield 'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":"Теплопроводность переносит энергию."}}]}\n\n'
        yield 'data: {"metadata":{"citations":[],"primary_source":null}}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")


@app.post("/hint", dependencies=[Depends(authorize_internal)])
async def hint(payload: dict):
    return {
        "hint": "Следите за температурой воды и дождитесь устойчивого кипения.",
        "hint_level": payload["hint_level"],
        "language": payload.get("language", "ru"),
    }


@app.post("/stt", dependencies=[Depends(authorize_internal)])
async def stt(file: UploadFile = File(...), language: str = Form("auto")):
    await file.read()
    return {"text": "Что такое кипение?", "language": "ru" if language == "auto" else language}


@app.post("/tts", dependencies=[Depends(authorize_internal)])
async def tts(payload: dict):
    return Response(content=b"RIFFtest", media_type="audio/wav", headers={"X-TTS-Backend": payload.get("backend", "supertonic")})


@app.post("/voice_ask", dependencies=[Depends(authorize_internal)])
async def voice_ask(file: UploadFile = File(...)):
    await file.read()
    return {
        "question": "Что такое кипение?",
        "answer": "Кипение представляет собой парообразование по всему объёму жидкости.",
        "citations": [],
        "primary_source": None,
        "audio_base64": base64.b64encode(b"RIFFtest").decode("ascii"),
        "audio_format": "audio/wav",
        "observability": {"latency_ms": {"stt": 10, "llm": 20, "tts": 10, "total": 40}},
    }


@app.get("/admin/ingestion/status", dependencies=[Depends(authorize)])
async def ingestion_status():
    await asyncio.sleep(float(os.environ.get("FAKE_INGESTION_STATUS_DELAY_S", "0")))
    return {
        "ocr_default": os.environ.get("FAKE_OCR_DEFAULT", "true").lower() == "true",
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
