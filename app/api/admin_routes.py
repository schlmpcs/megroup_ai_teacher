import shutil
import uuid
from collections import Counter
from pathlib import Path, PurePath
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field, TypeAdapter

from app.api.routes import _handle_llm_error
from app.api.upload_utils import read_upload, stream_upload
from app.core.config import settings
from app.core.languages import LanguageCode
from app.core.security import verify_admin_api_key
from app.services import corpus_meta, ingestion, ingestion_jobs
from app.services.corpus_meta import build_upload_metadata
from app.services.errors import LLMError
from app.services.llm import clear_answer_cache
from app.services.scenarios import list_scenarios

admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(verify_admin_api_key)],
)

PreviewPath = Annotated[str, Field(min_length=1, max_length=1024)]


class UploadPreviewRequest(BaseModel):
    paths: list[PreviewPath] = Field(
        min_length=1,
        max_length=settings.INGESTION_BATCH_MAX_FILES,
    )


class UploadManifestItem(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(default="", max_length=1024)
    doc_type: Literal["textbook", "lab_instruction"] | None = None
    subject: Literal["physics", "chemistry", "biology"] | None = None
    grade: int | None = Field(default=None, ge=7, le=11)
    lang: LanguageCode | None = None
    lab_number: int | None = Field(default=None, ge=1, le=99)
    ocr: bool = False

    def metadata_and_key(self) -> tuple[dict | None, str | None]:
        return build_upload_metadata(
            self.filename,
            doc_type=self.doc_type,
            subject=self.subject,
            grade=self.grade,
            lang=self.lang,
            lab_number=self.lab_number,
        )


class CorpusJobRequest(BaseModel):
    subtree: str = Field(default="", max_length=1024)
    ocr: bool | None = None
    prune: bool = False


_manifest_adapter = TypeAdapter(list[UploadManifestItem])


def _preview_path(path: str) -> dict:
    filename = PurePath(path.replace("\\", "/")).name
    parsed = corpus_meta.parse_path(path)
    errors: list[str] = []
    metadata = None
    doc_key = None
    if parsed is not None:
        try:
            metadata, doc_key = build_upload_metadata(
                filename,
                doc_type=parsed.get("doc_type"),
                subject=parsed.get("subject"),
                grade=parsed.get("grade"),
                lang=parsed.get("lang"),
                lab_number=parsed.get("lab_number"),
            )
        except ValueError as exc:
            errors.append(str(exc))
    return {
        "path": path,
        "filename": filename,
        "metadata": metadata,
        "doc_key": doc_key,
        "errors": errors,
    }


def _preview_paths(paths: list[str]) -> list[dict]:
    items = [_preview_path(path) for path in paths]
    identities = Counter(item["doc_key"] or item["filename"] for item in items)
    for item in items:
        identity = item["doc_key"] or item["filename"]
        if identities[identity] > 1:
            item["errors"].append(f"Duplicate document identity: {identity}")
    return items


def _prunable_documents(documents: list[dict], present_doc_ids: set[str]) -> list[dict]:
    return [
        document
        for document in documents
        if document.get("file_id")
        and document.get("source_path")
        and not document["source_path"].startswith("admin_uploads/")
        and corpus_meta.parse_path(document["source_path"]) is not None
        and document["file_id"] not in present_doc_ids
    ]


@admin_router.get("/corpus_status")
async def corpus_status_endpoint():
    try:
        return await ingestion.corpus_status()
    except LLMError as exc:
        _handle_llm_error(exc)


@admin_router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    doc_type: Optional[Literal["textbook", "lab_instruction"]] = Form(None),
    subject: Optional[Literal["physics", "chemistry", "biology"]] = Form(None),
    grade: Optional[int] = Form(None, ge=7, le=11),
    lang: Optional[LanguageCode] = Form(None),
    lab_number: Optional[int] = Form(None, ge=1, le=99),
    ocr: Optional[bool] = Form(None),
):
    filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ingestion.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. Supported: "
                f"{', '.join(sorted(ingestion.SUPPORTED_EXTENSIONS))}"
            ),
        )
    structured = any(
        value is not None for value in (doc_type, subject, grade, lang, lab_number)
    )
    try:
        metadata, doc_key = build_upload_metadata(
            filename,
            doc_type=doc_type,
            subject=subject,
            grade=grade,
            lang=lang,
            lab_number=lab_number,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raw = await read_upload(file, max_bytes=settings.MAX_DOCUMENT_UPLOAD_BYTES)
    try:
        result = await ingestion.upload_document(
            filename,
            raw,
            metadata=metadata,
            doc_key=doc_key,
            ocr=settings.OCR_ENABLED if ocr is None else ocr,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        _handle_llm_error(exc)
    finally:
        clear_answer_cache()
    return {**result, "metadata": metadata} if structured else result


@admin_router.post("/ingestion/preview")
async def preview_upload_metadata(request: UploadPreviewRequest):
    return {"items": _preview_paths(request.paths)}


@admin_router.post("/ingestion/jobs/upload", status_code=status.HTTP_202_ACCEPTED)
async def create_upload_job(
    files: list[UploadFile] = File(...),
    manifest: str = Form(...),
):
    try:
        items = _manifest_adapter.validate_json(manifest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid upload manifest") from exc
    if not files or len(files) != len(items):
        raise HTTPException(status_code=400, detail="Manifest and file counts must match")
    if len(files) > settings.INGESTION_BATCH_MAX_FILES:
        raise HTTPException(status_code=413, detail="Upload batch contains too many files")

    validated: list[dict] = []
    identities: set[str] = set()
    for position, (file, item) in enumerate(zip(files, items, strict=True)):
        filename = Path((file.filename or "").replace("\\", "/")).name
        if filename != Path(item.filename.replace("\\", "/")).name:
            raise HTTPException(status_code=400, detail="Manifest filename order does not match files")
        if Path(filename).suffix.lower() not in ingestion.SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        try:
            metadata, doc_key = item.metadata_and_key()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        identity = doc_key or filename
        if identity in identities:
            raise HTTPException(status_code=400, detail=f"Duplicate document identity: {identity}")
        identities.add(identity)
        validated.append(
            {
                "position": position,
                "filename": filename,
                "relative_path": item.relative_path,
                "metadata": metadata,
                "doc_key": doc_key,
                "ocr": item.ocr,
            }
        )

    job_id = uuid.uuid4().hex
    temp_dir = ingestion_jobs.data_dir() / "tmp" / job_id
    final_dir = ingestion_jobs.data_dir() / "uploads" / job_id
    temp_dir.mkdir(parents=True)
    total_bytes = 0
    try:
        for file, item in zip(files, validated, strict=True):
            item_id = uuid.uuid4().hex
            size = await stream_upload(
                file,
                temp_dir / item_id,
                max_bytes=settings.MAX_DOCUMENT_UPLOAD_BYTES,
            )
            total_bytes += size
            if total_bytes > settings.INGESTION_BATCH_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Upload batch exceeds maximum size")
            item.update(
                id=item_id,
                stored_path=f"uploads/{job_id}/{item_id}",
            )
        temp_dir.rename(final_dir)
        try:
            return ingestion_jobs.enqueue_upload_job(job_id, validated)
        except Exception:
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


@admin_router.post("/ingestion/corpus/preview")
async def preview_corpus(request: CorpusJobRequest):
    subtree = request.subtree.strip()
    if request.prune and subtree:
        raise HTTPException(
            status_code=400,
            detail="prune is allowed only for the full CORPUS_ROOT",
        )
    try:
        scan = ingestion.scan_corpus_tree(
            settings.CORPUS_ROOT,
            subtree=subtree,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    documents = await ingestion.list_documents()
    prunable = _prunable_documents(documents, scan["present_doc_ids"])
    return {
        "root": scan["root"],
        "subtree": scan["subtree"],
        "total": scan["total"],
        "recognized": len(scan["candidates"]),
        "skipped": scan["skipped"],
        "items": [item["metadata"] for item in scan["candidates"]],
        "prunable": len(prunable) if not subtree else 0,
        "duplicate_lab_ids": scan["duplicate_lab_ids"],
        "counts_by_type": scan["counts_by_type"],
        "counts_by_language": scan["counts_by_language"],
    }


@admin_router.post("/ingestion/jobs/corpus", status_code=status.HTTP_202_ACCEPTED)
async def create_corpus_job(request: CorpusJobRequest):
    subtree = request.subtree.strip()
    if request.prune and subtree:
        raise HTTPException(
            status_code=400,
            detail="prune is allowed only for the full CORPUS_ROOT",
        )
    try:
        ingestion.resolve_corpus_scope(settings.CORPUS_ROOT, subtree)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ingestion_jobs.enqueue_corpus_job(
        {
            "subtree": subtree,
            "ocr": settings.OCR_ENABLED if request.ocr is None else request.ocr,
            "prune": request.prune,
        }
    )


@admin_router.get("/ingestion/status")
async def ingestion_status():
    return {
        "queue": ingestion_jobs.queue_status(),
        "worker": ingestion_jobs.worker_status(),
    }


@admin_router.get("/ingestion/jobs")
async def list_ingestion_jobs(
    status_filter: str | None = Query(default=None, alias="status"),
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    if limit < 1 or limit > 200 or offset < 0:
        raise HTTPException(status_code=422, detail="Invalid pagination")
    return {
        "jobs": ingestion_jobs.list_jobs(
            status=status_filter,
            kind=kind,
            limit=limit,
            offset=offset,
        )
    }


@admin_router.get("/ingestion/jobs/{job_id}")
async def get_ingestion_job(job_id: str):
    job = ingestion_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@admin_router.post("/ingestion/jobs/{job_id}/cancel")
async def cancel_ingestion_job(job_id: str):
    try:
        return ingestion_jobs.request_cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@admin_router.post("/ingestion/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_ingestion_job(job_id: str):
    try:
        return ingestion_jobs.retry_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@admin_router.delete("/ingestion/jobs/{job_id}")
async def delete_ingestion_job(job_id: str):
    try:
        deleted = ingestion_jobs.delete_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True, "job_id": job_id}


@admin_router.get("/documents")
async def list_documents_endpoint():
    try:
        return {"documents": await ingestion.list_documents()}
    except LLMError as exc:
        _handle_llm_error(exc)


@admin_router.delete("/documents/{file_id}")
async def delete_document_endpoint(file_id: str):
    try:
        deleted = await ingestion.delete_document(file_id)
    except LLMError as exc:
        _handle_llm_error(exc)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    clear_answer_cache()
    return {"deleted": True, "file_id": file_id}


@admin_router.get("/scenarios")
async def list_scenarios_endpoint():
    return {"scenarios": list_scenarios()}
