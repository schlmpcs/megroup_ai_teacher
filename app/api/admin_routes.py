from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.routes import _handle_llm_error
from app.api.upload_utils import read_upload
from app.core.config import settings
from app.core.languages import LanguageCode
from app.core.security import verify_admin_api_key
from app.services import ingestion
from app.services.corpus_meta import build_upload_metadata
from app.services.errors import LLMError
from app.services.llm import clear_answer_cache
from app.services.scenarios import list_scenarios

admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(verify_admin_api_key)],
)


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
