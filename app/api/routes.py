import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Literal, NoReturn, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.security import verify_api_key
from app.services import ingestion
from app.services.corpus_meta import compose_lab_id
from app.services.llm import (
    LLMTimeoutError,
    LLMError,
    generate_answer,
    rephrase_hint,
    stream_answer,
)
from app.services.memory import build_input_messages, latest_user_message
from app.services.scenarios import (
    ScenarioNotFoundError,
    format_scenario_state,
    get_scenario_context,
    list_scenarios,
)
from app.services.voice import synthesize, transcribe

router = APIRouter(dependencies=[Depends(verify_api_key)])
logger = logging.getLogger("assistant.api")

# Set RATE_LIMIT_PER_MINUTE<=0 to disable rate limiting entirely (e.g. for bulk
# eval runs from a single client). The decorators stay in place but become
# no-ops when the limiter is disabled.
_rate_limit_enabled = settings.RATE_LIMIT_PER_MINUTE > 0
limiter = Limiter(key_func=get_remote_address, enabled=_rate_limit_enabled)
_consumer_limit = (
    f"{settings.RATE_LIMIT_PER_MINUTE}/minute" if _rate_limit_enabled else "1000000/minute"
)


# ── Request models ───────────────────────────────────────────────────────────


class ScenarioState(BaseModel):
    """Live per-request scene state from the simulator (ТЗ §3.2).

    Distinct from the static scenario document: this is what the user is doing
    *right now* — which step they are on and which objects they are holding.
    """

    current_step: Optional[str] = Field(default=None, max_length=settings.MAX_INPUT_CHARS)
    held_items: Optional[List[str]] = Field(default=None, max_length=50)


class Lab(BaseModel):
    """Structured current-lab context from the simulator (ТЗ §3.2).

    The other service sends which lab is running as structured fields; we
    compose the canonical ``lab_id`` (e.g. ``physics-10-ru-02``) from them. This
    lets the assistant scope retrieval to the subject and load the lab's
    procedure. ``lab_number`` may be omitted for a general subject/grade context.
    """

    subject: Literal["physics", "chemistry", "biology"]
    grade: int = Field(ge=7, le=11)
    lang: Literal["ru", "kk"] = "ru"
    lab_number: Optional[int] = Field(default=None, ge=1, le=20)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.MAX_INPUT_CHARS)
    scenario_id: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, ge=64, le=4096)
    scenario_state: Optional[ScenarioState] = None
    lab: Optional[Lab] = None
    stream: bool = False

    @model_validator(mode="after")
    def _not_blank(self):
        if not self.query.strip():
            raise ValueError("query must not be empty")
        return self


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=settings.MAX_INPUT_CHARS)


class ChatCompletionRequest(BaseModel):
    model: str = settings.OPENAI_MODEL
    messages: List[ChatMessage] = Field(min_length=1, max_length=50)
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=64, le=4096)
    # VR-specific extension: the simulator passes the current scenario here.
    scenario_id: Optional[str] = None
    scenario_state: Optional[ScenarioState] = None
    lab: Optional[Lab] = None

    @model_validator(mode="after")
    def _latest_is_user(self):
        latest = self.messages[-1]
        if latest.role != "user" or not latest.content.strip():
            raise ValueError("latest message must be a non-empty user message")
        return self


class HintRequest(BaseModel):
    hint_text: str = Field(min_length=1, max_length=settings.MAX_INPUT_CHARS)
    hint_level: int = Field(ge=1, le=3)
    scenario_id: Optional[str] = None
    scenario_state: Optional[ScenarioState] = None


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.MAX_INPUT_CHARS)
    voice: Optional[str] = None
    format: Optional[str] = None
    instructions: Optional[str] = None
    language: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _scenario_context_or_404(scenario_id: Optional[str]) -> Optional[str]:
    try:
        return get_scenario_context(scenario_id)
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _scenario_state_text(state: Optional[ScenarioState]) -> Optional[str]:
    """Render live scene state into a prompt block, or None if nothing useful."""
    if state is None:
        return None
    text = format_scenario_state(state.current_step, state.held_items)
    return text or None


def _lab_dict(lab: Optional[Lab]) -> Optional[dict]:
    """Flatten a ``Lab`` into the dict ``llm`` expects, composing ``lab_id``."""
    if lab is None:
        return None
    return {
        "subject": lab.subject,
        "grade": lab.grade,
        "lang": lab.lang,
        "lab_number": lab.lab_number,
        "lab_id": compose_lab_id(lab.subject, lab.grade, lab.lang, lab.lab_number),
    }


def _sse(obj: dict) -> str:
    """Encode one event as an SSE data frame."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# Sentence boundary for TTS chunking: end punctuation followed by whitespace.
# Decimal points ("3.14") never match; short fragments ("1." list markers,
# "Да.") are held below _TTS_MIN_CHARS and glued to the following sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_TTS_MIN_CHARS = 20


def _split_ready(buf: str, min_chars: int = _TTS_MIN_CHARS) -> tuple[list[str], str]:
    """Split off complete sentences ready for TTS; keep the unfinished tail.

    Returns ``(ready_chunks, remaining_buffer)``. Complete sentences shorter
    than ``min_chars`` are merged forward so the TTS sidecar is not called on
    tiny fragments.
    """
    parts = _SENTENCE_SPLIT_RE.split(buf)
    if len(parts) == 1:
        return [], buf
    tail = parts.pop()
    ready: list[str] = []
    acc = ""
    for part in parts:
        acc = f"{acc} {part}".strip()
        if len(acc) >= min_chars:
            ready.append(acc)
            acc = ""
    remaining = f"{acc} {tail}".strip() if acc else tail
    return ready, remaining


def _handle_llm_error(exc: LLMError) -> NoReturn:
    if isinstance(exc, LLMTimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM gateway timeout: {exc}",
        )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"LLM upstream error: {exc}",
    )


# ── Consumer endpoints ────────────────────────────────────────────────────────


@router.post("/ask")
@limiter.limit(_consumer_limit)
async def ask_endpoint(req: AskRequest, request: Request):
    """Grounded Q&A for the VR client. Returns answer + citations + scenario id.

    With ``"stream": true`` the response is SSE instead of one JSON object:
    ``{"type":"delta","text":...}`` frames as tokens arrive, one
    ``{"type":"done","citations":...,"primary_source":...,"usage":...}`` frame,
    then ``data: [DONE]``. Errors after the stream starts arrive as
    ``{"type":"error","message":...}`` frames.
    """
    scenario_context = _scenario_context_or_404(req.scenario_id)
    scenario_state = _scenario_state_text(req.scenario_state)

    if req.stream:

        async def event_gen():
            try:
                async for event in stream_answer(
                    req.query,
                    scenario_context=scenario_context,
                    max_tokens=req.max_tokens,
                    scenario_state=scenario_state,
                    lab=_lab_dict(req.lab),
                ):
                    if event["type"] == "done":
                        citations = event["citations"]
                        yield _sse(
                            {
                                "type": "done",
                                "citations": citations,
                                "primary_source": citations[0] if citations else None,
                                "scenario_id": req.scenario_id,
                                "usage": event["usage"],
                            }
                        )
                    else:
                        yield _sse(event)
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    start = time.time()
    try:
        result = await generate_answer(
            req.query,
            scenario_context=scenario_context,
            max_tokens=req.max_tokens,
            scenario_state=scenario_state,
            lab=_lab_dict(req.lab),
        )
    except LLMError as exc:
        _handle_llm_error(exc)
    llm_ms = (time.time() - start) * 1000

    return {
        "answer": result.answer,
        "citations": result.citations,
        "primary_source": result.primary_source,
        "scenario_id": req.scenario_id,
        "usage": result.usage,
        "observability": {"latency_ms": {"llm": llm_ms, "total": llm_ms}},
    }


def _sse_chat_chunk(model: str, delta: dict, finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": "chatcmpl-vr",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/v1/chat/completions")
@limiter.limit(_consumer_limit)
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat endpoint with scenario + file-search grounding."""
    scenario_context = _scenario_context_or_404(req.scenario_id)
    scenario_state = _scenario_state_text(req.scenario_state)
    history = build_input_messages([m.model_dump() for m in req.messages])
    user_query = latest_user_message(history) or ""

    if req.stream:

        async def event_gen():
            yield _sse_chat_chunk(req.model, {"role": "assistant", "content": ""})
            try:
                async for event in stream_answer(
                    user_query,
                    scenario_context=scenario_context,
                    chat_history=history,
                    max_tokens=req.max_tokens,
                    scenario_state=scenario_state,
                    lab=_lab_dict(req.lab),
                ):
                    if event["type"] == "delta":
                        yield _sse_chat_chunk(req.model, {"content": event["text"]})
                    elif event["type"] == "done":
                        meta = {
                            "citations": event["citations"],
                            "primary_source": event["citations"][0]
                            if event["citations"]
                            else None,
                        }
                        yield f"data: {json.dumps({'metadata': meta}, ensure_ascii=False)}\n\n"
                        yield _sse_chat_chunk(req.model, {}, finish_reason="stop")
                    elif event["type"] == "error":
                        err = {"error": {"message": event["message"], "type": "stream_error"}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    start = time.time()
    try:
        result = await generate_answer(
            user_query,
            scenario_context=scenario_context,
            chat_history=history,
            max_tokens=req.max_tokens,
            scenario_state=scenario_state,
            lab=_lab_dict(req.lab),
        )
    except LLMError as exc:
        _handle_llm_error(exc)
    llm_ms = (time.time() - start) * 1000

    return {
        "id": "chatcmpl-vr",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": result.usage,
        "metadata": {
            "citations": result.citations,
            "primary_source": result.primary_source,
            "scenario_id": req.scenario_id,
            "observability": {"latency_ms": {"llm": llm_ms, "total": llm_ms}},
        },
    }


@router.post("/hint")
@limiter.limit(_consumer_limit)
async def hint_endpoint(req: HintRequest, request: Request):
    """Rephrase a simulator-provided hint at the given verbosity level."""
    scenario_context = _scenario_context_or_404(req.scenario_id)
    scenario_state = _scenario_state_text(req.scenario_state)
    try:
        hint = await rephrase_hint(
            req.hint_text,
            req.hint_level,
            scenario_context=scenario_context,
            scenario_state=scenario_state,
        )
    except LLMError as exc:
        _handle_llm_error(exc)
    return {"hint": hint, "hint_level": req.hint_level, "scenario_id": req.scenario_id}


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_BYTES} bytes",
        )
    return raw


@router.post("/stt")
@limiter.limit(_consumer_limit)
async def stt_endpoint(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Speech-to-text: mic audio in → recognised text out."""
    raw = await _read_upload(file)
    try:
        text = await transcribe(
            raw,
            filename=file.filename or "audio.webm",
            language=language or settings.DEFAULT_LANGUAGE,
        )
    except LLMError as exc:
        _handle_llm_error(exc)
    return {"text": text}


@router.post("/tts")
@limiter.limit(_consumer_limit)
async def tts_endpoint(req: TTSRequest, request: Request):
    """Text-to-speech: returns synthesized audio bytes (teacher-tone voice)."""
    try:
        audio, media_type = await synthesize(
            req.text,
            voice=req.voice,
            response_format=req.format,
            instructions=req.instructions,
            language=req.language,
        )
    except LLMError as exc:
        _handle_llm_error(exc)
    return Response(content=audio, media_type=media_type)


@router.post("/voice_ask")
@limiter.limit(_consumer_limit)
async def voice_ask_endpoint(
    request: Request,
    file: UploadFile = File(...),
    scenario_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    current_step: Optional[str] = Form(None),
    held_items: Optional[List[str]] = Form(None),
    subject: Optional[str] = Form(None),
    grade: Optional[int] = Form(None),
    lang: Optional[str] = Form(None),
    lab_number: Optional[int] = Form(None),
    stream: bool = Form(False),
):
    """Full voice pipeline: audio question → STT → grounded answer → TTS audio.

    Returns JSON with the recognised question, the text answer, citations and
    the synthesized answer audio (base64). Per-stage latencies are reported so
    the ≤5s acceptance criterion can be monitored. Lab context (subject/grade/
    lang/lab_number) is optional multipart form fields, mirroring ``Lab``.

    With ``stream=true`` the response is SSE: a ``{"type":"question"}`` frame
    after STT, ``{"type":"delta","text":...}`` frames as answer tokens arrive
    (live captions), ``{"type":"audio","seq":N,"text":...,"audio_base64":...,
    "audio_format":...}`` frames as each sentence finishes TTS, one
    ``{"type":"done",...}`` frame with citations, then ``data: [DONE]``. The
    client plays audio frames back-to-back in ``seq`` order.
    """
    scenario_context = _scenario_context_or_404(scenario_id)
    scenario_state = _scenario_state_text(
        ScenarioState(current_step=current_step, held_items=held_items)
    )
    lab = _lab_dict(
        Lab(subject=subject, grade=grade, lang=lang or "ru", lab_number=lab_number)
        if subject and grade is not None
        else None
    )
    raw = await _read_upload(file)
    timings: dict = {}
    tts_language = language or settings.DEFAULT_LANGUAGE

    t0 = time.time()
    try:
        question = await transcribe(
            raw,
            filename=file.filename or "audio.webm",
            language=language or settings.DEFAULT_LANGUAGE,
        )
        timings["stt"] = (time.time() - t0) * 1000
    except LLMError as exc:
        _handle_llm_error(exc)

    if stream:

        async def event_gen():
            # ponytail: TTS awaited inline per sentence (deltas buffer in the
            # transport meanwhile); add a queue+task pipeline if TTS ever
            # becomes the bottleneck.
            yield _sse({"type": "question", "text": question})
            buf = ""
            seq = 0

            async def tts_frame(text: str) -> str:
                nonlocal seq
                audio, media_type = await synthesize(
                    text, voice=voice, language=tts_language
                )
                seq += 1
                return _sse(
                    {
                        "type": "audio",
                        "seq": seq,
                        "text": text,
                        "audio_base64": base64.b64encode(audio).decode("ascii"),
                        "audio_format": media_type,
                    }
                )

            try:
                async for event in stream_answer(
                    question,
                    scenario_context=scenario_context,
                    scenario_state=scenario_state,
                    lab=lab,
                ):
                    if event["type"] == "delta":
                        yield _sse({"type": "delta", "text": event["text"]})
                        buf += event["text"]
                        ready, buf = _split_ready(buf)
                        for sentence in ready:
                            yield await tts_frame(sentence)
                    elif event["type"] == "done":
                        if buf.strip():
                            yield await tts_frame(buf.strip())
                        citations = event["citations"]
                        yield _sse(
                            {
                                "type": "done",
                                "citations": citations,
                                "primary_source": citations[0] if citations else None,
                                "scenario_id": scenario_id,
                                "usage": event["usage"],
                                "observability": {"latency_ms": timings},
                            }
                        )
                    elif event["type"] == "error":
                        yield _sse({"type": "error", "message": event["message"]})
            except LLMError as exc:  # synthesize() failures mid-stream
                yield _sse({"type": "error", "message": str(exc)})
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    try:
        t0 = time.time()
        result = await generate_answer(
            question,
            scenario_context=scenario_context,
            scenario_state=scenario_state,
            lab=lab,
        )
        timings["llm"] = (time.time() - t0) * 1000

        t0 = time.time()
        audio, media_type = await synthesize(
            result.answer, voice=voice, language=language or settings.DEFAULT_LANGUAGE
        )
        timings["tts"] = (time.time() - t0) * 1000
    except LLMError as exc:
        _handle_llm_error(exc)

    timings["total"] = sum(timings.values())
    return {
        "question": question,
        "answer": result.answer,
        "citations": result.citations,
        "primary_source": result.primary_source,
        "scenario_id": scenario_id,
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "audio_format": media_type,
        "observability": {"latency_ms": timings},
    }


# ── Admin endpoints ────────────────────────────────────────────────────────


@router.get("/admin/corpus_status")
async def corpus_status_endpoint():
    try:
        return await ingestion.corpus_status()
    except LLMError as exc:
        _handle_llm_error(exc)


@router.post("/admin/documents", status_code=status.HTTP_201_CREATED)
async def upload_document_endpoint(file: UploadFile = File(...)):
    """Upload a PDF/DOCX/TXT into the knowledge-base vector store."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ingestion.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. Supported: "
                f"{', '.join(sorted(ingestion.SUPPORTED_EXTENSIONS))}"
            ),
        )
    raw = await _read_upload(file)
    try:
        return await ingestion.upload_document(filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LLMError as exc:
        _handle_llm_error(exc)


@router.get("/admin/documents")
async def list_documents_endpoint():
    try:
        return {"documents": await ingestion.list_documents()}
    except (ValueError, LLMError) as exc:
        if isinstance(exc, LLMError):
            _handle_llm_error(exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete("/admin/documents/{file_id}")
async def delete_document_endpoint(file_id: str):
    try:
        deleted = await ingestion.delete_document(file_id)
    except LLMError as exc:
        _handle_llm_error(exc)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return {"deleted": True, "file_id": file_id}


@router.get("/admin/scenarios")
async def list_scenarios_endpoint():
    return {"scenarios": list_scenarios()}
