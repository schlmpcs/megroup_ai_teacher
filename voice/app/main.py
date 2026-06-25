import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.stt.language import normalize_stt_language
from app.tts.language import normalize_tts_language
from app.ui import register_ui

logger = logging.getLogger(__name__)


class SttBackend(Protocol):
    loaded: list[str]

    def load_models(self) -> None:
        ...

    def transcribe(self, audio_bytes: bytes, language: str) -> dict:
        ...


class TtsBackend(Protocol):
    loaded: list[str]

    def load_models(self) -> None:
        ...

    def synthesize(self, text: str, language: str, speed: float) -> bytes:
        ...


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "ru"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, value: str) -> str:
        return normalize_tts_language(value)



def _default_stt_backend(settings: Settings) -> SttBackend:
    from app.stt.model import LocalWhisperSttBackend

    return LocalWhisperSttBackend(settings)


def _default_tts_backend(settings: Settings) -> TtsBackend:
    from app.tts.model import MmsTtsBackend

    return MmsTtsBackend(settings)


def create_app(
    stt_backend: SttBackend | None = None,
    tts_backend: TtsBackend | None = None,
    settings: Settings | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    upload_limit = max_upload_bytes if max_upload_bytes is not None else settings.max_upload_bytes
    stt_backend = stt_backend or _default_stt_backend(settings)
    tts_backend = tts_backend or _default_tts_backend(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(stt_backend.load_models)
        await asyncio.to_thread(tts_backend.load_models)
        yield

    app = FastAPI(title="VRRAG STT/TTS Service", lifespan=lifespan)
    app.state.stt_backend = stt_backend
    app.state.tts_backend = tts_backend
    register_ui(app)

    @app.get("/health")
    def health():
        def loaded_language_keys(backend, prefix: str) -> list[str]:
            models = getattr(backend, "_models", None)
            if isinstance(models, dict) and models:
                return list(models.keys())
            return [item.removeprefix(prefix) for item in backend.loaded if item.startswith(prefix)]

        return {
            "status": "ok",
            "stt_models": loaded_language_keys(stt_backend, "stt_"),
            "tts_models": loaded_language_keys(tts_backend, "tts_"),
        }

    @app.post("/stt/recognize")
    async def recognize(audio: UploadFile = File(...), language: str = Form(default="auto")):
        normalized_language = normalize_stt_language(language)
        audio_bytes = await audio.read()
        if len(audio_bytes) > upload_limit:
            raise HTTPException(status_code=413, detail="audio upload exceeds MAX_UPLOAD_BYTES")

        try:
            return await asyncio.to_thread(stt_backend.transcribe, audio_bytes, normalized_language)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("STT transcription failed")
            raise HTTPException(status_code=500, detail="STT transcription failed") from exc

    @app.post("/tts/synthesize")
    async def synthesize(req: SynthesizeRequest, format: str = Query(default="wav")):
        normalized_format = format.strip().lower()
        if normalized_format not in {"wav", "json"}:
            raise HTTPException(status_code=422, detail="format must be one of: wav, json")

        try:
            wav_bytes = await asyncio.to_thread(tts_backend.synthesize, req.text, req.language, req.speed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            raise HTTPException(status_code=500, detail="TTS synthesis failed") from exc

        if normalized_format == "json":
            return JSONResponse(
                {
                    "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                    "content_type": "audio/wav",
                    "language": req.language,
                }
            )

        return Response(content=wav_bytes, media_type="audio/wav")

    return app


app = create_app()
