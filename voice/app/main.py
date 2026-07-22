import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from contextlib import suppress
from pathlib import Path
from typing import Mapping, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from .config import Settings, get_settings
from .stt.language import normalize_stt_language
from .tts.language import normalize_tts_language
from .ui import register_ui

logger = logging.getLogger(__name__)


class SttBackend(Protocol):
    loaded: list[str]

    def load_models(self) -> None: ...

    def transcribe(self, audio_bytes: bytes, language: str) -> dict: ...

    def warm_up(
        self,
        probes: Mapping[str, bytes],
        *,
        blocking: bool,
        min_real_idle_s: float,
        min_warmup_interval_s: float,
    ) -> bool: ...


class TtsBackend(Protocol):
    loaded: list[str]

    def load_models(self) -> None: ...

    def select_backend(self, language: str, backend: str | None = None) -> str: ...

    def synthesize(
        self,
        text: str,
        language: str,
        speed: float,
        backend: str | None = None,
        voice: str | None = None,
    ) -> bytes: ...


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "ru"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    backend: str | None = None
    voice: str | None = None

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

    @field_validator("backend")
    @classmethod
    def backend_must_be_supported(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"mms", "qwen", "supertonic"}:
            raise ValueError("backend must be one of: mms, qwen, supertonic")
        return normalized

    @field_validator("voice")
    @classmethod
    def voice_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("voice must not be blank")
        return normalized


def _default_stt_backend(settings: Settings) -> SttBackend:
    from .stt.model import LocalWhisperSttBackend

    return LocalWhisperSttBackend(settings)


def _default_tts_backend(settings: Settings) -> TtsBackend:
    from .tts.model import LocalTtsBackend

    return LocalTtsBackend(settings)


def _load_stt_warmup_probes() -> dict[str, bytes]:
    # These clips were generated for this service with the local macOS system
    # voices Milena ("Проверка") and Aru ("Тексеру"), then converted to 16 kHz
    # mono PCM. Spoken probes avoid Whisper's long hallucinated decodes on tone
    # or silence inputs.
    probe_dir = Path(__file__).resolve().parent / "stt"
    return {
        "ru": (probe_dir / "warmup_ru.wav").read_bytes(),
        "kk": (probe_dir / "warmup_kk.wav").read_bytes(),
    }


async def _to_thread_until_complete(function, /, *args, **kwargs):
    """Await a worker even when shutdown cancels the surrounding task."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with suppress(Exception):
            await worker
        raise


async def _stt_keep_warm_loop(
    stt_backend: SttBackend,
    probes: Mapping[str, bytes],
    *,
    poll_s: float,
    real_idle_s: float,
    interval_s: float,
) -> None:
    while True:
        await asyncio.sleep(poll_s)
        try:
            warmed = await _to_thread_until_complete(
                stt_backend.warm_up,
                probes,
                blocking=False,
                min_real_idle_s=real_idle_s,
                min_warmup_interval_s=interval_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Periodic STT warm-up failed")
            continue

        if warmed:
            logger.info("Periodic STT warm-up completed for ru and kk")
        else:
            logger.debug(
                "Periodic STT warm-up skipped because STT is active or not due"
            )


def create_app(
    stt_backend: SttBackend | None = None,
    tts_backend: TtsBackend | None = None,
    settings: Settings | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    upload_limit = (
        max_upload_bytes if max_upload_bytes is not None else settings.max_upload_bytes
    )
    stt_backend = stt_backend or _default_stt_backend(settings)
    tts_backend = tts_backend or _default_tts_backend(settings)
    stt_warmup_probes = _load_stt_warmup_probes()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        keep_warm_task: asyncio.Task | None = None
        await asyncio.to_thread(stt_backend.load_models)
        await asyncio.to_thread(tts_backend.load_models)
        warmed = await _to_thread_until_complete(
            stt_backend.warm_up,
            stt_warmup_probes,
            blocking=True,
            min_real_idle_s=0.0,
            min_warmup_interval_s=0.0,
        )
        if not warmed:
            raise RuntimeError("Startup STT warm-up could not reserve the backend")
        logger.info("Startup STT warm-up completed for ru and kk")

        if settings.stt_keep_warm_enabled:
            keep_warm_task = asyncio.create_task(
                _stt_keep_warm_loop(
                    stt_backend,
                    stt_warmup_probes,
                    poll_s=settings.stt_keep_warm_poll_s,
                    real_idle_s=settings.stt_keep_warm_real_idle_s,
                    interval_s=settings.stt_keep_warm_interval_s,
                ),
                name="stt-keep-warm",
            )
        app.state.stt_keep_warm_task = keep_warm_task

        try:
            yield
        finally:
            if keep_warm_task is not None:
                keep_warm_task.cancel()
                with suppress(asyncio.CancelledError):
                    await keep_warm_task

    app = FastAPI(title="VRRAG STT/TTS Service", lifespan=lifespan)
    app.state.stt_backend = stt_backend
    app.state.tts_backend = tts_backend
    register_ui(app)

    @app.get("/health")
    def health():
        def loaded_language_keys(backend, prefix: str) -> list[str]:
            return [
                item.removeprefix(prefix)
                for item in backend.loaded
                if item.startswith(prefix)
            ]

        return {
            "status": "ok",
            "stt_models": loaded_language_keys(stt_backend, "stt_"),
            "tts_models": loaded_language_keys(tts_backend, "tts_"),
            "tts_backends": getattr(tts_backend, "available_backends", {}),
            "tts_default_backends": getattr(tts_backend, "default_backends", {}),
            "tts_number_normalization": {
                "ru": settings.tts_normalize_ru_numbers,
            },
        }

    @app.post("/stt/recognize")
    async def recognize(
        audio: UploadFile = File(...), language: str = Form(default="auto")
    ):
        normalized_language = normalize_stt_language(language)
        audio_bytes = await audio.read()
        if len(audio_bytes) > upload_limit:
            raise HTTPException(
                status_code=413, detail="audio upload exceeds MAX_UPLOAD_BYTES"
            )

        try:
            return await _to_thread_until_complete(
                stt_backend.transcribe, audio_bytes, normalized_language
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("STT transcription failed")
            raise HTTPException(
                status_code=500, detail="STT transcription failed"
            ) from exc

    @app.post("/tts/synthesize")
    async def synthesize(req: SynthesizeRequest, format: str = Query(default="wav")):
        normalized_format = format.strip().lower()
        if normalized_format not in {"wav", "json"}:
            raise HTTPException(
                status_code=422, detail="format must be one of: wav, json"
            )

        try:
            selected_backend = tts_backend.select_backend(req.language, req.backend)
            wav_bytes = await asyncio.to_thread(
                tts_backend.synthesize,
                req.text,
                req.language,
                req.speed,
                selected_backend,
                req.voice,
            )
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
                    "backend": selected_backend,
                    "voice": req.voice,
                }
            )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-TTS-Backend": selected_backend},
        )

    return app


app = create_app()
