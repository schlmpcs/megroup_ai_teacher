"""Dedicated service for the fixed young-male Kazakh OmniVoice profile."""

import asyncio
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("omnivoice")


def _device() -> str:
    requested = os.getenv("DEVICE", "cuda").strip().lower()
    return requested if requested == "cpu" or torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OMNIVOICE_MODEL", "shyngys879/KazakhTTS-OmniVoice")
    instruct: str = os.getenv(
        "OMNIVOICE_INSTRUCT", "male, young adult, moderate pitch"
    )
    steps: int = int(os.getenv("OMNIVOICE_STEPS", "24"))
    audio_tokenizer_path: str = os.getenv(
        "OMNIVOICE_AUDIO_TOKENIZER_PATH", "/models/hf_cache/higgs-audio-v2-tokenizer"
    )
    device: str = _device()
    hf_cache: str = os.getenv("HF_HOME", "/models/hf_cache")


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "kk"
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
    def kazakh_only(cls, value: str) -> str:
        if value.strip().lower() != "kk":
            raise ValueError("OmniVoice service supports only language=kk")
        return "kk"

    @field_validator("backend")
    @classmethod
    def omnivoice_only(cls, value: str | None) -> str | None:
        if value is not None and value.strip().lower() != "omnivoice":
            raise ValueError("OmniVoice service supports only backend=omnivoice")
        return "omnivoice" if value is not None else None

    @field_validator("voice")
    @classmethod
    def fixed_voice_only(cls, value: str | None) -> None:
        if value is not None:
            raise ValueError("The Kazakh OmniVoice profile is fixed")
        return None


class OmniVoiceBackend:
    """Loads the model once and serializes GPU synthesis requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._lock = threading.Lock()

    def load_model(self) -> None:
        from omnivoice import OmniVoice
        from omnivoice.models import omnivoice as omnivoice_module

        dtype = torch.float16 if self.settings.device.startswith("cuda") else torch.float32
        tokenizer_path = Path(self.settings.audio_tokenizer_path)
        if not (tokenizer_path / "config.json").is_file():
            raise RuntimeError(
                "OmniVoice audio tokenizer is missing at "
                f"{self.settings.audio_tokenizer_path}. Download "
                "eustlb/higgs-audio-v2-tokenizer into that mounted path."
            )

        # OmniVoice otherwise always calls snapshot_download() for this public
        # dependency, even when the model itself is already local. Let deploys
        # mount a verified local copy so a transient Hugging Face failure cannot
        # make the service unavailable.
        resolve_model_path = omnivoice_module._resolve_model_path

        def resolve_local_tokenizer(name_or_path: str) -> str:
            if name_or_path == "eustlb/higgs-audio-v2-tokenizer":
                return str(tokenizer_path)
            return resolve_model_path(name_or_path)

        omnivoice_module._resolve_model_path = resolve_local_tokenizer
        self.model = OmniVoice.from_pretrained(
            self.settings.model,
            device_map="cuda:0" if self.settings.device.startswith("cuda") else "cpu",
            dtype=dtype,
            cache_dir=self.settings.hf_cache,
        )

    def synthesize(self, text: str, speed: float) -> bytes:
        if self.model is None:
            raise RuntimeError("OmniVoice model is not loaded")
        with self._lock:
            audio = self.model.generate(
                text=text,
                language="Kazakh",
                instruct=self.settings.instruct,
                speed=speed,
                num_step=self.settings.steps,
            )
        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.asarray(audio[0], dtype=np.float32),
            24000,
            format="WAV",
            subtype="PCM_16",
        )
        return buffer.getvalue()


def create_app(
    backend: OmniVoiceBackend | None = None, settings: Settings | None = None
) -> FastAPI:
    settings = settings or Settings()
    backend = backend or OmniVoiceBackend(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(backend.load_model)
        yield

    app = FastAPI(title="Kazakh OmniVoice TTS", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "language": "kk",
            "backend": "omnivoice",
            "profile": settings.instruct,
            "model_loaded": backend.model is not None,
        }

    @app.post("/tts/synthesize")
    async def synthesize(req: SynthesizeRequest, format: str = Query(default="wav")):
        normalized_format = format.strip().lower()
        if normalized_format not in {"wav", "json"}:
            raise HTTPException(status_code=422, detail="format must be one of: wav, json")
        try:
            wav_bytes = await asyncio.to_thread(backend.synthesize, req.text, req.speed)
        except Exception as exc:
            logger.exception("OmniVoice synthesis failed")
            raise HTTPException(status_code=500, detail="OmniVoice synthesis failed") from exc

        if normalized_format == "json":
            import base64

            return JSONResponse(
                {
                    "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                    "content_type": "audio/wav",
                    "language": "kk",
                    "backend": "omnivoice",
                    "profile": settings.instruct,
                }
            )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-TTS-Backend": "omnivoice"},
        )

    return app


app = create_app()
