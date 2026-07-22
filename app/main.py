import logging
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.admin_routes import admin_router
from app.api.routes import limiter, router
from app.core.config import missing_required_env_vars, settings
from app.core.languages import SUPPORTED_LANGUAGES


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    app_logger = logging.getLogger("assistant")
    app_logger.setLevel(level)
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        app_logger.addHandler(handler)
    app_logger.propagate = False


setup_logging()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign/echo an X-Request-ID for each request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = missing_required_env_vars()
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
    logging.getLogger("assistant").info(
        "Retrieval backend: Qdrant %s (collection '%s'), embedder %s",
        settings.QDRANT_URL,
        settings.QDRANT_COLLECTION,
        settings.EMBEDDING_BASE_URL,
    )
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.include_router(router)
app.include_router(admin_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": settings.VERSION,
        "model": settings.OPENAI_MODEL,
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }


@app.get("/ready")
async def readiness_check():
    return {
        "status": "ready",
        "version": settings.VERSION,
        "qdrant_url": settings.QDRANT_URL,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "embedding_base_url": settings.EMBEDDING_BASE_URL,
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }
