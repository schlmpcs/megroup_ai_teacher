import hmac
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from admin_ui.auth import LoginLimiter, create_session, decode_session, verify_password
from admin_ui.config import AdminSettings, get_settings

_EXACT = {
    ("GET", "corpus_status"),
    ("GET", "documents"),
    ("GET", "scenarios"),
    ("GET", "ingestion/status"),
    ("POST", "ingestion/preview"),
    ("POST", "ingestion/corpus/preview"),
    ("POST", "ingestion/jobs/upload"),
    ("POST", "ingestion/jobs/corpus"),
    ("GET", "ingestion/jobs"),
}
_PATTERNS = (
    ("GET", re.compile(r"ingestion/jobs/[0-9a-f]{32}")),
    ("POST", re.compile(r"ingestion/jobs/[0-9a-f]{32}/(?:cancel|retry)")),
    ("DELETE", re.compile(r"ingestion/jobs/[0-9a-f]{32}")),
    ("DELETE", re.compile(r"documents/[A-Za-z0-9._:-]{1,128}")),
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings()
    yield


app = FastAPI(
    title="VR AI Assistant Admin",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static", check_dir=False),
    name="static",
)
login_limiter = LoginLimiter()


@app.middleware("http")
async def disable_session_caching(request: Request, call_next):
    response = await call_next(request)
    session_path = request.url.path in {"/auth/login", "/api/session", "/auth/logout"}
    if session_path or request.url.path.startswith("/api/admin/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class LoginRequest(BaseModel):
    username: str
    password: str


def _allowed(method: str, path: str) -> bool:
    return (method, path) in _EXACT or any(
        method == allowed_method and pattern.fullmatch(path)
        for allowed_method, pattern in _PATTERNS
    )


def session_payload(
    request: Request,
    settings: AdminSettings = Depends(get_settings),
) -> dict:
    token = request.cookies.get("admin_session")
    payload = decode_session(token or "", secret=settings.ADMIN_UI_SESSION_SECRET)
    if payload is None or not hmac.compare_digest(payload["username"], settings.ADMIN_UI_USERNAME):
        raise HTTPException(status_code=401, detail="Authentication required")
    return payload


def require_csrf(request: Request, session: dict = Depends(session_payload)) -> dict:
    supplied = request.headers.get("X-CSRF-Token", "")
    if not hmac.compare_digest(supplied, session["csrf"]):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return session


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/auth/login")
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    settings: AdminSettings = Depends(get_settings),
):
    client_ip = request.client.host if request.client else "unknown"
    if not login_limiter.allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    username_ok = hmac.compare_digest(payload.username, settings.ADMIN_UI_USERNAME)
    password_ok = verify_password(payload.password, settings.ADMIN_UI_PASSWORD_HASH)
    if not username_ok or not password_ok:
        login_limiter.record_failure(client_ip)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    login_limiter.clear(client_ip)
    token, csrf = create_session(
        payload.username,
        secret=settings.ADMIN_UI_SESSION_SECRET,
        ttl_s=settings.ADMIN_UI_SESSION_TTL_S,
    )
    response.set_cookie(
        "admin_session",
        token,
        max_age=settings.ADMIN_UI_SESSION_TTL_S,
        httponly=True,
        secure=settings.ADMIN_UI_COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    return {"authenticated": True, "csrf_token": csrf}


@app.get("/api/session")
async def session(session: dict = Depends(session_payload)):
    return {"authenticated": True, "username": session["username"], "csrf_token": session["csrf"]}


@app.post("/auth/logout")
async def logout(response: Response, session: dict = Depends(require_csrf)):
    response.delete_cookie("admin_session", path="/")
    return {"authenticated": False}


@app.api_route("/api/admin/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_admin(
    path: str,
    request: Request,
    session: dict = Depends(session_payload),
    settings: AdminSettings = Depends(get_settings),
):
    method = request.method.upper()
    if not _allowed(method, path):
        raise HTTPException(status_code=404, detail="Admin operation not found")
    if method in {"POST", "DELETE"}:
        require_csrf(request, session)
    headers = {
        "Authorization": f"Bearer {settings.BACKEND_ADMIN_API_KEY}",
        "Accept": request.headers.get("accept", "application/json"),
    }
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    transport = getattr(app.state, "backend_transport", None)
    async with httpx.AsyncClient(
        base_url=settings.BACKEND_BASE_URL,
        timeout=settings.BACKEND_TIMEOUT_S,
        transport=transport,
    ) as client:
        try:
            backend = await client.request(
                method,
                f"/admin/{path}",
                params=request.query_params,
                headers=headers,
                content=request.stream(),
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="Backend admin API unavailable") from exc
    response_headers = {}
    if backend.headers.get("content-type"):
        response_headers["Content-Type"] = backend.headers["content-type"]
    return Response(content=backend.content, status_code=backend.status_code, headers=response_headers)
