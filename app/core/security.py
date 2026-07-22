import hmac

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _token_matches(token: str, candidate: str) -> bool:
    return bool(candidate) and hmac.compare_digest(token.encode(), candidate.encode())


def _extract_token(api_key: str) -> str:
    """Accept both ``Authorization: Bearer <token>`` and ``Authorization: <token>``."""
    raw_value = api_key.strip()
    if not raw_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization Header",
        )

    if raw_value.lower().startswith("bearer"):
        parts = raw_value.split()
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Malformed Authorization Header",
            )
        return parts[1].strip()

    if " " in raw_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization Header",
        )

    return raw_value


def _verify_header(api_key: str | None, candidate: str) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header",
        )
    token = _extract_token(api_key)
    if not _token_matches(token, candidate):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return token


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the caller's bearer token against INTERNAL_API_KEY."""
    return _verify_header(api_key, settings.INTERNAL_API_KEY)


async def verify_admin_api_key(api_key: str = Security(api_key_header)) -> str:
    return _verify_header(api_key, settings.ADMIN_API_KEY)
