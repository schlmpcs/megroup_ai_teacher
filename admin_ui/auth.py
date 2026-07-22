import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque

_SCRYPT_N = 16_384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if not password:
        raise ValueError("Password must not be empty")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt:{_SCRYPT_N}:{_SCRYPT_R}:{_SCRYPT_P}:{_b64(salt)}:{_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split(":", 5)
        if algorithm != "scrypt":
            return False
        expected_digest = _unb64(expected)
        actual = hashlib.scrypt(
            password.encode(),
            salt=_unb64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected_digest),
        )
        return hmac.compare_digest(actual, expected_digest)
    except (TypeError, ValueError, binascii.Error):
        return False


def create_session(
    username: str,
    *,
    secret: str,
    ttl_s: int,
    now: int | None = None,
) -> tuple[str, str]:
    current = int(time.time()) if now is None else now
    csrf = secrets.token_urlsafe(24)
    payload = {
        "username": username,
        "issued_at": current,
        "expires_at": current + ttl_s,
        "csrf": csrf,
    }
    encoded = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = _b64(hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).digest())
    return f"{encoded}.{signature}", csrf


def decode_session(token: str, *, secret: str, now: int | None = None) -> dict | None:
    try:
        encoded, supplied = token.split(".", 1)
        expected = _b64(hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied, expected):
            return None
        payload = json.loads(_unb64(encoded))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error):
        return None
    current = int(time.time()) if now is None else now
    if payload.get("expires_at", 0) < current:
        return None
    if not isinstance(payload.get("username"), str) or not isinstance(payload.get("csrf"), str):
        return None
    return payload


class LoginLimiter:
    def __init__(self, max_failures: int = 5, window_s: int = 60):
        self.max_failures = max_failures
        self.window_s = window_s
        self.failures = defaultdict(deque)

    def _prune(self, client_ip: str, now: float) -> None:
        cutoff = now - self.window_s
        entries = self.failures[client_ip]
        while entries and entries[0] <= cutoff:
            entries.popleft()

    def allowed(self, client_ip: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        self._prune(client_ip, current)
        return len(self.failures[client_ip]) < self.max_failures

    def record_failure(self, client_ip: str, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self._prune(client_ip, current)
        self.failures[client_ip].append(current)

    def clear(self, client_ip: str) -> None:
        self.failures.pop(client_ip, None)
