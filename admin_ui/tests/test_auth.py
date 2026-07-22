import base64
import hashlib
import hmac
import json

import pytest

import admin_ui.auth as auth

from admin_ui.auth import (
    LoginLimiter,
    create_session,
    decode_session,
    hash_password,
    verify_password,
)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _session_token(payload, secret: str = "session-secret-1234567890") -> str:
    encoded = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = _b64(hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def test_scrypt_hash_round_trip():
    encoded = hash_password("correct horse", salt=b"0123456789abcdef")
    assert verify_password("correct horse", encoded) is True
    assert verify_password("wrong", encoded) is False


def test_hash_password_requires_16_byte_explicit_salt():
    with pytest.raises(ValueError):
        hash_password("correct horse", salt=b"short")


def test_verify_password_rejects_invalid_scrypt_metadata_before_deriving(monkeypatch):
    good = hash_password("correct horse", salt=b"0123456789abcdef")
    bad_values = [
        good.replace("scrypt:", "other:", 1),
        good.replace("scrypt:16384:", "scrypt:8192:", 1),
        good.replace("scrypt:16384:8:", "scrypt:16384:4:", 1),
        good.replace("scrypt:16384:8:1:", "scrypt:16384:8:2:", 1),
        "scrypt:16384:8:1:" + _b64(b"short") + ":" + good.rsplit(":", 1)[1],
        "scrypt:16384:8:1:" + good.rsplit(":", 2)[1] + ":" + _b64(b"short"),
    ]

    def fail_scrypt(*args, **kwargs):
        raise AssertionError("scrypt should not run for malformed hashes")

    monkeypatch.setattr(auth.hashlib, "scrypt", fail_scrypt)
    for encoded in bad_values:
        assert verify_password("correct horse", encoded) is False


def test_session_rejects_tampering_and_expiry():
    token, csrf = create_session(
        "admin",
        secret="session-secret-1234567890",
        ttl_s=60,
        now=1000,
    )
    assert decode_session(token, secret="session-secret-1234567890", now=1050)["csrf"] == csrf
    assert decode_session(token + "x", secret="session-secret-1234567890", now=1050) is None
    assert decode_session(token, secret="session-secret-1234567890", now=1061) is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"username": "", "csrf": "token", "issued_at": 1000, "expires_at": 1060},
        {"username": "admin", "csrf": "", "issued_at": 1000, "expires_at": 1060},
        {"username": "admin", "csrf": "token", "issued_at": True, "expires_at": 1060},
        {"username": "admin", "csrf": "token", "issued_at": 1000, "expires_at": False},
        {"username": "admin", "csrf": "token", "issued_at": "1000", "expires_at": 1060},
        {"username": "admin", "csrf": "token", "issued_at": 1000, "expires_at": "1060"},
    ],
)
def test_session_rejects_malformed_payload_shape(payload):
    token = _session_token(payload)
    assert decode_session(token, secret="session-secret-1234567890", now=1050) is None


def test_login_limiter_blocks_sixth_failure_in_one_minute():
    limiter = LoginLimiter(max_failures=5, window_s=60)
    for _ in range(5):
        limiter.record_failure("127.0.0.1", now=1000)
    assert limiter.allowed("127.0.0.1", now=1000) is False
    assert limiter.allowed("127.0.0.1", now=1061) is True


def test_login_limiter_prunes_empty_ip_entries():
    limiter = LoginLimiter(max_failures=5, window_s=60)
    limiter.record_failure("127.0.0.1", now=1000)
    limiter.record_failure("127.0.0.2", now=1000)
    assert set(limiter.failures) == {"127.0.0.1", "127.0.0.2"}
    assert limiter.allowed("127.0.0.3", now=1061) is True
    assert set(limiter.failures) == set()
