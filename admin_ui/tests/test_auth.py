import time

from admin_ui.auth import (
    LoginLimiter,
    create_session,
    decode_session,
    hash_password,
    verify_password,
)


def test_scrypt_hash_round_trip():
    encoded = hash_password("correct horse", salt=b"0123456789abcdef")
    assert verify_password("correct horse", encoded) is True
    assert verify_password("wrong", encoded) is False


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


def test_login_limiter_blocks_sixth_failure_in_one_minute():
    limiter = LoginLimiter(max_failures=5, window_s=60)
    for _ in range(5):
        limiter.record_failure("127.0.0.1", now=1000)
    assert limiter.allowed("127.0.0.1", now=1000) is False
    assert limiter.allowed("127.0.0.1", now=1061) is True
