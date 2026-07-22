from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_PLACEHOLDER = "generate-with-python-secrets-token-urlsafe-32"


class AdminSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ADMIN_UI_USERNAME: str = ""
    ADMIN_UI_PASSWORD_HASH: str = ""
    ADMIN_UI_SESSION_SECRET: str = ""
    ADMIN_UI_SESSION_TTL_S: int = 28_800
    ADMIN_UI_COOKIE_SECURE: bool = True
    BACKEND_BASE_URL: str = "http://api:8000"
    BACKEND_ADMIN_API_KEY: str = ""
    BACKEND_TIMEOUT_S: float = 300.0

    @field_validator(
        "ADMIN_UI_USERNAME",
        "ADMIN_UI_PASSWORD_HASH",
        "ADMIN_UI_SESSION_SECRET",
        "BACKEND_ADMIN_API_KEY",
    )
    @classmethod
    def required(cls, value: str, info) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} is required")
        if (
            info.field_name in {"ADMIN_UI_SESSION_SECRET", "BACKEND_ADMIN_API_KEY"}
            and stripped == _SECRET_PLACEHOLDER
        ):
            raise ValueError(f"{info.field_name} must not use the example placeholder")
        if info.field_name == "ADMIN_UI_SESSION_SECRET" and len(stripped) < 32:
            raise ValueError("ADMIN_UI_SESSION_SECRET must contain at least 32 characters")
        return value


@lru_cache
def get_settings() -> AdminSettings:
    return AdminSettings()
