"""Trusted assistant profiles selected by client-supplied assistant_type."""

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Final

from app.core.config import settings
from app.services import vectorstore

DEFAULT_ASSISTANT_TYPE: Final[str] = "vr_lab_teacher"
ASSISTANT_TYPE_PATTERN: Final[str] = r"^[a-z][a-z0-9_]{0,63}$"
_ASSISTANT_TYPE_RE = re.compile(ASSISTANT_TYPE_PATTERN)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_profiles (
    assistant_type TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL,
    qdrant_collection TEXT NOT NULL UNIQUE,
    corpus_root TEXT NOT NULL UNIQUE
);
"""
_initialized_databases: set[Path] = set()
_initialize_lock = Lock()


@dataclass(frozen=True)
class AssistantProfile:
    assistant_type: str
    system_prompt: str
    qdrant_collection: str
    corpus_root: str


@dataclass(frozen=True)
class ListedAssistantProfile:
    profile: AssistantProfile
    built_in: bool


class AssistantProfileConflictError(ValueError):
    pass


ASSISTANT_PROFILES = MappingProxyType(
    {
        DEFAULT_ASSISTANT_TYPE: AssistantProfile(
            assistant_type=DEFAULT_ASSISTANT_TYPE,
            system_prompt="You are a VR laboratory teaching assistant.",
            qdrant_collection=settings.QDRANT_COLLECTION,
            corpus_root=settings.CORPUS_ROOT,
        ),
        "other_assistant": AssistantProfile(
            assistant_type="other_assistant",
            system_prompt="You are the configured domain assistant.",
            qdrant_collection="other_assistant_kb",
            corpus_root="/data/other-corpus",
        ),
    }
)


def _database_path() -> Path:
    return Path(settings.INGESTION_DATA_DIR) / "assistant_profiles.sqlite3"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(), timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def initialize() -> None:
    database = _database_path()
    with _initialize_lock:
        if database in _initialized_databases and database.exists():
            return
        database.parent.mkdir(parents=True, exist_ok=True)
        with _connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
        _initialized_databases.add(database)


def _profile_from_row(row: sqlite3.Row) -> AssistantProfile:
    values = {
        field: row[field]
        for field in (
            "assistant_type",
            "system_prompt",
            "qdrant_collection",
            "corpus_root",
        )
    }
    if (
        not all(isinstance(value, str) and value for value in values.values())
        or not _ASSISTANT_TYPE_RE.fullmatch(values["assistant_type"])
        or values["system_prompt"] != values["system_prompt"].strip()
        or len(values["system_prompt"]) > settings.MAX_INPUT_CHARS
    ):
        raise ValueError(
            "Malformed stored assistant profile: "
            f"{values.get('assistant_type')!r}"
        )
    return AssistantProfile(**values)


def _dynamic_profiles() -> tuple[AssistantProfile, ...]:
    initialize()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM assistant_profiles ORDER BY assistant_type"
        ).fetchall()
    profiles = tuple(_profile_from_row(row) for row in rows)
    duplicate = next(
        (
            profile.assistant_type
            for profile in profiles
            if profile.assistant_type in ASSISTANT_PROFILES
        ),
        None,
    )
    if duplicate is not None:
        raise ValueError(f"Malformed stored assistant profile: {duplicate!r}")
    return profiles


def list_assistant_profiles() -> tuple[ListedAssistantProfile, ...]:
    built_ins = tuple(
        ListedAssistantProfile(profile=profile, built_in=True)
        for profile in ASSISTANT_PROFILES.values()
    )
    dynamic = tuple(
        ListedAssistantProfile(profile=profile, built_in=False)
        for profile in _dynamic_profiles()
    )
    return built_ins + dynamic


def available_assistant_types() -> tuple[str, ...]:
    return tuple(item.profile.assistant_type for item in list_assistant_profiles())


def get_assistant_profile(assistant_type: str | None = None) -> AssistantProfile:
    if assistant_type is None:
        assistant_type = DEFAULT_ASSISTANT_TYPE
    if not isinstance(assistant_type, str):
        raise ValueError(f"Unknown assistant_type: {assistant_type!r}")
    try:
        return ASSISTANT_PROFILES[assistant_type]
    except KeyError:
        initialize()
        with _connect() as connection:
            row = connection.execute(
                "SELECT * FROM assistant_profiles WHERE assistant_type = ?",
                (assistant_type,),
            ).fetchone()
        if row is not None:
            return _profile_from_row(row)
        raise ValueError(f"Unknown assistant_type: {assistant_type!r}")


def _validated_new_profile(
    assistant_type: str,
    system_prompt: str,
) -> AssistantProfile:
    if not isinstance(assistant_type, str) or not _ASSISTANT_TYPE_RE.fullmatch(
        assistant_type
    ):
        raise ValueError(f"Invalid assistant_type: {assistant_type!r}")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    prompt = system_prompt.strip()
    if not prompt:
        raise ValueError("system_prompt must not be empty")
    if len(prompt) > settings.MAX_INPUT_CHARS:
        raise ValueError(
            f"system_prompt must not exceed {settings.MAX_INPUT_CHARS} characters"
        )
    return AssistantProfile(
        assistant_type=assistant_type,
        system_prompt=prompt,
        qdrant_collection=f"{assistant_type}_kb",
        corpus_root=str(
            Path(settings.INGESTION_DATA_DIR)
            / "assistant_corpora"
            / assistant_type
        ),
    )


def _check_profile_collisions(profile: AssistantProfile) -> None:
    for item in list_assistant_profiles():
        existing = item.profile
        if existing.assistant_type == profile.assistant_type:
            raise AssistantProfileConflictError(
                f"assistant_type already exists: {profile.assistant_type!r}"
            )
        if existing.qdrant_collection == profile.qdrant_collection:
            raise AssistantProfileConflictError(
                "qdrant_collection already in use: "
                f"{profile.qdrant_collection!r}"
            )
        if existing.corpus_root == profile.corpus_root:
            raise AssistantProfileConflictError(
                f"corpus_root already in use: {profile.corpus_root!r}"
            )


async def create_assistant_profile(
    assistant_type: str,
    system_prompt: str,
) -> AssistantProfile:
    profile = _validated_new_profile(assistant_type, system_prompt)
    _check_profile_collisions(profile)
    await vectorstore.ensure_collection(
        collection_name=profile.qdrant_collection
    )
    Path(profile.corpus_root).mkdir(parents=True, exist_ok=True)
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_profiles (
                    assistant_type, system_prompt, qdrant_collection, corpus_root
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    profile.assistant_type,
                    profile.system_prompt,
                    profile.qdrant_collection,
                    profile.corpus_root,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise AssistantProfileConflictError(
            f"Assistant profile already exists: {profile.assistant_type!r}"
        ) from exc
    return profile


def validate_assistant_profiles(
    profiles: Mapping[str, AssistantProfile] = ASSISTANT_PROFILES,
) -> None:
    if DEFAULT_ASSISTANT_TYPE not in profiles:
        raise ValueError(f"Missing default assistant_type: {DEFAULT_ASSISTANT_TYPE}")
    seen_collections: set[str] = set()
    seen_corpus_roots: set[str] = set()
    for assistant_type, profile in profiles.items():
        if not _ASSISTANT_TYPE_RE.fullmatch(assistant_type):
            raise ValueError(f"Invalid assistant_type: {assistant_type!r}")
        if profile.assistant_type != assistant_type:
            raise ValueError(f"Mismatched assistant_type for {assistant_type}")
        if (
            not profile.system_prompt
            or profile.system_prompt != profile.system_prompt.strip()
            or len(profile.system_prompt) > settings.MAX_INPUT_CHARS
        ):
            raise ValueError(f"Invalid system_prompt for {assistant_type}")
        if not profile.qdrant_collection:
            raise ValueError(f"Missing qdrant_collection for {assistant_type}")
        if not profile.corpus_root:
            raise ValueError(f"Missing corpus_root for {assistant_type}")
        if profile.qdrant_collection in seen_collections:
            raise ValueError(
                f"duplicate qdrant_collection: {profile.qdrant_collection}"
            )
        seen_collections.add(profile.qdrant_collection)
        if profile.corpus_root in seen_corpus_roots:
            raise ValueError(f"duplicate corpus_root: {profile.corpus_root}")
        seen_corpus_roots.add(profile.corpus_root)


validate_assistant_profiles()
