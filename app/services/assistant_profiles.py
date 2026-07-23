"""Trusted assistant profiles selected by client-supplied assistant_type."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from app.core.config import settings

DEFAULT_ASSISTANT_TYPE: Final[str] = "vr_lab_teacher"


@dataclass(frozen=True)
class AssistantProfile:
    assistant_type: str
    system_prompt: str
    qdrant_collection: str
    corpus_root: str


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


def available_assistant_types() -> tuple[str, ...]:
    return tuple(ASSISTANT_PROFILES)


def get_assistant_profile(assistant_type: str | None = None) -> AssistantProfile:
    if assistant_type is None:
        assistant_type = DEFAULT_ASSISTANT_TYPE
    if not isinstance(assistant_type, str):
        raise ValueError(f"Unknown assistant_type: {assistant_type!r}")
    try:
        return ASSISTANT_PROFILES[assistant_type]
    except KeyError as exc:
        raise ValueError(f"Unknown assistant_type: {assistant_type!r}") from exc


def validate_assistant_profiles(
    profiles: Mapping[str, AssistantProfile] = ASSISTANT_PROFILES,
) -> None:
    if DEFAULT_ASSISTANT_TYPE not in profiles:
        raise ValueError(f"Missing default assistant_type: {DEFAULT_ASSISTANT_TYPE}")
    seen_collections: set[str] = set()
    for assistant_type, profile in profiles.items():
        if profile.assistant_type != assistant_type:
            raise ValueError(f"Mismatched assistant_type for {assistant_type}")
        if not profile.system_prompt:
            raise ValueError(f"Missing system_prompt for {assistant_type}")
        if not profile.qdrant_collection:
            raise ValueError(f"Missing qdrant_collection for {assistant_type}")
        if not profile.corpus_root:
            raise ValueError(f"Missing corpus_root for {assistant_type}")
        if profile.qdrant_collection in seen_collections:
            raise ValueError(
                f"duplicate qdrant_collection: {profile.qdrant_collection}"
            )
        seen_collections.add(profile.qdrant_collection)


validate_assistant_profiles()
