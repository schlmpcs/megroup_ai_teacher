"""Unit tests for the trusted assistant profile registry."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from app.core.config import settings
from app.services.assistant_profiles import (
    ASSISTANT_PROFILES,
    AssistantProfile,
    DEFAULT_ASSISTANT_TYPE,
    available_assistant_types,
    get_assistant_profile,
    validate_assistant_profiles,
)


def test_resolve_assistant_profile_defaults_to_vr_lab_teacher():
    profile = get_assistant_profile()

    assert profile.assistant_type == DEFAULT_ASSISTANT_TYPE
    assert profile.system_prompt == "You are a VR laboratory teaching assistant."
    assert profile.qdrant_collection == settings.QDRANT_COLLECTION
    assert profile.corpus_root == settings.CORPUS_ROOT


def test_resolve_assistant_profile_accepts_explicit_none():
    profile = get_assistant_profile(None)

    assert profile == ASSISTANT_PROFILES[DEFAULT_ASSISTANT_TYPE]


def test_resolve_assistant_profile_returns_other_assistant_example():
    profile = get_assistant_profile("other_assistant")

    assert profile == AssistantProfile(
        assistant_type="other_assistant",
        system_prompt="You are the configured domain assistant.",
        qdrant_collection="other_assistant_kb",
        corpus_root="/data/other-corpus",
    )


def test_resolve_assistant_profile_rejects_unknown_assistant_type():
    with pytest.raises(ValueError, match="Unknown assistant_type"):
        get_assistant_profile("unknown")


def test_assistant_profile_is_frozen():
    profile = get_assistant_profile()

    with pytest.raises(FrozenInstanceError):
        profile.assistant_type = "other_assistant"  # type: ignore[misc]


def test_available_assistant_types_is_trusted_registry_order():
    assert available_assistant_types() == (
        DEFAULT_ASSISTANT_TYPE,
        "other_assistant",
    )


def test_validate_assistant_profiles_passes():
    validate_assistant_profiles()


def test_validate_assistant_profiles_rejects_duplicate_qdrant_collections():
    profiles = MappingProxyType(
        {
            DEFAULT_ASSISTANT_TYPE: ASSISTANT_PROFILES[DEFAULT_ASSISTANT_TYPE],
            "alpha": AssistantProfile(
                assistant_type="alpha",
                system_prompt="Alpha assistant.",
                qdrant_collection="shared_kb",
                corpus_root="/data/alpha",
            ),
            "beta": AssistantProfile(
                assistant_type="beta",
                system_prompt="Beta assistant.",
                qdrant_collection="shared_kb",
                corpus_root="/data/beta",
            ),
        }
    )

    assert profiles["alpha"].assistant_type == "alpha"

    with pytest.raises(ValueError, match="duplicate qdrant_collection"):
        validate_assistant_profiles(profiles)
