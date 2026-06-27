"""Tests for the shared content-profile prompt/schema contract."""

from app.content_profile_prompt import (
    CONTENT_PROFILE_SCHEMA_TEXT,
    common_content_profile_field_guidance,
    content_profile_schema_instruction,
)


def test_content_profile_schema_includes_learning_effort() -> None:
    assert "estimated_time_minutes" in CONTENT_PROFILE_SCHEMA_TEXT
    assert "learning_effort_minutes" in CONTENT_PROFILE_SCHEMA_TEXT
    assert content_profile_schema_instruction().startswith("Use this exact schema:")


def test_common_content_profile_guidance_is_source_agnostic() -> None:
    guidance = common_content_profile_field_guidance(
        consumption_time_label="estimated_time_minutes: consume it."
    )

    assert "estimated_time_minutes: consume it." in guidance
    assert any("learning_effort_minutes" in item for item in guidance)
    assert any("categories" in item for item in guidance)
