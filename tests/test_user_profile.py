from pathlib import Path

from app.library.user_profile import parse_user_profile, read_user_profile, user_profile_path


def test_read_user_profile_creates_editable_template(tmp_path: Path) -> None:
    result = read_user_profile(library_root=tmp_path)

    assert result["status"] == "completed"
    assert result["created"]
    assert Path(result["path"]) == tmp_path / "user" / "profile.md"
    assert "current_goals" in result["profile"]
    assert user_profile_path(library_root=tmp_path).exists()


def test_parse_user_profile_extracts_lists_and_preferences() -> None:
    profile = parse_user_profile(
        """# Scratchpad User Profile

## Current Goals
- Learn practical LLM deployment

## Interests
- local LLMs
- security

## Avoided Topics
- crypto

## Preferences
- preferred_depth: medium
- preferred_session_minutes: 25
"""
    )

    assert profile == {
        "current_goals": ["Learn practical LLM deployment"],
        "interests": ["local LLMs", "security"],
        "avoided_topics": ["crypto"],
        "preferences": {
            "preferred_depth": "medium",
            "preferred_session_minutes": "25",
        },
    }
