from __future__ import annotations

from pathlib import Path
from typing import Any

from app.library.markdown_store import DEFAULT_LIBRARY_ROOT


PROFILE_RELATIVE_PATH = Path("user") / "profile.md"
PROFILE_TEMPLATE = """# Scratchpad User Profile

Edit this file to make recommendations more useful. Keep it human-readable.

## Current Goals

- Understand practical local LLM workflows

## Interests

- local LLMs
- software engineering
- evaluation

## Avoided Topics

- 

## Preferences

- preferred_depth: medium
- preferred_session_minutes: 20
"""


def user_profile_path(*, library_root: Path = DEFAULT_LIBRARY_ROOT) -> Path:
    return library_root / PROFILE_RELATIVE_PATH


def read_user_profile(
    *,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    path = user_profile_path(library_root=library_root)
    existed = path.exists()
    if not existed and create_if_missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(PROFILE_TEMPLATE, encoding="utf-8")

    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "profile": empty_profile(),
            "content": "",
            "created": False,
        }

    content = path.read_text(encoding="utf-8")
    return {
        "status": "completed",
        "path": str(path),
        "profile": parse_user_profile(content),
        "content": content,
        "created": not existed,
    }


def parse_user_profile(content: str) -> dict[str, Any]:
    profile = empty_profile()
    current_section = ""

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_section = normalize_section(line.removeprefix("## "))
            continue
        if not line.startswith("-"):
            continue

        value = line.lstrip("-").strip()
        if not value:
            continue
        if current_section == "current_goals":
            profile["current_goals"].append(value)
        elif current_section == "interests":
            profile["interests"].append(value)
        elif current_section == "avoided_topics":
            profile["avoided_topics"].append(value)
        elif current_section == "preferences":
            key, separator, raw_value = value.partition(":")
            if separator:
                profile["preferences"][key.strip()] = raw_value.strip()

    return profile


def empty_profile() -> dict[str, Any]:
    return {
        "current_goals": [],
        "interests": [],
        "avoided_topics": [],
        "preferences": {},
    }


def normalize_section(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")
