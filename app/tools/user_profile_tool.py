from __future__ import annotations

import json
from typing import Any

from app.library.user_profile import read_user_profile


USER_PROFILE_GET_SCHEMA = {
    "name": "user_profile_get",
    "description": (
        "Read the editable Scratchpad user profile from library/user/profile.md. "
        "Use it after loading the scratchpad-recommendation skill, or when the "
        "user explicitly asks to inspect their profile, goals, interests, avoided "
        "topics, preferred depth, or preferred session length. "
        "Creates a human-editable template if the profile does not exist."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "create_if_missing": {
                "type": "boolean",
                "description": "Whether to create the default profile template when missing. Defaults to true.",
            }
        },
        "additionalProperties": False,
    },
}


def user_profile_get_json(arguments: dict[str, Any]) -> str:
    create_if_missing = bool(arguments.get("create_if_missing", True))
    return json.dumps(
        read_user_profile(create_if_missing=create_if_missing),
        ensure_ascii=True,
    )
