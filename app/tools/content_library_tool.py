from __future__ import annotations

import json
from typing import Any

from app.library.markdown_store import content_list, content_save


CONTENT_SAVE_SCHEMA = {
    "name": "content_save",
    "description": (
        "Save a normalized content profile to the local Markdown library. "
        "Use this after url_analyze or youtube_analyze returns a content_profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_type": {"type": "string"},
            "source_id": {"type": ["string", "null"]},
            "url": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "subject": {"type": "string"},
            "depth_level": {"type": "string", "enum": ["light", "medium", "deep"]},
            "categories": {"type": "array", "items": {"type": "string"}},
            "estimated_time_minutes": {"type": "integer"},
            "confidence": {"type": "number"},
            "status": {
                "type": "string",
                "description": "Reading state. Defaults to unread.",
            },
            "metadata": {"type": "object"},
            "notes": {"type": "string"},
        },
        "required": [
            "source_type",
            "url",
            "title",
            "summary",
            "subject",
            "depth_level",
            "estimated_time_minutes",
        ],
        "additionalProperties": False,
    },
}

CONTENT_LIST_SCHEMA = {
    "name": "content_list",
    "description": (
        "List saved Markdown library items. Filter by subject, category, depth, "
        "status, maximum time, or free-text query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "depth_level": {"type": "string", "enum": ["light", "medium", "deep"]},
            "status": {"type": "string"},
            "max_estimated_time_minutes": {"type": "integer"},
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "additionalProperties": False,
    },
}


def content_save_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(content_save(arguments), ensure_ascii=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)


def content_list_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(content_list(arguments), ensure_ascii=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)
