from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from app.library.markdown_store import content_list, content_save, content_status_update
from app.tools.url_analyze_tool import url_analyze
from app.tools.youtube_analyze_tool import youtube_analyze


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

CONTENT_ADD_SCHEMA = {
    "name": "content_add",
    "description": (
        "Analyze a URL into a normalized content_profile and save it to the "
        "local Markdown library in one step."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "A YouTube URL or non-YouTube URL to analyze and save.",
            },
            "status": {
                "type": "string",
                "description": "Initial reading state. Defaults to unread.",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes to include in the Markdown body.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}

CONTENT_LIST_SCHEMA = {
    "name": "content_list",
    "description": (
        "List saved Markdown library items. Filter by subject, category, depth, "
        "status, time window, free-text query, and sort order."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "depth_level": {
                "oneOf": [
                    {"type": "string", "enum": ["light", "medium", "deep"]},
                    {
                        "type": "array",
                        "items": {"type": "string", "enum": ["light", "medium", "deep"]},
                    },
                ],
            },
            "status": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
            },
            "exclude_status": {"type": "array", "items": {"type": "string"}},
            "min_estimated_time_minutes": {"type": "integer"},
            "max_estimated_time_minutes": {"type": "integer"},
            "query": {"type": "string"},
            "sort": {
                "type": "string",
                "enum": [
                    "created_at",
                    "updated_at",
                    "estimated_time_minutes",
                    "confidence",
                    "relevance",
                ],
            },
            "limit": {"type": "integer"},
        },
        "additionalProperties": False,
    },
}

CONTENT_STATUS_UPDATE_SCHEMA = {
    "name": "content_status_update",
    "description": (
        "Update a saved Markdown library item's reading status by id, URL, or source identity. "
        "Use this to mark items unread, started, done, archived, or abandoned."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "url": {"type": "string"},
            "source_type": {"type": "string"},
            "source_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["unread", "started", "done", "archived", "abandoned"],
            },
            "notes": {"type": "string"},
        },
        "additionalProperties": False,
    },
}


def content_add(arguments: dict[str, Any], *, library_root: Optional[Path] = None) -> dict[str, Any]:
    url = str(arguments.get("url") or "").strip()
    if not url:
        return {"status": "error", "error": "Missing required argument: url"}

    analyzer_arguments = {"url": url, "task": "content_profile"}
    raw_result = youtube_analyze(analyzer_arguments) if is_youtube_url(url) else url_analyze(analyzer_arguments)
    profile = json.loads(raw_result)
    if profile.get("status") != "completed":
        return profile

    save_input = dict(profile)
    save_input.pop("status", None)
    save_input["status"] = str(arguments.get("status") or "unread").strip()
    if arguments.get("notes"):
        save_input["notes"] = str(arguments["notes"])

    if library_root is None:
        saved = content_save(save_input)
    else:
        saved = content_save(save_input, library_root=library_root)

    return {
        "status": "saved",
        "id": saved["id"],
        "path": saved["path"],
        "created": saved["created"],
        "duplicate": saved["duplicate"],
        "item": saved["item"],
    }


def is_youtube_url(url: str) -> bool:
    return bool(
        re.search(
            r"(^[a-zA-Z0-9_-]{11}$|youtu\.be/|youtube\.com/(watch|shorts|embed|live))",
            url,
        )
    )


def content_add_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(content_add(arguments), ensure_ascii=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)


def content_save_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(content_save(arguments), ensure_ascii=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)


def content_status_update_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(
            content_status_update(
                item_id=arguments.get("id"),
                url=arguments.get("url"),
                source_type=arguments.get("source_type"),
                source_id=arguments.get("source_id"),
                status=arguments.get("status"),
                notes=arguments.get("notes"),
            ),
            ensure_ascii=True,
        )
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)


def content_list_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(content_list(arguments), ensure_ascii=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)
