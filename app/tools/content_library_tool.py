from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from app.library.markdown_store import content_list, content_save, content_status_update, content_update
from app.tools.url_analyze_tool import url_analyze
from app.tools.youtube_analyze_tool import youtube_analyze


DEFAULT_CONTENT_LIST_STATUSES = ["unread", "started"]


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
            "learning_effort_minutes": {"type": ["integer", "null"]},
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

ANALYZE_SOURCE_SCHEMA = {
    "name": "analyze_source",
    "description": (
        "Analyze a URL or YouTube video into a normalized content profile without saving it. "
        "Use this when the user wants to know what a source is about or whether it is worth "
        "reading/watching before adding it to the library."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "A YouTube URL or non-YouTube URL to analyze without saving.",
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
        "status, time window, free-text query, and sort order. If status is omitted, "
        "defaults to unread and started items so recommendations do not include done, "
        "archived, or abandoned content unless explicitly requested."
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
                "description": "Reading status filter. Defaults to ['unread', 'started'] when omitted.",
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

CONTENT_UPDATE_SCHEMA = {
    "name": "content_update",
    "description": (
        "Update saved content metadata/profile fields by id, URL, or source identity. "
        "Use this for corrections to title, summary, subject, categories, depth, time, "
        "confidence, metadata, or notes. For reading state only, prefer content_status_update."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "url": {"type": "string"},
            "source_type": {"type": "string"},
            "source_id": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "subject": {"type": "string"},
            "depth_level": {"type": "string", "enum": ["light", "medium", "deep"]},
            "categories": {"type": "array", "items": {"type": "string"}},
            "estimated_time_minutes": {"type": "integer"},
            "learning_effort_minutes": {"type": ["integer", "null"]},
            "confidence": {"type": "number"},
            "metadata": {"type": "object"},
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
    missing_profile_fields = [
        field
        for field in ("summary", "subject", "depth_level", "estimated_time_minutes")
        if save_input.get(field) in {None, ""}
    ]
    if missing_profile_fields:
        return {
            "status": "error",
            "error": "Analysis did not produce a saveable content profile.",
            "missing_fields": missing_profile_fields,
            "analysis": profile,
        }

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
        "git": saved["git"],
    }


def analyze_source(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url") or "").strip()
    if not url:
        return {"status": "error", "error": "Missing required argument: url"}

    analyzer_arguments = {"url": url, "task": "content_profile"}
    raw_result = youtube_analyze(analyzer_arguments) if is_youtube_url(url) else url_analyze(analyzer_arguments)
    return json.loads(raw_result)


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


def analyze_source_json(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(analyze_source(arguments), ensure_ascii=True)
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


def content_update_json(arguments: dict[str, Any]) -> str:
    try:
        update_fields = {
            key: arguments[key]
            for key in (
                "title",
                "summary",
                "subject",
                "depth_level",
                "categories",
                "estimated_time_minutes",
                "learning_effort_minutes",
                "confidence",
                "metadata",
            )
            if key in arguments
        }
        return json.dumps(
            content_update(
                item_id=arguments.get("id"),
                url=arguments.get("url"),
                source_type=arguments.get("source_type"),
                source_id=arguments.get("source_id"),
                updates=update_fields,
                notes=arguments.get("notes"),
            ),
            ensure_ascii=True,
        )
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)


def content_list_json(arguments: dict[str, Any]) -> str:
    try:
        filters = apply_content_list_defaults(arguments)
        result = content_list(filters)
        if "status" not in arguments:
            result["applied_defaults"] = {"status": DEFAULT_CONTENT_LIST_STATUSES}
        return json.dumps(result, ensure_ascii=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True)


def apply_content_list_defaults(arguments: dict[str, Any]) -> dict[str, Any]:
    filters = dict(arguments)
    if "status" not in filters:
        filters["status"] = list(DEFAULT_CONTENT_LIST_STATUSES)
    return filters
