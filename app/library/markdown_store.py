from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.content import ContentItem, READING_STATUS_VALUES


DEFAULT_LIBRARY_ROOT = Path(os.getenv("SCRATCHPAD_LIBRARY_DIR", "library"))


def content_save(item: dict[str, Any], *, library_root: Path = DEFAULT_LIBRARY_ROOT) -> dict[str, Any]:
    normalized = normalize_item(item)
    items_dir = library_root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    item_id = normalized.get("id") or build_item_id(normalized)
    normalized["id"] = item_id
    path = items_dir / f"{item_id}.md"

    existing = read_item_file(path) if path.exists() else None
    now = utc_now()
    normalized["created_at"] = normalized.get("created_at") or (
        existing["frontmatter"].get("created_at") if existing else now
    )
    normalized["updated_at"] = now
    if existing and normalized["status"] == "unread":
        normalized["status"] = str(existing["frontmatter"].get("status") or normalized["status"])

    notes = str(item.get("notes") or "")
    if existing and "notes" not in item:
        notes = extract_notes_section(existing["body"])
    body = build_markdown_body(normalized, notes)
    path.write_text(render_frontmatter(normalized) + "\n" + body, encoding="utf-8")

    return {
        "status": "saved",
        "id": item_id,
        "path": str(path),
        "created": existing is None,
        "duplicate": existing is not None,
        "item": normalized,
    }


def content_status_update(
    *,
    item_id: Optional[str] = None,
    url: Optional[str] = None,
    source_type: Optional[str] = None,
    source_id: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
) -> dict[str, Any]:
    if status is None and notes is None:
        return {"status": "error", "error": "Provide at least one field to update."}
    if status is not None and status not in READING_STATUS_VALUES:
        return {
            "status": "error",
            "error": f"status must be one of: {', '.join(sorted(READING_STATUS_VALUES))}",
        }

    found = find_item_path(
        item_id=item_id,
        url=url,
        source_type=source_type,
        source_id=source_id,
        library_root=library_root,
    )
    if found is None:
        return {"status": "error", "error": "No matching content item found."}

    parsed = read_item_file(found)
    frontmatter = dict(parsed["frontmatter"])
    if status is not None:
        frontmatter["status"] = status
    frontmatter["updated_at"] = utc_now()

    body = parsed["body"]
    if notes is not None:
        body = replace_notes_section(body, notes)

    found.write_text(render_frontmatter(frontmatter) + "\n" + body.strip() + "\n", encoding="utf-8")
    updated = read_item_file(found)["frontmatter"]
    updated["path"] = str(found)
    return {"status": "updated", "id": updated.get("id"), "path": str(found), "item": updated}


def content_list(
    filters: dict[str, Any] | None = None,
    *,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
) -> dict[str, Any]:
    filters = filters or {}
    items_dir = library_root / "items"
    if not items_dir.exists():
        return {"status": "completed", "items": [], "count": 0}

    items: list[dict[str, Any]] = []
    for path in sorted(items_dir.glob("*.md")):
        parsed = read_item_file(path)
        item = dict(parsed["frontmatter"])
        item["path"] = str(path)
        item["_body"] = parsed["body"]
        if matches_filters(item, filters):
            item.pop("_body", None)
            items.append(item)

    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    limit = coerce_int(filters.get("limit"), default=20)
    return {"status": "completed", "items": items[:limit], "count": len(items)}


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    return ContentItem.from_mapping(item).to_dict()


def build_item_id(item: dict[str, Any]) -> str:
    source_type = slugify(str(item.get("source_type") or "item"))
    source_id = item.get("source_id")
    stable_key = f"{source_type}:{source_id}" if source_id else f"url:{item['url']}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:10]
    slug_source = item.get("subject") or item.get("title") or source_type
    slug = slugify(str(slug_source)) or "item"
    return f"{source_type}-{slug}-{digest}"


def slugify(text: str) -> str:
    lowered = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = slug.strip("-")
    return re.sub(r"-{2,}", "-", slug)[:80]


def render_frontmatter(item: dict[str, Any]) -> str:
    lines = ["---"]
    for key in [
        "id",
        "source_type",
        "source_id",
        "url",
        "title",
        "summary",
        "subject",
        "depth_level",
        "categories",
        "estimated_time_minutes",
        "confidence",
        "status",
        "created_at",
        "updated_at",
        "metadata",
    ]:
        lines.append(f"{key}: {format_frontmatter_value(item.get(key))}")
    lines.append("---")
    return "\n".join(lines)


def format_frontmatter_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def build_markdown_body(item: dict[str, Any], notes: str) -> str:
    sections = [
        f"# {item['title']}",
        "",
        item["summary"],
        "",
        f"Source: {item['url']}",
    ]
    if notes.strip():
        sections.extend(["", "## Notes", "", notes.strip()])
    return "\n".join(sections).strip() + "\n"


def read_item_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {"frontmatter": {}, "body": text}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"frontmatter": {}, "body": text}
    _, frontmatter, body = parts
    return {"frontmatter": parse_frontmatter(frontmatter), "body": body.strip()}


def find_item_path(
    *,
    item_id: Optional[str] = None,
    url: Optional[str] = None,
    source_type: Optional[str] = None,
    source_id: Optional[str] = None,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
) -> Path | None:
    items_dir = library_root / "items"
    if not items_dir.exists():
        return None

    if item_id:
        target = items_dir / f"{item_id}.md"
        if target.exists():
            return target

    for path in sorted(items_dir.glob("*.md")):
        frontmatter = read_item_file(path)["frontmatter"]
        if url and frontmatter.get("url") == url:
            return path
        if (
            source_type
            and source_id
            and frontmatter.get("source_type") == source_type
            and frontmatter.get("source_id") == source_id
        ):
            return path
    return None


def replace_notes_section(body: str, notes: str) -> str:
    body = body.strip()
    if "## Notes" not in body:
        return f"{body}\n\n## Notes\n\n{notes.strip()}\n" if notes.strip() else f"{body}\n"

    before, _separator, _after = body.partition("## Notes")
    if not notes.strip():
        return before.strip() + "\n"
    return f"{before.strip()}\n\n## Notes\n\n{notes.strip()}\n"


def extract_notes_section(body: str) -> str:
    if "## Notes" not in body:
        return ""
    _before, _separator, after = body.partition("## Notes")
    return after.strip()


def parse_frontmatter(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        metadata[key.strip()] = parse_frontmatter_value(raw_value.strip())
    return metadata


def parse_frontmatter_value(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def matches_filters(item: dict[str, Any], filters: dict[str, Any]) -> bool:
    if filters.get("subject"):
        needle = str(filters["subject"]).lower()
        if needle not in str(item.get("subject") or "").lower():
            return False

    if filters.get("depth_level") and item.get("depth_level") != filters["depth_level"]:
        return False

    if filters.get("status") and item.get("status") != filters["status"]:
        return False

    max_time = filters.get("max_estimated_time_minutes")
    if max_time is not None:
        max_time_value = coerce_int(max_time, default=0)
        if coerce_int(item.get("estimated_time_minutes"), default=0) > max_time_value:
            return False

    if filters.get("categories"):
        requested = filters["categories"]
        if isinstance(requested, str):
            requested = [requested]
        item_categories = {str(category).lower() for category in item.get("categories", [])}
        if not any(str(category).lower() in item_categories for category in requested):
            return False

    if filters.get("query"):
        query = str(filters["query"]).lower()
        haystack = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("summary") or ""),
                str(item.get("subject") or ""),
                " ".join(str(category) for category in item.get("categories", [])),
                str(item.get("_body") or ""),
            ]
        ).lower()
        if query not in haystack:
            return False

    return True


def coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
