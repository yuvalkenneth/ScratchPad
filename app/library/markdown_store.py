from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.content import ContentItem, READING_STATUS_VALUES
from app.library.git_history import commit_library_paths
from app.library.query import query_items


DEFAULT_LIBRARY_ROOT = Path(os.getenv("SCRATCHPAD_LIBRARY_DIR", "library"))


def content_save(item: dict[str, Any], *, library_root: Path = DEFAULT_LIBRARY_ROOT) -> dict[str, Any]:
    normalized = normalize_item(item)
    items_dir = library_root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    existing_path = find_existing_content_path(normalized, library_root=library_root)
    item_id = normalized.get("id") or (
        read_item_file(existing_path)["frontmatter"].get("id") if existing_path else None
    )
    item_id = item_id or build_item_id(normalized)
    normalized["id"] = item_id
    path = existing_path or items_dir / f"{item_id}.md"

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
    git_result = commit_library_paths(
        library_root=library_root,
        paths=[path],
        message=content_save_commit_message(normalized, created=existing is None),
    )

    return {
        "status": "saved",
        "id": item_id,
        "path": str(path),
        "created": existing is None,
        "duplicate": existing is not None,
        "item": normalized,
        "git": git_result,
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
    previous_status = frontmatter.get("status")
    if status is not None:
        frontmatter["status"] = status
    frontmatter["updated_at"] = utc_now()

    body = parsed["body"]
    if notes is not None:
        body = replace_notes_section(body, notes)

    found.write_text(render_frontmatter(frontmatter) + "\n" + body.strip() + "\n", encoding="utf-8")
    updated = read_item_file(found)["frontmatter"]
    updated["path"] = str(found)
    git_result = commit_library_paths(
        library_root=library_root,
        paths=[found],
        message=content_update_commit_message(
            updated,
            previous_status=previous_status,
            requested_status=status,
            notes_updated=notes is not None,
        ),
    )
    return {
        "status": "updated",
        "id": updated.get("id"),
        "path": str(found),
        "item": updated,
        "git": git_result,
    }


def content_list(
    filters: dict[str, Any] | None = None,
    *,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
) -> dict[str, Any]:
    items_dir = library_root / "items"
    if not items_dir.exists():
        return {"status": "completed", "items": [], "count": 0}

    items: list[dict[str, Any]] = []
    for path in sorted(items_dir.glob("*.md")):
        parsed = read_item_file(path)
        item = dict(parsed["frontmatter"])
        item["path"] = str(path)
        item["_body"] = parsed["body"]
        items.append(item)

    result = query_items(items, filters)
    return {"status": "completed", **result}


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


def content_save_commit_message(item: dict[str, Any], *, created: bool) -> str:
    action = "Add" if created else "Update"
    return f"{action} content: {item_title(item)}"


def content_update_commit_message(
    item: dict[str, Any],
    *,
    previous_status: Any,
    requested_status: Optional[str],
    notes_updated: bool,
) -> str:
    title = item_title(item)
    status_changed = requested_status is not None and requested_status != previous_status
    if status_changed and not notes_updated:
        return f"Update status: {title} -> {requested_status}"
    if notes_updated and not status_changed:
        return f"Update notes: {title}"
    return f"Update content: {title}"


def item_title(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("id") or "Untitled").strip()


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


def find_existing_content_path(
    item: dict[str, Any],
    *,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
) -> Path | None:
    item_id = str(item.get("id") or "").strip()
    if item_id:
        path = library_root / "items" / f"{item_id}.md"
        if path.exists():
            return path

    source_id = item.get("source_id")
    source_id_value = str(source_id) if source_id is not None else None
    return find_item_path(
        url=str(item.get("url") or "").strip() or None,
        source_type=str(item.get("source_type") or "").strip() or None,
        source_id=source_id_value,
        library_root=library_root,
    )


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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
