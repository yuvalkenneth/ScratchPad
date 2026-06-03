from __future__ import annotations

import re
from typing import Any


DEFAULT_LIMIT = 20
DEFAULT_SORT = "created_at"
SORT_VALUES = {"created_at", "updated_at", "estimated_time_minutes", "confidence", "relevance"}


def query_items(items: list[dict[str, Any]], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    filters = filters or {}
    sort = str(filters.get("sort") or DEFAULT_SORT)
    if sort not in SORT_VALUES:
        raise ValueError(f"sort must be one of: {', '.join(sorted(SORT_VALUES))}")

    matched: list[dict[str, Any]] = []
    for item in items:
        match = match_item(item, filters)
        if not match["matched"]:
            continue
        result = dict(item)
        result.pop("_body", None)
        result["match_score"] = match["score"]
        result["match_reasons"] = match["reasons"]
        matched.append(result)

    matched.sort(key=lambda item: sort_key(item, sort), reverse=sort != "estimated_time_minutes")
    limit = coerce_int(filters.get("limit"), default=DEFAULT_LIMIT)
    return {"items": matched[:limit], "count": len(matched)}


def match_item(item: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    score = 0

    if not matches_any_value(item.get("status"), filters.get("status")):
        return no_match()
    if filters.get("status"):
        reasons.append(f"status:{item.get('status')}")

    if filters.get("exclude_status") and matches_any_value(item.get("status"), filters.get("exclude_status")):
        return no_match()

    if not matches_any_value(item.get("depth_level"), filters.get("depth_level")):
        return no_match()
    if filters.get("depth_level"):
        reasons.append(f"depth:{item.get('depth_level')}")

    min_time = filters.get("min_estimated_time_minutes")
    if min_time is not None and estimated_time(item) < coerce_int(min_time, default=0):
        return no_match()

    max_time = filters.get("max_estimated_time_minutes")
    if max_time is not None and estimated_time(item) > coerce_int(max_time, default=0):
        return no_match()
    if min_time is not None or max_time is not None:
        reasons.append(f"time:{estimated_time(item)}m")

    if filters.get("subject"):
        subject = normalize_text(str(item.get("subject") or ""))
        needle = normalize_text(str(filters["subject"]))
        if needle not in subject:
            return no_match()
        score += 6
        reasons.append("subject")

    if filters.get("categories"):
        requested = normalized_values(filters["categories"])
        item_categories = normalized_values(item.get("categories", []))
        matches = sorted(requested & item_categories)
        if not matches:
            return no_match()
        score += 5 * len(matches)
        reasons.append("categories:" + ",".join(matches))

    query = str(filters.get("query") or "").strip()
    if query:
        query_match = score_text_query(item, query)
        if query_match["score"] <= 0:
            return no_match()
        score += query_match["score"]
        reasons.extend(query_match["reasons"])

    if item.get("status") == "started":
        score += 2
    if item.get("status") == "unread":
        score += 1
    score += min(2, max(0, coerce_float(item.get("confidence"), default=0.0) * 2))

    return {"matched": True, "score": round(score, 3), "reasons": reasons}


def score_text_query(item: dict[str, Any], query: str) -> dict[str, Any]:
    query_text = normalize_text(query)
    terms = [term for term in query_text.split() if len(term) > 1]
    fields = {
        "title": str(item.get("title") or ""),
        "subject": str(item.get("subject") or ""),
        "categories": " ".join(str(category) for category in item.get("categories", [])),
        "summary": str(item.get("summary") or ""),
        "body": str(item.get("_body") or ""),
    }
    weights = {
        "title": 8,
        "subject": 6,
        "categories": 5,
        "summary": 3,
        "body": 1,
    }

    score = 0
    reasons: list[str] = []
    for field, value in fields.items():
        normalized = normalize_text(value)
        if not normalized:
            continue
        field_score = 0
        if query_text and query_text in normalized:
            field_score += weights[field]
        field_score += sum(1 for term in terms if term in normalized)
        if field_score:
            score += field_score
            reasons.append(f"query:{field}")

    return {"score": score, "reasons": reasons}


def sort_key(item: dict[str, Any], sort: str) -> Any:
    if sort == "relevance":
        return (
            coerce_float(item.get("match_score"), default=0.0),
            str(item.get("created_at") or ""),
        )
    if sort == "estimated_time_minutes":
        return estimated_time(item)
    if sort == "confidence":
        return (
            coerce_float(item.get("confidence"), default=0.0),
            str(item.get("created_at") or ""),
        )
    return str(item.get(sort) or "")


def matches_any_value(value: Any, expected: Any) -> bool:
    expected_values = normalized_values(expected)
    if not expected_values:
        return True
    return normalize_text(str(value or "")) in expected_values


def normalized_values(value: Any) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        return {normalize_text(value)}
    if isinstance(value, list):
        return {normalize_text(str(item)) for item in value if str(item).strip()}
    return {normalize_text(str(value))}


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def estimated_time(item: dict[str, Any]) -> int:
    return coerce_int(item.get("estimated_time_minutes"), default=0)


def no_match() -> dict[str, Any]:
    return {"matched": False, "score": 0, "reasons": []}


def coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
