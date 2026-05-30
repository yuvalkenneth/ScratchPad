from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


DEPTH_LEVELS = {"light", "medium", "deep"}
READING_STATUS_VALUES = {"unread", "started", "done", "archived", "abandoned"}
ANALYZER_STATUS_VALUES = {"completed"}
CONTENT_ITEM_FIELDS = {
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
    "metadata",
    "created_at",
    "updated_at",
    "notes",
}
REQUIRED_CONTENT_ITEM_FIELDS = {
    "source_type",
    "url",
    "title",
    "summary",
    "subject",
    "depth_level",
    "estimated_time_minutes",
}


@dataclass
class ContentProfile:
    summary: str
    subject: str
    depth_level: str
    categories: list[str]
    estimated_time_minutes: int
    confidence: float

    @classmethod
    def from_saved_item(cls, item: dict[str, Any]) -> "ContentProfile":
        summary = str(item["summary"]).strip()
        subject = str(item["subject"]).strip()
        depth_level = str(item.get("depth_level") or "").strip().lower()
        if depth_level not in DEPTH_LEVELS:
            raise ValueError("depth_level must be one of: light, medium, deep")

        try:
            estimated_time_minutes = int(item.get("estimated_time_minutes"))
        except (TypeError, ValueError):
            raise ValueError("estimated_time_minutes must be an integer")
        if estimated_time_minutes < 1:
            raise ValueError("estimated_time_minutes must be at least 1")

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return cls(
            summary=summary,
            subject=subject,
            depth_level=depth_level,
            categories=coerce_categories(item.get("categories") or []),
            estimated_time_minutes=estimated_time_minutes,
            confidence=max(0.0, min(1.0, confidence)),
        )

    @classmethod
    def from_model_output(
        cls,
        profile: dict[str, Any],
        *,
        estimated_time_minutes: int,
        trust_model_time: bool,
    ) -> "ContentProfile":
        summary = str(profile.get("summary") or "").strip()
        subject = str(profile.get("subject") or "").strip()
        depth_level = str(profile.get("depth_level") or "").strip().lower()
        if depth_level not in DEPTH_LEVELS:
            depth_level = "medium"

        time_value = estimated_time_minutes
        if trust_model_time:
            try:
                time_value = int(profile.get("estimated_time_minutes"))
            except (TypeError, ValueError):
                time_value = estimated_time_minutes

        try:
            confidence_value = float(profile.get("confidence"))
        except (TypeError, ValueError):
            confidence_value = 0.5

        return cls(
            summary=summary,
            subject=subject,
            depth_level=depth_level,
            categories=coerce_categories(profile.get("categories") or [])[:4],
            estimated_time_minutes=max(1, time_value),
            confidence=max(0.0, min(1.0, confidence_value)),
        )

    @classmethod
    def fallback(cls, *, estimated_time_minutes: int) -> "ContentProfile":
        return cls(
            summary="",
            subject="",
            depth_level="medium",
            categories=[],
            estimated_time_minutes=max(1, estimated_time_minutes),
            confidence=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContentItem:
    source_type: str
    source_id: Any
    url: str
    title: str
    profile: ContentProfile
    id: str | None = None
    status: str = "unread"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Any = None
    updated_at: Any = None

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "ContentItem":
        missing = sorted(
            field for field in REQUIRED_CONTENT_ITEM_FIELDS if item.get(field) in {None, ""}
        )
        if missing:
            raise ValueError(f"Missing required content item fields: {', '.join(missing)}")

        return cls(
            id=str(item.get("id") or "").strip() or None,
            source_type=str(item["source_type"]).strip(),
            source_id=item.get("source_id"),
            url=str(item["url"]).strip(),
            title=str(item["title"]).strip(),
            profile=ContentProfile.from_saved_item(item),
            status=coerce_reading_status(item.get("status")),
            metadata=coerce_metadata(item),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            **self.profile.to_dict(),
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def coerce_reading_status(value: Any) -> str:
    status = str(value or "unread").strip()
    if status in ANALYZER_STATUS_VALUES:
        return "unread"
    if status not in READING_STATUS_VALUES:
        raise ValueError(f"status must be one of: {', '.join(sorted(READING_STATUS_VALUES))}")
    return status


def coerce_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}

    extras = {
        key: value
        for key, value in item.items()
        if key not in CONTENT_ITEM_FIELDS and value is not None
    }
    if extras:
        metadata = {**metadata, **extras}
    return metadata


def coerce_categories(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON was not an object.")
    return parsed


def coerce_content_profile(
    profile: dict[str, Any],
    *,
    estimated_time_minutes: int,
    trust_model_time: bool,
) -> dict[str, Any]:
    return ContentProfile.from_model_output(
        profile,
        estimated_time_minutes=estimated_time_minutes,
        trust_model_time=trust_model_time,
    ).to_dict()


def fallback_content_profile(*, estimated_time_minutes: int) -> dict[str, Any]:
    return ContentProfile.fallback(estimated_time_minutes=estimated_time_minutes).to_dict()


def build_content_profile_payload(
    *,
    source_type: str,
    source_id: Any,
    url: str,
    title: str,
    estimated_time_minutes: int,
    raw_analysis: str,
    trust_model_time: bool,
    status: str = "completed",
    task: str = "content_profile",
    metadata: Any = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "source_type": source_type,
        "source_id": source_id,
        "url": url,
        "title": title,
        "task": task,
    }
    if metadata is not None:
        payload["metadata"] = metadata

    try:
        profile = coerce_content_profile(
            extract_json_object(raw_analysis),
            estimated_time_minutes=estimated_time_minutes,
            trust_model_time=trust_model_time,
        )
    except Exception:
        profile = fallback_content_profile(estimated_time_minutes=estimated_time_minutes)
        payload["raw_analysis"] = raw_analysis

    payload.update(profile)
    if extra_fields:
        payload["metadata"] = {**coerce_metadata(payload), **extra_fields}
    return payload
