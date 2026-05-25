from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


DEPTH_LEVELS = {"light", "medium", "deep"}


@dataclass
class ContentProfile:
    summary: str
    subject: str
    depth_level: str
    categories: list[str]
    estimated_time_minutes: int
    confidence: float

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

        raw_categories = profile.get("categories") or []
        if isinstance(raw_categories, str):
            categories = [item.strip() for item in raw_categories.split(",") if item.strip()]
        elif isinstance(raw_categories, list):
            categories = [str(item).strip() for item in raw_categories if str(item).strip()]
        else:
            categories = []

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
            categories=categories[:4],
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
    if extra_fields:
        payload.update(extra_fields)
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
    return payload
