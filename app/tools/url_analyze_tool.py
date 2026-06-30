from __future__ import annotations

import json
from typing import Any

import httpx

from app.fetchers.utils import estimate_time_minutes, http_error_payload
from app.fetchers.router import fetch_source
from app.content_profile_prompt import (
    common_content_profile_field_guidance,
    content_profile_schema_instruction,
)
from app.llm.config import LLMConfig
from app.llm.openai_compatible import complete_text
from app.llm.catalog import config_from_model_ref
from app.content import build_content_profile_payload


CONTENT_PROFILE_CONTEXT = (
    "This profile will be saved in a local learning library and later used to help "
    "a user decide what to read, revisit, search, filter, or get recommended."
)

URL_ANALYZE_SCHEMA = {
    "name": "url_analyze",
    "description": (
        "Fetch a web page internally, extract compact readable text, and classify it into "
        "a content profile without exposing the raw page text to the main chat context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "A publicly accessible article, blog post, or documentation URL.",
            },
            "task": {
                "type": "string",
                "description": (
                    "Analysis task to perform. Use 'content_profile' for product-facing "
                    "classification output."
                ),
                "enum": ["content_profile"],
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}

def _analysis_prompt(title: str, url: str) -> str:
    lines = [
        "You classify scraped web page content into a compact content profile.",
        CONTENT_PROFILE_CONTEXT,
        "Return a compact JSON object only.",
        content_profile_schema_instruction(),
        *common_content_profile_field_guidance(
            consumption_time_label=(
                "estimated_time_minutes: minutes needed to consume and understand enough to "
                "decide whether to revisit, save, or act on this item."
            )
        ),
        "Output JSON only. Do not wrap it in markdown fences.",
    ]
    if title:
        lines.append(f"Page title: {title}")
    lines.append(f"URL: {url}")
    return "\n".join(lines)


def _complete_text(
    config: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int = 800,
) -> str:
    return complete_text(config, messages, max_tokens=max_tokens, temperature=0.2)


def _http_error_payload(exc: Exception) -> str:
    return http_error_payload(exc)


def url_analyze(arguments: dict[str, Any]) -> str:
    url = str(arguments.get("url") or "").strip()
    task = str(arguments.get("task") or "content_profile").strip()
    if not url:
        return json.dumps({"status": "error", "error": "Missing required argument: url"})
    if task != "content_profile":
        return json.dumps({"status": "error", "error": f"Unsupported analysis task: {task}"})

    try:
        source_data = fetch_source(url)
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)
    except httpx.RequestError as exc:
        return _http_error_payload(exc)
    except Exception as exc:
        return _http_error_payload(exc)

    text = str(source_data.get("text") or "").strip()
    title = str(source_data.get("title") or "").strip()
    if not text:
        return json.dumps({"status": "error", "error": "Could not extract readable text from URL."})

    word_count = int(source_data.get("word_count") or len(text.split()))
    estimated_time_minutes = int(source_data.get("estimated_time_minutes") or estimate_time_minutes(text))
    metadata = source_data.get("metadata")
    extraction_quality = metadata.get("extraction_quality") if isinstance(metadata, dict) else None
    user_content = text
    if extraction_quality == "metadata_only":
        user_content = (
            "Extraction note: only page metadata was available; the rendered article body "
            "was not present in the static HTML. Keep the profile conservative and lower confidence.\n\n"
            f"{text}"
        )
    config = config_from_model_ref(None)
    messages = [
        {"role": "system", "content": _analysis_prompt(title, url)},
        {"role": "user", "content": user_content},
    ]
    raw_analysis = _complete_text(config, messages)

    payload = build_content_profile_payload(
        source_type=str(source_data.get("source_type") or "web"),
        source_id=source_data.get("source_id"),
        url=str(source_data.get("url") or url),
        title=title,
        estimated_time_minutes=estimated_time_minutes,
        raw_analysis=raw_analysis,
        trust_model_time=True,
        metadata=metadata,
        extra_fields={"word_count": word_count},
    )
    if extraction_quality == "metadata_only":
        payload["confidence"] = min(float(payload.get("confidence") or 0.0), 0.6)

    return json.dumps(payload, ensure_ascii=True)
