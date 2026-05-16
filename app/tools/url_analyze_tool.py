from __future__ import annotations

import json
import os
from typing import Any

import httpx
from openai import OpenAI

from app.fetchers.common import estimate_time_minutes, http_error_payload
from app.fetchers.router import fetch_source
from app.llm.config import LLMConfig
from app.tools.content_profile import build_content_profile_payload


PROVIDER_TO_API_KEY_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "llama_cpp": "LLAMA_CPP_API_KEY",
}

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
        "Return a compact JSON object only.",
        (
            'Use this exact schema: {"summary":"string","subject":"string",'
            '"depth_level":"light|medium|deep","categories":["string"],'
            '"estimated_time_minutes":0,"confidence":0.0}.'
        ),
        "Keep the summary to 1-2 sentences.",
        "Choose the single best primary subject.",
        "Only use depth_level values light, medium, or deep.",
        "Use 1-4 short categories.",
        "Base estimated_time_minutes on likely reading time from the provided content, rounded to an integer.",
        "Output JSON only. Do not wrap it in markdown fences.",
    ]
    if title:
        lines.append(f"Page title: {title}")
    lines.append(f"URL: {url}")
    return "\n".join(lines)


def _complete_text(
    config: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int = 400,
) -> str:
    api_key_env_var = PROVIDER_TO_API_KEY_ENV_VAR.get(config.provider)
    api_key = os.getenv(api_key_env_var) if api_key_env_var else None
    if not api_key:
        api_key = config.api_key or "local"
    client = OpenAI(api_key=api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


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
    config = LLMConfig.from_env()
    messages = [
        {"role": "system", "content": _analysis_prompt(title, url)},
        {"role": "user", "content": text},
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
        metadata=source_data.get("metadata"),
        extra_fields={"word_count": word_count},
    )

    return json.dumps(payload, ensure_ascii=True)
