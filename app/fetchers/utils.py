from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx


MAX_PAGE_CHARS = 18_000
WORDS_PER_MINUTE = 200
HTTP_TIMEOUT_SECONDS = 15
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
    )
}


def fetch_url_html(url: str) -> str:
    with httpx.Client(
        headers=DEFAULT_HTTP_HEADERS,
        follow_redirects=True,
        timeout=HTTP_TIMEOUT_SECONDS,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def fetch_json(url: str, headers: Optional[dict[str, str]] = None) -> Any:
    request_headers = {
        **DEFAULT_HTTP_HEADERS,
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    with httpx.Client(
        headers=request_headers,
        follow_redirects=True,
        timeout=HTTP_TIMEOUT_SECONDS,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str) -> str:
    if len(text) <= MAX_PAGE_CHARS:
        return text
    return text[:MAX_PAGE_CHARS].rsplit(" ", 1)[0].strip()


def truncate_lines(items: list[str], *, limit: int) -> list[str]:
    return [item for item in items if item][:limit]


def estimate_time_minutes(text: str) -> int:
    word_count = len(text.split())
    return max(1, round(word_count / WORDS_PER_MINUTE))


def host_for_url(url: str) -> str:
    return (httpx.URL(url).host or "").lower()


def http_error_payload(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return json.dumps(
            {"status": "error", "error": f"HTTP error while fetching URL: {exc.response.status_code}"}
        )
    if isinstance(exc, httpx.RequestError):
        return json.dumps(
            {"status": "error", "error": f"Network error while fetching URL: {exc}"}
        )
    return json.dumps({"status": "error", "error": f"Failed to fetch URL: {exc}"})
