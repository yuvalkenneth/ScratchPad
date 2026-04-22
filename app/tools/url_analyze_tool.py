from __future__ import annotations

import json
import os
import re
from base64 import b64decode
from html import unescape
from html.parser import HTMLParser
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openai import OpenAI

from app.llm.config import LLMConfig


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

MAX_PAGE_CHARS = 18_000
WORDS_PER_MINUTE = 200
GENERIC_X_ERROR_SNIPPETS = (
    "something went wrong, but don't fret",
    "something went wrong, but don’t fret",
    "some privacy related extensions may cause issues on x.com",
    "try again",
)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta: dict[str, str] = {}
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        if tag == "meta":
            key = (attrs_dict.get("property") or attrs_dict.get("name") or "").strip().lower()
            content = (attrs_dict.get("content") or "").strip()
            if key and content and key not in self.meta:
                self.meta[key] = unescape(content)
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = unescape(data).strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        else:
            self.text_parts.append(text)


def _fetch_url_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=15) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _fetch_json(url: str, headers: Optional[dict[str, str]] = None) -> Any:
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=15) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="replace"))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_PAGE_CHARS:
        return text
    return text[:MAX_PAGE_CHARS].rsplit(" ", 1)[0].strip()


def _host_for_url(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _is_x_domain(url: str) -> bool:
    host = _host_for_url(url)
    return host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}


def _is_github_domain(url: str) -> bool:
    host = _host_for_url(url)
    return host in {"github.com", "www.github.com"}


def _parse_github_repo_url(url: str) -> Optional[tuple[str, str]]:
    if not _is_github_domain(url):
        return None

    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    repo = re.sub(r"\.git$", "", repo)
    if not owner or not repo:
        return None
    return owner, repo


def _clean_x_title(title: str) -> str:
    cleaned = _normalize_whitespace(title)
    cleaned = re.sub(r"\s*/\s*X\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+/\s+Twitter\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_x_description(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    for suffix in (
        " - X",
        " on X",
        " / X",
        " - Twitter",
        " on Twitter",
        " / Twitter",
    ):
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned.strip(" -")


def _looks_like_generic_x_shell(text: str) -> bool:
    lowered = text.lower()
    return any(snippet in lowered for snippet in GENERIC_X_ERROR_SNIPPETS)


def _extract_x_page_content(title: str, text: str, meta: dict[str, str]) -> dict[str, str]:
    meta_title = (
        meta.get("og:title")
        or meta.get("twitter:title")
        or meta.get("title")
        or title
    )
    meta_description = (
        meta.get("og:description")
        or meta.get("twitter:description")
        or meta.get("description")
        or ""
    )

    cleaned_title = _clean_x_title(meta_title)
    cleaned_description = _clean_x_description(meta_description)

    parts: list[str] = []
    for candidate in (cleaned_title, cleaned_description):
        candidate = candidate.strip()
        if candidate and candidate not in parts:
            parts.append(candidate)

    fallback_text = _normalize_whitespace(text)
    if fallback_text and not _looks_like_generic_x_shell(fallback_text):
        parts.append(fallback_text)

    combined_text = _truncate_text(_normalize_whitespace(" ".join(parts)))
    return {"title": cleaned_title or title, "text": combined_text}


def _extract_page_content(url: str, html: str) -> dict[str, str]:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)

    title = _normalize_whitespace(" ".join(extractor.title_parts))
    text = _normalize_whitespace(" ".join(extractor.text_parts))
    if _is_x_domain(url):
        return _extract_x_page_content(title, text, extractor.meta)

    text = _truncate_text(text)
    return {"title": title, "text": text}


def _truncate_lines(items: list[str], *, limit: int) -> list[str]:
    return [item for item in items if item][:limit]


def _strip_readme_noise(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"<img[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"`{3,}.*?`{3,}", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", lambda m: m.group(0).split("](", 1)[0].lstrip("["), cleaned)
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    return cleaned.strip()


def _decode_github_readme(payload: dict[str, Any]) -> str:
    content = str(payload.get("content") or "")
    encoding = str(payload.get("encoding") or "").lower()
    if not content:
        return ""
    if encoding == "base64":
        try:
            return b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return content


def _build_github_repo_text(
    repo_payload: dict[str, Any],
    languages_payload: dict[str, Any],
    readme_text: str,
    root_entries: list[str],
) -> str:
    owner = str((repo_payload.get("owner") or {}).get("login") or "").strip()
    name = str(repo_payload.get("name") or "").strip()
    description = _normalize_whitespace(str(repo_payload.get("description") or ""))
    homepage = _normalize_whitespace(str(repo_payload.get("homepage") or ""))
    topics = [str(item).strip() for item in (repo_payload.get("topics") or []) if str(item).strip()]

    language_names = [str(name).strip() for name in languages_payload.keys() if str(name).strip()]
    language_names = _truncate_lines(language_names, limit=6)
    root_entries = _truncate_lines(root_entries, limit=12)

    readme_excerpt = _strip_readme_noise(readme_text)
    readme_excerpt = _truncate_text(readme_excerpt)

    parts = [
        f"Repository: {owner}/{name}" if owner and name else "",
        f"Description: {description}" if description else "",
        f"Topics: {', '.join(topics[:8])}" if topics else "",
        f"Primary language: {repo_payload.get('language')}" if repo_payload.get("language") else "",
        f"Languages: {', '.join(language_names)}" if language_names else "",
        f"Homepage: {homepage}" if homepage else "",
        f"Default branch: {repo_payload.get('default_branch')}" if repo_payload.get("default_branch") else "",
        f"Root contents: {', '.join(root_entries)}" if root_entries else "",
        f"README excerpt: {readme_excerpt}" if readme_excerpt else "",
    ]
    return _truncate_text(_normalize_whitespace(" ".join(part for part in parts if part)))


def _fetch_github_repo_source(url: str) -> Optional[dict[str, Any]]:
    parsed = _parse_github_repo_url(url)
    if not parsed:
        return None

    owner, repo = parsed
    api_base = f"https://api.github.com/repos/{owner}/{repo}"
    repo_payload = _fetch_json(
        api_base,
        headers={"Accept": "application/vnd.github+json"},
    )
    languages_payload = _fetch_json(
        f"{api_base}/languages",
        headers={"Accept": "application/vnd.github+json"},
    )

    readme_text = ""
    try:
        readme_payload = _fetch_json(
            f"{api_base}/readme",
            headers={"Accept": "application/vnd.github+json"},
        )
        if isinstance(readme_payload, dict):
            readme_text = _decode_github_readme(readme_payload)
    except HTTPError as exc:
        if exc.code != 404:
            raise

    root_entries: list[str] = []
    try:
        contents_payload = _fetch_json(
            f"{api_base}/contents",
            headers={"Accept": "application/vnd.github+json"},
        )
        if isinstance(contents_payload, list):
            for entry in contents_payload:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                entry_type = str(entry.get("type") or "").strip().lower()
                if not name:
                    continue
                root_entries.append(f"{name}/" if entry_type == "dir" else name)
    except HTTPError as exc:
        if exc.code != 404:
            raise

    title = f"{owner}/{repo}"
    text = _build_github_repo_text(repo_payload, languages_payload, readme_text, root_entries)
    metadata = {
        "owner": owner,
        "repo": repo,
        "description": repo_payload.get("description") or "",
        "primary_language": repo_payload.get("language") or "",
        "languages": languages_payload if isinstance(languages_payload, dict) else {},
        "topics": repo_payload.get("topics") or [],
        "default_branch": repo_payload.get("default_branch") or "",
        "homepage": repo_payload.get("homepage") or "",
        "stargazers_count": repo_payload.get("stargazers_count"),
        "forks_count": repo_payload.get("forks_count"),
        "open_issues_count": repo_payload.get("open_issues_count"),
        "license": ((repo_payload.get("license") or {}).get("spdx_id") or ""),
        "archived": bool(repo_payload.get("archived")),
        "pushed_at": repo_payload.get("pushed_at") or "",
        "root_entries": root_entries[:12],
    }
    return {
        "source_type": "repo",
        "source_id": title,
        "url": url,
        "title": title,
        "text": text,
        "word_count": len(text.split()),
        "estimated_time_minutes": _estimate_time_minutes(text),
        "metadata": metadata,
    }


def _estimate_time_minutes(text: str) -> int:
    word_count = len(text.split())
    return max(1, round(word_count / WORDS_PER_MINUTE))


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


def _extract_json_object(text: str) -> dict[str, Any]:
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


def _coerce_profile(
    profile: dict[str, Any],
    *,
    estimated_time_minutes: int,
) -> dict[str, Any]:
    summary = str(profile.get("summary") or "").strip()
    subject = str(profile.get("subject") or "").strip()
    depth_level = str(profile.get("depth_level") or "").strip().lower()
    if depth_level not in {"light", "medium", "deep"}:
        depth_level = "medium"

    raw_categories = profile.get("categories") or []
    if isinstance(raw_categories, str):
        categories = [item.strip() for item in raw_categories.split(",") if item.strip()]
    elif isinstance(raw_categories, list):
        categories = [str(item).strip() for item in raw_categories if str(item).strip()]
    else:
        categories = []
    categories = categories[:4]

    raw_estimated_time = profile.get("estimated_time_minutes")
    try:
        estimated_time_value = int(raw_estimated_time)
    except (TypeError, ValueError):
        estimated_time_value = estimated_time_minutes
    estimated_time_value = max(1, estimated_time_value)

    confidence = profile.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.5
    confidence_value = max(0.0, min(1.0, confidence_value))

    return {
        "summary": summary,
        "subject": subject,
        "depth_level": depth_level,
        "categories": categories,
        "estimated_time_minutes": estimated_time_value,
        "confidence": confidence_value,
    }


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


def url_analyze(arguments: dict[str, Any]) -> str:
    url = str(arguments.get("url") or "").strip()
    task = str(arguments.get("task") or "content_profile").strip()
    if not url:
        return json.dumps({"status": "error", "error": "Missing required argument: url"})
    if task != "content_profile":
        return json.dumps({"status": "error", "error": f"Unsupported analysis task: {task}"})

    source_data: dict[str, Any]
    github_target = _parse_github_repo_url(url)
    if github_target:
        try:
            source_data = _fetch_github_repo_source(url) or {}
        except HTTPError as exc:
            return json.dumps({"status": "error", "error": f"HTTP error while fetching URL: {exc.code}"})
        except URLError as exc:
            return json.dumps({"status": "error", "error": f"Network error while fetching URL: {exc.reason}"})
        except Exception as exc:
            return json.dumps({"status": "error", "error": f"Failed to fetch URL: {exc}"})
    else:
        try:
            html = _fetch_url_html(url)
        except HTTPError as exc:
            return json.dumps({"status": "error", "error": f"HTTP error while fetching URL: {exc.code}"})
        except URLError as exc:
            return json.dumps({"status": "error", "error": f"Network error while fetching URL: {exc.reason}"})
        except Exception as exc:
            return json.dumps({"status": "error", "error": f"Failed to fetch URL: {exc}"})

        page = _extract_page_content(url, html)
        text = page["text"]
        title = page["title"]
        source_data = {
            "source_type": "web",
            "source_id": None,
            "url": url,
            "title": title,
            "text": text,
            "word_count": len(text.split()),
            "estimated_time_minutes": _estimate_time_minutes(text) if text else 1,
        }

    text = str(source_data.get("text") or "").strip()
    title = str(source_data.get("title") or "").strip()
    if not text:
        return json.dumps({"status": "error", "error": "Could not extract readable text from URL."})

    word_count = int(source_data.get("word_count") or len(text.split()))
    estimated_time_minutes = int(source_data.get("estimated_time_minutes") or _estimate_time_minutes(text))
    config = LLMConfig.from_env()
    messages = [
        {"role": "system", "content": _analysis_prompt(title, url)},
        {"role": "user", "content": text},
    ]
    raw_analysis = _complete_text(config, messages)

    payload: dict[str, Any] = {
        "status": "completed",
        "source_type": source_data.get("source_type") or "web",
        "source_id": source_data.get("source_id"),
        "url": url,
        "title": title,
        "task": task,
        "word_count": word_count,
        "estimated_time_minutes": estimated_time_minutes,
    }
    if source_data.get("metadata") is not None:
        payload["metadata"] = source_data["metadata"]
    try:
        parsed = _extract_json_object(raw_analysis)
        payload["profile"] = _coerce_profile(
            parsed,
            estimated_time_minutes=estimated_time_minutes,
        )
    except Exception:
        payload["profile"] = {
            "summary": "",
            "subject": "",
            "depth_level": "medium",
            "categories": [],
            "estimated_time_minutes": estimated_time_minutes,
            "confidence": 0.0,
        }
        payload["raw_analysis"] = raw_analysis

    return json.dumps(payload, ensure_ascii=True)
