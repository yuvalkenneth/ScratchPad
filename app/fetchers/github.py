from __future__ import annotations

import re
from base64 import b64decode
from typing import Any, Optional

import httpx

from app.fetchers.common import estimate_time_minutes, fetch_json, host_for_url, normalize_whitespace, truncate_lines, truncate_text


def is_github_domain(url: str) -> bool:
    host = host_for_url(url)
    return host in {"github.com", "www.github.com"}


def parse_github_repo_url(url: str) -> Optional[tuple[str, str]]:
    if not is_github_domain(url):
        return None

    parts = [part for part in httpx.URL(url).path.split("/") if part]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    repo = re.sub(r"\.git$", "", repo)
    if not owner or not repo:
        return None
    return owner, repo


def strip_readme_noise(text: str) -> str:
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
    return normalize_whitespace(cleaned).strip()


def decode_github_readme(payload: dict[str, Any]) -> str:
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


def build_github_repo_text(
    repo_payload: dict[str, Any],
    languages_payload: dict[str, Any],
    readme_text: str,
    root_entries: list[str],
) -> str:
    owner = str((repo_payload.get("owner") or {}).get("login") or "").strip()
    name = str(repo_payload.get("name") or "").strip()
    description = normalize_whitespace(str(repo_payload.get("description") or ""))
    homepage = normalize_whitespace(str(repo_payload.get("homepage") or ""))
    topics = [str(item).strip() for item in (repo_payload.get("topics") or []) if str(item).strip()]

    language_names = [str(name).strip() for name in languages_payload.keys() if str(name).strip()]
    language_names = truncate_lines(language_names, limit=6)
    root_entries = truncate_lines(root_entries, limit=12)

    readme_excerpt = truncate_text(strip_readme_noise(readme_text))

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
    return truncate_text(normalize_whitespace(" ".join(part for part in parts if part)))


def fetch_github_repo_source(url: str) -> Optional[dict[str, Any]]:
    parsed = parse_github_repo_url(url)
    if not parsed:
        return None

    owner, repo = parsed
    api_base = f"https://api.github.com/repos/{owner}/{repo}"
    repo_payload = fetch_json(api_base, headers={"Accept": "application/vnd.github+json"})
    languages_payload = fetch_json(f"{api_base}/languages", headers={"Accept": "application/vnd.github+json"})

    readme_text = ""
    try:
        readme_payload = fetch_json(f"{api_base}/readme", headers={"Accept": "application/vnd.github+json"})
        if isinstance(readme_payload, dict):
            readme_text = decode_github_readme(readme_payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise

    root_entries: list[str] = []
    try:
        contents_payload = fetch_json(f"{api_base}/contents", headers={"Accept": "application/vnd.github+json"})
        if isinstance(contents_payload, list):
            for entry in contents_payload:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                entry_type = str(entry.get("type") or "").strip().lower()
                if not name:
                    continue
                root_entries.append(f"{name}/" if entry_type == "dir" else name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise

    title = f"{owner}/{repo}"
    text = build_github_repo_text(repo_payload, languages_payload, readme_text, root_entries)
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
        "source_type": "github",
        "source_id": title,
        "url": url,
        "title": title,
        "text": text,
        "word_count": len(text.split()),
        "estimated_time_minutes": estimate_time_minutes(text),
        "metadata": metadata,
    }
