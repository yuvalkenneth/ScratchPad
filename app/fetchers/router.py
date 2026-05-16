from __future__ import annotations

from typing import Any

from app.fetchers.github import fetch_github_repo_source, parse_github_repo_url
from app.fetchers.reddit import fetch_reddit_source, normalize_reddit_post_url
from app.fetchers.web import fetch_web_source


def fetch_source(url: str) -> dict[str, Any]:
    if parse_github_repo_url(url):
        return fetch_github_repo_source(url) or {}
    if normalize_reddit_post_url(url):
        return fetch_reddit_source(url) or {}
    return fetch_web_source(url)
