from __future__ import annotations

import re
from typing import Any, Optional

import httpx
from app.fetchers.common import estimate_time_minutes, fetch_json, host_for_url, normalize_whitespace, truncate_text

REDDIT_HTTP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "scratchpad-bot/0.1 (+https://example.invalid)",
}


def is_reddit_domain(url: str) -> bool:
    host = host_for_url(url)
    return host in {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com", "redd.it"}


def normalize_reddit_post_url(url: str) -> Optional[str]:
    parsed = httpx.URL(url)
    host = (parsed.host or "").lower()
    parts = [part for part in parsed.path.split("/") if part]

    if host == "redd.it":
        if not parts:
            return None
        return f"https://www.reddit.com/comments/{parts[0]}"

    if host not in {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"}:
        return None

    if len(parts) >= 4 and parts[0] == "r" and parts[2] == "comments":
        subreddit = parts[1]
        post_id = parts[3]
        if not subreddit or not post_id:
            return None
        if len(parts) >= 5 and parts[4]:
            return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/{parts[4]}"
        return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}"

    if len(parts) >= 2 and parts[0] == "comments":
        return f"https://www.reddit.com/comments/{parts[1]}"

    return None


def clean_reddit_body(text: str) -> str:
    cleaned = re.sub(r"&nbsp;", " ", text)
    cleaned = re.sub(r"\[deleted\]|\[removed\]", " ", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def extract_reddit_comments(comment_listing: Any) -> list[str]:
    if not isinstance(comment_listing, dict):
        return []
    data = comment_listing.get("data")
    if not isinstance(data, dict):
        return []
    children = data.get("children")
    if not isinstance(children, list):
        return []

    comments: list[str] = []
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t1":
            continue
        comment_data = child.get("data")
        if not isinstance(comment_data, dict):
            continue
        body = clean_reddit_body(str(comment_data.get("body") or ""))
        author = str(comment_data.get("author") or "").strip()
        if not body:
            continue
        comments.append(f"{author}: {body}" if author else body)
        if len(comments) >= 5:
            break
    return comments


def fetch_reddit_source(url: str) -> Optional[dict[str, Any]]:
    canonical_url = normalize_reddit_post_url(url)
    if not canonical_url:
        return None

    reddit_json = fetch_json(f"{canonical_url}.json?raw_json=1", headers=REDDIT_HTTP_HEADERS)
    if not isinstance(reddit_json, list) or len(reddit_json) < 1:
        return None

    post_listing = reddit_json[0]
    if not isinstance(post_listing, dict):
        return None
    post_data = ((post_listing.get("data") or {}).get("children") or [])
    if not isinstance(post_data, list) or not post_data:
        return None
    post = ((post_data[0] or {}).get("data") or {})
    if not isinstance(post, dict):
        return None

    title = normalize_whitespace(str(post.get("title") or ""))
    selftext = clean_reddit_body(str(post.get("selftext") or ""))
    subreddit = str(post.get("subreddit") or "").strip()
    author = str(post.get("author") or "").strip()
    permalink = str(post.get("permalink") or "").strip()
    canonical_post_url = f"https://www.reddit.com{permalink}" if permalink.startswith("/") else canonical_url
    comments = extract_reddit_comments(reddit_json[1] if len(reddit_json) > 1 else None)

    parts = [
        f"Subreddit: r/{subreddit}" if subreddit else "",
        f"Author: u/{author}" if author else "",
        f"Title: {title}" if title else "",
        f"Post: {selftext}" if selftext else "",
        f"Top comments: {' | '.join(comments)}" if comments else "",
    ]
    text = truncate_text(normalize_whitespace(" ".join(part for part in parts if part)))
    metadata = {
        "subreddit": subreddit,
        "author": author,
        "score": post.get("score"),
        "num_comments": post.get("num_comments"),
        "created_utc": post.get("created_utc"),
        "permalink": permalink,
        "is_self": bool(post.get("is_self")),
        "comments_sample": comments,
    }
    return {
        "source_type": "reddit",
        "source_id": str(post.get("id") or "") or None,
        "url": canonical_post_url,
        "title": title or canonical_post_url,
        "text": text,
        "word_count": len(text.split()),
        "estimated_time_minutes": estimate_time_minutes(text),
        "metadata": metadata,
    }
