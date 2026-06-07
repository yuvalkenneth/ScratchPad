from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Optional

from app.fetchers.common import fetch_url_html, host_for_url, normalize_whitespace, truncate_text


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


def is_x_domain(url: str) -> bool:
    host = host_for_url(url)
    return host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}


def clean_x_title(title: str) -> str:
    cleaned = normalize_whitespace(title)
    cleaned = re.sub(r"\s*/\s*X\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+/\s+Twitter\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def clean_x_description(text: str) -> str:
    cleaned = normalize_whitespace(text)
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


def looks_like_generic_x_shell(text: str) -> bool:
    lowered = text.lower()
    return any(snippet in lowered for snippet in GENERIC_X_ERROR_SNIPPETS)


def extract_x_page_content(title: str, text: str, meta: dict[str, str]) -> dict[str, str]:
    meta_title = meta.get("og:title") or meta.get("twitter:title") or meta.get("title") or title
    meta_description = (
        meta.get("og:description")
        or meta.get("twitter:description")
        or meta.get("description")
        or ""
    )

    cleaned_title = clean_x_title(meta_title)
    cleaned_description = clean_x_description(meta_description)

    parts: list[str] = []
    for candidate in (cleaned_title, cleaned_description):
        candidate = candidate.strip()
        if candidate and candidate not in parts:
            parts.append(candidate)

    fallback_text = normalize_whitespace(text)
    if fallback_text and not looks_like_generic_x_shell(fallback_text):
        parts.append(fallback_text)

    combined_text = truncate_text(normalize_whitespace(" ".join(parts)))
    return {"title": cleaned_title or title, "text": combined_text}


def extract_page_content(url: str, html: str) -> dict[str, str]:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)

    title = normalize_whitespace(" ".join(extractor.title_parts))
    text = normalize_whitespace(" ".join(extractor.text_parts))
    if is_x_domain(url):
        return extract_x_page_content(title, text, extractor.meta)
    if not text:
        return extract_meta_page_content(title, extractor.meta)

    return {"title": title, "text": truncate_text(text), "extraction_quality": "full_text"}


def extract_meta_page_content(title: str, meta: dict[str, str]) -> dict[str, str]:
    meta_title = meta.get("og:title") or meta.get("twitter:title") or meta.get("title") or title
    meta_description = (
        meta.get("og:description")
        or meta.get("twitter:description")
        or meta.get("description")
        or ""
    )
    return {
        "title": normalize_whitespace(meta_title),
        "text": truncate_text(normalize_whitespace(meta_description)),
        "extraction_quality": "metadata_only",
    }


def fetch_web_source(url: str) -> dict[str, object]:
    html = fetch_url_html(url)
    page = extract_page_content(url, html)
    text = page["text"]
    return {
        "source_type": "web",
        "source_id": None,
        "url": url,
        "title": page["title"],
        "text": text,
        "word_count": len(text.split()),
        "metadata": {"extraction_quality": page.get("extraction_quality", "full_text")},
    }
