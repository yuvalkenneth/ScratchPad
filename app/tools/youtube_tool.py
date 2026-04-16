from __future__ import annotations

import re
from typing import Any, Optional


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    patterns = [
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return value


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def fetch_transcript_segments(video_id: str, languages: Optional[list[str]] = None) -> list[dict[str, Any]]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    if languages:
        result = api.fetch(video_id, languages=languages)
    else:
        result = api.fetch(video_id)

    return [
        {"text": segment.text, "start": segment.start, "duration": segment.duration}
        for segment in result
    ]
