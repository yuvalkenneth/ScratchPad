import json

from app.fetchers.web import extract_page_content
from app.content import ContentProfile
import app.tools.url_analyze_tool as url_tool
from app.tools.url_analyze_tool import url_analyze
import app.tools.youtube_analyze_tool as youtube_tool
from app.tools.youtube_analyze_tool import _chunk_text, youtube_analyze


CONTENT_PROFILE_KEYS = {
    "status",
    "source_type",
    "source_id",
    "url",
    "title",
    "summary",
    "subject",
    "depth_level",
    "categories",
    "estimated_time_minutes",
    "confidence",
}


def test_extract_page_content_ignores_script_text() -> None:
    html = """
    <html>
      <head>
        <title>Example Article</title>
        <script>var hidden = 'ignore me';</script>
      </head>
      <body>
        <h1>Working with SQLite</h1>
        <p>This article explains local-first persistence.</p>
      </body>
    </html>
    """

    page = extract_page_content("https://example.com/sqlite-guide", html)

    assert page["title"] == "Example Article"
    assert "Working with SQLite" in page["text"]
    assert "ignore me" not in page["text"]


def test_url_analyze_returns_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        url_tool,
        "fetch_source",
        lambda *_args, **_kwargs: {
            "source_type": "github",
            "source_id": "owner/repo",
            "url": "https://github.com/owner/repo",
            "title": "owner/repo",
            "text": "Repository: owner/repo Description: Local-first reading assistant.",
            "word_count": 6,
            "estimated_time_minutes": 1,
            "metadata": {"owner": "owner", "repo": "repo"},
        },
    )
    monkeypatch.setattr(
        url_tool,
        "_complete_text",
        lambda *_args, **_kwargs: json.dumps(
            {
                "summary": "A repository for a local-first reading assistant.",
                "subject": "local-first software",
                "depth_level": "medium",
                "categories": ["software"],
                "confidence": 0.8,
            }
        ),
    )

    result = json.loads(
        url_analyze(
            {
                "url": "https://github.com/owner/repo",
                "task": "content_profile",
            }
        )
    )

    assert CONTENT_PROFILE_KEYS.issubset(result.keys())
    assert result["source_type"] == "github"
    assert result["source_id"] == "owner/repo"
    assert result["metadata"] == {
        "owner": "owner",
        "repo": "repo",
        "task": "content_profile",
        "word_count": 6,
    }


def test_url_analyze_falls_back_when_json_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        url_tool,
        "fetch_source",
        lambda *_args, **_kwargs: {
            "source_type": "web",
            "source_id": None,
            "url": "https://example.com/test-page",
            "title": "Test Page",
            "text": "Short readable article text.",
            "word_count": 4,
            "estimated_time_minutes": 1,
        },
    )
    monkeypatch.setattr(url_tool, "_complete_text", lambda *_args, **_kwargs: "not valid json")

    result = json.loads(
        url_analyze(
            {
                "url": "https://example.com/test-page",
                "task": "content_profile",
            }
        )
    )

    assert CONTENT_PROFILE_KEYS.issubset(result.keys())
    assert result["depth_level"] == "medium"
    assert result["confidence"] == 0.0
    assert "raw_analysis" in result


def test_chunk_text_splits_long_input() -> None:
    text = ("a" * 7000) + "\n" + ("b" * 7000)
    chunks = _chunk_text(text, chunk_chars=8000)

    assert len(chunks) > 1
    assert all(chunks)


def test_youtube_content_profile_returns_db_ready_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        youtube_tool,
        "fetch_transcript_segments",
        lambda *_args, **_kwargs: [
            {"text": "Intro to linear models", "start": 0.0, "duration": 2.0},
            {"text": "Then the lecture moves to neural nets", "start": 118.0, "duration": 2.0},
        ],
    )
    monkeypatch.setattr(
        youtube_tool,
        "_complete_text",
        lambda *_args, **_kwargs: json.dumps(
            {
                "summary": "A technical lecture introducing model families.",
                "subject": "machine learning",
                "depth_level": "deep",
                "categories": ["ml", "lecture", "neural-networks"],
                "estimated_time_minutes": 99,
                "confidence": 0.83,
            }
        ),
    )

    result = json.loads(
        youtube_analyze(
            {
                "url": "https://youtube.com/watch?v=abcdefghijk",
                "task": "content_profile",
            }
        )
    )

    assert CONTENT_PROFILE_KEYS.issubset(result.keys())
    assert result["source_type"] == "youtube"
    assert result["source_id"] == "abcdefghijk"
    assert result["estimated_time_minutes"] == 2


def test_youtube_content_profile_falls_back_when_json_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        youtube_tool,
        "fetch_transcript_segments",
        lambda *_args, **_kwargs: [
            {"text": "Short transcript", "start": 0.0, "duration": 30.0},
        ],
    )
    monkeypatch.setattr(youtube_tool, "_complete_text", lambda *_args, **_kwargs: "not valid json")

    result = json.loads(
        youtube_analyze(
            {
                "url": "https://youtube.com/watch?v=abcdefghijk",
                "task": "content_profile",
            }
        )
    )

    assert CONTENT_PROFILE_KEYS.issubset(result.keys())
    assert result["depth_level"] == "medium"
    assert result["estimated_time_minutes"] == 1
    assert result["confidence"] == 0.0
    assert "raw_analysis" in result


def test_content_profile_coerces_model_output() -> None:
    profile = ContentProfile.from_model_output(
        {
            "summary": "  A useful summary.  ",
            "subject": " Local-first software ",
            "depth_level": "DEEP",
            "categories": "software, sync, local-first, notes, ignored",
            "estimated_time_minutes": "12",
            "confidence": "1.5",
        },
        estimated_time_minutes=5,
        trust_model_time=True,
    )

    assert profile.summary == "A useful summary."
    assert profile.subject == "Local-first software"
    assert profile.depth_level == "deep"
    assert profile.categories == ["software", "sync", "local-first", "notes"]
    assert profile.estimated_time_minutes == 12
    assert profile.confidence == 1.0


def test_content_profile_fallback_is_safe() -> None:
    profile = ContentProfile.fallback(estimated_time_minutes=0)

    assert profile.depth_level == "medium"
    assert profile.estimated_time_minutes == 1
    assert profile.confidence == 0.0
    assert profile.to_dict()["categories"] == []
