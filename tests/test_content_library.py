import json
from pathlib import Path

import app.tools.content_library_tool as content_library_tool
from app.library.markdown_store import content_list, content_save, content_status_update
from app.tools.content_library_tool import content_add


def profile_fields(
    *,
    summary: str = "A short guide to embedded databases.",
    subject: str = "sqlite",
    depth_level: str = "light",
    categories: list[str] | None = None,
    estimated_time_minutes: int = 6,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "summary": summary,
        "subject": subject,
        "depth_level": depth_level,
        "categories": categories or ["databases"],
        "estimated_time_minutes": estimated_time_minutes,
        "confidence": confidence,
    }


def content_item(
    *,
    source_type: str = "web",
    source_id: str | None = None,
    url: str = "https://example.com/sqlite",
    title: str = "SQLite Guide",
    item_profile: dict[str, object] | None = None,
    status: str = "unread",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    item_profile = item_profile or profile_fields()
    return {
        "source_type": source_type,
        "source_id": source_id,
        "url": url,
        "title": title,
        **item_profile,
        "status": status,
        "metadata": metadata or {},
    }


def test_content_save_writes_flat_markdown_item(tmp_path: Path) -> None:
    result = content_save(
        {
            **content_item(
                url="https://example.com/local-first-notes",
                title="Local-first Notes",
                item_profile=profile_fields(
                    summary="A practical overview of local-first note taking.",
                    subject="local-first software",
                    depth_level="medium",
                    categories=["software", "notes"],
                    estimated_time_minutes=8,
                    confidence=0.81,
                ),
                metadata={"author": "Example"},
            ),
            "notes": "Worth revisiting.",
        },
        library_root=tmp_path,
    )

    saved_path = Path(result["path"])
    text = saved_path.read_text()

    assert result["status"] == "saved"
    assert saved_path.name.startswith("web-local-first-software-")
    assert 'source_type: "web"' in text
    assert 'categories: ["software", "notes"]' in text
    assert "## Notes" in text


def test_content_save_accepts_analyzer_status_and_preserves_extras_in_metadata(
    tmp_path: Path,
) -> None:
    result = content_save(
        {
            **content_item(url="https://example.com/analyzed", title="Analyzed Item"),
            "status": "completed",
            "task": "content_profile",
            "word_count": 321,
        },
        library_root=tmp_path,
    )

    assert result["item"]["status"] == "unread"
    assert result["item"]["metadata"]["task"] == "content_profile"
    assert result["item"]["metadata"]["word_count"] == 321


def test_content_save_updates_existing_url_preserving_status_and_notes(tmp_path: Path) -> None:
    first = content_save(
        {
            **content_item(
                url="https://example.com/same-url",
                title="Original Title",
                item_profile=profile_fields(
                    summary="Original summary.",
                    estimated_time_minutes=4,
                    confidence=0.7,
                ),
                status="started",
            ),
            "status": "started",
            "notes": "Already looked at this.",
        },
        library_root=tmp_path,
    )
    second = content_save(
        {
            **content_item(
                url="https://example.com/same-url",
                title="Updated Title",
                item_profile=profile_fields(
                    summary="Updated summary.",
                    estimated_time_minutes=5,
                    confidence=0.8,
                ),
            ),
        },
        library_root=tmp_path,
    )
    saved_text = Path(second["path"]).read_text()

    assert first["id"] == second["id"]
    assert second["duplicate"]
    assert second["item"]["status"] == "started"
    assert "# Updated Title" in saved_text
    assert "Already looked at this." in saved_text


def test_content_list_filters_by_topic_time_and_query(tmp_path: Path) -> None:
    content_save(
        content_item(url="https://example.com/sqlite", title="SQLite Guide"),
        library_root=tmp_path,
    )
    content_save(
        content_item(
            source_type="youtube",
            source_id="abcdefghijk",
            url="https://www.youtube.com/watch?v=abcdefghijk",
            title="RL Lecture",
            item_profile=profile_fields(
                summary="A long lecture about reinforcement learning.",
                subject="reinforcement learning",
                depth_level="deep",
                categories=["ml"],
                estimated_time_minutes=45,
                confidence=0.86,
            ),
        ),
        library_root=tmp_path,
    )

    result = content_list(
        {
            "query": "embedded",
            "max_estimated_time_minutes": 10,
            "categories": ["databases"],
        },
        library_root=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["count"] == 1
    assert result["items"][0]["title"] == "SQLite Guide"


def test_content_status_update_changes_status_and_notes(tmp_path: Path) -> None:
    saved = content_save(
        content_item(url="https://example.com/sqlite", title="SQLite Guide"),
        library_root=tmp_path,
    )

    status_result = content_status_update(
        url="https://example.com/sqlite",
        status="done",
        library_root=tmp_path,
    )
    notes_result = content_status_update(
        item_id=saved["id"],
        notes="This is useful for local apps.",
        library_root=tmp_path,
    )
    saved_text = Path(notes_result["path"]).read_text()

    assert status_result["item"]["status"] == "done"
    assert "## Notes" in saved_text
    assert "This is useful for local apps." in saved_text


def test_content_add_saves_and_deduplicates_analyzed_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        content_library_tool,
        "url_analyze",
        lambda *_args, **_kwargs: json.dumps(
            {
                "status": "completed",
                "source_type": "web",
                "source_id": None,
                "url": "https://example.com/sqlite-guide",
                "title": "SQLite for Local Apps",
                **profile_fields(
                    summary="A quick introduction to SQLite for local application development.",
                    estimated_time_minutes=4,
                    confidence=0.76,
                ),
            }
        ),
    )

    first = content_add(
        {
            "url": "https://example.com/sqlite-guide",
            "notes": "First pass.",
        },
        library_root=tmp_path,
    )
    second = content_add({"url": "https://example.com/sqlite-guide"}, library_root=tmp_path)
    saved_text = Path(second["path"]).read_text()
    file_count = len(list((tmp_path / "items").glob("*.md")))

    assert first["id"] == second["id"]
    assert second["duplicate"]
    assert "First pass." in saved_text
    assert file_count == 1
