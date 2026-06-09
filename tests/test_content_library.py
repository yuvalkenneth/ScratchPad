import json
import subprocess
from pathlib import Path

import app.library.git_history as git_history
import app.tools.content_library_tool as content_library_tool
from app.library.markdown_store import content_list, content_save, content_status_update, content_update
from app.tools.content_library_tool import analyze_source, content_add, content_list_json


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


def git_log_messages(library_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(library_root), "log", "--format=%s"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


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
    assert result["git"]["committed"]
    assert result["git"]["message"] == "Add content: Local-first Notes"
    assert git_log_messages(tmp_path)[0] == "Add content: Local-first Notes"


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
    assert second["git"]["committed"]
    assert git_log_messages(tmp_path)[0] == "Update content: Updated Title"


def test_content_save_updates_existing_url_when_subject_slug_changes(tmp_path: Path) -> None:
    first = content_save(
        content_item(
            url="https://example.com/same-url-new-subject",
            title="Original",
            item_profile=profile_fields(subject="original topic"),
        ),
        library_root=tmp_path,
    )
    second = content_save(
        content_item(
            url="https://example.com/same-url-new-subject",
            title="Updated",
            item_profile=profile_fields(subject="updated topic"),
        ),
        library_root=tmp_path,
    )

    files = list((tmp_path / "items").glob("*.md"))

    assert first["id"] == second["id"]
    assert second["duplicate"]
    assert len(files) == 1
    assert second["path"] == first["path"]
    assert second["item"]["subject"] == "updated topic"


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


def test_content_list_queries_status_time_window_and_sorting(tmp_path: Path) -> None:
    content_save(
        content_item(
            url="https://example.com/short-agent-evals",
            title="Short Agent Evals",
            item_profile=profile_fields(
                summary="A practical note about local LLM agent evaluation.",
                subject="agent evals",
                categories=["llm", "evals"],
                estimated_time_minutes=8,
                confidence=0.9,
            ),
            status="unread",
        ),
        library_root=tmp_path,
    )
    content_save(
        content_item(
            url="https://example.com/started-agent-evals",
            title="Started Agent Evals",
            item_profile=profile_fields(
                summary="A deeper article about evaluating local agents with rubrics.",
                subject="agent evals",
                categories=["llm", "evals"],
                estimated_time_minutes=18,
                confidence=0.7,
            ),
            status="started",
        ),
        library_root=tmp_path,
    )
    content_save(
        content_item(
            url="https://example.com/done-agent-evals",
            title="Done Agent Evals",
            item_profile=profile_fields(
                summary="A completed article about agent evaluation.",
                subject="agent evals",
                categories=["llm", "evals"],
                estimated_time_minutes=6,
            ),
            status="done",
        ),
        library_root=tmp_path,
    )

    result = content_list(
        {
            "status": ["unread", "started"],
            "exclude_status": ["done", "archived", "abandoned"],
            "min_estimated_time_minutes": 5,
            "max_estimated_time_minutes": 20,
            "query": "local agent evals",
            "sort": "estimated_time_minutes",
        },
        library_root=tmp_path,
    )

    assert [item["title"] for item in result["items"]] == [
        "Short Agent Evals",
        "Started Agent Evals",
    ]
    assert all("match_score" in item for item in result["items"])
    assert "query:title" in result["items"][0]["match_reasons"]


def test_content_list_json_defaults_to_unread_and_started(tmp_path: Path, monkeypatch) -> None:
    content_save(
        content_item(url="https://example.com/unread", title="Unread Item", status="unread"),
        library_root=tmp_path,
    )
    content_save(
        content_item(url="https://example.com/started", title="Started Item", status="started"),
        library_root=tmp_path,
    )
    content_save(
        content_item(url="https://example.com/done", title="Done Item", status="done"),
        library_root=tmp_path,
    )
    monkeypatch.setattr(content_library_tool, "content_list", lambda filters: content_list(filters, library_root=tmp_path))

    result = json.loads(content_list_json({}))

    assert result["applied_defaults"] == {"status": ["unread", "started"]}
    assert [item["title"] for item in result["items"]] == ["Started Item", "Unread Item"]


def test_content_list_json_respects_explicit_status(tmp_path: Path, monkeypatch) -> None:
    content_save(
        content_item(url="https://example.com/unread", title="Unread Item", status="unread"),
        library_root=tmp_path,
    )
    content_save(
        content_item(url="https://example.com/done", title="Done Item", status="done"),
        library_root=tmp_path,
    )
    monkeypatch.setattr(content_library_tool, "content_list", lambda filters: content_list(filters, library_root=tmp_path))

    result = json.loads(content_list_json({"status": "done"}))

    assert "applied_defaults" not in result
    assert [item["title"] for item in result["items"]] == ["Done Item"]


def test_content_list_relevance_sort_prefers_better_text_match(tmp_path: Path) -> None:
    content_save(
        content_item(
            url="https://example.com/generic",
            title="Local Tools",
            item_profile=profile_fields(
                summary="Mentions evaluation briefly.",
                subject="developer tools",
                categories=["tools"],
                estimated_time_minutes=5,
            ),
        ),
        library_root=tmp_path,
    )
    content_save(
        content_item(
            url="https://example.com/exact",
            title="Local LLM Evaluation",
            item_profile=profile_fields(
                summary="Local LLM evaluation methods for agent workflows.",
                subject="local llm evaluation",
                categories=["llm", "evals"],
                estimated_time_minutes=12,
            ),
        ),
        library_root=tmp_path,
    )

    result = content_list(
        {
            "query": "local llm evaluation",
            "sort": "relevance",
            "limit": 1,
        },
        library_root=tmp_path,
    )

    assert result["count"] == 2
    assert result["items"][0]["title"] == "Local LLM Evaluation"
    assert result["items"][0]["match_score"] > 0


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
    assert status_result["git"]["message"] == "Update status: SQLite Guide -> done"
    assert notes_result["git"]["message"] == "Update notes: SQLite Guide"
    assert git_log_messages(tmp_path)[:2] == [
        "Update notes: SQLite Guide",
        "Update status: SQLite Guide -> done",
    ]


def test_content_update_changes_profile_fields_and_commits(tmp_path: Path) -> None:
    saved = content_save(
        content_item(url="https://example.com/sqlite", title="SQLite Guide"),
        library_root=tmp_path,
    )

    result = content_update(
        item_id=saved["id"],
        updates={
            "title": "SQLite for Local Apps",
            "summary": "A corrected summary about SQLite in local apps.",
            "subject": "local app persistence",
            "categories": ["sqlite", "local-first"],
            "confidence": 0.75,
        },
        notes="Corrected after review.",
        library_root=tmp_path,
    )
    saved_text = Path(result["path"]).read_text()

    assert result["status"] == "updated"
    assert result["id"] == saved["id"]
    assert result["item"]["title"] == "SQLite for Local Apps"
    assert result["item"]["subject"] == "local app persistence"
    assert result["item"]["categories"] == ["sqlite", "local-first"]
    assert result["item"]["confidence"] == 0.75
    assert "Corrected after review." in saved_text
    assert result["git"]["committed"]
    assert git_log_messages(tmp_path)[0] == "Update content: SQLite for Local Apps"


def test_content_save_reports_nonfatal_git_failure(tmp_path: Path, monkeypatch) -> None:
    def fail_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["git"], stderr="git failed")

    monkeypatch.setattr(git_history, "run_git", fail_git)

    result = content_save(
        content_item(url="https://example.com/git-failure", title="Git Failure"),
        library_root=tmp_path,
    )

    assert result["status"] == "saved"
    assert Path(result["path"]).exists()
    assert result["git"]["enabled"]
    assert not result["git"]["committed"]
    assert result["git"]["commit"] is None
    assert result["git"]["error"] == "git failed"


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
    assert first["git"]["committed"]
    assert second["duplicate"]
    assert second["git"]["enabled"]
    assert "First pass." in saved_text
    assert file_count == 1


def test_analyze_source_profiles_url_without_saving(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        content_library_tool,
        "url_analyze",
        lambda *_args, **_kwargs: json.dumps(
            {
                "status": "completed",
                "source_type": "web",
                "source_id": None,
                "url": "https://example.com/source",
                "title": "Source",
                **profile_fields(
                    summary="A source worth inspecting before saving.",
                    subject="source analysis",
                    estimated_time_minutes=3,
                    confidence=0.8,
                ),
            }
        ),
    )

    result = analyze_source({"url": "https://example.com/source"})

    assert result["status"] == "completed"
    assert result["subject"] == "source analysis"
    assert not (tmp_path / "items").exists()


def test_content_add_reports_unsaveable_analysis_without_writing(
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
                "url": "https://example.com/bad-profile",
                "title": "Bad Profile",
                "summary": "",
                "subject": "",
                "depth_level": "medium",
                "estimated_time_minutes": 1,
                "confidence": 0.0,
            }
        ),
    )

    result = content_add({"url": "https://example.com/bad-profile"}, library_root=tmp_path)

    assert result["status"] == "error"
    assert result["missing_fields"] == ["summary", "subject"]
    assert not (tmp_path / "items").exists()
