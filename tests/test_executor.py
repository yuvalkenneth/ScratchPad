import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.library.markdown_store import content_list, content_save, content_status_update
from app.llm.client import LLMClient
import app.tools.content_library_tool as content_library_tool
from app.tools.content_library_tool import content_add
from app.tools.executor import Executor, WORKSPACE, should_ask_permission
from app.tools.registry import get_tool_definitions, get_tools_prompt_text
import app.tools.youtube_analyze_tool as analyze_tool
from app.tools.youtube_analyze_tool import _chunk_text, youtube_analyze
import app.tools.url_analyze_tool as url_tool
from app.fetchers.web import extract_page_content
from app.tools.url_analyze_tool import url_analyze


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "youtube_profile_eval_cases.json"
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


class PermissionPolicyTests(unittest.TestCase):
    def test_network_command_requires_approval(self) -> None:
        ask, reason = should_ask_permission("curl https://example.com")
        self.assertTrue(ask)
        self.assertIn("approval-required", reason)

    def test_sudo_requires_separate_denial_path(self) -> None:
        ask, _ = should_ask_permission("sudo ls")
        self.assertFalse(ask)

    def test_absolute_path_outside_workspace_requires_approval(self) -> None:
        ask, reason = should_ask_permission("cat /tmp/outside.txt")
        self.assertTrue(ask)
        self.assertIn("outside the workspace", reason)


class ExecutorTests(unittest.TestCase):
    def test_shell_runs_in_workspace_by_default(self) -> None:
        executor = Executor()
        result = executor.run_shell("pwd")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), str(WORKSPACE))

    def test_python_runs_and_captures_output(self) -> None:
        executor = Executor()
        result = executor.run_python("print('hello')")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "hello")

    def test_outside_workspace_cwd_is_denied(self) -> None:
        executor = Executor()
        result = executor.run_shell("pwd", cwd="/tmp")
        self.assertEqual(result["status"], "denied")


class MarkdownLibraryTests(unittest.TestCase):
    def test_content_save_writes_flat_markdown_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = content_save(
                {
                    "source_type": "web",
                    "source_id": None,
                    "url": "https://example.com/local-first-notes",
                    "title": "Local-first Notes",
                    "summary": "A practical overview of local-first note taking.",
                    "subject": "local-first software",
                    "depth_level": "medium",
                    "categories": ["software", "notes"],
                    "estimated_time_minutes": 8,
                    "confidence": 0.81,
                    "metadata": {"author": "Example"},
                    "notes": "Worth revisiting.",
                },
                library_root=Path(directory),
            )

            saved_path = Path(result["path"])
            self.assertEqual(result["status"], "saved")
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.parent.name, "items")
            self.assertTrue(saved_path.name.startswith("web-local-first-software-"))

            text = saved_path.read_text()
            self.assertIn('source_type: "web"', text)
            self.assertIn('categories: ["software", "notes"]', text)
            self.assertIn("## Notes", text)

    def test_content_save_updates_existing_item_by_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = content_save(
                {
                    "source_type": "github",
                    "source_id": "owner/repo",
                    "url": "https://github.com/owner/repo",
                    "title": "owner/repo",
                    "summary": "First summary.",
                    "subject": "local-first software",
                    "depth_level": "medium",
                    "categories": ["software"],
                    "estimated_time_minutes": 5,
                    "confidence": 0.7,
                },
                library_root=root,
            )
            second = content_save(
                {
                    "source_type": "github",
                    "source_id": "owner/repo",
                    "url": "https://github.com/owner/repo",
                    "title": "owner/repo",
                    "summary": "Updated summary.",
                    "subject": "local-first software",
                    "depth_level": "medium",
                    "categories": ["software"],
                    "estimated_time_minutes": 6,
                    "confidence": 0.8,
                },
                library_root=root,
            )

            self.assertEqual(first["id"], second["id"])
            self.assertFalse(second["created"])
            self.assertTrue(second["duplicate"])
            self.assertEqual(len(list((root / "items").glob("*.md"))), 1)

    def test_content_save_preserves_status_and_notes_on_duplicate_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = content_save(
                {
                    "source_type": "web",
                    "url": "https://example.com/same-url",
                    "title": "Original Title",
                    "summary": "Original summary.",
                    "subject": "sqlite",
                    "depth_level": "light",
                    "categories": ["databases"],
                    "estimated_time_minutes": 4,
                    "confidence": 0.7,
                    "status": "started",
                    "notes": "Already looked at this.",
                },
                library_root=root,
            )
            second = content_save(
                {
                    "source_type": "web",
                    "url": "https://example.com/same-url",
                    "title": "Updated Title",
                    "summary": "Updated summary.",
                    "subject": "sqlite",
                    "depth_level": "light",
                    "categories": ["databases"],
                    "estimated_time_minutes": 5,
                    "confidence": 0.8,
                },
                library_root=root,
            )

            saved_text = Path(second["path"]).read_text()

            self.assertEqual(first["id"], second["id"])
            self.assertFalse(second["created"])
            self.assertTrue(second["duplicate"])
            self.assertEqual(second["item"]["status"], "started")
            self.assertIn("# Updated Title", saved_text)
            self.assertIn("Already looked at this.", saved_text)
            self.assertEqual(len(list((root / "items").glob("*.md"))), 1)

    def test_content_list_filters_by_topic_time_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_save(
                {
                    "source_type": "web",
                    "url": "https://example.com/sqlite",
                    "title": "SQLite Guide",
                    "summary": "A short guide to embedded databases.",
                    "subject": "sqlite",
                    "depth_level": "light",
                    "categories": ["databases"],
                    "estimated_time_minutes": 6,
                    "confidence": 0.9,
                },
                library_root=root,
            )
            content_save(
                {
                    "source_type": "youtube",
                    "source_id": "abcdefghijk",
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                    "title": "RL Lecture",
                    "summary": "A long lecture about reinforcement learning.",
                    "subject": "reinforcement learning",
                    "depth_level": "deep",
                    "categories": ["ml"],
                    "estimated_time_minutes": 45,
                    "confidence": 0.86,
                },
                library_root=root,
            )

            result = content_list(
                {
                    "query": "embedded",
                    "max_estimated_time_minutes": 10,
                    "categories": ["databases"],
                },
                library_root=root,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["items"][0]["title"], "SQLite Guide")

    def test_content_status_update_changes_status_by_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = content_save(
                {
                    "source_type": "web",
                    "url": "https://example.com/sqlite",
                    "title": "SQLite Guide",
                    "summary": "A short guide to embedded databases.",
                    "subject": "sqlite",
                    "depth_level": "light",
                    "categories": ["databases"],
                    "estimated_time_minutes": 6,
                    "confidence": 0.9,
                },
                library_root=root,
            )

            result = content_status_update(
                url="https://example.com/sqlite",
                status="done",
                library_root=root,
            )

            self.assertEqual(result["status"], "updated")
            self.assertEqual(result["id"], saved["id"])
            self.assertEqual(result["item"]["status"], "done")

    def test_content_status_update_changes_notes_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = content_save(
                {
                    "source_type": "web",
                    "url": "https://example.com/sqlite",
                    "title": "SQLite Guide",
                    "summary": "A short guide to embedded databases.",
                    "subject": "sqlite",
                    "depth_level": "light",
                    "categories": ["databases"],
                    "estimated_time_minutes": 6,
                    "confidence": 0.9,
                },
                library_root=root,
            )

            result = content_status_update(
                item_id=saved["id"],
                notes="This is useful for local apps.",
                library_root=root,
            )
            saved_text = Path(result["path"]).read_text()

            self.assertEqual(result["status"], "updated")
            self.assertIn("## Notes", saved_text)
            self.assertIn("This is useful for local apps.", saved_text)

    def test_content_add_saves_url_analyzer_output_as_markdown(self) -> None:
        original_url_analyze = content_library_tool.url_analyze
        content_library_tool.url_analyze = lambda *_args, **_kwargs: json.dumps(
            {
                "status": "completed",
                "source_type": "web",
                "source_id": None,
                "url": "https://example.com/sqlite-guide",
                "title": "SQLite for Local Apps",
                "summary": "A quick introduction to SQLite for local application development.",
                "subject": "sqlite",
                "depth_level": "light",
                "categories": ["databases"],
                "estimated_time_minutes": 4,
                "confidence": 0.76,
                "metadata": {"word_count": 800},
            }
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = content_add(
                    {
                        "url": "https://example.com/sqlite-guide",
                        "notes": "Check whether this is worth adding to the local-first set.",
                    },
                    library_root=Path(directory),
                )

                saved_path = Path(result["path"])
                text = saved_path.read_text()
        finally:
            content_library_tool.url_analyze = original_url_analyze

        self.assertEqual(result["status"], "saved")
        self.assertFalse(result["duplicate"])
        self.assertTrue(saved_path.name.startswith("web-sqlite-"))
        self.assertIn('source_type: "web"', text)
        self.assertIn('url: "https://example.com/sqlite-guide"', text)
        self.assertIn('subject: "sqlite"', text)
        self.assertIn('depth_level: "light"', text)
        self.assertIn('categories: ["databases"]', text)
        self.assertIn("# SQLite for Local Apps", text)
        self.assertIn("A quick introduction to SQLite", text)
        self.assertIn("## Notes", text)

    def test_content_add_saves_youtube_analyzer_output_as_markdown(self) -> None:
        original_youtube_analyze = content_library_tool.youtube_analyze
        content_library_tool.youtube_analyze = lambda *_args, **_kwargs: json.dumps(
            {
                "status": "completed",
                "source_type": "youtube",
                "source_id": "abcdefghijk",
                "url": "https://www.youtube.com/watch?v=abcdefghijk",
                "title": "abcdefghijk",
                "summary": "A technical lecture introducing model families.",
                "subject": "machine learning",
                "depth_level": "deep",
                "categories": ["ml", "lecture"],
                "estimated_time_minutes": 12,
                "confidence": 0.83,
                "metadata": {},
            }
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = content_add(
                    {
                        "url": "https://www.youtube.com/watch?v=abcdefghijk",
                        "status": "started",
                    },
                    library_root=Path(directory),
                )

                saved_path = Path(result["path"])
                text = saved_path.read_text()
        finally:
            content_library_tool.youtube_analyze = original_youtube_analyze

        self.assertEqual(result["status"], "saved")
        self.assertTrue(saved_path.name.startswith("youtube-machine-learning-"))
        self.assertIn('source_type: "youtube"', text)
        self.assertIn('source_id: "abcdefghijk"', text)
        self.assertIn('status: "started"', text)
        self.assertIn('subject: "machine learning"', text)

    def test_content_add_duplicate_url_updates_existing_markdown_file(self) -> None:
        original_url_analyze = content_library_tool.url_analyze
        content_library_tool.url_analyze = lambda *_args, **_kwargs: json.dumps(
            {
                "status": "completed",
                "source_type": "web",
                "source_id": None,
                "url": "https://example.com/sqlite-guide",
                "title": "SQLite for Local Apps",
                "summary": "A quick introduction to SQLite for local application development.",
                "subject": "sqlite",
                "depth_level": "light",
                "categories": ["databases"],
                "estimated_time_minutes": 4,
                "confidence": 0.76,
            }
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first = content_add(
                    {
                        "url": "https://example.com/sqlite-guide",
                        "notes": "First pass.",
                    },
                    library_root=root,
                )
                second = content_add(
                    {"url": "https://example.com/sqlite-guide"},
                    library_root=root,
                )
                saved_text = Path(second["path"]).read_text()
                file_count = len(list((root / "items").glob("*.md")))
        finally:
            content_library_tool.url_analyze = original_url_analyze

        self.assertEqual(first["id"], second["id"])
        self.assertFalse(second["created"])
        self.assertTrue(second["duplicate"])
        self.assertIn("First pass.", saved_text)
        self.assertEqual(file_count, 1)


class FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self._index = 0

    async def create(self, **_: object) -> object:
        response = self._responses[self._index]
        self._index += 1
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_response(message: object) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(name: str, arguments: str, call_id: str = "call_1") -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class LLMClientToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_message_when_model_stops_without_final_content(self) -> None:
        client = LLMClient(model_name="test-model")
        responses = [
            make_response(
                SimpleNamespace(
                    content="",
                    tool_calls=[make_tool_call("skill_view", '{"name":"youtube-content"}')],
                )
            ),
            make_response(SimpleNamespace(content="", tool_calls=[])),
        ]
        client._get_client = lambda: FakeClient(responses)  # type: ignore[method-assign]

        result = await client.get_response([{"role": "user", "content": "summarize this video"}])

        self.assertIn("without producing a final answer", result)

    async def test_detects_repeated_tool_calls(self) -> None:
        client = LLMClient(model_name="test-model", max_tool_rounds=8)
        repeated_call = make_tool_call("skill_view", '{"name":"youtube-content"}')
        responses = [
            make_response(SimpleNamespace(content="", tool_calls=[repeated_call])),
            make_response(SimpleNamespace(content="", tool_calls=[repeated_call])),
            make_response(SimpleNamespace(content="", tool_calls=[repeated_call])),
        ]
        client._get_client = lambda: FakeClient(responses)  # type: ignore[method-assign]

        result = await client.get_response([{"role": "user", "content": "summarize this video"}])

        self.assertIn("appears stuck", result)


class YouTubeAnalyzeToolTests(unittest.TestCase):
    def test_profile_eval_fixture_set_is_well_formed(self) -> None:
        with FIXTURE_PATH.open() as handle:
            cases = json.load(handle)

        self.assertIsInstance(cases, list)
        self.assertGreaterEqual(len(cases), 4)
        for case in cases:
            self.assertIn("name", case)
            self.assertIn("url", case)
            self.assertIn("segments", case)
            self.assertIn("expected", case)
            self.assertTrue(case["segments"])
            self.assertTrue(case["expected"]["subject_options"])
            self.assertIn(case["expected"]["depth_level"], {"light", "medium", "deep"})
            self.assertGreaterEqual(case["expected"]["estimated_time_minutes"], 1)

    def test_chunk_text_splits_long_input(self) -> None:
        text = ("a" * 7000) + "\n" + ("b" * 7000)
        chunks = _chunk_text(text, chunk_chars=8000)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))

    def test_analyze_returns_structured_result(self) -> None:
        original_fetch = analyze_tool.fetch_transcript_segments
        original_complete = analyze_tool._complete_text
        analyze_tool.fetch_transcript_segments = lambda *_args, **_kwargs: [
            {"text": "Intro to linear models", "start": 0.0, "duration": 2.0},
            {"text": "Then the lecture moves to neural nets", "start": 2.0, "duration": 2.0},
        ]
        analyze_tool._complete_text = lambda *_args, **_kwargs: "Merged analysis output"
        try:
            result = json.loads(
                youtube_analyze(
                    {
                        "url": "https://youtube.com/watch?v=abcdefghijk",
                        "task": "detailed_summary",
                    }
                )
            )
        finally:
            analyze_tool.fetch_transcript_segments = original_fetch
            analyze_tool._complete_text = original_complete

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["video_id"], "abcdefghijk")
        self.assertEqual(result["task"], "detailed_summary")
        self.assertEqual(result["analysis"], "Merged analysis output")
        self.assertEqual(result["summary_strategy"], "single_pass")

    def test_content_profile_returns_db_ready_fields(self) -> None:
        original_fetch = analyze_tool.fetch_transcript_segments
        original_complete = analyze_tool._complete_text
        analyze_tool.fetch_transcript_segments = lambda *_args, **_kwargs: [
            {"text": "Intro to linear models", "start": 0.0, "duration": 2.0},
            {"text": "Then the lecture moves to neural nets", "start": 118.0, "duration": 2.0},
        ]
        analyze_tool._complete_text = lambda *_args, **_kwargs: json.dumps(
            {
                "summary": "A technical lecture introducing model families.",
                "subject": "machine learning",
                "depth_level": "deep",
                "categories": ["ml", "lecture", "neural-networks"],
                "estimated_time_minutes": 99,
                "confidence": 0.83,
            }
        )
        try:
            result = json.loads(
                youtube_analyze(
                    {
                        "url": "https://youtube.com/watch?v=abcdefghijk",
                        "task": "content_profile",
                    }
                )
            )
        finally:
            analyze_tool.fetch_transcript_segments = original_fetch
            analyze_tool._complete_text = original_complete

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["task"], "content_profile")
        self.assertTrue(CONTENT_PROFILE_KEYS.issubset(result.keys()))
        self.assertEqual(result["source_type"], "youtube")
        self.assertEqual(result["source_id"], "abcdefghijk")
        self.assertEqual(result["url"], "https://www.youtube.com/watch?v=abcdefghijk")
        self.assertEqual(result["title"], "abcdefghijk")
        self.assertEqual(result["subject"], "machine learning")
        self.assertEqual(result["depth_level"], "deep")
        self.assertEqual(result["estimated_time_minutes"], 2)
        self.assertEqual(result["categories"], ["ml", "lecture", "neural-networks"])

    def test_content_profile_falls_back_when_json_is_invalid(self) -> None:
        original_fetch = analyze_tool.fetch_transcript_segments
        original_complete = analyze_tool._complete_text
        analyze_tool.fetch_transcript_segments = lambda *_args, **_kwargs: [
            {"text": "Short transcript", "start": 0.0, "duration": 30.0},
        ]
        analyze_tool._complete_text = lambda *_args, **_kwargs: "not valid json"
        try:
            result = json.loads(
                youtube_analyze(
                    {
                        "url": "https://youtube.com/watch?v=abcdefghijk",
                        "task": "content_profile",
                    }
                )
            )
        finally:
            analyze_tool.fetch_transcript_segments = original_fetch
            analyze_tool._complete_text = original_complete

        self.assertEqual(result["status"], "completed")
        self.assertTrue(CONTENT_PROFILE_KEYS.issubset(result.keys()))
        self.assertEqual(result["depth_level"], "medium")
        self.assertEqual(result["estimated_time_minutes"], 1)
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("raw_analysis", result)

    def test_tools_prompt_mentions_youtube_analyze_routing(self) -> None:
        prompt_text = get_tools_prompt_text()

        self.assertIn("youtube_analyze", prompt_text)
        self.assertIn("content_profile", prompt_text)
        self.assertNotIn("youtube_transcript_fetch", prompt_text)

    def test_tool_definitions_hide_internal_transcript_fetch(self) -> None:
        definitions = get_tool_definitions()
        tool_names = [item["function"]["name"] for item in definitions]

        self.assertIn("youtube_analyze", tool_names)
        self.assertNotIn("youtube_transcript_fetch", tool_names)


class URLAnalyzeToolTests(unittest.TestCase):
    def test_extract_page_content_ignores_script_text(self) -> None:
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

        self.assertEqual(page["title"], "Example Article")
        self.assertIn("Working with SQLite", page["text"])
        self.assertNotIn("ignore me", page["text"])

    def test_url_analyze_returns_profile(self) -> None:
        original_fetch = url_tool.fetch_source
        original_complete = url_tool._complete_text
        url_tool.fetch_source = lambda *_args, **_kwargs: {
            "source_type": "web",
            "source_id": None,
            "url": "https://example.com/sqlite-guide",
            "title": "SQLite for Local Apps",
            "text": "SQLite is a compact embedded database for local applications.",
            "word_count": 8,
            "estimated_time_minutes": 1,
        }
        url_tool._complete_text = lambda *_args, **_kwargs: json.dumps(
            {
                "summary": "A quick introduction to SQLite for local application development.",
                "subject": "sqlite",
                "depth_level": "light",
                "categories": ["databases"],
                "confidence": 0.76,
            }
        )
        try:
            result = json.loads(
                url_analyze(
                    {
                        "url": "https://example.com/sqlite-guide",
                        "task": "content_profile",
                    }
                )
            )
        finally:
            url_tool.fetch_source = original_fetch
            url_tool._complete_text = original_complete

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["task"], "content_profile")
        self.assertTrue(CONTENT_PROFILE_KEYS.issubset(result.keys()))
        self.assertEqual(result["source_type"], "web")
        self.assertEqual(result["url"], "https://example.com/sqlite-guide")
        self.assertEqual(result["title"], "SQLite for Local Apps")
        self.assertEqual(result["subject"], "sqlite")
        self.assertEqual(result["depth_level"], "light")
        self.assertEqual(result["categories"], ["databases"])
        self.assertGreaterEqual(result["estimated_time_minutes"], 1)

    def test_url_analyze_falls_back_when_json_is_invalid(self) -> None:
        original_fetch = url_tool.fetch_source
        original_complete = url_tool._complete_text
        url_tool.fetch_source = lambda *_args, **_kwargs: {
            "source_type": "web",
            "source_id": None,
            "url": "https://example.com/test-page",
            "title": "Test Page",
            "text": "Short readable article text.",
            "word_count": 4,
            "estimated_time_minutes": 1,
        }
        url_tool._complete_text = lambda *_args, **_kwargs: "not valid json"
        try:
            result = json.loads(
                url_analyze(
                    {
                        "url": "https://example.com/test-page",
                        "task": "content_profile",
                    }
                )
            )
        finally:
            url_tool.fetch_source = original_fetch
            url_tool._complete_text = original_complete

        self.assertEqual(result["status"], "completed")
        self.assertTrue(CONTENT_PROFILE_KEYS.issubset(result.keys()))
        self.assertEqual(result["source_type"], "web")
        self.assertEqual(result["depth_level"], "medium")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("raw_analysis", result)

    def test_url_analyze_preserves_source_metadata_for_github(self) -> None:
        original_fetch = url_tool.fetch_source
        original_complete = url_tool._complete_text
        url_tool.fetch_source = lambda *_args, **_kwargs: {
            "source_type": "github",
            "source_id": "owner/repo",
            "url": "https://github.com/owner/repo",
            "title": "owner/repo",
            "text": "Repository: owner/repo Description: Local-first reading assistant.",
            "word_count": 6,
            "estimated_time_minutes": 1,
            "metadata": {"owner": "owner", "repo": "repo"},
        }
        url_tool._complete_text = lambda *_args, **_kwargs: json.dumps(
            {
                "summary": "A repository for a local-first reading assistant.",
                "subject": "local-first software",
                "depth_level": "medium",
                "categories": ["software"],
                "confidence": 0.8,
            }
        )
        try:
            result = json.loads(
                url_analyze(
                    {
                        "url": "https://github.com/owner/repo",
                        "task": "content_profile",
                    }
                )
            )
        finally:
            url_tool.fetch_source = original_fetch
            url_tool._complete_text = original_complete

        self.assertTrue(CONTENT_PROFILE_KEYS.issubset(result.keys()))
        self.assertEqual(result["source_type"], "github")
        self.assertEqual(result["source_id"], "owner/repo")
        self.assertEqual(result["metadata"], {"owner": "owner", "repo": "repo"})

    def test_url_analyze_preserves_source_metadata_for_reddit(self) -> None:
        original_fetch = url_tool.fetch_source
        original_complete = url_tool._complete_text
        url_tool.fetch_source = lambda *_args, **_kwargs: {
            "source_type": "reddit",
            "source_id": "abc123",
            "url": "https://www.reddit.com/r/LocalFirst/comments/abc123/example",
            "title": "Local-first note taking",
            "text": "Title: Local-first note taking Post: What are good patterns?",
            "word_count": 8,
            "estimated_time_minutes": 1,
            "metadata": {"subreddit": "LocalFirst", "num_comments": 12},
        }
        url_tool._complete_text = lambda *_args, **_kwargs: json.dumps(
            {
                "summary": "A discussion about local-first note-taking patterns.",
                "subject": "local-first note taking",
                "depth_level": "light",
                "categories": ["discussion"],
                "confidence": 0.72,
            }
        )
        try:
            result = json.loads(
                url_analyze(
                    {
                        "url": "https://www.reddit.com/r/LocalFirst/comments/abc123/example",
                        "task": "content_profile",
                    }
                )
            )
        finally:
            url_tool.fetch_source = original_fetch
            url_tool._complete_text = original_complete

        self.assertTrue(CONTENT_PROFILE_KEYS.issubset(result.keys()))
        self.assertEqual(result["source_type"], "reddit")
        self.assertEqual(result["source_id"], "abc123")
        self.assertEqual(result["metadata"], {"subreddit": "LocalFirst", "num_comments": 12})

    def test_tools_prompt_mentions_url_analyze(self) -> None:
        prompt_text = get_tools_prompt_text()

        self.assertIn("url_analyze", prompt_text)
        self.assertIn("non-YouTube URLs", prompt_text)

    def test_tool_definitions_include_url_analyze(self) -> None:
        definitions = get_tool_definitions()
        tool_names = [item["function"]["name"] for item in definitions]

        self.assertIn("url_analyze", tool_names)

    def test_tool_definitions_include_content_library_tools(self) -> None:
        definitions = get_tool_definitions()
        tool_names = [item["function"]["name"] for item in definitions]

        self.assertIn("content_add", tool_names)
        self.assertIn("content_save", tool_names)
        self.assertIn("content_list", tool_names)
        self.assertIn("content_status_update", tool_names)


if __name__ == "__main__":
    unittest.main()
