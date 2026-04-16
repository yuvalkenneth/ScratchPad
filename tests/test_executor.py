import json
import unittest
from types import SimpleNamespace

from app.llm.client import LLMClient
from app.tools.executor import Executor, WORKSPACE, should_ask_permission
from app.tools.registry import get_tool_definitions, get_tools_prompt_text
import app.tools.youtube_analyze_tool as analyze_tool
from app.tools.youtube_analyze_tool import _chunk_text, youtube_analyze


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
        self.assertIn("profile", result)
        self.assertEqual(result["profile"]["subject"], "machine learning")
        self.assertEqual(result["profile"]["depth_level"], "deep")
        self.assertEqual(result["profile"]["estimated_time_minutes"], 2)
        self.assertEqual(result["profile"]["categories"], ["ml", "lecture", "neural-networks"])

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
        self.assertEqual(result["profile"]["depth_level"], "medium")
        self.assertEqual(result["profile"]["estimated_time_minutes"], 1)
        self.assertEqual(result["profile"]["confidence"], 0.0)
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


if __name__ == "__main__":
    unittest.main()
