import unittest
from types import SimpleNamespace

from app.llm.client import LLMClient
from app.tools.executor import Executor, WORKSPACE, should_ask_permission


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


if __name__ == "__main__":
    unittest.main()
