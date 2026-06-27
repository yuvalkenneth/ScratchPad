"""Tests for LLM runtime prompt policy, env resolution, and tool-call guards."""

import os
import asyncio
from types import SimpleNamespace

from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.llm.config import load_env_file
from app.llm.openai_compatible import is_non_retryable_quota_error, resolve_api_key, resolve_base_url
from app.llm.prompting import build_system_prompt


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


def test_returns_message_when_model_stops_without_final_content() -> None:
    async def run_case() -> str:
        client = LLMClient(config=LLMConfig(model_name="test-model"))
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
        return await client.get_response([{"role": "user", "content": "summarize this video"}])

    assert "without producing a final answer" in asyncio.run(run_case())


def test_detects_repeated_tool_calls() -> None:
    async def run_case() -> str:
        client = LLMClient(config=LLMConfig(model_name="test-model"), max_tool_rounds=8)
        repeated_call = make_tool_call("skill_view", '{"name":"youtube-content"}')
        responses = [
            make_response(SimpleNamespace(content="", tool_calls=[repeated_call])),
            make_response(SimpleNamespace(content="", tool_calls=[repeated_call])),
            make_response(SimpleNamespace(content="", tool_calls=[repeated_call])),
        ]
        client._get_client = lambda: FakeClient(responses)  # type: ignore[method-assign]
        return await client.get_response([{"role": "user", "content": "summarize this video"}])

    assert "appears stuck" in asyncio.run(run_case())


def test_default_system_prompt_identifies_scratchpad_and_tool_policy() -> None:
    prompt = build_system_prompt()

    assert "You are Scratchpad" in prompt
    assert "local-first" in prompt
    assert "Current local time:" in prompt


def test_resolve_api_key_prefers_provider_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")

    api_key = resolve_api_key(
        LLMConfig(provider="openai", model_name="test", api_key="config-key")
    )

    assert api_key == "provider-key"


def test_detects_non_retryable_daily_quota_errors() -> None:
    assert is_non_retryable_quota_error(
        RuntimeError("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )
    assert not is_non_retryable_quota_error(RuntimeError("Please retry in 27s"))


def test_load_env_file_sets_missing_values_without_overriding(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_BASE_URL=https://example.com/v1\nLLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    load_env_file(env_path)

    assert os.environ["GEMINI_BASE_URL"] == "https://example.com/v1"
    assert os.environ["LLM_MODEL"] == "from-env"


def test_config_from_env_leaves_non_llama_base_url_to_provider_resolution(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    config = LLMConfig.from_env()

    assert config.provider == "gemini"
    assert config.base_url == ""


def test_resolve_base_url_uses_config_then_env_then_provider_default(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example.test/v1")

    assert (
        resolve_base_url(
            LLMConfig(
                provider="openai",
                model_name="test",
                base_url="https://config.example.test/v1",
            )
        )
        == "https://config.example.test/v1"
    )
    assert (
        resolve_base_url(LLMConfig(provider="openai", model_name="test", base_url=""))
        == "https://env.example.test/v1"
    )

    monkeypatch.delenv("LLM_BASE_URL")
    assert (
        resolve_base_url(LLMConfig(provider="openai", model_name="test", base_url=""))
        == "https://api.openai.com/v1"
    )


def test_resolve_gemini_uses_gemini_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gemini.example.test/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://generic.example.test/v1")

    config = LLMConfig(provider="gemini", model_name="gemini-3.5-flash", base_url="")

    assert resolve_base_url(config) == "https://gemini.example.test/v1"
    assert resolve_api_key(config) == "gemini-key"
