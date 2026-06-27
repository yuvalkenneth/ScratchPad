"""Regression tests for agent-level tool routing with fake model responses."""

import asyncio
from types import SimpleNamespace

import app.llm.client as llm_client
from app.llm.client import LLMClient
from app.llm.config import LLMConfig


class RecordingCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.index = 0
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses[self.index]
        self.index += 1
        return response


class RecordingClient:
    def __init__(self, completions: RecordingCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def make_response(message: object) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(name: str, arguments: str, call_id: str = "call_1") -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def run_tool_choice_case(
    monkeypatch,
    *,
    user_message: str,
    tool_name: str,
    arguments: str,
) -> tuple[str, list[dict[str, object]], list[tuple[str, dict[str, object]]], list[dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_run_tool(name: str, args: dict[str, object]) -> str:
        calls.append((name, args))
        return '{"status":"ok"}'

    monkeypatch.setattr(llm_client, "run_tool", fake_run_tool)
    completions = RecordingCompletions(
        [
            make_response(
                SimpleNamespace(
                    content="",
                    tool_calls=[make_tool_call(tool_name, arguments)],
                )
            ),
            make_response(SimpleNamespace(content="Handled.", tool_calls=[])),
        ]
    )
    client = LLMClient(config=LLMConfig(model_name="test-model"))
    client._get_client = lambda: RecordingClient(completions)  # type: ignore[method-assign]
    messages = [{"role": "user", "content": user_message}]

    response = asyncio.run(client.get_response(messages))

    return response, completions.requests, calls, messages


def test_recommendation_intent_loads_skill_then_uses_content_list(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_run_tool(name: str, args: dict[str, object]) -> str:
        calls.append((name, args))
        if name == "skill_view":
            return '{"content":"Call user_profile_get, then content_list before recommending saved items."}'
        if name == "user_profile_get":
            return (
                '{"status":"completed","profile":{"current_goals":["learn LLM deployment"],'
                '"interests":["local LLMs"],"avoided_topics":[],"preferences":'
                '{"preferred_session_minutes":"20"}}}'
            )
        if name == "content_list":
            return (
                '{"status":"completed","items":[{"title":"LLM endpoint deployment",'
                '"subject":"LLM deployment","categories":["LLMs"],"depth_level":"light",'
                '"estimated_time_minutes":5,"learning_effort_minutes":15,'
                '"status":"unread","match_reasons":["time:5m"]}],'
                '"count":1}'
            )
        return '{"status":"ok"}'

    monkeypatch.setattr(llm_client, "run_tool", fake_run_tool)
    completions = RecordingCompletions(
        [
            make_response(
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        make_tool_call(
                            "skill_view",
                            '{"name":"scratchpad-recommendation"}',
                            call_id="call_skill",
                        )
                    ],
                )
            ),
            make_response(
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        make_tool_call(
                            "user_profile_get",
                            '{}',
                            call_id="call_profile",
                        )
                    ],
                )
            ),
            make_response(
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        make_tool_call(
                            "content_list",
                            '{"status":["unread","started"],"max_estimated_time_minutes":20,"sort":"relevance"}',
                            call_id="call_list",
                        )
                    ],
                )
            ),
            make_response(
                SimpleNamespace(
                    content="Pick LLM endpoint deployment: unread, 5 minutes, light.",
                    tool_calls=[],
                )
            ),
        ]
    )
    client = LLMClient(config=LLMConfig(model_name="test-model"))
    client._get_client = lambda: RecordingClient(completions)  # type: ignore[method-assign]
    messages = [{"role": "user", "content": "What should I read in 20 minutes?"}]

    response = asyncio.run(client.get_response(messages))

    assert response == "Pick LLM endpoint deployment: unread, 5 minutes, light."
    assert calls == [
        ("skill_view", {"name": "scratchpad-recommendation"}),
        ("user_profile_get", {}),
        (
            "content_list",
            {
                "status": ["unread", "started"],
                "max_estimated_time_minutes": 20,
                "sort": "relevance",
            },
        ),
    ]


def first_request_tool_names(requests: list[dict[str, object]]) -> set[str]:
    tools = requests[0]["tools"]
    assert isinstance(tools, list)
    return {tool["function"]["name"] for tool in tools}  # type: ignore[index]


def test_save_url_intent_uses_content_add(monkeypatch) -> None:
    response, requests, calls, _messages = run_tool_choice_case(
        monkeypatch,
        user_message="Save https://example.com/article to my library",
        tool_name="content_add",
        arguments='{"url":"https://example.com/article"}',
    )

    assert response == "Handled."
    assert calls == [("content_add", {"url": "https://example.com/article"})]
    assert "content_add" in first_request_tool_names(requests)
    assert "content_save" not in first_request_tool_names(requests)


def test_inspect_source_intent_uses_analyze_source(monkeypatch) -> None:
    _response, requests, calls, _messages = run_tool_choice_case(
        monkeypatch,
        user_message="What is this and is it worth reading? https://example.com/article",
        tool_name="analyze_source",
        arguments='{"url":"https://example.com/article"}',
    )

    assert calls == [("analyze_source", {"url": "https://example.com/article"})]
    assert "analyze_source" in first_request_tool_names(requests)
    assert "url_analyze" not in first_request_tool_names(requests)
    assert "youtube_analyze" not in first_request_tool_names(requests)


def test_status_update_intent_uses_content_status_update(monkeypatch) -> None:
    _response, _requests, calls, _messages = run_tool_choice_case(
        monkeypatch,
        user_message="Mark the Ramp article as done",
        tool_name="content_status_update",
        arguments='{"url":"https://builders.ramp.com/post/stack-benchmarking","status":"done"}',
    )

    assert calls == [
        (
            "content_status_update",
            {
                "url": "https://builders.ramp.com/post/stack-benchmarking",
                "status": "done",
            },
        )
    ]


def test_detail_correction_intent_uses_content_update(monkeypatch) -> None:
    _response, requests, calls, _messages = run_tool_choice_case(
        monkeypatch,
        user_message="Fix the YouTube item title and categories",
        tool_name="content_update",
        arguments='{"source_type":"youtube","source_id":"PZ9u6DR8qOU","title":"Claude Code interview","categories":["Claude Code","AI coding tools"]}',
    )

    assert calls == [
        (
            "content_update",
            {
                "source_type": "youtube",
                "source_id": "PZ9u6DR8qOU",
                "title": "Claude Code interview",
                "categories": ["Claude Code", "AI coding tools"],
            },
        )
    ]
    assert "content_update" in first_request_tool_names(requests)


def test_listing_intent_can_use_content_list_directly(monkeypatch) -> None:
    _response, _requests, calls, _messages = run_tool_choice_case(
        monkeypatch,
        user_message="List my unread saved items under 20 minutes.",
        tool_name="content_list",
        arguments='{"status":["unread","started"],"max_estimated_time_minutes":20,"sort":"relevance"}',
    )

    assert calls == [
        (
            "content_list",
            {
                "status": ["unread", "started"],
                "max_estimated_time_minutes": 20,
                "sort": "relevance",
            },
        )
    ]
