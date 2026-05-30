from app.tools.registry import get_tool_definitions, get_tools_prompt_text
from app.tools.skills_tool import skill_view


def tool_names() -> set[str]:
    return {item["function"]["name"] for item in get_tool_definitions()}


def test_executor_tools_are_hidden_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SCRATCHPAD_ENABLE_EXECUTOR_TOOLS", raising=False)

    names = tool_names()
    prompt_text = get_tools_prompt_text()

    assert "run_shell" not in names
    assert "run_python" not in names
    assert "run_shell" not in prompt_text
    assert "run_python" not in prompt_text


def test_executor_tools_can_be_enabled_for_dev_mode(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCHPAD_ENABLE_EXECUTOR_TOOLS", "1")

    names = tool_names()
    prompt_text = get_tools_prompt_text()

    assert "run_shell" in names
    assert "run_python" in names
    assert "run_shell" in prompt_text
    assert "run_python" in prompt_text


def test_youtube_skill_routes_url_only_inputs_to_youtube_analyze() -> None:
    content = skill_view("youtube-content")["content"]

    assert "use `youtube_analyze` first" in content
    assert "Only present quotes that are directly supported" in content
