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


def test_default_tool_surface_hides_low_level_analysis_and_save_tools(monkeypatch) -> None:
    monkeypatch.delenv("SCRATCHPAD_ENABLE_EXECUTOR_TOOLS", raising=False)

    names = tool_names()
    prompt_text = get_tools_prompt_text()

    assert "get_time" not in names
    assert "analyze_source" in names
    assert "content_add" in names
    assert "content_list" in names
    assert "content_update" in names
    assert "content_status_update" in names
    assert "url_analyze" not in names
    assert "youtube_analyze" not in names
    assert "content_save" not in names
    assert "- analyze_source:" in prompt_text
    assert "- url_analyze:" not in prompt_text
    assert "- youtube_analyze:" not in prompt_text
    assert "- content_save:" not in prompt_text
    assert "- get_time:" not in prompt_text


def test_executor_tools_can_be_enabled_for_dev_mode(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCHPAD_ENABLE_EXECUTOR_TOOLS", "1")

    names = tool_names()
    prompt_text = get_tools_prompt_text()

    assert "run_shell" in names
    assert "run_python" in names
    assert "url_analyze" not in names
    assert "youtube_analyze" not in names
    assert "content_save" not in names
    assert "run_shell" in prompt_text
    assert "run_python" in prompt_text


def test_youtube_skill_routes_url_only_inputs_to_youtube_analyze() -> None:
    content = skill_view("youtube-content")["content"]

    assert "use `youtube_analyze` first" in content
    assert "Only present quotes that are directly supported" in content


def test_save_url_requests_route_to_content_add() -> None:
    prompt_text = get_tools_prompt_text()

    assert "call content_add directly" in prompt_text
    assert "The request is complete only after content_add reports a saved result" in prompt_text


def test_inspect_url_requests_route_to_analyze_source() -> None:
    prompt_text = get_tools_prompt_text()

    assert "whether it is worth reading/watching before saving, call analyze_source" in prompt_text


def test_content_status_update_prompt_lists_status_values() -> None:
    prompt_text = get_tools_prompt_text()

    assert "Status values: unread, started, done, archived, abandoned" in prompt_text


def test_content_update_prompt_handles_profile_corrections() -> None:
    prompt_text = get_tools_prompt_text()

    assert "- content_update:" in prompt_text
    assert "correct saved item details" in prompt_text
    assert "title, summary, subject, categories" in prompt_text


def test_content_list_prompt_mentions_default_status_filter() -> None:
    prompt_text = get_tools_prompt_text()

    assert "Defaults to status=[unread, started]" in prompt_text
    assert "default unread/started status filter" in prompt_text
