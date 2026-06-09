from scripts.eval_tool_choice import (
    effective_tool_arguments,
    evaluate_arguments,
    parse_tool_arguments,
)


def test_parse_tool_arguments_returns_object() -> None:
    assert parse_tool_arguments('{"status":"done"}') == {"status": "done"}


def test_effective_arguments_apply_content_list_status_default() -> None:
    assert effective_tool_arguments("content_list", {"max_estimated_time_minutes": 20}) == {
        "max_estimated_time_minutes": 20,
        "status": ["unread", "started"],
    }


def test_argument_eval_supports_equal_include_and_not_include() -> None:
    result = evaluate_arguments(
        expected={
            "must_equal": {"max_estimated_time_minutes": 20},
            "must_include": {"status": ["unread", "started"]},
            "must_not_include": {"status": ["done", "archived"]},
        },
        actual={
            "max_estimated_time_minutes": 20,
            "status": ["unread", "started"],
        },
    )

    assert result["passed"]
    assert len(result["checks"]) == 3


def test_argument_eval_reports_failures() -> None:
    result = evaluate_arguments(
        expected={
            "must_equal": {"status": "done"},
            "must_not_include": {"status": ["archived"]},
        },
        actual={"status": "archived"},
    )

    assert not result["passed"]
    assert [check["passed"] for check in result["checks"]] == [False, False]
