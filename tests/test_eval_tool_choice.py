from scripts.eval_tool_choice import (
    build_report,
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


def test_build_report_scores_tool_selection_as_multiclass_classification() -> None:
    report = build_report(
        [
            {
                "id": "a",
                "passed": True,
                "tool_pass": True,
                "argument_pass": True,
                "expected_tool": "content_list",
                "actual_tool": "content_list",
                "argument_checks": [{"passed": True}],
                "argument_json_valid": True,
                "used_tool_defaults": False,
                "extra_tool_count": 0,
                "latency_seconds": 1.0,
            },
            {
                "id": "b",
                "passed": False,
                "tool_pass": False,
                "argument_pass": True,
                "expected_tool": "content_update",
                "actual_tool": "content_list",
                "argument_checks": [],
                "argument_json_valid": True,
                "used_tool_defaults": True,
                "extra_tool_count": 0,
                "latency_seconds": 2.0,
            },
            {
                "id": "c",
                "passed": True,
                "tool_pass": True,
                "argument_pass": True,
                "expected_tool": "no_tool",
                "actual_tool": "no_tool",
                "argument_checks": [],
                "argument_json_valid": True,
                "used_tool_defaults": False,
                "extra_tool_count": 0,
                "latency_seconds": 3.0,
            },
        ],
        provider="llama_cpp",
        model="test-model",
    )

    assert report["tool_selection_accuracy"] == 0.6667
    assert report["argument_accuracy"] == 1.0
    assert report["default_reliance_rate"] == 0.3333
    assert report["confusion_matrix"]["content_update"]["content_list"] == 1
    assert report["per_class"]["content_list"]["precision"] == 0.5
    assert report["per_class"]["content_update"]["recall"] == 0.0
    assert report["latency"]["average_seconds"] == 2.0
