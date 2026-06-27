"""Tests for tool-choice eval scoring, grouping, and failure taxonomy."""

from scripts.evals.tool_choice import (
    build_report,
    classify_failure_types,
    effective_tool_arguments,
    evaluate_arguments,
    filter_cases_by_split,
    parse_tool_arguments,
    request_messages_from_case,
    summarize_result_subset,
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


def test_filter_cases_by_split_keeps_only_requested_split() -> None:
    cases = [
        {"id": "train_case", "user": "Save https://example.com", "expected_tool": "content_add", "split": "train"},
        {
            "id": "heldout_case",
            "user": "Inspect https://example.com",
            "expected_tool": "analyze_source",
            "split": "heldout",
        },
    ]

    assert [case["id"] for case in filter_cases_by_split(cases, "heldout")] == ["heldout_case"]
    assert filter_cases_by_split(cases, None) == cases


def test_request_messages_from_case_supports_multi_turn_context() -> None:
    messages = request_messages_from_case(
        {
            "id": "context",
            "messages": [
                {"role": "user", "content": "Here is https://example.com"},
                {"role": "assistant", "content": "I can save it."},
                {"role": "user", "content": "Save it."},
            ],
            "expected_tool": "content_add",
        }
    )

    assert messages[0]["role"] == "system"
    assert [message["role"] for message in messages[1:]] == ["user", "assistant", "user"]


def test_build_report_scores_tool_selection_as_multiclass_classification() -> None:
    report = build_report(
        [
            {
                "id": "a",
                "case_metadata": {
                    "difficulty": "easy",
                    "context_kind": "stateless",
                    "category": "content_list",
                    "intent": "library_query",
                    "split": "heldout",
                },
                "passed": True,
                "tool_pass": True,
                "argument_pass": True,
                "expected_tool": "content_list",
                "actual_tool": "content_list",
                "argument_checks": [{"passed": True}],
                "argument_json_valid": True,
                "used_tool_defaults": False,
                "extra_tool_count": 0,
                "failure_types": [],
                "latency_seconds": 1.0,
            },
            {
                "id": "b",
                "case_metadata": {
                    "difficulty": "hard",
                    "context_kind": "contextual",
                    "category": "contextual_reference",
                    "intent": "metadata_update",
                    "split": "heldout",
                },
                "passed": False,
                "tool_pass": False,
                "argument_pass": True,
                "expected_tool": "content_update",
                "actual_tool": "content_list",
                "argument_checks": [],
                "argument_json_valid": True,
                "used_tool_defaults": True,
                "extra_tool_count": 0,
                "failure_types": ["wrong_tool"],
                "latency_seconds": 2.0,
            },
            {
                "id": "c",
                "case_metadata": {
                    "difficulty": "ambiguous",
                    "context_kind": "stateless",
                    "category": "ambiguous_no_tool",
                    "intent": "clarify_missing_context",
                    "split": "heldout",
                },
                "passed": True,
                "tool_pass": True,
                "argument_pass": True,
                "expected_tool": "no_tool",
                "actual_tool": "no_tool",
                "argument_checks": [],
                "argument_json_valid": True,
                "used_tool_defaults": False,
                "extra_tool_count": 0,
                "failure_types": [],
                "latency_seconds": 3.0,
            },
        ],
        provider="llama_cpp",
        model="test-model",
        split="heldout",
    )

    assert report["split"] == "heldout"
    assert report["tool_selection_accuracy"] == 0.6667
    assert report["argument_accuracy"] == 1.0
    assert report["default_reliance_rate"] == 0.3333
    assert report["confusion_matrix"]["content_update"]["content_list"] == 1
    assert report["per_class"]["content_list"]["precision"] == 0.5
    assert report["per_class"]["content_update"]["recall"] == 0.0
    assert report["failure_type_counts"] == {"wrong_tool": 1}
    assert report["latency"]["average_seconds"] == 2.0
    assert report["groups"]["difficulty"]["hard"]["overall_accuracy"] == 0.0
    assert report["groups"]["context_kind"]["contextual"]["wrong_tool_rate"] == 1.0


def test_summarize_result_subset_calculates_group_metrics() -> None:
    summary = summarize_result_subset(
        [
            {
                "passed": False,
                "tool_pass": False,
                "argument_pass": False,
                "expected_tool": "no_tool",
                "actual_tool": "content_list",
                "argument_checks": [],
                "argument_json_valid": True,
                "extra_tool_count": 0,
                "failure_types": ["tool_false_positive"],
                "latency_seconds": 1.5,
            }
        ]
    )

    assert summary["overall_accuracy"] == 0.0
    assert summary["tool_false_positive_rate"] == 1.0
    assert summary["failure_type_counts"] == {"tool_false_positive": 1}


def test_classify_failure_types_names_tool_and_argument_failures() -> None:
    assert classify_failure_types(
        expected_tool="content_list",
        actual_tool="no_tool",
        argument_json_valid=True,
        argument_pass=True,
        extra_tool_count=0,
    ) == ["no_tool_false_negative"]
    assert classify_failure_types(
        expected_tool="no_tool",
        actual_tool="content_list",
        argument_json_valid=False,
        argument_pass=False,
        extra_tool_count=1,
    ) == ["tool_false_positive", "invalid_tool_arguments_json", "extra_tool_call"]
