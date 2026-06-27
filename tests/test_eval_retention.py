"""Tests for retention eval labels, text constraints, and summaries."""

import json

from scripts.evals.retention import (
    build_report,
    evaluate_text_expectations,
    load_cases,
    retention_label,
)


def test_retention_cases_load_and_use_allowed_labels() -> None:
    cases = load_cases()

    assert 20 <= len(cases) <= 40
    assert {case["retention_kind"] for case in cases} == {
        "conceptual",
        "instruction_following",
        "url_abstention",
        "simple_coding_math",
        "general_assistant",
    }


def test_evaluate_text_expectations_scores_content_constraints() -> None:
    checks = evaluate_text_expectations(
        "LoRA uses low-rank adapters for fine-tuning.",
        {"must_contain_any": ["adapter", "overfitting"], "max_words": 10},
    )

    assert [check["passed"] for check in checks] == [True, True]


def test_retention_label_fails_tool_calls_and_degrades_weak_answers() -> None:
    assert retention_label(tool_called=True, content="Good answer", checks=[]) == "fail"
    assert retention_label(tool_called=False, content="", checks=[]) == "fail"
    assert retention_label(
        tool_called=False,
        content="Weak answer",
        checks=[{"passed": False}],
    ) == "degraded"
    assert retention_label(
        tool_called=False,
        content="Strong answer",
        checks=[{"passed": True}],
    ) == "pass"


def test_build_retention_report_groups_by_kind() -> None:
    report = build_report(
        [
            {
                "id": "a",
                "retention_kind": "conceptual",
                "label": "pass",
                "passed": True,
                "tool_called": False,
                "latency_seconds": 1.0,
            },
            {
                "id": "b",
                "retention_kind": "conceptual",
                "label": "degraded",
                "passed": False,
                "tool_called": False,
                "latency_seconds": 2.0,
            },
            {
                "id": "c",
                "retention_kind": "url_abstention",
                "label": "fail",
                "passed": False,
                "tool_called": True,
                "latency_seconds": 3.0,
            },
        ],
        provider="llama_cpp",
        model="qwen",
    )

    assert json.dumps(report)
    assert report["label_counts"] == {"pass": 1, "degraded": 1, "fail": 1}
    assert report["retention_pass_rate"] == 0.3333
    assert report["tool_false_positive_rate"] == 0.3333
    assert report["by_kind"]["conceptual"]["pass"] == 1
    assert report["latency"]["average_seconds"] == 2.0
