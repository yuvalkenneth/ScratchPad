"""Tests for base-vs-SFT comparison scorecards."""

from scripts.training.compare_reports import build_scorecard


def test_build_scorecard_compares_tool_and_retention_metrics() -> None:
    scorecard = build_scorecard(
        base_tool_report={
            "model": "qwen-base",
            "overall_accuracy": 0.5,
            "tool_selection_accuracy": 0.7,
            "argument_accuracy": 0.4,
            "argument_json_validity_rate": 1.0,
            "tool_false_positive_rate": 0.1,
            "wrong_tool_rate": 0.2,
            "extra_tool_rate": 0.0,
            "failure_type_counts": {"wrong_tool": 4},
            "per_class": {"content_add": {"f1": 0.5}},
            "groups": {
                "difficulty": {
                    "hard": {
                        "total_cases": 2,
                        "overall_accuracy": 0.25,
                        "tool_selection_accuracy": 0.5,
                        "argument_accuracy": 0.0,
                        "argument_json_validity_rate": 1.0,
                        "tool_false_positive_rate": 0.0,
                        "wrong_tool_rate": 0.5,
                        "extra_tool_rate": 0.0,
                        "failure_type_counts": {"wrong_tool": 1},
                    }
                }
            },
            "latency": {"average_seconds": 2.0},
        },
        sft_tool_report={
            "model": "qwen-sft",
            "overall_accuracy": 0.8,
            "tool_selection_accuracy": 0.9,
            "argument_accuracy": 0.75,
            "argument_json_validity_rate": 1.0,
            "tool_false_positive_rate": 0.05,
            "wrong_tool_rate": 0.1,
            "extra_tool_rate": 0.0,
            "failure_type_counts": {"wrong_tool": 2, "argument_mismatch": 1},
            "per_class": {"content_add": {"f1": 0.8}},
            "groups": {
                "difficulty": {
                    "hard": {
                        "total_cases": 2,
                        "overall_accuracy": 0.5,
                        "tool_selection_accuracy": 0.75,
                        "argument_accuracy": 0.25,
                        "argument_json_validity_rate": 1.0,
                        "tool_false_positive_rate": 0.0,
                        "wrong_tool_rate": 0.25,
                        "extra_tool_rate": 0.0,
                        "failure_type_counts": {"wrong_tool": 0},
                    }
                }
            },
            "latency": {"average_seconds": 2.5},
        },
        base_retention_report={
            "retention_pass_rate": 0.9,
            "tool_false_positive_rate": 0.0,
            "label_counts": {"pass": 9, "degraded": 1, "fail": 0},
            "by_kind": {"conceptual": {"pass": 2, "degraded": 0, "fail": 0}},
            "latency": {"average_seconds": 1.0},
        },
        sft_retention_report={
            "retention_pass_rate": 0.8,
            "tool_false_positive_rate": 0.1,
            "label_counts": {"pass": 8, "degraded": 1, "fail": 1},
            "by_kind": {"conceptual": {"pass": 1, "degraded": 1, "fail": 0}},
            "latency": {"average_seconds": 1.2},
        },
    )

    assert scorecard["type"] == "tool_choice_sft_v1_scorecard"
    assert scorecard["model"] == {"base": "qwen-base", "sft": "qwen-sft"}
    assert scorecard["tool_choice"]["metrics"]["overall_accuracy"]["delta"] == 0.3
    assert scorecard["tool_choice"]["failure_type_counts"]["wrong_tool"]["delta"] == -2
    assert scorecard["tool_choice"]["per_class_f1"]["content_add"]["delta"] == 0.3
    hard_group = scorecard["tool_choice"]["groups"]["difficulty"]["hard"]
    assert hard_group["overall_accuracy"]["delta"] == 0.25
    assert hard_group["failure_type_counts"]["wrong_tool"]["delta"] == -1
    assert scorecard["retention"]["metrics"]["retention_pass_rate"]["delta"] == -0.1
    assert scorecard["retention"]["label_counts"]["fail"]["delta"] == 1
