from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOOL_METRICS = (
    "overall_accuracy",
    "tool_selection_accuracy",
    "argument_accuracy",
    "argument_json_validity_rate",
    "tool_false_positive_rate",
    "wrong_tool_rate",
    "extra_tool_rate",
)
RETENTION_METRICS = (
    "retention_pass_rate",
    "tool_false_positive_rate",
)


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"Report must be a JSON object: {path}")
    return report


def metric_delta(base: float | int | None, sft: float | int | None) -> float | None:
    if base is None or sft is None:
        return None
    return round(float(sft) - float(base), 4)


def compare_metric_group(
    base: dict[str, Any],
    sft: dict[str, Any],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    return {
        metric: {
            "base": base.get(metric),
            "sft": sft.get(metric),
            "delta": metric_delta(base.get(metric), sft.get(metric)),
        }
        for metric in metric_names
    }


def compare_failure_type_counts(
    base: dict[str, Any],
    sft: dict[str, Any],
) -> dict[str, dict[str, int]]:
    base_counts = base.get("failure_type_counts", {})
    sft_counts = sft.get("failure_type_counts", {})
    labels = sorted(set(base_counts) | set(sft_counts))
    return {
        label: {
            "base": int(base_counts.get(label, 0)),
            "sft": int(sft_counts.get(label, 0)),
            "delta": int(sft_counts.get(label, 0)) - int(base_counts.get(label, 0)),
        }
        for label in labels
    }


def compare_per_class_f1(
    base: dict[str, Any],
    sft: dict[str, Any],
) -> dict[str, dict[str, float]]:
    base_per_class = base.get("per_class", {})
    sft_per_class = sft.get("per_class", {})
    labels = sorted(set(base_per_class) | set(sft_per_class))
    return {
        label: {
            "base": float(base_per_class.get(label, {}).get("f1", 0.0)),
            "sft": float(sft_per_class.get(label, {}).get("f1", 0.0)),
            "delta": round(
                float(sft_per_class.get(label, {}).get("f1", 0.0))
                - float(base_per_class.get(label, {}).get("f1", 0.0)),
                4,
            ),
        }
        for label in labels
    }


def compare_tool_groups(
    base: dict[str, Any],
    sft: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    base_groups = base.get("groups", {})
    sft_groups = sft.get("groups", {})
    group_names = sorted(set(base_groups) | set(sft_groups))
    return {
        group_name: {
            group_value: compare_metric_group(
                base_groups.get(group_name, {}).get(group_value, {}),
                sft_groups.get(group_name, {}).get(group_value, {}),
                TOOL_METRICS,
            )
            | {
                "total_cases": {
                    "base": base_groups.get(group_name, {}).get(group_value, {}).get("total_cases", 0),
                    "sft": sft_groups.get(group_name, {}).get(group_value, {}).get("total_cases", 0),
                    "delta": int(sft_groups.get(group_name, {}).get(group_value, {}).get("total_cases", 0))
                    - int(base_groups.get(group_name, {}).get(group_value, {}).get("total_cases", 0)),
                },
                "failure_type_counts": compare_failure_type_counts(
                    base_groups.get(group_name, {}).get(group_value, {}),
                    sft_groups.get(group_name, {}).get(group_value, {}),
                ),
            }
            for group_value in sorted(
                set(base_groups.get(group_name, {}))
                | set(sft_groups.get(group_name, {}))
            )
        }
        for group_name in group_names
    }


def compare_label_counts(
    base: dict[str, Any],
    sft: dict[str, Any],
) -> dict[str, dict[str, int]]:
    base_counts = base.get("label_counts", {})
    sft_counts = sft.get("label_counts", {})
    labels = sorted(set(base_counts) | set(sft_counts))
    return {
        label: {
            "base": int(base_counts.get(label, 0)),
            "sft": int(sft_counts.get(label, 0)),
            "delta": int(sft_counts.get(label, 0)) - int(base_counts.get(label, 0)),
        }
        for label in labels
    }


def build_scorecard(
    *,
    base_tool_report: dict[str, Any],
    sft_tool_report: dict[str, Any],
    base_retention_report: dict[str, Any],
    sft_retention_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "tool_choice_sft_v1_scorecard",
        "model": {
            "base": base_tool_report.get("model"),
            "sft": sft_tool_report.get("model"),
        },
        "tool_choice": {
            "metrics": compare_metric_group(base_tool_report, sft_tool_report, TOOL_METRICS),
            "failure_type_counts": compare_failure_type_counts(base_tool_report, sft_tool_report),
            "per_class_f1": compare_per_class_f1(base_tool_report, sft_tool_report),
            "groups": compare_tool_groups(base_tool_report, sft_tool_report),
            "latency": {
                "average_seconds": {
                    "base": base_tool_report.get("latency", {}).get("average_seconds"),
                    "sft": sft_tool_report.get("latency", {}).get("average_seconds"),
                    "delta": metric_delta(
                        base_tool_report.get("latency", {}).get("average_seconds"),
                        sft_tool_report.get("latency", {}).get("average_seconds"),
                    ),
                }
            },
        },
        "retention": {
            "metrics": compare_metric_group(base_retention_report, sft_retention_report, RETENTION_METRICS),
            "label_counts": compare_label_counts(base_retention_report, sft_retention_report),
            "by_kind": {
                "base": base_retention_report.get("by_kind", {}),
                "sft": sft_retention_report.get("by_kind", {}),
            },
            "latency": {
                "average_seconds": {
                    "base": base_retention_report.get("latency", {}).get("average_seconds"),
                    "sft": sft_retention_report.get("latency", {}).get("average_seconds"),
                    "delta": metric_delta(
                        base_retention_report.get("latency", {}).get("average_seconds"),
                        sft_retention_report.get("latency", {}).get("average_seconds"),
                    ),
                }
            },
        },
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare base vs SFT tool-choice and retention reports.")
    parser.add_argument("--base-tool-report", type=Path, required=True)
    parser.add_argument("--sft-tool-report", type=Path, required=True)
    parser.add_argument("--base-retention-report", type=Path, required=True)
    parser.add_argument("--sft-retention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scorecard = build_scorecard(
        base_tool_report=load_report(args.base_tool_report),
        sft_tool_report=load_report(args.sft_tool_report),
        base_retention_report=load_report(args.base_retention_report),
        sft_retention_report=load_report(args.sft_retention_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scorecard, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison scorecard to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
