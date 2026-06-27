"""Tests for optional MLflow metric extraction helpers."""

from scripts.observability.experiment_tracking import flatten_numeric_metrics


def test_flatten_numeric_metrics_skips_large_result_payloads() -> None:
    metrics = flatten_numeric_metrics(
        {
            "overall_accuracy": 0.5,
            "latency": {"average_seconds": 1.2},
            "groups": {
                "difficulty": {
                    "hard": {
                        "overall_accuracy": 0.25,
                    }
                }
            },
            "results": [{"id": "case"}],
            "confusion_matrix": {"a": {"b": 1}},
        }
    )

    assert metrics["overall_accuracy"] == 0.5
    assert metrics["latency.average_seconds"] == 1.2
    assert metrics["groups.difficulty.hard.overall_accuracy"] == 0.25
    assert all(not key.startswith("results") for key in metrics)
    assert all(not key.startswith("confusion_matrix") for key in metrics)
