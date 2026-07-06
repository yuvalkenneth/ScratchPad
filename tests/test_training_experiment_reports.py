"""Tests for fine-tuning experiment report rendering."""

from scripts.training.render_experiment_report import (
    eval_metric_summary,
    loss_gap_series,
    normalize_log_history,
    render_html_report,
    summarize_training_logs,
)
from scripts.training.render_experiment_matrix_report import collect_run_rows, render_matrix_html


def test_summarize_training_logs_extracts_classic_metrics() -> None:
    rows = normalize_log_history(
        {
            "log_history": [
                {"step": 1, "loss": 1.2, "learning_rate": 0.0002},
                {"step": 2, "loss": 0.9, "grad_norm": 0.5},
                {"step": 2, "eval_loss": 1.0},
            ]
        }
    )

    summary = summarize_training_logs(rows)

    assert summary["train_loss_points"] == 2
    assert summary["eval_loss_points"] == 1
    assert summary["first_train_loss"] == 1.2
    assert summary["last_train_loss"] == 0.9


def test_eval_metric_summary_handles_scorecard_deltas() -> None:
    metrics = eval_metric_summary(
        {
            "tool_choice": {
                "metrics": {
                    "overall_accuracy": {"delta": 0.2},
                    "argument_accuracy": {"delta": 0.15},
                }
            },
            "retention": {
                "metrics": {
                    "retention_pass_rate": {"delta": -0.05},
                }
            },
        }
    )

    assert metrics["tool_overall_accuracy_delta"] == 0.2
    assert metrics["tool_argument_accuracy_delta"] == 0.15
    assert metrics["retention_pass_rate_delta"] == -0.05


def test_render_html_report_includes_teaching_notes_and_plots() -> None:
    html = render_html_report(
        manifest={
            "experiment_id": "synthetic-run",
            "status": "complete",
            "git_sha": "abc123",
            "spec": {
                "method": "sft",
                "expected_outcome": "loss should fall",
                "training": {"seed": 13},
                "lora": {"rank": 8, "alpha": 16},
            },
        },
        trainer_rows=[
            {"step": 1, "loss": 1.2, "learning_rate": 0.0002, "grad_norm": 0.8},
            {"step": 2, "loss": 0.9, "eval_loss": 1.0, "tokens_per_second": 400.0},
        ],
        eval_payload={"overall_accuracy": 0.75, "tool_false_positive_rate": 0.05},
    )

    assert "What this shows" in html
    assert "Train loss by step" in html
    assert "Validation loss by eval step" in html
    assert "Validation minus train loss gap" in html
    assert "Samples per second" in html
    assert "overall_accuracy" in html


def test_loss_gap_series_uses_matching_steps() -> None:
    gap = loss_gap_series(
        [(1.0, 1.2), (2.0, 0.9)],
        [(2.0, 1.1), (3.0, 1.0)],
    )

    assert gap == [(2.0, 0.20000000000000007)]


def test_collect_run_rows_and_render_matrix_html(tmp_path) -> None:
    run_dir = tmp_path / "lora-r8-alpha16"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        """{
  "experiment_id": "lora-r8-alpha16",
  "spec": {
    "method": "sft",
    "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"], "quantization": null}
  }
}
""",
        encoding="utf-8",
    )
    (run_dir / "trainer_log.json").write_text(
        """{"log_history": [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.8, "eval_loss": 0.9}]}""",
        encoding="utf-8",
    )
    (run_dir / "scorecard.json").write_text(
        """{"tool_choice": {"metrics": {"overall_accuracy": {"delta": 0.2}}}}""",
        encoding="utf-8",
    )

    rows = collect_run_rows(tmp_path)
    html = render_matrix_html(rows)

    assert rows[0]["experiment_id"] == "lora-r8-alpha16"
    assert rows[0]["rank"] == 8
    assert rows[0]["tool_overall_accuracy_delta"] == 0.2
    assert "Comparison Table" in html
