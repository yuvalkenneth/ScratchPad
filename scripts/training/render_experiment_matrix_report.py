"""Render a matrix-level comparison dashboard for FT learning runs."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.observability.render_runtime_report import point_path
from scripts.training.experiment_config import DEFAULT_RUNS_ROOT
from scripts.training.render_experiment_report import (
    eval_metric_summary,
    load_optional_json,
    normalize_log_history,
    summarize_training_logs,
)


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Load one run manifest when present."""
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def collect_run_rows(runs_root: Path = DEFAULT_RUNS_ROOT) -> list[dict[str, Any]]:
    """Collect comparable LoRA/QLoRA fields from run artifacts."""
    rows: list[dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        manifest = load_manifest(run_dir)
        if manifest is None:
            continue
        spec = manifest.get("spec", {})
        lora = spec.get("lora", {}) if isinstance(spec, dict) else {}
        training_rows = normalize_log_history(load_optional_json(run_dir / "trainer_log.json"))
        eval_payload = load_optional_json(run_dir / "scorecard.json") or load_optional_json(run_dir / "eval_metrics.json")
        training_summary = summarize_training_logs(training_rows)
        metric_summary = eval_metric_summary(eval_payload)
        rows.append(
            {
                "experiment_id": manifest.get("experiment_id"),
                "method": spec.get("method") if isinstance(spec, dict) else None,
                "rank": lora.get("rank") if isinstance(lora, dict) else None,
                "alpha": lora.get("alpha") if isinstance(lora, dict) else None,
                "dropout": lora.get("dropout") if isinstance(lora, dict) else None,
                "quantization": lora.get("quantization") if isinstance(lora, dict) else None,
                "target_modules": ",".join(lora.get("target_modules", [])) if isinstance(lora, dict) else "",
                "last_train_loss": training_summary.get("last_train_loss"),
                "last_eval_loss": training_summary.get("last_eval_loss"),
                "min_eval_loss": training_summary.get("min_eval_loss"),
                **metric_summary,
            }
        )
    return rows


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Extract a numeric column from row dictionaries."""
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        values.append(float(value))
    return values


def render_sparkline(title: str, values: list[float]) -> str:
    """Render a tiny matrix comparison line when values exist."""
    if not values:
        return f"<p class=\"missing\">No {html.escape(title)} values yet.</p>"
    path = point_path(values, width=700, height=180)
    return f"""
    <figure>
      <figcaption>{html.escape(title)}</figcaption>
      <svg viewBox="0 0 700 180" role="img" aria-label="{html.escape(title)}">
        <path d="{path}" fill="none" stroke="#2563eb" stroke-width="3" />
      </svg>
    </figure>
"""


def render_rows_table(rows: list[dict[str, Any]]) -> str:
    """Render comparable run rows."""
    if not rows:
        return '<p class="missing">No run manifests were found. Create them with run_sft_experiment.py --write-manifest.</p>'
    columns = [
        "experiment_id",
        "rank",
        "alpha",
        "dropout",
        "quantization",
        "last_train_loss",
        "last_eval_loss",
        "min_eval_loss",
        "overall_accuracy",
        "tool_overall_accuracy_delta",
        "tool_argument_accuracy_delta",
        "retention_pass_rate_delta",
    ]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_matrix_html(rows: list[dict[str, Any]]) -> str:
    """Render a standalone dashboard for comparing FT runs."""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scratchpad FT Matrix Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #18212f; background: #f8fafc; }}
    section {{ background: #fff; border: 1px solid #dbe3ed; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5eaf0; padding: 8px; text-align: left; vertical-align: top; }}
    svg {{ width: 100%; max-width: 760px; border: 1px solid #d6dde6; border-radius: 8px; background: #fbfcfd; }}
    .missing {{ color: #8a4b00; background: #fff8e8; padding: 10px 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Scratchpad FT Matrix Report</h1>
  <section>
    <h2>How To Read This</h2>
    <p>Use this report to compare LoRA and QLoRA knobs after each run has a manifest, trainer log, and eval scorecard. Rank and alpha explain adapter capacity/update scale; loss explains fitting; eval deltas decide whether the run is worth keeping.</p>
  </section>
  <section>
    <h2>Comparison Table</h2>
    {render_rows_table(rows)}
  </section>
  <section>
    <h2>Classic Comparison Curves</h2>
    {render_sparkline("Rank by run order", numeric_values(rows, "rank"))}
    {render_sparkline("Final train loss by run order", numeric_values(rows, "last_train_loss"))}
    {render_sparkline("Final eval loss by run order", numeric_values(rows, "last_eval_loss"))}
    {render_sparkline("Tool overall accuracy delta by run order", numeric_values(rows, "tool_overall_accuracy_delta"))}
    {render_sparkline("Retention pass-rate delta by run order", numeric_values(rows, "retention_pass_rate_delta"))}
  </section>
</body>
</html>
"""


def run(argv: list[str] | None = None) -> int:
    """Run the matrix report renderer CLI."""
    parser = argparse.ArgumentParser(description="Render a fine-tuning matrix comparison dashboard.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = collect_run_rows(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_matrix_html(rows), encoding="utf-8")
    print(f"Wrote FT matrix report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
