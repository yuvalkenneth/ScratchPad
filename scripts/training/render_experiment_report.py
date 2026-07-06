"""Render fine-tuning experiment artifacts as a teaching-oriented HTML report."""

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


TEACHING_NOTES = {
    "train_loss": {
        "shows": "Whether the model is fitting the training examples over steps.",
        "good": "A steady downward trend that does not require extreme steps to move.",
        "worry": "Flat loss, sudden spikes, or a collapse to near-zero while validation loss worsens.",
        "decision": "Adjust learning rate, steps, data quality, or LoRA capacity.",
    },
    "eval_loss": {
        "shows": "Whether training improvements transfer to validation examples.",
        "good": "Validation loss falls with training loss, then plateaus.",
        "worry": "Validation loss rises while training loss keeps falling.",
        "decision": "Stop earlier, add dropout, reduce rank, or improve data coverage.",
    },
    "learning_rate": {
        "shows": "The actual optimizer schedule over global steps.",
        "good": "Warmup and decay match the intended schedule.",
        "worry": "Learning rate is too high during unstable loss spikes.",
        "decision": "Tune warmup, scheduler, or peak LR before changing model size.",
    },
    "grad_norm": {
        "shows": "Gradient scale and training stability.",
        "good": "No sustained explosions, with occasional spikes explainable by batches.",
        "worry": "Exploding or vanishing gradients over many steps.",
        "decision": "Use clipping, lower LR, or inspect outlier examples.",
    },
    "runtime": {
        "shows": "Throughput, step time, and resource pressure.",
        "good": "Stable step time and memory after warmup.",
        "worry": "Memory climbs over time or throughput degrades sharply.",
        "decision": "Change batch size, gradient accumulation, sequence length, or QLoRA.",
    },
    "eval_metrics": {
        "shows": "Task quality, not just language-model loss.",
        "good": "Heldout tool metrics improve without retention regressions.",
        "worry": "Accuracy improves only by over-calling tools or breaking retention.",
        "decision": "Keep, reject, or rerun with data/parameter changes.",
    },
}


def load_json(path: Path) -> Any:
    """Load JSON from a path."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json(path: Path) -> Any | None:
    """Load JSON when present."""
    if not path.exists():
        return None
    return load_json(path)


def normalize_log_history(payload: Any) -> list[dict[str, Any]]:
    """Normalize common trainer log payload shapes into a list of rows."""
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("log_history") or payload.get("logs") or []
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def numeric_series(rows: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    """Extract `(step, value)` points for a numeric log key."""
    points: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        step = row.get("step", row.get("global_step", index))
        if isinstance(step, bool) or not isinstance(step, int | float):
            step = index
        points.append((float(step), float(value)))
    return points


def exponential_moving_average(values: list[float], *, weight: float = 0.35) -> list[float]:
    """Smooth a series while preserving the first value."""
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append((weight * value) + ((1 - weight) * smoothed[-1]))
    return smoothed


def summarize_training_logs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize available trainer-log keys for reports and tests."""
    train_loss = numeric_series(rows, "loss")
    eval_loss = numeric_series(rows, "eval_loss")
    return {
        "steps_logged": len({point[0] for point in train_loss + eval_loss}),
        "train_loss_points": len(train_loss),
        "eval_loss_points": len(eval_loss),
        "first_train_loss": train_loss[0][1] if train_loss else None,
        "last_train_loss": train_loss[-1][1] if train_loss else None,
        "first_eval_loss": eval_loss[0][1] if eval_loss else None,
        "last_eval_loss": eval_loss[-1][1] if eval_loss else None,
        "min_train_loss": min((point[1] for point in train_loss), default=None),
        "min_eval_loss": min((point[1] for point in eval_loss), default=None),
    }


def metric_at_path(payload: dict[str, Any], path: str) -> float | int | None:
    """Read a nested numeric metric by dotted path."""
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, bool) or not isinstance(current, int | float):
        return None
    return current


def eval_metric_summary(payload: Any) -> dict[str, float | int]:
    """Extract the main eval metrics used in FT scorecards."""
    if not isinstance(payload, dict):
        return {}
    direct_metrics = [
        "overall_accuracy",
        "tool_selection_accuracy",
        "argument_accuracy",
        "argument_json_validity_rate",
        "tool_false_positive_rate",
        "wrong_tool_rate",
        "retention_pass_rate",
    ]
    summary: dict[str, float | int] = {}
    for key in direct_metrics:
        value = metric_at_path(payload, key)
        if value is not None:
            summary[key] = value
    scorecard_paths = {
        "tool_overall_accuracy_delta": "tool_choice.metrics.overall_accuracy.delta",
        "tool_argument_accuracy_delta": "tool_choice.metrics.argument_accuracy.delta",
        "retention_pass_rate_delta": "retention.metrics.retention_pass_rate.delta",
    }
    for output_key, path in scorecard_paths.items():
        value = metric_at_path(payload, path)
        if value is not None:
            summary[output_key] = value
    return summary


def loss_gap_series(
    train_loss: list[tuple[float, float]],
    eval_loss: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return validation-minus-train loss for steps logged in both series."""
    train_by_step = {step: value for step, value in train_loss}
    return [
        (step, eval_value - train_by_step[step])
        for step, eval_value in eval_loss
        if step in train_by_step
    ]


def svg_line_chart(
    title: str,
    points: list[tuple[float, float]],
    *,
    secondary_values: list[float] | None = None,
    width: int = 900,
    height: int = 260,
) -> str:
    """Render a small dependency-free SVG line chart."""
    if not points:
        return f"<p class=\"missing\">No {html.escape(title)} points were found.</p>"
    values = [point[1] for point in points]
    primary_path = point_path(values, width=width, height=height)
    secondary_path = point_path(secondary_values or [], width=width, height=height)
    min_value = min(values + (secondary_values or []))
    max_value = max(values + (secondary_values or []))
    secondary = (
        f'<path d="{secondary_path}" fill="none" stroke="#f97316" stroke-width="2" stroke-dasharray="5 5" />'
        if secondary_path
        else ""
    )
    return f"""
    <figure>
      <figcaption>{html.escape(title)} <span>{min_value:.4g} to {max_value:.4g}</span></figcaption>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <path d="{primary_path}" fill="none" stroke="#2563eb" stroke-width="3" />
        {secondary}
      </svg>
    </figure>
"""


def teaching_note(key: str) -> str:
    """Render a standard teaching note for a report section."""
    note = TEACHING_NOTES[key]
    return f"""
    <div class="teaching-note">
      <p><strong>What this shows:</strong> {html.escape(note["shows"])}</p>
      <p><strong>What good looks like:</strong> {html.escape(note["good"])}</p>
      <p><strong>What would worry us:</strong> {html.escape(note["worry"])}</p>
      <p><strong>Decision rule:</strong> {html.escape(note["decision"])}</p>
    </div>
"""


def render_metric_table(metrics: dict[str, float | int]) -> str:
    """Render a compact metric table."""
    if not metrics:
        return '<p class="missing">No eval metric payload was found.</p>'
    rows = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    return f"<table><tbody>{rows}</tbody></table>"


def render_html_report(
    *,
    manifest: dict[str, Any],
    trainer_rows: list[dict[str, Any]],
    eval_payload: Any | None = None,
) -> str:
    """Render a complete standalone HTML report for one FT experiment."""
    spec = manifest.get("spec", {})
    lora = spec.get("lora", {}) if isinstance(spec, dict) else {}
    training = spec.get("training", {}) if isinstance(spec, dict) else {}
    train_loss = numeric_series(trainer_rows, "loss")
    eval_loss = numeric_series(trainer_rows, "eval_loss")
    learning_rate = numeric_series(trainer_rows, "learning_rate")
    grad_norm = numeric_series(trainer_rows, "grad_norm")
    epoch = numeric_series(trainer_rows, "epoch")
    step_time = numeric_series(trainer_rows, "step_time_seconds")
    samples_per_second = numeric_series(trainer_rows, "samples_per_second")
    tokens_per_second = numeric_series(trainer_rows, "tokens_per_second")
    memory = numeric_series(trainer_rows, "gpu_memory_peak_gib")
    gpu_utilization = numeric_series(trainer_rows, "gpu_utilization_percent")
    summary = summarize_training_logs(trainer_rows)
    eval_summary = eval_metric_summary(eval_payload)
    train_values = [point[1] for point in train_loss]
    gap = loss_gap_series(train_loss, eval_loss)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(str(manifest.get("experiment_id", "FT Experiment")))} Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #18212f; background: #f8fafc; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ background: #fff; border: 1px solid #dbe3ed; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    pre {{ background: #f4f6f8; padding: 16px; border-radius: 8px; overflow: auto; }}
    svg {{ width: 100%; max-width: 960px; border: 1px solid #d6dde6; border-radius: 8px; background: #fbfcfd; }}
    figcaption {{ font-weight: 650; margin: 8px 0; }}
    figcaption span {{ color: #526172; font-weight: 400; margin-left: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5eaf0; padding: 8px; text-align: left; }}
    .teaching-note {{ background: #f5f7fb; border-left: 4px solid #2563eb; padding: 10px 14px; margin: 12px 0; }}
    .missing {{ color: #8a4b00; background: #fff8e8; padding: 10px 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{html.escape(str(manifest.get("experiment_id", "FT Experiment")))}</h1>
  <section>
    <h2>Run Summary</h2>
    <pre>{html.escape(json.dumps({
        "status": manifest.get("status"),
        "git_sha": manifest.get("git_sha"),
        "method": spec.get("method") if isinstance(spec, dict) else None,
        "expected_outcome": spec.get("expected_outcome") if isinstance(spec, dict) else None,
        "training": training,
        "lora": lora,
        "training_log_summary": summary,
    }, indent=2))}</pre>
  </section>
  <section>
    <h2>Training Loss</h2>
    {teaching_note("train_loss")}
    {svg_line_chart("Train loss by step", train_loss, secondary_values=exponential_moving_average(train_values))}
  </section>
  <section>
    <h2>Validation Loss</h2>
    {teaching_note("eval_loss")}
    {svg_line_chart("Validation loss by eval step", eval_loss)}
    {svg_line_chart("Validation minus train loss gap", gap)}
  </section>
  <section>
    <h2>Optimizer Stability</h2>
    {teaching_note("learning_rate")}
    {svg_line_chart("Learning rate by step", learning_rate)}
    {teaching_note("grad_norm")}
    {svg_line_chart("Gradient norm by step", grad_norm)}
    {svg_line_chart("Epoch progress by step", epoch)}
  </section>
  <section>
    <h2>Runtime</h2>
    {teaching_note("runtime")}
    {svg_line_chart("Step time seconds", step_time)}
    {svg_line_chart("Samples per second", samples_per_second)}
    {svg_line_chart("Tokens per second", tokens_per_second)}
    {svg_line_chart("GPU peak memory GiB", memory)}
    {svg_line_chart("GPU utilization percent", gpu_utilization)}
  </section>
  <section>
    <h2>Eval Metrics</h2>
    {teaching_note("eval_metrics")}
    {render_metric_table(eval_summary)}
  </section>
</body>
</html>
"""


def run(argv: list[str] | None = None) -> int:
    """Run the FT experiment HTML renderer CLI."""
    parser = argparse.ArgumentParser(description="Render a fine-tuning experiment report.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trainer-log", type=Path)
    parser.add_argument("--eval-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_json(args.manifest)
    trainer_path = args.trainer_log or Path(manifest.get("artifact_paths", {}).get("trainer_log", ""))
    eval_path = args.eval_metrics or Path(manifest.get("artifact_paths", {}).get("eval_metrics", ""))
    if not trainer_path.is_absolute():
        trainer_path = REPO_ROOT / trainer_path
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path
    trainer_rows = normalize_log_history(load_optional_json(trainer_path))
    eval_payload = load_optional_json(eval_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html_report(manifest=manifest, trainer_rows=trainer_rows, eval_payload=eval_payload),
        encoding="utf-8",
    )
    print(f"Wrote FT experiment report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
