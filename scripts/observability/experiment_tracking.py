"""Optional MLflow logging helpers for Scratchpad eval reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def current_git_sha() -> str | None:
    """Return the current Git SHA when the checkout is available."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def flatten_numeric_metrics(
    value: Any,
    *,
    prefix: str = "",
    max_depth: int = 4,
) -> dict[str, float]:
    """Flatten report metrics into MLflow-compatible numeric keys."""
    if max_depth < 0:
        return {}
    if isinstance(value, bool):
        return {}
    if isinstance(value, int | float):
        return {prefix: float(value)} if prefix else {}
    if not isinstance(value, dict):
        return {}

    metrics: dict[str, float] = {}
    for key, child in value.items():
        if key in {"results", "confusion_matrix"}:
            continue
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        metrics.update(flatten_numeric_metrics(child, prefix=child_prefix, max_depth=max_depth - 1))
    return metrics


def mlflow_log_report(
    report: dict[str, Any],
    *,
    experiment_name: str,
    run_name: str | None = None,
    report_path: Path | None = None,
    artifacts: list[Path] | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Log an eval report, numeric metrics, params, and artifacts to MLflow."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging requested, but mlflow is not installed. "
            "Install it or omit --mlflow-experiment."
        ) from exc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logged_params = {
            "report_type": report.get("type"),
            "profile": report.get("profile"),
            "provider": report.get("provider"),
            "model": report.get("model"),
            "split": report.get("split"),
            "git_sha": current_git_sha(),
        }
        if params:
            logged_params.update(params)
        for key, value in logged_params.items():
            if value is not None:
                mlflow.log_param(key, value)
        for key, value in flatten_numeric_metrics(report).items():
            mlflow.log_metric(key.replace(" ", "_"), value)
        artifact_paths = []
        if report_path is not None:
            artifact_paths.append(report_path)
        if artifacts:
            artifact_paths.extend(artifacts)
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        mlflow.log_text(json.dumps(report, ensure_ascii=True, indent=2), "report.json")
        return run.info.run_id
