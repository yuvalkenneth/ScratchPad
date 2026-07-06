"""Experiment configuration helpers for fine-tuning learning runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.observability.experiment_tracking import current_git_sha


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = REPO_ROOT / "experiments" / "ft_learning" / "tool_choice_lora_qlora_matrix.json"
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "ft_learning" / "runs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str | None:
    """Return the SHA-256 digest for a file, or None when it is absent."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    """Render a path relative to the repo when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a possibly relative path from the repo root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_experiment_matrix(path: Path = DEFAULT_MATRIX_PATH) -> list[dict[str, Any]]:
    """Load the matrix file and expand defaults into concrete experiment specs."""
    matrix = load_json_object(path)
    defaults = matrix.get("defaults", {})
    experiments = matrix.get("experiments", [])
    if not isinstance(defaults, dict):
        raise ValueError("Experiment matrix defaults must be an object.")
    if not isinstance(experiments, list):
        raise ValueError("Experiment matrix experiments must be a list.")
    expanded = []
    seen_ids: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment matrix entry must be an object.")
        spec = deep_merge(defaults, experiment)
        validate_experiment_spec(spec)
        experiment_id = spec["experiment_id"]
        if experiment_id in seen_ids:
            raise ValueError(f"Duplicate experiment_id in matrix: {experiment_id}")
        seen_ids.add(experiment_id)
        expanded.append(spec)
    return expanded


def validate_experiment_spec(spec: dict[str, Any]) -> None:
    """Validate the minimum schema needed to run or report an experiment."""
    for key in ("experiment_id", "method", "model", "dataset", "training", "lora"):
        if key not in spec:
            raise ValueError(f"Experiment spec is missing required field: {key}")
    if not str(spec["experiment_id"]).strip():
        raise ValueError("experiment_id cannot be empty.")
    if spec["method"] not in {"base_eval", "sft"}:
        raise ValueError(f"Unsupported method: {spec['method']}")
    for key in ("model", "dataset", "training", "lora"):
        if not isinstance(spec[key], dict):
            raise ValueError(f"{key} must be an object.")
    if not str(spec["model"].get("base_model") or "").strip():
        raise ValueError("model.base_model is required.")
    for split_key in ("train_path", "validation_path", "heldout_path"):
        if not str(spec["dataset"].get(split_key) or "").strip():
            raise ValueError(f"dataset.{split_key} is required.")
    if int(spec["training"].get("seed", -1)) < 0:
        raise ValueError("training.seed must be a non-negative integer.")
    if spec["method"] == "sft" and not spec["lora"].get("enabled", False):
        raise ValueError("SFT experiments in this matrix must enable LoRA or QLoRA.")


def experiment_by_id(specs: list[dict[str, Any]], experiment_id: str) -> dict[str, Any]:
    """Find one expanded experiment spec by id."""
    for spec in specs:
        if spec["experiment_id"] == experiment_id:
            return spec
    raise ValueError(f"Unknown experiment_id: {experiment_id}")


def artifact_paths(
    experiment_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> dict[str, Path]:
    """Return the canonical artifact paths for one experiment run."""
    run_dir = runs_root / experiment_id
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "trainer_log": run_dir / "trainer_log.json",
        "eval_metrics": run_dir / "eval_metrics.json",
        "scorecard": run_dir / "scorecard.json",
        "report": run_dir / "report.html",
        "plots_dir": run_dir / "plots",
        "adapter_dir": run_dir / "adapter",
    }


def dataset_manifest(dataset: dict[str, Any]) -> dict[str, Any]:
    """Build split path and hash metadata for an experiment dataset."""
    manifest: dict[str, Any] = {}
    for key in ("cases_path", "train_path", "validation_path", "heldout_path"):
        if key not in dataset:
            continue
        path = resolve_repo_path(dataset[key])
        manifest[key] = {
            "path": repo_relative(path),
            "sha256": file_sha256(path),
            "exists": path.exists(),
        }
    return manifest


def build_run_manifest(
    spec: dict[str, Any],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    status: str = "planned",
) -> dict[str, Any]:
    """Create the persisted manifest for one fine-tuning experiment."""
    paths = artifact_paths(spec["experiment_id"], runs_root=runs_root)
    manifest = {
        "type": "scratchpad_ft_experiment_manifest",
        "experiment_id": spec["experiment_id"],
        "status": status,
        "git_sha": current_git_sha(),
        "run_dir": repo_relative(paths["run_dir"]),
        "artifact_paths": {key: repo_relative(path) for key, path in paths.items()},
        "spec": spec,
        "dataset": dataset_manifest(spec["dataset"]),
    }
    return manifest


def write_run_manifest(
    spec: dict[str, Any],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    status: str = "planned",
) -> Path:
    """Write one experiment manifest and return its path."""
    paths = artifact_paths(spec["experiment_id"], runs_root=runs_root)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(build_run_manifest(spec, runs_root=runs_root, status=status), ensure_ascii=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return paths["manifest"]
