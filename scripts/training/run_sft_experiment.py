"""Plan and materialize Scratchpad SFT experiment artifacts.

The heavy training stack is intentionally optional. This CLI validates the
matrix, records deterministic run manifests, and gives the exact experiment
surface that a GPU job should execute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.experiment_config import (
    DEFAULT_MATRIX_PATH,
    DEFAULT_RUNS_ROOT,
    build_run_manifest,
    experiment_by_id,
    load_experiment_matrix,
    repo_relative,
    write_run_manifest,
)
from scripts.training.sft_trainer import run_sft_training


def summarize_spec(spec: dict[str, object]) -> dict[str, object]:
    """Return the compact fields users need before launching a run."""
    model = spec.get("model", {})
    dataset = spec.get("dataset", {})
    training = spec.get("training", {})
    lora = spec.get("lora", {})
    return {
        "experiment_id": spec["experiment_id"],
        "method": spec["method"],
        "expected_outcome": spec.get("expected_outcome"),
        "base_model": model.get("base_model") if isinstance(model, dict) else None,
        "train_path": dataset.get("train_path") if isinstance(dataset, dict) else None,
        "validation_path": dataset.get("validation_path") if isinstance(dataset, dict) else None,
        "heldout_path": dataset.get("heldout_path") if isinstance(dataset, dict) else None,
        "seed": training.get("seed") if isinstance(training, dict) else None,
        "max_steps": training.get("max_steps") if isinstance(training, dict) else None,
        "epochs": training.get("num_train_epochs") if isinstance(training, dict) else None,
        "lora": lora if isinstance(lora, dict) else {},
    }


def run(argv: list[str] | None = None) -> int:
    """Run the SFT experiment CLI."""
    parser = argparse.ArgumentParser(description="Validate and materialize Scratchpad SFT experiment runs.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--experiment-id", help="Only validate/materialize one experiment from the matrix.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Create run directories and manifest.json files. Without this, the command is read-only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the expanded run manifest without launching training.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the selected SFT experiment with the optional TRL/PEFT training stack.",
    )
    args = parser.parse_args(argv)

    specs = load_experiment_matrix(args.matrix)
    selected = [experiment_by_id(specs, args.experiment_id)] if args.experiment_id else specs
    if args.execute and not args.experiment_id:
        parser.error("--execute requires --experiment-id so expensive training is explicit.")

    summaries = [summarize_spec(spec) for spec in selected]
    print(json.dumps({"experiments": summaries}, ensure_ascii=True, indent=2))

    if args.execute:
        for spec in selected:
            result = run_sft_training(spec, runs_root=args.runs_root)
            print(json.dumps({"completed": result}, ensure_ascii=True, indent=2))
    elif args.write_manifest:
        for spec in selected:
            manifest_path = write_run_manifest(spec, runs_root=args.runs_root, status="planned")
            print(f"Wrote manifest: {repo_relative(manifest_path)}")
    elif args.dry_run:
        manifests = [build_run_manifest(spec, runs_root=args.runs_root, status="dry_run") for spec in selected]
        print(json.dumps({"manifests": manifests}, ensure_ascii=True, indent=2))

    if not args.dry_run and not args.write_manifest and not args.execute:
        print("Validated matrix. Add --dry-run to print manifests or --write-manifest to create run artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
