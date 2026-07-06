"""Tests for fine-tuning experiment config helpers."""

from pathlib import Path

from scripts.training.experiment_config import (
    artifact_paths,
    build_run_manifest,
    experiment_by_id,
    load_experiment_matrix,
)


def test_load_experiment_matrix_expands_defaults() -> None:
    specs = load_experiment_matrix()
    spec = experiment_by_id(specs, "lora-r8-alpha16")

    assert spec["model"]["base_model"] == "models/hf/unsloth--Qwen3.5-0.8B"
    assert spec["dataset"]["heldout_path"].endswith("heldout.jsonl")
    assert spec["lora"]["rank"] == 8
    assert spec["lora"]["alpha"] == 16


def test_build_run_manifest_records_artifacts_and_dataset_hashes(tmp_path: Path) -> None:
    specs = load_experiment_matrix()
    spec = experiment_by_id(specs, "lora-r4-alpha8")
    manifest = build_run_manifest(spec, runs_root=tmp_path, status="dry_run")

    assert manifest["type"] == "scratchpad_ft_experiment_manifest"
    assert manifest["experiment_id"] == "lora-r4-alpha8"
    assert manifest["artifact_paths"]["report"].endswith("report.html")
    assert "train_path" in manifest["dataset"]
    assert manifest["dataset"]["train_path"]["exists"] in {True, False}


def test_artifact_paths_are_stable(tmp_path: Path) -> None:
    paths = artifact_paths("example-run", runs_root=tmp_path)

    assert paths["run_dir"] == tmp_path / "example-run"
    assert paths["manifest"] == tmp_path / "example-run" / "manifest.json"
    assert paths["adapter_dir"] == tmp_path / "example-run" / "adapter"
