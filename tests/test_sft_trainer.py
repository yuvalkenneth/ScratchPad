"""Tests for optional SFT trainer helpers that do not import heavy ML deps."""

import json
from pathlib import Path

import pytest

from scripts.training.sft_trainer import (
    build_training_arguments,
    load_jsonl,
    rows_to_text_examples,
    training_arguments_kwargs,
    write_trainer_log,
)


def test_load_jsonl_and_rows_to_text_examples(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"id": "case-1", "text": "hello", "metadata": {"split": "train"}}) + "\n",
        encoding="utf-8",
    )

    rows = load_jsonl(path)
    examples = rows_to_text_examples(rows, source_path=path)

    assert examples == [{"text": "hello", "id": "case-1", "metadata": {"split": "train"}}]


def test_rows_to_text_examples_requires_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--output-format text"):
        rows_to_text_examples([{"id": "case-1", "messages": []}], source_path=tmp_path / "train.jsonl")


def test_training_arguments_kwargs_maps_matrix_fields(tmp_path: Path) -> None:
    kwargs = training_arguments_kwargs(
        {
            "training": {
                "seed": 13,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 0.0002,
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine",
                "max_steps": 120,
                "num_train_epochs": 1,
                "logging_steps": 5,
                "eval_steps": 20,
                "save_steps": 40,
            }
        },
        output_dir=tmp_path / "checkpoints",
    )

    assert kwargs["output_dir"].endswith("checkpoints")
    assert kwargs["learning_rate"] == 0.0002
    assert kwargs["max_steps"] == 120
    assert kwargs["report_to"] == []


def test_build_training_arguments_prefers_sft_config() -> None:
    class FakeSFTConfig:
        def __init__(self, output_dir: str, max_length: int, dataset_text_field: str, eval_strategy: str) -> None:
            self.output_dir = output_dir
            self.max_length = max_length
            self.dataset_text_field = dataset_text_field
            self.eval_strategy = eval_strategy

    args = build_training_arguments(
        {"SFTConfig": FakeSFTConfig},
        {"output_dir": "out", "unused": "ignored"},
        max_seq_length=2048,
    )

    assert args.output_dir == "out"
    assert args.max_length == 2048
    assert args.dataset_text_field == "text"
    assert args.eval_strategy == "steps"


def test_write_trainer_log_uses_report_shape(tmp_path: Path) -> None:
    path = tmp_path / "trainer_log.json"

    write_trainer_log(
        path,
        log_history=[{"step": 1, "loss": 1.2}],
        metrics={"train_loss": 1.0, "wall_time_seconds": 10.0},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["type"] == "scratchpad_sft_trainer_log"
    assert payload["log_history"] == [{"step": 1, "loss": 1.2}]
    assert payload["metrics"]["wall_time_seconds"] == 10.0
