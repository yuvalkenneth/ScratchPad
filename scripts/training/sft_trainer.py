"""Optional TRL/Unsloth SFT execution for Scratchpad fine-tuning labs."""

from __future__ import annotations

import json
import inspect
import time
from pathlib import Path
from typing import Any

from scripts.training.experiment_config import artifact_paths, build_run_manifest, repo_relative, resolve_repo_path


OPTIONAL_TRAINING_DEPS = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
)
TRAINING_INSTALL_HINT = (
    "Install the optional training stack in the GPU environment, for example:\n"
    "uv run --with torch --with transformers --with datasets --with peft --with trl "
    "--with accelerate python scripts/training/run_sft_experiment.py --execute --experiment-id lora-r8-alpha16"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from a training split."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            rows.append(row)
    return rows


def rows_to_text_examples(rows: list[dict[str, Any]], *, source_path: Path) -> list[dict[str, Any]]:
    """Convert exported rows into the text-column shape expected by SFTTrainer."""
    examples = []
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{source_path}:{index + 1} is missing a non-empty text field. "
                "Export with scripts/training/export_sft_tool_choice.py --output-format text."
            )
        examples.append(
            {
                "text": text,
                "id": row.get("id"),
                "metadata": row.get("metadata", {}),
            }
        )
    return examples


def training_arguments_kwargs(spec: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Map matrix training fields into Transformers TrainingArguments kwargs."""
    training = spec["training"]
    return {
        "output_dir": str(output_dir),
        "seed": int(training["seed"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "lr_scheduler_type": str(training["lr_scheduler_type"]),
        "max_steps": int(training["max_steps"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "logging_steps": int(training["logging_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "save_steps": int(training["save_steps"]),
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
    }


def write_trainer_log(path: Path, *, log_history: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    """Persist trainer logs in the normalized shape consumed by reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "scratchpad_sft_trainer_log",
        "log_history": log_history,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def import_training_stack() -> dict[str, Any]:
    """Import optional training dependencies only when an experiment executes."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
        try:
            from trl import SFTConfig
        except ImportError:
            SFTConfig = None
    except ImportError as exc:
        raise RuntimeError(TRAINING_INSTALL_HINT) from exc
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        FastLanguageModel = None
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainingArguments": TrainingArguments,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "FastLanguageModel": FastLanguageModel,
    }


def kwargs_for_callable(callable_: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to parameters accepted by a callable when introspection works."""
    try:
        parameters = inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def build_training_arguments(stack: dict[str, Any], kwargs: dict[str, Any], *, max_seq_length: int) -> Any:
    """Build TRL SFTConfig or Transformers TrainingArguments across versions."""
    sft_config_cls = stack.get("SFTConfig")
    if sft_config_cls is not None:
        config_kwargs = {
            **kwargs,
            "dataset_text_field": "text",
            "max_length": max_seq_length,
            "max_seq_length": max_seq_length,
            "eval_strategy": "steps",
            "evaluation_strategy": "steps",
        }
        return sft_config_cls(**kwargs_for_callable(sft_config_cls, config_kwargs))
    training_arguments_cls = stack["TrainingArguments"]
    try:
        return training_arguments_cls(**kwargs, eval_strategy="steps")
    except TypeError:
        return training_arguments_cls(**kwargs, evaluation_strategy="steps")


def load_model_and_tokenizer(spec: dict[str, Any], stack: dict[str, Any]) -> tuple[Any, Any]:
    """Load the base model, tokenizer, and attach LoRA adapters."""
    model_name = spec["model"]["base_model"]
    training = spec["training"]
    lora = spec["lora"]
    quantization = lora.get("quantization")
    fast_language_model = stack.get("FastLanguageModel")

    if fast_language_model is not None:
        model, tokenizer = fast_language_model.from_pretrained(
            model_name=model_name,
            max_seq_length=int(training["max_seq_length"]),
            load_in_4bit=bool(quantization),
        )
        model = fast_language_model.get_peft_model(
            model,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            random_state=int(training["seed"]),
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    tokenizer = stack["AutoTokenizer"].from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
    }
    if quantization:
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=stack["torch"].bfloat16,
        )
    model = stack["AutoModelForCausalLM"].from_pretrained(model_name, **model_kwargs)
    if quantization:
        model = stack["prepare_model_for_kbit_training"](model)

    peft_config = stack["LoraConfig"](
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = stack["get_peft_model"](model, peft_config)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def build_sft_trainer(
    *,
    spec: dict[str, Any],
    run_dir: Path,
    stack: dict[str, Any],
) -> Any:
    """Build a TRL SFTTrainer for a concrete experiment spec."""
    dataset = spec["dataset"]
    train_path = resolve_repo_path(dataset["train_path"])
    validation_path = resolve_repo_path(dataset["validation_path"])
    train_examples = rows_to_text_examples(load_jsonl(train_path), source_path=train_path)
    eval_examples = rows_to_text_examples(
        load_jsonl(validation_path),
        source_path=validation_path,
    )
    train_dataset = stack["Dataset"].from_list(train_examples)
    eval_dataset = stack["Dataset"].from_list(eval_examples)
    model, tokenizer = load_model_and_tokenizer(spec, stack)
    args = build_training_arguments(
        stack,
        training_arguments_kwargs(spec, output_dir=run_dir / "checkpoints"),
        max_seq_length=int(spec["training"]["max_seq_length"]),
    )
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_cls = stack["SFTTrainer"]
    trainer_parameters = inspect.signature(trainer_cls).parameters
    if "dataset_text_field" in trainer_parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_parameters:
        trainer_kwargs["max_seq_length"] = int(spec["training"]["max_seq_length"])
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return trainer_cls(**trainer_kwargs)


def run_sft_training(spec: dict[str, Any], *, runs_root: Path) -> dict[str, Any]:
    """Execute one SFT run and write manifest, trainer log, and adapter artifacts."""
    if spec["method"] != "sft":
        raise ValueError(f"Only method='sft' can be trained by this runner, got {spec['method']!r}.")
    paths = artifact_paths(spec["experiment_id"], runs_root=runs_root)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    write_manifest(paths["manifest"], build_run_manifest(spec, runs_root=runs_root, status="running"))

    started_at = time.perf_counter()
    stack = import_training_stack()
    trainer = build_sft_trainer(spec=spec, run_dir=paths["run_dir"], stack=stack)
    train_result = trainer.train()
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(paths["adapter_dir"]))
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    metrics["wall_time_seconds"] = elapsed_seconds
    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    write_trainer_log(paths["trainer_log"], log_history=log_history, metrics=metrics)
    write_manifest(paths["manifest"], build_run_manifest(spec, runs_root=runs_root, status="completed"))
    return {
        "manifest": repo_relative(paths["manifest"]),
        "trainer_log": repo_relative(paths["trainer_log"]),
        "adapter_dir": repo_relative(paths["adapter_dir"]),
        "metrics": metrics,
    }
