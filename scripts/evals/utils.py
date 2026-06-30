"""Shared helpers for model-backed eval CLIs."""

from __future__ import annotations

import argparse
import os

from app.llm.catalog import config_from_model_ref
from app.llm.config import LLMConfig
from app.llm.runtime import ensure_provider_ready, is_llama_cpp_provider


MODEL_REF_HELP = "Model ref from config/models.json, e.g. custom:llamacpp:qwen3.5:9b."


def add_model_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-ref", help=MODEL_REF_HELP)
    parser.add_argument(
        "--provider",
        help="Provider for the model under test. Defaults to EVAL_PROVIDER or LLM_PROVIDER.",
    )
    parser.add_argument(
        "--model",
        help="Model id for the model under test. Defaults to EVAL_MODEL or LLM_MODEL.",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for the model under test.")
    parser.add_argument("--api-key", help="API key for the model under test. Prefer env vars for real keys.")
    parser.add_argument("--start-script", help="llama.cpp start script for the model under test.")


def add_auto_start_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not auto-start llama.cpp even when provider is llama_cpp.",
    )


def eval_config_from_args(args: argparse.Namespace) -> LLMConfig:
    """Resolve the model-under-test config from CLI args, env vars, or model refs."""
    model_ref = effective_model_ref(args)
    model_arg = str(args.model or "")
    if not model_ref and model_arg.startswith("custom:"):
        model_ref = model_arg
    if model_ref:
        return config_from_model_ref(
            model_ref,
            provider=args.provider or os.getenv("EVAL_PROVIDER"),
            model=None if model_arg.startswith("custom:") else args.model or os.getenv("EVAL_MODEL"),
            base_url=args.base_url or os.getenv("EVAL_BASE_URL"),
            api_key=args.api_key or os.getenv("EVAL_API_KEY"),
            start_script=args.start_script or os.getenv("EVAL_START_SCRIPT"),
        )

    default = LLMConfig.from_env()
    provider = args.provider or os.getenv("EVAL_PROVIDER") or default.provider
    explicit_base_url = args.base_url or os.getenv("EVAL_BASE_URL")
    return LLMConfig(
        provider=provider,
        model_name=args.model or os.getenv("EVAL_MODEL") or default.model_name,
        base_url=explicit_base_url or (default.base_url if is_llama_cpp_provider(provider) else ""),
        api_key=args.api_key or os.getenv("EVAL_API_KEY") or default.api_key or "local",
        start_script=args.start_script or os.getenv("EVAL_START_SCRIPT") or default.start_script,
        request_settings=dict(default.request_settings),
    )


def prepare_eval_provider(config: LLMConfig, *, auto_start: bool) -> LLMConfig:
    """Start a local llama.cpp provider when the eval explicitly allows it."""
    if auto_start and is_llama_cpp_provider(config.provider):
        return ensure_provider_ready(config)
    return config


def effective_model_ref(args: argparse.Namespace) -> str | None:
    return args.model_ref or os.getenv("EVAL_MODEL_REF")
