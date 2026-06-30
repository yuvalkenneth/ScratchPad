"""Tests for shared eval CLI/model configuration helpers."""

from argparse import ArgumentParser, Namespace

from app.llm.config import LLMConfig
from scripts.evals import utils


def test_add_model_config_args_parses_shared_flags() -> None:
    parser = ArgumentParser()
    utils.add_model_config_args(parser)
    utils.add_auto_start_arg(parser)

    args = parser.parse_args(
        [
            "--model-ref",
            "custom:llamacpp:qwen3.5:9b",
            "--provider",
            "custom:llamacpp",
            "--model",
            "override",
            "--base-url",
            "http://localhost:8080/v1",
            "--api-key",
            "local",
            "--start-script",
            "/tmp/start.sh",
            "--no-auto-start",
        ]
    )

    assert args.model_ref == "custom:llamacpp:qwen3.5:9b"
    assert args.provider == "custom:llamacpp"
    assert args.model == "override"
    assert args.base_url == "http://localhost:8080/v1"
    assert args.api_key == "local"
    assert args.start_script == "/tmp/start.sh"
    assert args.no_auto_start


def test_eval_config_from_args_resolves_catalog_model_ref(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_MODEL_REF", raising=False)
    args = Namespace(
        model_ref="custom:llamacpp:qwen3.5:9b",
        provider=None,
        model=None,
        base_url=None,
        api_key=None,
        start_script=None,
    )

    config = utils.eval_config_from_args(args)

    assert config.provider == "custom:llamacpp"
    assert config.model_name == "qwen3.5:9b"
    assert config.request_settings["max_tokens"] == 4096


def test_prepare_eval_provider_only_starts_llama_cpp_when_enabled(monkeypatch) -> None:
    calls: list[LLMConfig] = []
    monkeypatch.setattr(utils, "ensure_provider_ready", lambda config: calls.append(config) or config)
    config = LLMConfig(provider="custom:llamacpp", model_name="qwen")

    assert utils.prepare_eval_provider(config, auto_start=False) is config
    assert calls == []

    assert utils.prepare_eval_provider(config, auto_start=True) is config
    assert calls == [config]
