"""Tests for model catalog loading, overrides, and env-backed values."""

import json
from pathlib import Path

import pytest

from app.llm.catalog import config_from_model_ref, get_custom_provider, load_model_catalog


def write_catalog(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_provider_model_catalog_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_catalog(
        path,
        {
            "default": "custom:local:qwen",
            "providers": {
                "local": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "defaults": {"temperature": 0.7, "max_tokens": 4096},
                    "models": {
                        "qwen": {
                            "settings": {"temperature": 0.4},
                        }
                    },
                }
            },
        },
    )

    catalog = load_model_catalog((path,))
    config = config_from_model_ref(None, paths=(path,))

    assert catalog.default == "custom:local:qwen"
    assert "local" in catalog.providers
    assert config.provider == "custom:local"
    assert config.model_name == "qwen"
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.request_settings == {"temperature": 0.4, "max_tokens": 4096}


def test_local_catalog_file_deep_merges_committed_defaults(tmp_path: Path) -> None:
    committed = tmp_path / "models.json"
    local = tmp_path / "models.local.json"
    write_catalog(
        committed,
        {
            "default": "custom:llamacpp:qwen",
            "providers": {
                "llamacpp": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "defaults": {"temperature": 0.7, "max_tokens": 4096},
                    "models": {"qwen": {"settings": {"temperature": 0.5}}},
                }
            },
        },
    )
    write_catalog(
        local,
        {
            "providers": {
                "llamacpp": {
                    "models": {
                        "qwen": {
                            "start_script": "/tmp/start-qwen.sh",
                            "settings": {"top_p": 0.8},
                        }
                    }
                }
            }
        },
    )

    config = config_from_model_ref("custom:llamacpp:qwen", paths=(committed, local))

    assert config.provider == "custom:llamacpp"
    assert config.model_name == "qwen"
    assert config.start_script == "/tmp/start-qwen.sh"
    assert config.request_settings == {
        "temperature": 0.5,
        "max_tokens": 4096,
        "top_p": 0.8,
    }


def test_provider_uses_env_backed_api_key(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "models.json"
    write_catalog(
        path,
        {
            "providers": {
                "gemini": {
                    "base_url": "https://gemini.example.test/v1",
                    "key_env": "GEMINI_API_KEY",
                    "models": {"gemini-3.5-flash": {}},
                }
            },
        },
    )
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    config = config_from_model_ref("custom:gemini:gemini-3.5-flash", paths=(path,))

    assert config.provider == "custom:gemini"
    assert config.model_name == "gemini-3.5-flash"
    assert config.base_url == "https://gemini.example.test/v1"
    assert config.api_key == "secret"


def test_model_ref_allows_model_names_with_colons(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_catalog(
        path,
        {
            "providers": {
                "llamacpp": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "models": {"qwen3.5:9b": {}},
                }
            },
        },
    )

    config = config_from_model_ref("custom:llamacpp:qwen3.5:9b", paths=(path,))

    assert config.model_name == "qwen3.5:9b"


def test_config_from_model_ref_allows_explicit_overrides(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_catalog(
        path,
        {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "models": {"gpt-5.4": {"settings": {"reasoning_effort": "medium"}}},
                }
            },
        },
    )
    config = config_from_model_ref(
        "custom:openai:gpt-5.4",
        model="override-model",
        base_url="https://override.example.test/v1",
        request_settings={"reasoning_effort": "high"},
        paths=(path,),
    )

    assert config.provider == "custom:openai"
    assert config.model_name == "override-model"
    assert config.base_url == "https://override.example.test/v1"
    assert config.request_settings["reasoning_effort"] == "high"


def test_unknown_provider_error_lists_available_providers(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_catalog(
        path,
        {
            "providers": {
                "known": {
                    "base_url": "http://localhost:8080/v1",
                    "models": {"qwen": {}},
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Available providers: known"):
        get_custom_provider("missing", (path,))
