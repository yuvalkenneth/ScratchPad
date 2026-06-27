"""Tests for model profile loading, overrides, and env-backed values."""

import json
from pathlib import Path

import pytest

from app.llm.profiles import config_from_profile, get_model_profile, load_model_profiles


def write_profiles(path: Path, profiles: dict[str, object]) -> None:
    path.write_text(json.dumps({"profiles": profiles}), encoding="utf-8")


def test_loads_profile_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_profiles(
        path,
        {
            "local-small": {
                "provider": "llama_cpp",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8080/v1",
            }
        },
    )

    profiles = load_model_profiles((path,))

    assert profiles["local-small"].provider == "llama_cpp"
    assert profiles["local-small"].model_name == "qwen"


def test_local_profile_file_overrides_committed_defaults(tmp_path: Path) -> None:
    committed = tmp_path / "models.json"
    local = tmp_path / "models.local.json"
    write_profiles(
        committed,
        {
            "qwen-local": {
                "provider": "llama_cpp",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8080/v1",
            }
        },
    )
    write_profiles(
        local,
        {
            "qwen-local": {
                "model": "qwen4b",
                "start_script": "/tmp/start-qwen.sh",
            }
        },
    )

    profile = get_model_profile("qwen-local", (committed, local))

    assert profile.provider == "llama_cpp"
    assert profile.model_name == "qwen4b"
    assert profile.start_script == "/tmp/start-qwen.sh"


def test_profile_uses_env_backed_url_and_api_key(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "models.json"
    write_profiles(
        path,
        {
            "gemini-flash": {
                "provider": "gemini",
                "model": "gemini-3.5-flash",
                "base_url_env": "GEMINI_BASE_URL",
                "api_key_env": "GEMINI_API_KEY",
            }
        },
    )
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gemini.example.test/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    config = get_model_profile("gemini-flash", (path,)).to_config()

    assert config.provider == "gemini"
    assert config.model_name == "gemini-3.5-flash"
    assert config.base_url == "https://gemini.example.test/v1"
    assert config.api_key == "secret"


def test_config_from_profile_allows_explicit_overrides(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_profiles(
        path,
        {
            "base": {
                "provider": "llama_cpp",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8080/v1",
            }
        },
    )
    config = config_from_profile(
        "base",
        model="override-model",
        base_url="https://override.example.test/v1",
        paths=(path,),
    )

    assert config.provider == "llama_cpp"
    assert config.model_name == "override-model"
    assert config.base_url == "https://override.example.test/v1"


def test_unknown_profile_error_lists_available_profiles(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_profiles(path, {"known": {"provider": "llama_cpp", "model": "qwen"}})

    with pytest.raises(ValueError, match="Available profiles: known"):
        get_model_profile("missing", (path,))
