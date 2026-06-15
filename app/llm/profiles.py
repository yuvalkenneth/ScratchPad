from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm.config import LLMConfig, load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATHS = (
    REPO_ROOT / "config" / "models.json",
    REPO_ROOT / "config" / "models.local.json",
)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    model_name: str
    base_url: str = ""
    api_key: str = "local"
    start_script: str | None = None
    server_mode: str = "user_managed"
    base_url_env: str | None = None
    api_key_env: str | None = None
    start_script_env: str | None = None
    description: str = ""

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "ModelProfile":
        model_name = str(data.get("model_name") or data.get("model") or "").strip()
        provider = str(data.get("provider") or "").strip()
        if not provider:
            raise ValueError(f"Model profile {name!r} is missing provider.")
        if not model_name:
            raise ValueError(f"Model profile {name!r} is missing model/model_name.")
        return cls(
            name=name,
            provider=provider,
            model_name=model_name,
            base_url=str(data.get("base_url") or "").strip(),
            api_key=str(data.get("api_key") or "local").strip() or "local",
            start_script=_optional_str(data.get("start_script")),
            server_mode=str(data.get("server_mode") or "user_managed").strip(),
            base_url_env=_optional_str(data.get("base_url_env")),
            api_key_env=_optional_str(data.get("api_key_env")),
            start_script_env=_optional_str(data.get("start_script_env")),
            description=str(data.get("description") or "").strip(),
        )

    def to_config(self) -> LLMConfig:
        load_env_file()
        return LLMConfig(
            provider=self.provider,
            model_name=self.model_name,
            base_url=_env_or_value(self.base_url_env, self.base_url),
            api_key=_env_or_value(self.api_key_env, self.api_key) or "local",
            start_script=_env_or_value(self.start_script_env, self.start_script),
        )


def load_model_profiles(paths: tuple[Path, ...] = DEFAULT_PROFILE_PATHS) -> dict[str, ModelProfile]:
    raw_profiles: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", payload)
        if not isinstance(profiles, dict):
            raise ValueError(f"Model profile file must contain a profiles object: {path}")
        for name, data in profiles.items():
            if not isinstance(data, dict):
                raise ValueError(f"Model profile {name!r} in {path} must be an object.")
            raw_profiles[str(name)] = {**raw_profiles.get(str(name), {}), **data}
    return {
        name: ModelProfile.from_mapping(name, data)
        for name, data in sorted(raw_profiles.items())
    }


def get_model_profile(name: str, paths: tuple[Path, ...] = DEFAULT_PROFILE_PATHS) -> ModelProfile:
    profiles = load_model_profiles(paths)
    try:
        return profiles[name]
    except KeyError as exc:
        available = ", ".join(profiles) or "none"
        raise ValueError(f"Unknown model profile {name!r}. Available profiles: {available}") from exc


def config_from_profile(
    profile_name: str | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    start_script: str | None = None,
    paths: tuple[Path, ...] = DEFAULT_PROFILE_PATHS,
) -> LLMConfig:
    base = get_model_profile(profile_name, paths).to_config() if profile_name else LLMConfig.from_env()
    return LLMConfig(
        provider=provider or base.provider,
        model_name=model or base.model_name,
        base_url=base_url if base_url is not None else base.base_url,
        api_key=api_key if api_key is not None else base.api_key,
        start_script=start_script if start_script is not None else base.start_script,
    )


def resolved_config_metadata(config: LLMConfig, *, profile: str | None = None) -> dict[str, Any]:
    return {
        "profile": profile,
        "provider": config.provider,
        "model": config.model_name,
        "base_url": config.base_url,
        "has_api_key": bool(config.api_key and config.api_key != "local"),
        "has_start_script": bool(config.start_script),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env_or_value(env_name: str | None, value: str | None) -> str:
    if env_name:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    return value or ""
