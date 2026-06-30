from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.llm.config import LLMConfig, load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATHS = (
    REPO_ROOT / "config" / "models.json",
    REPO_ROOT / "config" / "models.local.json",
)


@dataclass(frozen=True)
class CatalogModel:
    name: str
    display_name: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    start_script: str | None = None
    description: str = ""

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "CatalogModel":
        return cls(
            name=name,
            display_name=str(data.get("display_name") or "").strip(),
            settings=dict(data.get("settings") or {}),
            start_script=_optional_str(data.get("start_script")),
            description=str(data.get("description") or "").strip(),
        )


@dataclass(frozen=True)
class CustomProvider:
    name: str
    base_url: str
    api_key: str = "local"
    key_env: str | None = None
    start_script: str | None = None
    start_script_env: str | None = None
    server_mode: str = "user_managed"
    defaults: dict[str, Any] = field(default_factory=dict)
    models: dict[str, CatalogModel] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "CustomProvider":
        models = data.get("models") or {}
        if not isinstance(models, dict):
            raise ValueError(f"Provider {name!r} models must be an object.")
        return cls(
            name=name,
            base_url=str(data.get("base_url") or "").strip(),
            api_key=str(data.get("api_key") or "local").strip() or "local",
            key_env=_optional_str(data.get("key_env") or data.get("api_key_env")),
            start_script=_optional_str(data.get("start_script")),
            start_script_env=_optional_str(data.get("start_script_env")),
            server_mode=str(data.get("server_mode") or "user_managed").strip(),
            defaults=dict(data.get("defaults") or {}),
            models={
                str(model_name): CatalogModel.from_mapping(str(model_name), model_data)
                for model_name, model_data in models.items()
                if isinstance(model_data, dict)
            },
            description=str(data.get("description") or "").strip(),
        )


@dataclass(frozen=True)
class ModelCatalog:
    default: str | None
    providers: dict[str, CustomProvider]


@dataclass(frozen=True)
class ModelRef:
    namespace: str
    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> "ModelRef":
        parts = value.split(":", 2)
        if len(parts) != 3 or parts[0] != "custom" or not parts[1] or not parts[2]:
            raise ValueError(
                "Model ref must use the form custom:<provider>:<model>, "
                f"got {value!r}."
            )
        return cls(namespace=parts[0], provider=parts[1], model=parts[2])

    def to_string(self) -> str:
        return f"{self.namespace}:{self.provider}:{self.model}"


def load_model_catalog(paths: tuple[Path, ...] = DEFAULT_CATALOG_PATHS) -> ModelCatalog:
    raw: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Model catalog file must contain an object: {path}")
        raw = deep_merge(raw, payload)

    providers_payload = raw.get("providers") or {}
    default = _optional_str(raw.get("default"))

    if not isinstance(providers_payload, dict):
        raise ValueError("Model catalog must contain a providers object.")

    providers = {
        str(name): CustomProvider.from_mapping(str(name), data)
        for name, data in providers_payload.items()
        if isinstance(data, dict)
    }
    return ModelCatalog(default=default, providers=providers)


def config_from_model_ref(
    model_ref: str | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    start_script: str | None = None,
    request_settings: dict[str, Any] | None = None,
    paths: tuple[Path, ...] = DEFAULT_CATALOG_PATHS,
) -> LLMConfig:
    load_env_file()
    catalog = load_model_catalog(paths)
    ref_value = model_ref or os.getenv("LLM_MODEL_REF") or catalog.default
    if not ref_value:
        base = LLMConfig.from_env()
        return override_config(
            base,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            start_script=start_script,
            request_settings=request_settings,
        )

    ref = ModelRef.parse(ref_value)
    provider_name = provider_ref_name(provider) if provider else ref.provider
    provider_config = get_custom_provider(provider_name, paths)
    model_name = model or ref.model
    model_config = provider_config.models.get(model_name)

    settings = dict(provider_config.defaults)
    if model_config:
        settings.update(model_config.settings)
    if request_settings:
        settings.update(request_settings)

    provider_start_script = _env_or_value(
        provider_config.start_script_env,
        provider_config.start_script,
    )
    model_start_script = model_config.start_script if model_config else None
    return LLMConfig(
        provider=f"custom:{provider_name}",
        model_name=model_name,
        base_url=base_url if base_url is not None else provider_config.base_url,
        api_key=api_key if api_key is not None else resolve_catalog_api_key(provider_config),
        start_script=start_script if start_script is not None else model_start_script or provider_start_script,
        request_settings=settings,
    )


def get_custom_provider(name: str, paths: tuple[Path, ...] = DEFAULT_CATALOG_PATHS) -> CustomProvider:
    catalog = load_model_catalog(paths)
    try:
        return catalog.providers[name]
    except KeyError as exc:
        available = ", ".join(catalog.providers) or "none"
        raise ValueError(f"Unknown provider {name!r}. Available providers: {available}") from exc


def resolved_config_metadata(config: LLMConfig, *, model_ref: str | None = None) -> dict[str, Any]:
    return {
        "model_ref": model_ref,
        "provider": config.provider,
        "model": config.model_name,
        "base_url": config.base_url,
        "has_api_key": bool(config.api_key and config.api_key != "local"),
        "has_start_script": bool(config.start_script),
        "request_settings": dict(config.request_settings),
    }


def override_config(
    config: LLMConfig,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    start_script: str | None = None,
    request_settings: dict[str, Any] | None = None,
) -> LLMConfig:
    settings = dict(config.request_settings)
    if request_settings:
        settings.update(request_settings)
    return LLMConfig(
        provider=provider or config.provider,
        model_name=model or config.model_name,
        base_url=base_url if base_url is not None else config.base_url,
        api_key=api_key if api_key is not None else config.api_key,
        start_script=start_script if start_script is not None else config.start_script,
        request_settings=settings,
    )


def resolve_catalog_api_key(provider: CustomProvider) -> str:
    if provider.key_env:
        env_value = os.getenv(provider.key_env)
        if env_value:
            return env_value
    return provider.api_key or "local"


def provider_ref_name(provider: str) -> str:
    if provider.startswith("custom:"):
        return provider.split(":", 1)[1]
    return provider


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
