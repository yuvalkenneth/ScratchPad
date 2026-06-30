import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def load_env_file(path: Path | str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


@dataclass
class LLMConfig:
    provider: str = "llama_cpp"
    model_name: str = "qwen"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local"
    start_script: Optional[str] = None
    request_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_env_file()
        provider = os.getenv("LLM_PROVIDER", "llama_cpp")
        default_base_url = "http://127.0.0.1:8080/v1" if provider == "llama_cpp" else ""
        return cls(
            provider=provider,
            model_name=os.getenv("LLM_MODEL", "qwen"),
            base_url=os.getenv("LLM_BASE_URL", default_base_url),
            api_key=os.getenv("LLM_API_KEY", "local") or "local",
            start_script=os.getenv("LLM_START_SCRIPT"),
            request_settings=env_request_settings(),
        )


def env_request_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for env_name, key in [
        ("LLM_TEMPERATURE", "temperature"),
        ("LLM_TOP_P", "top_p"),
        ("LLM_MAX_TOKENS", "max_tokens"),
        ("LLM_REASONING_EFFORT", "reasoning_effort"),
        ("LLM_FREQUENCY_PENALTY", "frequency_penalty"),
        ("LLM_PRESENCE_PENALTY", "presence_penalty"),
    ]:
        value = os.getenv(env_name)
        if value is None or value == "":
            continue
        settings[key] = coerce_setting_value(value)
    return settings


def coerce_setting_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
