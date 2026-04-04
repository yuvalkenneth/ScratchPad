import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "llama_cpp"
    model_name: str = "qwen"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local"
    start_script: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "llama_cpp"),
            model_name=os.getenv("LLM_MODEL", "qwen"),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("LLM_API_KEY", "local") or "local",
            start_script=os.getenv("LLM_START_SCRIPT"),
        )
