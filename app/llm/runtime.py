import os
import subprocess

from app.llm.config import LLMConfig


def ensure_provider_ready(config: LLMConfig) -> LLMConfig:
    if not is_llama_cpp_provider(config.provider):
        return config

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "scripts",
        "ensure_local_server.sh",
    )

    env = os.environ.copy()
    if config.start_script:
        env["LLM_START_SCRIPT"] = config.start_script
    env["LLM_BASE_URL"] = config.base_url

    subprocess.run([script_path], check=True, env=env)
    return config


def is_llama_cpp_provider(provider: str) -> bool:
    normalized = provider.strip().lower().replace("_", "-")
    return normalized in {"llama-cpp", "llamacpp", "custom:llamacpp", "custom:llama-cpp"}
