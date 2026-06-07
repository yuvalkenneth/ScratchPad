from __future__ import annotations

import os
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI, RateLimitError

from app.llm.config import LLMConfig


PROVIDER_TO_API_KEY_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "llama_cpp": "LLAMA_CPP_API_KEY",
}

PROVIDERS_TO_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "llama_cpp": "http://127.0.0.1:8080/v1",
}


def resolve_api_key(config: LLMConfig | Any) -> str:
    provider = getattr(config, "provider", "llama_cpp")
    api_key_env_var = PROVIDER_TO_API_KEY_ENV_VAR.get(provider)
    api_key = os.getenv(api_key_env_var) if api_key_env_var else None
    if api_key:
        return api_key
    config_api_key = getattr(config, "api_key", None)
    if config_api_key and config_api_key != "local":
        return config_api_key
    return os.getenv("LLM_API_KEY") or config_api_key or "local"


def resolve_base_url(config: LLMConfig | Any) -> str:
    base_url = getattr(config, "base_url", None) or getattr(config, "api_url", None)
    if base_url:
        return base_url
    env_base_url = os.getenv("LLM_BASE_URL")
    if env_base_url:
        return env_base_url
    provider = getattr(config, "provider", "llama_cpp")
    if provider == "gemini":
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        if gemini_base_url:
            return gemini_base_url
    default = PROVIDERS_TO_BASE_URL.get(provider)
    if default is None:
        raise ValueError(f"Base URL for provider {provider} not found.")
    return default


def make_sync_client(config: LLMConfig | Any) -> OpenAI:
    return OpenAI(api_key=resolve_api_key(config), base_url=resolve_base_url(config))


def make_async_client(config: LLMConfig | Any) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=resolve_api_key(config), base_url=resolve_base_url(config))


def complete_text(
    config: LLMConfig | Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    top_p: float | None = None,
) -> str:
    request_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        request_kwargs["top_p"] = top_p

    response = create_completion_with_retries(config, request_kwargs)
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def create_completion_with_retries(config: LLMConfig | Any, request_kwargs: dict[str, Any]) -> Any:
    attempts = int(os.getenv("LLM_RATE_LIMIT_RETRIES", "3"))
    delay_seconds = float(os.getenv("LLM_RATE_LIMIT_DELAY_SECONDS", "20"))
    client = make_sync_client(config)
    for attempt in range(attempts + 1):
        try:
            return client.chat.completions.create(**request_kwargs)
        except RateLimitError:
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds)
    raise RuntimeError("unreachable")
