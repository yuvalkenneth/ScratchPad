import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.llm.config import LLMConfig

PROVIDER_TO_API_KEY_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "llama_cpp": "LLAMA_CPP_API_KEY",
}

PROVIDERS_TO_BASE_URL: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "azure": "https://{your-resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version=2024-06-01",
    "openai": "https://api.openai.com/v1",
    "llama_cpp": "http://127.0.0.1:8080/v1",
    "gemini": "https://api.gemini.com/v1",
}


class LLMClient(BaseModel):
    api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    model_name: str
    provider: str = "openai"

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LLMClient":
        return cls(
            provider=config.provider,
            model_name=config.model_name,
            api_url=config.base_url,
        )

    def _get_client(self) -> AsyncOpenAI:
        api_key_env_var = PROVIDER_TO_API_KEY_ENV_VAR.get(self.provider)
        api_key = os.getenv(api_key_env_var) if api_key_env_var else None
        if not api_key:
            api_key = os.getenv("LLM_API_KEY", "local")

        base_url = self.api_url or os.getenv("LLM_BASE_URL")
        if not base_url:
            base_url = PROVIDERS_TO_BASE_URL.get(self.provider)
        if base_url is None:
            raise ValueError(f"Base URL for provider {self.provider} not found.")

        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def get_response(self, messages: List[Dict[str, Any]]) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block.text for block in content if getattr(block, "type", None) == "text"
            ]
            return "\n".join(text_parts)
        return str(content)
