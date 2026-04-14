import os
import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.llm.config import LLMConfig
from app.tools.registry import get_tool_definitions, run_tool

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

DEFAULT_MAX_TOOL_ROUNDS = 8


class LLMClient(BaseModel):
    api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    model_name: str
    provider: str = "openai"
    use_tools: bool = True
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS

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
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_tools:
            request_kwargs["tools"] = get_tool_definitions()
            request_kwargs["tool_choice"] = "auto"

        tool_rounds = 0
        repeated_tool_signature: Optional[str] = None
        repeated_tool_count = 0

        while True:
            response = await client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                content = self._extract_content(message.content).strip()
                if content:
                    return content
                return "The model stopped after using tools without producing a final answer."

            tool_rounds += 1
            if tool_rounds > self.max_tool_rounds:
                return (
                    "The model got stuck in a tool loop and hit the tool-round limit "
                    f"({self.max_tool_rounds})."
                )

            current_signature = self._tool_call_signature(tool_calls)
            if current_signature == repeated_tool_signature:
                repeated_tool_count += 1
            else:
                repeated_tool_signature = current_signature
                repeated_tool_count = 1

            if repeated_tool_count >= 3:
                return "The model repeated the same tool call several times and appears stuck."

            messages.append(self._assistant_message_to_dict(message))

            for tool_call in tool_calls:
                raw_arguments = tool_call.function.arguments or "{}"
                arguments = json.loads(raw_arguments)
                tool_output = run_tool(tool_call.function.name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_output,
                    }
                )
            request_kwargs["messages"] = messages

    def _tool_call_signature(self, tool_calls: List[Any]) -> str:
        signature_payload = []
        for tool_call in tool_calls:
            signature_payload.append(
                {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "{}",
                }
            )
        return json.dumps(signature_payload, sort_keys=True, ensure_ascii=True)

    def _assistant_message_to_dict(self, message: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": "assistant",
            "content": self._extract_content(message.content),
        }

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]
        return payload

    def _extract_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block.text for block in content if getattr(block, "type", None) == "text"
            ]
            return "\n".join(text_parts)
        return str(content)
