import asyncio
from email.mime import base
import http
import json
import logging
import os
import sqlite3
import time
import uuid
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

PROVIDER_TO_API_KEY_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "llama_cpp" : "LLAMA_CPP_API_KEY"
}

PROVIDERS_TO_BASE_URL: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "azure": "https://{your-resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version=2024-06-01",
    "openai": "https://api.openai.com/v1",
    "llama_cpp": "http://localhost:8000/v1",
    "gemini": "https://api.gemini.com/v1"

}

class LLMClient(BaseModel):
    api_url: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    model_name: str
    provider=str
    
    def _get_client(self) -> AsyncOpenAI:
        api_key = os.getenv(PROVIDER_TO_API_KEY_ENV_VAR.get(self.provider))
        if api_key is None:
            raise ValueError(f"API key for provider {self.provider} not found in environment variables.")
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
            max_tokens=self.max_tokens
        )
        return response['choices'][0]['message']['content']
    