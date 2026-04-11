import os

from app.tools.registry import get_tools_prompt_text


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local assistant.\n"
    "Use tools when they help answer accurately."
)


def build_system_prompt() -> str:
    base_prompt = os.getenv("LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    sections = [
        base_prompt.strip(),
        get_tools_prompt_text(),
    ]
    return "\n\n".join(section for section in sections if section)
