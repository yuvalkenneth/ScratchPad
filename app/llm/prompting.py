import os

from app.tools.registry import get_tools_prompt_text


DEFAULT_SYSTEM_PROMPT = (
    "You are Scratchpad, a local-first assistant for thinking, reading, coding, "
    "and maintaining a personal knowledge library.\n"
    "Be concise, practical, and explicit about uncertainty.\n"
    "Use tools when they materially improve accuracy, provide local context, or "
    "complete the user's requested action. Do not call tools just to restate "
    "known information, and inspect tool results before relying on them."
)


def build_system_prompt() -> str:
    base_prompt = os.getenv("LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    sections = [
        base_prompt.strip(),
        get_tools_prompt_text(),
    ]
    return "\n\n".join(section for section in sections if section)
