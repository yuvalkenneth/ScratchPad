import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.tools.skills_tool import (
    SKILLS_LIST_SCHEMA,
    SKILL_VIEW_SCHEMA,
    get_skills_prompt_text,
    skill_view_json,
    skills_list_json,
)


ToolHandler = Callable[[dict[str, Any]], str]


def get_time(_: dict[str, Any]) -> str:
    return datetime.now(timezone.utc).isoformat()


def list_files(arguments: dict[str, Any]) -> str:
    raw_path = arguments.get("path", ".")
    target = Path(raw_path).expanduser().resolve()

    if not target.exists():
        return f"Path does not exist: {target}"
    if not target.is_dir():
        return f"Path is not a directory: {target}"

    entries = sorted(item.name for item in target.iterdir())
    return json.dumps({"path": str(target), "entries": entries}, ensure_ascii=True)


TOOLS: dict[str, dict[str, Any]] = {
    "get_time": {
        "definition": {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current UTC time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        "handler": get_time,
    },
    "list_files": {
        "definition": {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory path on the local machine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to inspect.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        "handler": list_files,
    },
    "skills_list": {
        "definition": {
            "type": "function",
            "function": SKILLS_LIST_SCHEMA,
        },
        "handler": skills_list_json,
    },
    "skill_view": {
        "definition": {
            "type": "function",
            "function": SKILL_VIEW_SCHEMA,
        },
        "handler": skill_view_json,
    },
}


def get_tool_definitions() -> list[dict[str, Any]]:
    return [tool["definition"] for tool in TOOLS.values()]


def get_tools_prompt_text() -> str:
    lines = [
        "Available tools:",
        "- get_time: Get the current UTC time.",
        "- list_files: List files in a local directory.",
        "- skills_list: List available skills with names and descriptions.",
        "- skill_view: Load the full content of a skill or one of its linked files.",
        "Use skills_list before skill_view when you need to inspect available skills or confirm naming.",
        get_skills_prompt_text(),
    ]
    return "\n".join(lines)


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")
    return tool["handler"](arguments)
