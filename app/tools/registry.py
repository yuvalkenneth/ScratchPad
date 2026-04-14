import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.tools.executor import Executor
from app.tools.skills_tool import (
    SKILL_VIEW_SCHEMA,
    get_skills_prompt_text,
    skill_view_json,
)


ToolHandler = Callable[[dict[str, Any]], str]
EXECUTOR = Executor()


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


def run_shell(arguments: dict[str, Any]) -> str:
    result = EXECUTOR.run_shell(
        arguments["cmd"],
        cwd=arguments.get("cwd"),
    )
    return json.dumps(result, ensure_ascii=True)


def run_python(arguments: dict[str, Any]) -> str:
    result = EXECUTOR.run_python(
        arguments["code"],
        cwd=arguments.get("cwd"),
    )
    return json.dumps(result, ensure_ascii=True)


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
    "run_shell": {
        "definition": {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": (
                    "Run a shell command inside the local workspace. "
                    "Commands may be denied or require approval based on policy."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "Shell command to execute with bash -lc.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": (
                                "Optional working directory. Defaults to the workspace root "
                                "and is restricted by workspace policy."
                            ),
                        },
                    },
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
        },
        "handler": run_shell,
    },
    "run_python": {
        "definition": {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": (
                    "Run a Python snippet inside the local workspace with uv run python -c "
                    "so the project's managed environment is used. "
                    "Commands may be denied or require approval based on policy."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code to execute with uv run python -c.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": (
                                "Optional working directory. Defaults to the workspace root "
                                "and is restricted by workspace policy."
                            ),
                        },
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        "handler": run_python,
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
        "- run_shell: Run a shell command in the workspace; may return denied or needs_approval.",
        "- run_python: Run Python code in the workspace; may return denied or needs_approval.",
        "- skill_view: Load the full content of a skill or one of its linked files.",
        "Use run_shell and run_python for local execution when needed, and inspect the returned status field before assuming the command ran.",
        "For Python commands, prefer `uv run python` over raw `python` or `python3` so the project venv is used.",
        "Prefer workspace-relative paths for local scripts and files, and run them from the workspace root. Do not assume helper environment variables such as SKILL_DIR exist unless a tool explicitly provides them.",
        get_skills_prompt_text(),
    ]
    return "\n".join(lines)


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")
    return tool["handler"](arguments)
