import json
import os
from pathlib import Path
from typing import Any, Callable

from app.tools.executor import Executor
from app.tools.youtube_analyze_tool import (
    YOUTUBE_ANALYZE_SCHEMA,
    youtube_analyze,
)
from app.tools.url_analyze_tool import (
    URL_ANALYZE_SCHEMA,
    url_analyze,
)
from app.tools.skills_tool import (
    SKILLS_LIST_SCHEMA,
    SKILL_VIEW_SCHEMA,
    get_skills_prompt_text,
    skills_list_json,
    skill_view_json,
)
from app.tools.content_library_tool import (
    ANALYZE_SOURCE_SCHEMA,
    CONTENT_ADD_SCHEMA,
    CONTENT_LIST_SCHEMA,
    CONTENT_SAVE_SCHEMA,
    CONTENT_STATUS_UPDATE_SCHEMA,
    CONTENT_UPDATE_SCHEMA,
    analyze_source_json,
    content_add_json,
    content_list_json,
    content_save_json,
    content_status_update_json,
    content_update_json,
)


ToolHandler = Callable[[dict[str, Any]], str]
EXECUTOR = Executor()
EXECUTOR_TOOL_NAMES = {"run_shell", "run_python"}
INTERNAL_TOOL_NAMES = {"url_analyze", "youtube_analyze", "content_save"}


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
    "youtube_analyze": {
        "definition": {
            "type": "function",
            "function": YOUTUBE_ANALYZE_SCHEMA,
        },
        "handler": youtube_analyze,
    },
    "url_analyze": {
        "definition": {
            "type": "function",
            "function": URL_ANALYZE_SCHEMA,
        },
        "handler": url_analyze,
    },
    "content_save": {
        "definition": {
            "type": "function",
            "function": CONTENT_SAVE_SCHEMA,
        },
        "handler": content_save_json,
    },
    "analyze_source": {
        "definition": {
            "type": "function",
            "function": ANALYZE_SOURCE_SCHEMA,
        },
        "handler": analyze_source_json,
    },
    "content_add": {
        "definition": {
            "type": "function",
            "function": CONTENT_ADD_SCHEMA,
        },
        "handler": content_add_json,
    },
    "content_list": {
        "definition": {
            "type": "function",
            "function": CONTENT_LIST_SCHEMA,
        },
        "handler": content_list_json,
    },
    "content_status_update": {
        "definition": {
            "type": "function",
            "function": CONTENT_STATUS_UPDATE_SCHEMA,
        },
        "handler": content_status_update_json,
    },
    "content_update": {
        "definition": {
            "type": "function",
            "function": CONTENT_UPDATE_SCHEMA,
        },
        "handler": content_update_json,
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


def executor_tools_enabled() -> bool:
    return os.getenv("SCRATCHPAD_ENABLE_EXECUTOR_TOOLS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def visible_tools() -> dict[str, dict[str, Any]]:
    hidden = set(INTERNAL_TOOL_NAMES)
    if not executor_tools_enabled():
        hidden.update(EXECUTOR_TOOL_NAMES)
    return {name: tool for name, tool in TOOLS.items() if name not in hidden}


def get_tool_definitions() -> list[dict[str, Any]]:
    return [tool["definition"] for tool in visible_tools().values()]


def get_tools_prompt_text() -> str:
    lines = [
        "Available tools:",
        "- list_files: List files in a local directory.",
        "- analyze_source: Analyze a URL or YouTube video into a normalized content profile without saving it.",
        "- content_add: Analyze a URL and save the normalized content_profile into the local Markdown library in one step.",
        "- content_list: List saved Markdown library items by subject, category, depth, status, time, or free-text query. Defaults to status=[unread, started] when status is omitted.",
        "- content_update: Correct saved content metadata/profile fields such as title, summary, subject, categories, depth, time, confidence, metadata, or notes.",
        "- content_status_update: Update a saved Markdown library item's reading status or notes by id, URL, or source identity. Status values: unread, started, done, archived, abandoned.",
        "- skills_list: List available skills with compact metadata.",
        "- skill_view: Load the full content of a skill or one of its linked files.",
    ]
    if executor_tools_enabled():
        lines.extend(
            [
                "- run_shell: Run a shell command in the workspace; may return denied or needs_approval.",
                "- run_python: Run Python code in the workspace; may return denied or needs_approval.",
            ]
        )
    lines.extend([
        "For YouTube URLs, do not load a skill first unless you already have transcript data and need a specific transcript-transformation workflow.",
        "When the user asks what a URL/video is about or whether it is worth reading/watching before saving, call analyze_source.",
        "When the user asks to add, save, store, remember, or put a URL in the library, call content_add directly.",
        "For save requests, do not stop after analysis. The request is complete only after content_add reports a saved result.",
        "Use content_add for URL save requests so analysis, Markdown persistence, and library git history happen together.",
        "If content_add reports duplicate=true, tell the user the existing item was updated instead of creating a second copy.",
        "When the user asks to correct saved item details such as title, summary, subject, categories, depth, time, confidence, or metadata, use content_update.",
        "When the user says they started, finished, archived, or abandoned an item, use content_status_update.",
        "When the user asks what to read, watch, learn, study, revisit, or pick next from saved material, first load skill_view with name=\"scratchpad-recommendation\", then follow that skill and call content_list before answering.",
        "For recommendation/listing requests, if the user did not ask for done, archived, or abandoned items, rely on the content_list default unread/started status filter or pass status=[\"unread\", \"started\"].",
    ])
    if executor_tools_enabled():
        lines.extend(
            [
                "Use run_shell and run_python for local execution when explicitly needed, and inspect the returned status field before assuming the command ran.",
                "For Python commands, prefer `uv run python` over raw `python` or `python3` so the project venv is used.",
                "Prefer workspace-relative paths for local scripts and files, and run them from the workspace root. Do not assume helper environment variables such as SKILL_DIR exist unless a tool explicitly provides them.",
            ]
        )
    lines.append(get_skills_prompt_text())
    return "\n".join(lines)


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    tool = visible_tools().get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")
    return tool["handler"](arguments)
