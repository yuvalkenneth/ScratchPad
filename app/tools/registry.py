import json
from datetime import datetime, timezone
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
    CONTENT_ADD_SCHEMA,
    CONTENT_LIST_SCHEMA,
    CONTENT_SAVE_SCHEMA,
    CONTENT_STATUS_UPDATE_SCHEMA,
    content_add_json,
    content_list_json,
    content_save_json,
    content_status_update_json,
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
        "- run_shell: Run a shell command in the workspace; may return denied or needs_approval.",
        "- run_python: Run Python code in the workspace; may return denied or needs_approval.",
        "- youtube_analyze: Analyze a YouTube video internally using transcript fetch, chunking, and dedicated LLM passes. It uses the same active provider/server/model as the main chat. Optional arguments: task, question, language, include_timestamps.",
        "- url_analyze: Fetch a web page internally, extract readable text, and classify it into a compact content profile.",
        "- content_add: Analyze a URL and save the normalized content_profile into the local Markdown library in one step.",
        "- content_save: Save a normalized content_profile into the local Markdown library.",
        "- content_list: List saved Markdown library items by subject, category, depth, status, time, or free-text query.",
        "- content_status_update: Update a saved Markdown library item's reading status or notes by id, URL, or source identity.",
        "- skills_list: List available skills with compact metadata.",
        "- skill_view: Load the full content of a skill or one of its linked files.",
        "For YouTube URLs, do not load a skill first unless you already have transcript data and need a specific transcript-transformation workflow.",
        "Use youtube_analyze with task='content_profile' for product-facing classification such as summary, subject, depth_level, categories, and estimated_time_minutes.",
        "Use youtube_analyze for YouTube summaries, explanations, chapters, study notes, key points, quotes, and other whole-video analysis tasks.",
        "youtube_analyze handles transcript retrieval internally so the raw transcript stays out of the main chat context.",
        "Use url_analyze with task='content_profile' for non-YouTube URLs when the user wants a compact summary plus subject, depth_level, and estimated_time_minutes.",
        "When the user asks to add or save a URL, prefer content_add so analysis and Markdown persistence happen together.",
        "If content_add reports duplicate=true, tell the user the existing item was updated instead of creating a second copy.",
        "After a user asks to save an already-analyzed profile, pass the top-level content_profile fields to content_save.",
        "When the user says they started, finished, archived, or abandoned an item, use content_status_update.",
        "When the user asks what to read or wants saved material, use content_list before answering.",
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
