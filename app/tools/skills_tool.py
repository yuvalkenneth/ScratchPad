"""
Skills Tool Module

This module provides tools for listing and viewing skill documents.
Skills are organized as directories containing a SKILL.md file (the main
instructions) and optional supporting files like references, templates,
and examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": (
        "List available skills with compact metadata. Use this before loading a "
        "full skill so you can choose the right one."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": (
        "Load a skill's full content or access a linked file within a skill. "
        "Omit file_path to get the main SKILL.md content plus linked file names."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills).",
            },
            "file_path": {
                "type": "string",
                "description": (
                    "OPTIONAL: Path to a linked file within the skill, for example "
                    "'references/api.md' or 'scripts/validate.py'."
                ),
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}


def _skills_root() -> Path:
    return Path("skills")


def _iter_skill_files() -> list[Path]:
    root = _skills_root()
    if not root.exists():
        return []
    return sorted(path for path in root.glob("**/SKILL.md") if path.is_file())


def _extract_frontmatter_parts(content: str) -> tuple[dict[str, Any], str]:
    if not content.startswith("---\n"):
        return {}, content.strip()

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content.strip()

    _, frontmatter, body = parts
    return _parse_frontmatter(frontmatter.strip("\n")), body.strip()


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    lines = frontmatter.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        i += 1

        if not stripped or stripped.startswith("#") or ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value in {"|", ">"}:
            block_lines: list[str] = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith((" ", "\t")):
                    block_lines.append(next_line.strip())
                    i += 1
                    continue
                break
            metadata[key] = "\n".join(block_lines) if value == "|" else " ".join(block_lines)
            continue

        metadata[key] = _parse_scalar(value)

    return metadata


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",")]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    return value


def _load_skill_metadata(skill_md_path: Path) -> dict[str, Any]:
    content = skill_md_path.read_text()
    metadata, _ = _extract_frontmatter_parts(content)
    skill_name = metadata.get("name") or skill_md_path.parent.name
    description = metadata.get("description") or ""
    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]

    skill_dir = skill_md_path.parent
    relative_parts = skill_dir.relative_to(_skills_root()).parts
    category = metadata.get("category")
    if not category:
        category = relative_parts[0] if len(relative_parts) > 1 else "uncategorized"

    return {
        "name": skill_name,
        "description": description,
        "path": str(skill_dir),
        "category": category,
        "tags": tags,
    }


def skills_list() -> dict[str, Any]:
    skills = [_load_skill_metadata(path) for path in _iter_skill_files()]
    return {"skills": skills}


def _find_skill_dir(name: str) -> Path:
    for skill_md_path in _iter_skill_files():
        metadata = _load_skill_metadata(skill_md_path)
        if metadata["name"] == name:
            return skill_md_path.parent
    raise FileNotFoundError(f"Skill '{name}' not found.")


def skill_view(name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    skill_dir = _find_skill_dir(name)
    skill_md_path = skill_dir / "SKILL.md"

    if file_path:
        target_path = (skill_dir / file_path).resolve()
        if skill_dir.resolve() not in target_path.parents:
            raise ValueError(f"Invalid file path outside skill: {file_path}")
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"File '{file_path}' not found in skill '{name}'.")
        return {"content": target_path.read_text()}

    skill_md_content = skill_md_path.read_text()
    metadata, main_content = _extract_frontmatter_parts(skill_md_content)

    linked_files: dict[str, list[str]] = {}
    for category in ["references", "templates", "scripts", "assets"]:
        category_path = skill_dir / category
        if category_path.is_dir():
            linked_files[category] = sorted(
                str(path.relative_to(skill_dir))
                for path in category_path.rglob("*")
                if path.is_file()
            )

    return {
        "content": main_content,
        "metadata": metadata,
        "linked_files": linked_files,
    }


def skills_list_json(_: dict[str, Any]) -> str:
    return json.dumps(skills_list(), ensure_ascii=True)


def skill_view_json(arguments: dict[str, Any]) -> str:
    return json.dumps(
        skill_view(arguments["name"], arguments.get("file_path")),
        ensure_ascii=True,
    )


def get_skills_prompt_text() -> str:
    skills = skills_list()["skills"]
    if not skills:
        return (
            "## Skills (mandatory)\n"
            "Before replying, scan the skills below. If one clearly matches the task, "
            "load it with skill_view(name) and follow its instructions.\n"
            "If none match, proceed normally without loading a skill.\n\n"
            "<available_skills>\n"
            "  none\n"
            "</available_skills>"
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for skill in skills:
        grouped.setdefault(skill["category"], []).append(skill)

    lines = [
        "## Skills (mandatory)",
        "Before replying, scan the skills below. If one clearly matches the task, load it with skill_view(name) and follow its instructions.",
        "After difficult or iterative tasks, offer to save the workflow as a skill.",
        "If none match, proceed normally without loading a skill.",
        "",
        "<available_skills>",
    ]
    for category in sorted(grouped):
        lines.append(f"  {category}:")
        for skill in sorted(grouped[category], key=lambda item: item["name"]):
            description = skill["description"] or "No description."
            tags = skill.get("tags") or []
            tag_suffix = f" [tags: {', '.join(tags)}]" if tags else ""
            lines.append(f"    - {skill['name']}: {description}{tag_suffix}")
    lines.append("</available_skills>")
    return "\n".join(lines)
