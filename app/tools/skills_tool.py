"""
Skills Tool Module

This module provides tools for listing and viewing skill documents.
Skills are organized as directories containing a SKILL.md file (the main instructions)
and optional supporting files like references, templates, and examples.

Inspired by Anthropic's Claude Skills system with progressive disclosure architecture:
- Metadata (name ≤64 chars, description ≤1024 chars) - shown in skills_list
- Full Instructions - loaded via skill_view when needed
- Linked Files (references, templates) - loaded on demand

Directory Structure:
    skills/
    ├── my-skill/
    │   ├── SKILL.md           # Main instructions (required)
    │   ├── references/        # Supporting documentation
    │   │   ├── api.md
    │   │   └── examples.md
    │   ├── templates/         # Templates for output
    │   │   └── template.md
    │   └── assets/            # Supplementary files (agentskills.io standard)
    └── category/              # Category folder for organization
        └── another-skill/
            └── SKILL.md

SKILL.md Format (YAML Frontmatter, agentskills.io compatible):
    ---
    name: skill-name              # Required, max 64 chars
    description: Brief description # Required, max 1024 chars
    version: 1.0.0                # Optional
    license: MIT                  # Optional (agentskills.io)
    platforms: [macos]            # Optional — restrict to specific OS platforms
                                  #   Valid: macos, linux, windows
                                  #   Omit to load on all platforms (default)
    prerequisites:                # Optional — legacy runtime requirements
      env_vars: [API_KEY]         #   Legacy env var names are normalized into
                                  #   required_environment_variables on load.
      commands: [curl, jq]        #   Command checks remain advisory only.
    compatibility: Requires X     # Optional (agentskills.io)
    metadata:                     # Optional, arbitrary key-value (agentskills.io)
      hermes:
        tags: [fine-tuning, llm]
        related_skills: [peft, lora]
    ---

    # Skill Title

    Full instructions and content here...

Available tools:
- skills_list: List skills with metadata (progressive disclosure tier 1)
- skill_view: Load full skill content (progressive disclosure tier 2-3)

Usage:
    from tools.skills_tool import skills_list, skill_view, check_skills_requirements

    # List all skills (returns metadata only - token efficient)
    result = skills_list()

    # View a skill's main content (loads full instructions)
    content = skill_view("axolotl")

    # View a reference file within a skill (loads linked file)
    content = skill_view("axolotl", "references/dataset-formats.md")
"""

import os
import yaml
from typing import List, Dict, Optional
SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills)",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

def skill_view(name: str, file_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load a skill's main content or a linked file.

    Args:
        name: The skill name (use skills_list to see available skills)
        file_path: OPTIONAL path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.
    Returns:
        A dict containing the requested content. If file_path is omitted, returns the main SKILL.md content under the 'content' key, along with a 'linked_files' dict showing available references/templates/scripts. If file_path is provided, returns the content of that specific file under the 'content' key.
    """
    base_path = os.path.join("skills", name)
    skill_md_path = os.path.join(base_path, "SKILL.md")

    if not os.path.exists(skill_md_path):
        raise FileNotFoundError(f"Skill '{name}' not found.")

    if file_path:
        # Load a linked file (reference, template, or script)
        target_path = os.path.join(base_path, file_path)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"File '{file_path}' not found in skill '{name}'.")
        with open(target_path, "r") as f:
            content = f.read()
        return {"content": content}

    # Load main SKILL.md content
    with open(skill_md_path, "r") as f:
        skill_md_content = f.read()

    # Parse YAML frontmatter for metadata and linked files
    if skill_md_content.startswith("---"):
        _, frontmatter, main_content = skill_md_content.split("---", 2)
        metadata = yaml.safe_load(frontmatter)
    else:
        metadata = {}
        main_content = skill_md_content

    # Identify linked files in references/, templates/, scripts/
    linked_files = {}
    for category in ["references", "templates", "scripts"]:
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            linked_files[category] = [
                f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))
            ]

    return {
        "content": main_content.strip(),
        "metadata": metadata,
        "linked_files": linked_files,
    }
