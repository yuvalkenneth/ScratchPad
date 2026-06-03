from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


GIT_USER_NAME = "Scratchpad"
GIT_USER_EMAIL = "scratchpad@local"


def commit_library_paths(
    *,
    library_root: Path,
    paths: list[Path],
    message: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": True,
        "committed": False,
        "commit": None,
        "message": message,
        "error": None,
    }

    try:
        root = library_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        relative_paths = [relative_library_path(root, path) for path in paths]

        run_git(root, "init")
        run_git(root, "config", "user.name", GIT_USER_NAME)
        run_git(root, "config", "user.email", GIT_USER_EMAIL)
        run_git(root, "add", "--", *relative_paths)
        run_git(root, "commit", "-m", message)
        commit_hash = run_git(root, "rev-parse", "HEAD").stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        result["error"] = git_error_text(exc)
        return result

    result["committed"] = True
    result["commit"] = commit_hash
    return result


def relative_library_path(library_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(library_root))
    except ValueError as exc:
        raise ValueError(f"Path is outside library root: {resolved}") from exc


def run_git(library_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(library_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def git_error_text(exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        details = (exc.stderr or exc.stdout or "").strip()
        return details or str(exc)
    return str(exc)
