from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional


WORKSPACE = Path(__file__).resolve().parents[2]
MAX_OUTPUT_CHARS = 50_000
DEFAULT_TIMEOUT_SECONDS = 30
SAFE_ENV_KEYS = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
}
SENSITIVE_PATH_PREFIXES = (
    Path("/etc"),
    Path("/root"),
    Path.home() / ".ssh",
    Path.home() / ".env",
)

ASK_PATTERNS = [
    re.compile(r"(^|\s)(curl|wget)\b"),
    re.compile(r"(^|\s)git\s+clone\b"),
    re.compile(r"(^|\s)(pip|pip3)\s+install\b"),
    re.compile(r"(^|\s)(npm|pnpm|yarn)\s+(install|add)\b"),
    re.compile(r"(^|\s)(nohup|disown|setsid)\b"),
    re.compile(r"&\s*$"),
]

DENY_PATTERNS = [
    re.compile(r"(^|\s)sudo\b"),
    re.compile(r"(^|\s)su\b"),
    re.compile(r"\brm\s+-[^\n]*\brf\b"),
    re.compile(r"(^|\s)(chmod|chown|dd|mkfs)\b"),
    re.compile(r"(^|\s)(doas|pkexec)\b"),
]


def _is_within_workspace(path: Path, workspace: Path = WORKSPACE) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
        return True
    except ValueError:
        return False


def _truncate_output(text: Optional[str]) -> str:
    if not text:
        return ""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:MAX_OUTPUT_CHARS]}\n...[truncated {omitted} chars]"


def _build_safe_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in SAFE_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value

    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _looks_like_path(token: str) -> bool:
    if token in {".", "..", "~"}:
        return True
    return token.startswith(("/", "./", "../", "~/"))


def _extract_candidate_paths(command_text: str, workspace: Path = WORKSPACE) -> list[Path]:
    candidates: list[Path] = []
    try:
        tokens = shlex.split(command_text)
    except ValueError:
        tokens = command_text.split()

    for token in tokens:
        if not _looks_like_path(token):
            continue
        if "://" in token:
            continue

        expanded = Path(token).expanduser()
        if expanded.is_absolute():
            candidates.append(expanded)
        else:
            candidates.append((workspace / expanded).resolve())

    for match in re.finditer(r"(?:^|(?<=[\s=\"']))(?P<path>/[^\s\"']+)", command_text):
        raw_path = match.group("path")
        if "://" in raw_path:
            continue
        candidates.append(Path(raw_path).expanduser())

    return candidates


def _find_sensitive_path(command_text: str, workspace: Path = WORKSPACE) -> Optional[Path]:
    for candidate in _extract_candidate_paths(command_text, workspace):
        for sensitive in SENSITIVE_PATH_PREFIXES:
            if candidate == sensitive or sensitive in candidate.parents:
                return candidate
    return None


def _find_outside_workspace_path(
    command_text: str,
    workspace: Path = WORKSPACE,
) -> Optional[Path]:
    for candidate in _extract_candidate_paths(command_text, workspace):
        if candidate.is_absolute() and not _is_within_workspace(candidate, workspace):
            return candidate
    return None


def _background_reason(command_text: str) -> Optional[str]:
    if re.search(r"&\s*$", command_text):
        return "Background execution requires approval."
    if re.search(r"(^|\s)(nohup|disown|setsid)\b", command_text):
        return "Long-running detached processes require approval."
    return None


def _command_policy(command_text: str, workspace: Path = WORKSPACE) -> tuple[str, str]:
    normalized = command_text.strip()
    if not normalized:
        return "allow", ""

    sensitive_path = _find_sensitive_path(normalized, workspace)
    if sensitive_path is not None:
        return "deny", f"Access to sensitive path is denied: {sensitive_path}"

    outside_path = _find_outside_workspace_path(normalized, workspace)
    if outside_path is not None:
        return "ask", f"Command references a path outside the workspace: {outside_path}"

    for pattern in DENY_PATTERNS:
        if pattern.search(normalized):
            return "deny", f"Command matches a denied pattern: {pattern.pattern}"

    background_reason = _background_reason(normalized)
    if background_reason:
        return "ask", background_reason

    for pattern in ASK_PATTERNS:
        if pattern.search(normalized):
            return "ask", f"Command matches an approval-required pattern: {pattern.pattern}"

    return "allow", ""


def should_ask_permission(cmd: str) -> tuple[bool, str]:
    decision, reason = _command_policy(cmd, WORKSPACE)
    return decision == "ask", reason


class Executor:
    def __init__(
        self,
        workspace: Path | str = WORKSPACE,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        allow_outside_workspace: bool = False,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.allow_outside_workspace = allow_outside_workspace

    def run_shell(self, cmd: str, cwd: Optional[str] = None) -> dict[str, Any]:
        return self._run(
            ["bash", "-lc", cmd],
            original_input=cmd,
            cwd=cwd,
            language="shell",
        )

    def run_python(self, code: str, cwd: Optional[str] = None) -> dict[str, Any]:
        # Route Python execution through uv so the project's managed environment is used.
        command_text = f"uv run python -c {shlex.quote(code)}"
        return self._run(
            ["uv", "run", "python", "-c", code],
            original_input=command_text,
            cwd=cwd,
            language="python",
        )

    def _run(
        self,
        argv: list[str],
        *,
        original_input: str,
        cwd: Optional[str],
        language: str,
    ) -> dict[str, Any]:
        resolved_cwd = self._resolve_cwd(cwd)
        if isinstance(resolved_cwd, dict):
            return resolved_cwd
        command_text = original_input
        if self.allow_outside_workspace:
            decision, reason = _command_policy_without_workspace_block(command_text, self.workspace)
        else:
            decision, reason = _command_policy(command_text, self.workspace)

        if decision == "deny":
            return {"status": "denied", "reason": reason}
        if decision == "ask":
            return {"status": "needs_approval", "reason": reason}

        return self._execute(argv, resolved_cwd, language)

    def _resolve_cwd(self, cwd: Optional[str]) -> Path | dict[str, Any]:
        target = self.workspace if cwd is None else Path(cwd).expanduser()
        if not target.is_absolute():
            target = (self.workspace / target).resolve()
        else:
            target = target.resolve()

        if not self.allow_outside_workspace and not _is_within_workspace(target, self.workspace):
            return {
                "status": "denied",
                "reason": f"Working directory is outside the workspace: {target}",
            }

        if not target.exists():
            return {"status": "error", "reason": f"Working directory does not exist: {target}"}
        if not target.is_dir():
            return {"status": "error", "reason": f"Working directory is not a directory: {target}"}
        return target

    def _execute(self, argv: list[str], cwd: Path, language: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                env=_build_safe_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return {
                "status": "completed",
                "language": language,
                "cwd": str(cwd),
                "exit_code": completed.returncode,
                "stdout": _truncate_output(completed.stdout),
                "stderr": _truncate_output(completed.stderr),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return {
                "status": "completed",
                "language": language,
                "cwd": str(cwd),
                "exit_code": None,
                "stdout": _truncate_output(stdout),
                "stderr": _truncate_output(stderr),
                "timed_out": True,
            }


def _command_policy_without_workspace_block(
    command_text: str,
    workspace: Path = WORKSPACE,
) -> tuple[str, str]:
    normalized = command_text.strip()
    if not normalized:
        return "allow", ""

    sensitive_path = _find_sensitive_path(normalized, workspace)
    if sensitive_path is not None:
        return "deny", f"Access to sensitive path is denied: {sensitive_path}"

    for pattern in DENY_PATTERNS:
        if pattern.search(normalized):
            return "deny", f"Command matches a denied pattern: {pattern.pattern}"

    background_reason = _background_reason(normalized)
    if background_reason:
        return "ask", background_reason

    for pattern in ASK_PATTERNS:
        if pattern.search(normalized):
            return "ask", f"Command matches an approval-required pattern: {pattern.pattern}"

    return "allow", ""
