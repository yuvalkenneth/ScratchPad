"""Dispatch Scratchpad eval commands through one small user-facing CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evals import benchmark, content_profiles, recommendations, retention, tool_choice, workflows
from scripts.training import compare_reports


COMMANDS: dict[str, Callable[[list[str] | None], int]] = {
    "tool-choice": tool_choice.run,
    "content-profiles": content_profiles.run,
    "retention": retention.run,
    "recommendations": recommendations.run,
    "workflows": workflows.run,
    "benchmark": benchmark.run,
    "compare": compare_reports.run,
}


def run(argv: list[str] | None = None) -> int:
    """Route the first CLI argument to the matching eval implementation."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run Scratchpad evals and eval-related report utilities."
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
    command = argv[0]
    if command not in COMMANDS:
        parser.parse_args([command])
    return COMMANDS[command](argv[1:])


if __name__ == "__main__":
    raise SystemExit(run())
