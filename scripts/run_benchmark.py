from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "evals" / "runs"


@dataclass
class BenchmarkCommand:
    name: str
    command: list[str]
    report_path: Path | None = None


def build_commands(args: argparse.Namespace) -> list[BenchmarkCommand]:
    reports_dir = Path(args.reports_dir)
    commands = [
        BenchmarkCommand(
            name="workflow",
            command=[sys.executable, "scripts/eval_workflows.py", "--json"],
        ),
        BenchmarkCommand(
            name="recommendations",
            command=[sys.executable, "scripts/eval_recommendations.py", "--json"],
        ),
    ]

    if args.skip_model_evals:
        return commands

    tool_report = reports_dir / f"{args.label}-tool-choice.json"
    tool_command = [
        sys.executable,
        "scripts/eval_tool_choice.py",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--report",
        str(tool_report),
        "--json",
    ]
    if args.base_url:
        tool_command.extend(["--base-url", args.base_url])
    if args.api_key:
        tool_command.extend(["--api-key", args.api_key])
    if args.start_script:
        tool_command.extend(["--start-script", args.start_script])
    commands.append(BenchmarkCommand("tool_choice", tool_command, tool_report))

    content_command = [
        sys.executable,
        "scripts/eval_content_profiles.py",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--json",
    ]
    if args.base_url:
        content_command.extend(["--base-url", args.base_url])
    if args.api_key:
        content_command.extend(["--api-key", args.api_key])
    if args.start_script:
        content_command.extend(["--start-script", args.start_script])
    if args.content_limit:
        content_command.extend(["--limit", str(args.content_limit)])
    commands.append(BenchmarkCommand("content_profiles", content_command))
    return commands


def run_command(command: BenchmarkCommand) -> dict[str, Any]:
    started_at = time.perf_counter()
    completed = subprocess.run(
        command.command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "name": command.name,
        "command": command.command,
        "returncode": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started_at, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report_path": str(command.report_path) if command.report_path else None,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Scratchpad as a local-model benchmark harness."
    )
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--provider", default="llama_cpp")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--start-script")
    parser.add_argument("--content-limit", type=int, default=5)
    parser.add_argument("--skip-model-evals", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    manifest_path = args.reports_dir / f"{args.label}-benchmark-manifest.json"

    if args.dry_run:
        manifest = {
            "type": "scratchpad_benchmark_manifest",
            "label": args.label,
            "dry_run": True,
            "commands": [asdict(command) | {"report_path": str(command.report_path) if command.report_path else None} for command in commands],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote dry-run manifest: {manifest_path}")
        return 0

    results = [run_command(command) for command in commands]
    manifest = {
        "type": "scratchpad_benchmark_manifest",
        "label": args.label,
        "dry_run": False,
        "passed": all(result["returncode"] == 0 for result in results),
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark manifest: {manifest_path}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
