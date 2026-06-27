"""Build and run grouped Scratchpad benchmark commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.profiles import config_from_profile, resolved_config_metadata

DEFAULT_REPORTS_DIR = REPO_ROOT / "evals" / "runs"


@dataclass
class BenchmarkCommand:
    """A subprocess command plus optional report path for benchmark manifests."""

    name: str
    command: list[str]
    report_path: Path | None = None


def build_commands(args: argparse.Namespace) -> list[BenchmarkCommand]:
    """Create the deterministic and model-backed commands for one benchmark run."""
    reports_dir = Path(args.reports_dir)
    commands = [
        BenchmarkCommand(
            name="workflow",
            command=[sys.executable, "scripts/eval.py", "workflows", "--json"],
        ),
        BenchmarkCommand(
            name="recommendations",
            command=[sys.executable, "scripts/eval.py", "recommendations", "--json"],
        ),
    ]

    if args.skip_model_evals:
        return commands

    tool_report = reports_dir / f"{args.label}-tool-choice.json"
    tool_command = [
        sys.executable,
        "scripts/eval.py",
        "tool-choice",
        "--report",
        str(tool_report),
        "--json",
    ]
    append_model_args(tool_command, args)
    if args.base_url:
        tool_command.extend(["--base-url", args.base_url])
    if args.api_key:
        tool_command.extend(["--api-key", args.api_key])
    if args.start_script:
        tool_command.extend(["--start-script", args.start_script])
    commands.append(BenchmarkCommand("tool_choice", tool_command, tool_report))

    content_command = [
        sys.executable,
        "scripts/eval.py",
        "content-profiles",
        "--json",
    ]
    append_model_args(content_command, args)
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


def append_model_args(command: list[str], args: argparse.Namespace) -> None:
    """Append shared model/profile CLI arguments to an eval command."""
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.model:
        command.extend(["--model", args.model])
    if not args.profile and not args.provider:
        command.extend(["--provider", "llama_cpp"])
    if not args.profile and not args.model:
        command.extend(["--model", "qwen"])


def run_command(command: BenchmarkCommand) -> dict[str, Any]:
    """Execute one benchmark subprocess and capture output for the manifest."""
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
    """Run the benchmark CLI or write a dry-run manifest."""
    parser = argparse.ArgumentParser(
        description="Run Scratchpad as a local-model benchmark harness."
    )
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--profile", help="Named model profile from config/models.json or config/models.local.json.")
    parser.add_argument("--provider")
    parser.add_argument("--model")
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
    resolved_config = config_from_profile(
        args.profile,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        start_script=args.start_script,
    )
    model_config = resolved_config_metadata(resolved_config, profile=args.profile)

    if args.dry_run:
        manifest = {
            "type": "scratchpad_benchmark_manifest",
            "label": args.label,
            "dry_run": True,
            "model_config": model_config,
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
        "model_config": model_config,
        "passed": all(result["returncode"] == 0 for result in results),
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark manifest: {manifest_path}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
