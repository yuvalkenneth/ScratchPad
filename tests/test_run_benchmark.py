from argparse import Namespace
from pathlib import Path

from scripts.run_benchmark import build_commands, run


def test_build_commands_can_skip_model_evals(tmp_path: Path) -> None:
    commands = build_commands(
        Namespace(
            reports_dir=tmp_path,
            label="test",
            skip_model_evals=True,
            provider="llama_cpp",
            model="qwen",
            base_url=None,
            api_key=None,
            start_script=None,
            content_limit=5,
        )
    )

    assert [command.name for command in commands] == ["workflow", "recommendations"]


def test_build_commands_adds_model_eval_reports(tmp_path: Path) -> None:
    commands = build_commands(
        Namespace(
            reports_dir=tmp_path,
            label="qwen4b",
            skip_model_evals=False,
            provider="llama_cpp",
            model="Qwen3.5-4B-BF16",
            base_url="http://localhost:8080/v1",
            api_key=None,
            start_script="/tmp/run-model.sh",
            content_limit=3,
        )
    )

    assert [command.name for command in commands] == [
        "workflow",
        "recommendations",
        "tool_choice",
        "content_profiles",
    ]
    assert commands[2].report_path == tmp_path / "qwen4b-tool-choice.json"
    assert "--limit" in commands[3].command


def test_benchmark_dry_run_writes_manifest(tmp_path: Path) -> None:
    result = run(["--label", "dry", "--reports-dir", str(tmp_path), "--skip-model-evals", "--dry-run"])

    assert result == 0
    manifest = tmp_path / "dry-benchmark-manifest.json"
    assert manifest.exists()
