"""Validate tokenizer-rendered SFT rows before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evals.tool_choice import DEFAULT_CASES_PATH
from scripts.training.export_sft_tool_choice import build_sft_rows, load_chat_template_tokenizer


def infer_family(tokenizer_name_or_path: str) -> str:
    """Infer template-marker expectations from a tokenizer path or id."""
    lowered = tokenizer_name_or_path.lower()
    if "qwen" in lowered:
        return "qwen"
    return "generic"


def assert_required_substrings(text: str, required: list[str], *, row_id: str) -> None:
    """Fail when a rendered row is missing expected template/tool markers."""
    missing = [substring for substring in required if substring not in text]
    if missing:
        raise ValueError(f"Rendered row {row_id} is missing required template markers: {missing}")


def validate_rendered_row(
    row: dict[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    """Validate one rendered row and return a compact inspection summary."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Rendered row {row.get('id')} is missing non-empty text.")

    expected_tool = row["metadata"]["expected_tool"]
    target = row["metadata"].get("target")
    if not isinstance(target, dict):
        raise ValueError(f"Rendered row {row.get('id')} is missing canonical target metadata.")
    if expected_tool == "no_tool":
        assert_required_substrings(text, ["No tool call is needed."], row_id=str(row["id"]))
    else:
        arguments = target.get("arguments") or {}
        required = [
            "<tools>",
            "</tools>",
            "<tool_call>",
            f"<function={expected_tool}>",
            "</function>",
            "</tool_call>",
        ]
        for name, value in arguments.items():
            if isinstance(value, (list, dict)):
                rendered_value = json.dumps(value, ensure_ascii=True)
            else:
                rendered_value = str(value)
            required.extend([f"<parameter={name}>", rendered_value, "</parameter>"])
        assert_required_substrings(text, required, row_id=str(row["id"]))

    markers: list[str] = []
    if family == "qwen":
        markers = [
            "<|im_start|>system",
            "<|im_start|>user",
            "<|im_start|>assistant",
            "<|im_end|>",
        ]
    if markers:
        assert_required_substrings(text, markers, row_id=str(row["id"]))

    return {
        "id": row["id"],
        "expected_tool": expected_tool,
        "characters": len(text),
        "starts_with": text[:80],
        "ends_with": text[-80:],
    }


def run(argv: list[str] | None = None) -> int:
    """Run the chat-template validation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Render Scratchpad SFT rows through a tokenizer chat template and validate "
            "that expected model-family markers and targets are present."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path or id.")
    parser.add_argument(
        "--family",
        choices=("auto", "generic", "qwen"),
        default="auto",
        help="Template marker expectations to validate.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of rows to render and validate.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--show-text", action="store_true", help="Print full rendered examples.")
    args = parser.parse_args(argv)

    family = infer_family(args.tokenizer) if args.family == "auto" else args.family
    tokenizer = load_chat_template_tokenizer(args.tokenizer)
    rows = build_sft_rows(
        args.cases,
        output_format="text",
        tokenizer=tokenizer,
    )
    selected_rows = rows[: args.limit]
    validations = [validate_rendered_row(row, family=family) for row in selected_rows]

    result = {
        "type": "chat_template_validation",
        "tokenizer": args.tokenizer,
        "chat_template": "tokenizer-default",
        "family": family,
        "target_format": "openai-tools",
        "validated_rows": len(validations),
        "rows": validations,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print(
            f"Validated {len(validations)} rendered rows with family={family} "
            f"tokenizer={args.tokenizer}"
        )
        for validation, row in zip(validations, selected_rows):
            print(
                f"- {validation['id']}: expected_tool={validation['expected_tool']} "
                f"characters={validation['characters']}"
            )
            if args.show_text:
                print(row["text"])
                print("---")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
