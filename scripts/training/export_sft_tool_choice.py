"""Export Scratchpad tool-choice cases into SFT training formats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.prompting import DEFAULT_SYSTEM_PROMPT
from app.tools.registry import get_tool_definitions
from scripts.evals.tool_choice import DEFAULT_CASES_PATH, NO_TOOL_LABEL, SPLIT_LABELS, load_cases


DEFAULT_OUTPUT_PATH = REPO_ROOT / "training" / "datasets" / "tool_choice" / "sft.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "datasets" / "tool_choice"
OUTPUT_FORMATS = ("messages", "text")


class ChatTemplateTokenizer(Protocol):
    """Minimal tokenizer protocol needed for chat-template rendering."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str:
        ...


def build_training_system_prompt() -> str:
    """Build the system prompt used in exported tool-routing examples."""
    return "\n\n".join(
        [
            DEFAULT_SYSTEM_PROMPT.strip(),
            "Current local time: <LOCAL_TIME>",
            "Use the provided tools when they are needed. If no tool is needed, answer normally.",
        ]
    )


def target_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Convert eval expectations into the canonical training target."""
    expected_tool = str(case["expected_tool"])
    expected_arguments = case.get("expected_arguments") or {}
    arguments: dict[str, Any] = {}
    for field, value in expected_arguments.get("must_equal", {}).items():
        arguments[field] = value
    for field, value in expected_arguments.get("must_include", {}).items():
        arguments[field] = value
    return {
        "tool": expected_tool,
        "arguments": arguments if expected_tool != NO_TOOL_LABEL else {},
    }


def openai_tool_call_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Render one case target as an OpenAI-style tool call."""
    target = target_from_case(case)
    return {
        "type": "function",
        "function": {
            "name": target["tool"],
            "arguments": target["arguments"],
        },
    }


def messages_from_case(
    case: dict[str, Any],
    *,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-style SFT messages for a single tool-choice case."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if "messages" in case:
        messages.extend(case["messages"])
    else:
        messages.append({"role": "user", "content": case["user"]})
    if case["expected_tool"] == NO_TOOL_LABEL:
        messages.append(
            {
                "role": "assistant",
                "content": case.get("assistant_response")
                or "I can answer this directly from general knowledge without using a Scratchpad tool.",
            }
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [openai_tool_call_from_case(case)],
            }
        )
    return messages


def render_messages_as_text(
    messages: list[dict[str, Any]],
    *,
    tokenizer: ChatTemplateTokenizer,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render messages through the target tokenizer's chat template."""
    kwargs: dict[str, Any] = {}
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )


def load_chat_template_tokenizer(
    tokenizer_name_or_path: str,
    *,
    chat_template: str | None = None,
) -> ChatTemplateTokenizer:
    """Load a tokenizer and optionally apply an Unsloth chat template."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Text export requires transformers. Install it in the training environment "
            "or use --output-format messages."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if chat_template:
        try:
            from unsloth.chat_templates import get_chat_template
        except ImportError as exc:
            raise RuntimeError(
                "--chat-template requires unsloth in the training environment. "
                "Omit it to use the tokenizer's built-in chat template."
            ) from exc
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    return tokenizer


def sft_row_from_case(
    case: dict[str, Any],
    *,
    system_prompt: str,
    source: str,
    output_format: str = "messages",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> dict[str, Any]:
    """Convert one eval case into one SFT JSONL row."""
    messages = messages_from_case(
        case,
        system_prompt=system_prompt,
    )
    tools = get_tool_definitions()
    row: dict[str, Any] = {
        "id": case["id"],
        "metadata": {
            "source": source,
            "expected_tool": case["expected_tool"],
            "target": target_from_case(case),
            "target_format": "openai-tools",
            "output_format": output_format,
            "native_tools": True,
        },
    }
    for metadata_key in ("intent", "category", "difficulty", "retention_kind", "split", "context_kind"):
        if metadata_key in case:
            row["metadata"][metadata_key] = case[metadata_key]
    if output_format == "messages":
        row["messages"] = messages
    elif output_format == "text":
        if tokenizer is None:
            raise ValueError("Text output format requires a tokenizer.")
        row["text"] = render_messages_as_text(messages, tokenizer=tokenizer, tools=tools)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return row


def build_sft_rows(
    cases_path: Path,
    *,
    output_format: str = "messages",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> list[dict[str, Any]]:
    """Build all SFT rows from a case file."""
    system_prompt = build_training_system_prompt()
    source = str(cases_path.relative_to(REPO_ROOT)) if cases_path.is_relative_to(REPO_ROOT) else str(cases_path)
    return [
        sft_row_from_case(
            case,
            system_prompt=system_prompt,
            source=source,
            output_format=output_format,
            tokenizer=tokenizer,
        )
        for case in load_cases(cases_path)
    ]


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split rows into train/validation/heldout, preferring explicit metadata."""
    splits = {split: [] for split in SPLIT_LABELS}
    for index, row in enumerate(rows):
        explicit_split = row.get("metadata", {}).get("split")
        if explicit_split:
            splits[explicit_split].append(row)
        elif index % 5 == 3:
            splits["validation"].append(row)
        elif index % 5 == 4:
            splits["heldout"].append(row)
        else:
            splits["train"].append(row)
    return splits


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows as newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def export_rows(
    cases_path: Path,
    output_path: Path,
    *,
    output_format: str = "messages",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> int:
    """Export one unsplit JSONL file and return the row count."""
    rows = build_sft_rows(
        cases_path,
        output_format=output_format,
        tokenizer=tokenizer,
    )
    write_jsonl(rows, output_path)
    return len(rows)


def export_split_rows(
    cases_path: Path,
    output_dir: Path,
    *,
    output_format: str = "messages",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> dict[str, int]:
    """Export train/validation/heldout JSONL files and return split counts."""
    rows = build_sft_rows(
        cases_path,
        output_format=output_format,
        tokenizer=tokenizer,
    )
    splits = split_rows(rows)
    for split, split_rows_ in splits.items():
        write_jsonl(split_rows_, output_dir / f"{split}.jsonl")
    return {split: len(split_rows_) for split, split_rows_ in splits.items()}


def run(argv: list[str] | None = None) -> int:
    """Run the SFT export CLI."""
    parser = argparse.ArgumentParser(
        description="Export Scratchpad tool-choice eval cases as chat-style SFT JSONL."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default="messages",
        help=(
            "messages writes chat messages JSONL; text renders messages through a tokenizer "
            "chat template for SFT trainers that expect a text column."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        help="HF model/tokenizer path or id used with --output-format text.",
    )
    parser.add_argument(
        "--chat-template",
        help="Optional Unsloth chat template name to apply before text rendering, e.g. chatml or gemma-3.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        help="Write deterministic train/validation/heldout JSONL files to this directory.",
    )
    args = parser.parse_args(argv)
    tokenizer = None
    if args.output_format == "text":
        if not args.tokenizer:
            parser.error("--output-format text requires --tokenizer")
        tokenizer = load_chat_template_tokenizer(args.tokenizer, chat_template=args.chat_template)

    if args.split_dir:
        counts = export_split_rows(
            args.cases,
            args.split_dir,
            output_format=args.output_format,
            tokenizer=tokenizer,
        )
        print(
            "Wrote split SFT rows to "
            f"{args.split_dir}: "
            + ", ".join(f"{split}={count}" for split, count in counts.items())
        )
    else:
        row_count = export_rows(
            args.cases,
            args.output,
            output_format=args.output_format,
            tokenizer=tokenizer,
        )
        print(f"Wrote {row_count} SFT rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
