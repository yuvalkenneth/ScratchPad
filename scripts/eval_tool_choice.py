from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.config import LLMConfig
from app.llm.openai_compatible import create_completion_with_retries
from app.llm.prompting import build_system_prompt
from app.llm.runtime import ensure_provider_ready
from app.tools.content_library_tool import apply_content_list_defaults
from app.tools.registry import get_tool_definitions


DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "tool_choice" / "cases.json"


def eval_config_from_args(args: argparse.Namespace) -> LLMConfig:
    default = LLMConfig.from_env()
    provider = args.provider or os.getenv("EVAL_PROVIDER") or default.provider
    explicit_base_url = args.base_url or os.getenv("EVAL_BASE_URL")
    return LLMConfig(
        provider=provider,
        model_name=args.model or os.getenv("EVAL_MODEL") or default.model_name,
        base_url=explicit_base_url or (default.base_url if provider == "llama_cpp" else ""),
        api_key=args.api_key or os.getenv("EVAL_API_KEY") or default.api_key or "local",
        start_script=args.start_script or os.getenv("EVAL_START_SCRIPT") or default.start_script,
    )


def prepare_eval_provider(config: LLMConfig, *, auto_start: bool) -> LLMConfig:
    if auto_start and config.provider.strip().lower() == "llama_cpp":
        return ensure_provider_ready(config)
    return config


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Expected tool-choice cases file to contain a list.")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    for key in ("id", "user", "expected_tool"):
        if not str(case.get(key) or "").strip():
            raise ValueError(f"Tool-choice case is missing required field: {key}")
    expected_arguments = case.get("expected_arguments")
    if expected_arguments is not None and not isinstance(expected_arguments, dict):
        raise ValueError(f"Tool-choice case {case['id']} expected_arguments must be an object.")


def parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    parsed = json.loads(raw_arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return parsed


def effective_tool_arguments(tool_name: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "content_list":
        return apply_content_list_defaults(arguments)
    return dict(arguments)


def evaluate_arguments(
    *,
    expected: dict[str, Any] | None,
    actual: dict[str, Any],
) -> dict[str, Any]:
    if not expected:
        return {"passed": True, "checks": []}

    checks: list[dict[str, Any]] = []
    for field, expected_value in expected.get("must_equal", {}).items():
        actual_value = actual.get(field)
        checks.append(
            {
                "type": "must_equal",
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
                "passed": actual_value == expected_value,
            }
        )
    for field, expected_values in expected.get("must_include", {}).items():
        actual_value = actual.get(field)
        actual_values = actual_value if isinstance(actual_value, list) else [actual_value]
        missing = [value for value in expected_values if value not in actual_values]
        checks.append(
            {
                "type": "must_include",
                "field": field,
                "expected": expected_values,
                "actual": actual_value,
                "passed": not missing,
                "missing": missing,
            }
        )
    for field, forbidden_values in expected.get("must_not_include", {}).items():
        actual_value = actual.get(field)
        actual_values = actual_value if isinstance(actual_value, list) else [actual_value]
        present = [value for value in forbidden_values if value in actual_values]
        checks.append(
            {
                "type": "must_not_include",
                "field": field,
                "forbidden": forbidden_values,
                "actual": actual_value,
                "passed": not present,
                "present": present,
            }
        )

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def run_case(
    case: dict[str, Any],
    *,
    config: LLMConfig,
    temperature: float,
    top_p: float | None,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": case["user"]},
        ],
        "temperature": temperature,
        "max_tokens": 256,
        "tools": get_tool_definitions(),
        "tool_choice": "auto",
    }
    if top_p is not None:
        request_kwargs["top_p"] = top_p

    response = create_completion_with_retries(config, request_kwargs)
    choice = response.choices[0]
    message = choice.message
    tool_calls = getattr(message, "tool_calls", None) or []
    called_tools = [tool_call.function.name for tool_call in tool_calls]
    first_tool = called_tools[0] if called_tools else None
    first_arguments = tool_calls[0].function.arguments if tool_calls else None
    parsed_arguments = parse_tool_arguments(first_arguments)
    effective_arguments = effective_tool_arguments(first_tool, parsed_arguments)
    content = getattr(message, "content", "") or ""
    tool_pass = first_tool == case["expected_tool"]
    argument_evaluation = evaluate_arguments(
        expected=case.get("expected_arguments"),
        actual=effective_arguments,
    )
    argument_pass = bool(argument_evaluation["passed"])
    passed = tool_pass and argument_pass

    return {
        "id": case["id"],
        "passed": passed,
        "tool_pass": tool_pass,
        "argument_pass": argument_pass,
        "expected_tool": case["expected_tool"],
        "first_tool": first_tool,
        "first_arguments": first_arguments,
        "parsed_arguments": parsed_arguments,
        "effective_arguments": effective_arguments,
        "argument_checks": argument_evaluation["checks"],
        "called_tools": called_tools,
        "finish_reason": getattr(choice, "finish_reason", None),
        "assistant_content": content,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a model chooses the expected Scratchpad tool for simple requests."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--provider", help="Provider for the model under test. Defaults to EVAL_PROVIDER or LLM_PROVIDER.")
    parser.add_argument("--model", help="Model id for the model under test. Defaults to EVAL_MODEL or LLM_MODEL.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for the model under test.")
    parser.add_argument("--api-key", help="API key for the model under test. Prefer env vars for real keys.")
    parser.add_argument("--start-script", help="llama.cpp start script for the model under test.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not auto-start llama.cpp even when provider is llama_cpp.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSONL instead of text.")
    args = parser.parse_args(argv)

    config = prepare_eval_provider(eval_config_from_args(args), auto_start=not args.no_auto_start)
    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No tool-choice eval case found with id: {args.case_id}")

    failures = 0
    for case in cases:
        result = run_case(case, config=config, temperature=args.temperature, top_p=args.top_p)
        failures += 0 if result["passed"] else 1
        if args.json:
            print(json.dumps(result, ensure_ascii=True))
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"CASE {result['id']}: {status} "
                f"expected={result['expected_tool']} first_tool={result['first_tool']}"
            )
            if result["called_tools"]:
                print(f"  called_tools: {result['called_tools']}")
                print(f"  first_arguments: {result['first_arguments']}")
                if result["effective_arguments"] != result["parsed_arguments"]:
                    print(f"  effective_arguments: {result['effective_arguments']}")
                if result["argument_checks"]:
                    print(f"  argument_pass: {result['argument_pass']}")
                    for check in result["argument_checks"]:
                        print(f"  arg_check: {check}")
            elif result["assistant_content"]:
                print(f"  assistant_content: {result['assistant_content']}")
            if result["finish_reason"]:
                print(f"  finish_reason: {result['finish_reason']}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
