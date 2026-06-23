from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.config import LLMConfig
from app.llm.openai_compatible import create_completion_with_retries
from app.llm.prompting import build_system_prompt
from app.llm.profiles import config_from_profile
from app.llm.runtime import ensure_provider_ready
from app.tools.content_library_tool import apply_content_list_defaults
from app.tools.registry import get_tool_definitions
from scripts.experiment_tracking import mlflow_log_report


DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "tool_choice" / "cases.json"
NO_TOOL_LABEL = "no_tool"
SPLIT_LABELS = ("train", "validation", "heldout")
DIFFICULTY_LABELS = ("easy", "medium", "hard", "ambiguous")
GROUP_FIELDS = ("split", "difficulty", "category", "intent", "context_kind")


def eval_config_from_args(args: argparse.Namespace) -> LLMConfig:
    profile_name = args.profile or os.getenv("EVAL_PROFILE")
    if profile_name:
        return config_from_profile(
            profile_name,
            provider=args.provider or os.getenv("EVAL_PROVIDER"),
            model=args.model or os.getenv("EVAL_MODEL"),
            base_url=args.base_url or os.getenv("EVAL_BASE_URL"),
            api_key=args.api_key or os.getenv("EVAL_API_KEY"),
            start_script=args.start_script or os.getenv("EVAL_START_SCRIPT"),
        )

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
    for key in ("id", "expected_tool"):
        if not str(case.get(key) or "").strip():
            raise ValueError(f"Tool-choice case is missing required field: {key}")
    has_user = bool(str(case.get("user") or "").strip())
    messages = case.get("messages")
    has_messages = isinstance(messages, list) and bool(messages)
    if not has_user and not has_messages:
        raise ValueError(f"Tool-choice case {case['id']} needs user or messages.")
    if has_messages:
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Tool-choice case {case['id']} message {index} must be an object.")
            if message.get("role") not in {"user", "assistant", "tool"}:
                raise ValueError(f"Tool-choice case {case['id']} message {index} has invalid role.")
            if not str(message.get("content") or "").strip():
                raise ValueError(f"Tool-choice case {case['id']} message {index} has empty content.")
    expected_arguments = case.get("expected_arguments")
    if expected_arguments is not None and not isinstance(expected_arguments, dict):
        raise ValueError(f"Tool-choice case {case['id']} expected_arguments must be an object.")
    for key in ("intent", "category", "difficulty", "retention_kind", "split"):
        if key in case and not str(case[key]).strip():
            raise ValueError(f"Tool-choice case {case['id']} has empty metadata field: {key}")
    if "difficulty" in case and case["difficulty"] not in DIFFICULTY_LABELS:
        raise ValueError(
            f"Tool-choice case {case['id']} has invalid difficulty {case['difficulty']!r}; "
            f"expected one of {', '.join(DIFFICULTY_LABELS)}."
        )
    if "split" in case and case["split"] not in SPLIT_LABELS:
        raise ValueError(
            f"Tool-choice case {case['id']} has invalid split {case['split']!r}; "
            f"expected one of {', '.join(SPLIT_LABELS)}."
        )


def filter_cases_by_split(cases: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    if split is None:
        return cases
    filtered = [case for case in cases if case.get("split") == split]
    if not filtered:
        raise ValueError(f"No tool-choice eval cases found for split: {split}")
    return filtered


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


def uses_tool_defaults(tool_name: str | None, parsed: dict[str, Any], effective: dict[str, Any]) -> bool:
    return tool_name == "content_list" and parsed != effective


def request_messages_from_case(case: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    if "messages" in case:
        messages.extend(case["messages"])
    else:
        messages.append({"role": "user", "content": case["user"]})
    return messages


def run_case(
    case: dict[str, Any],
    *,
    config: LLMConfig,
    temperature: float,
    top_p: float | None,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": request_messages_from_case(case),
        "temperature": temperature,
        "max_tokens": 256,
        "tools": get_tool_definitions(),
        "tool_choice": "auto",
    }
    if top_p is not None:
        request_kwargs["top_p"] = top_p

    started_at = time.perf_counter()
    response = create_completion_with_retries(config, request_kwargs)
    latency_seconds = time.perf_counter() - started_at
    choice = response.choices[0]
    message = choice.message
    tool_calls = getattr(message, "tool_calls", None) or []
    called_tools = [tool_call.function.name for tool_call in tool_calls]
    first_tool = called_tools[0] if called_tools else None
    first_arguments = tool_calls[0].function.arguments if tool_calls else None
    parse_error = None
    try:
        parsed_arguments = parse_tool_arguments(first_arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        parsed_arguments = {}
        parse_error = str(exc)
    effective_arguments = effective_tool_arguments(first_tool, parsed_arguments)
    content = getattr(message, "content", "") or ""
    expected_tool = case["expected_tool"]
    actual_tool = first_tool or NO_TOOL_LABEL
    tool_pass = actual_tool == expected_tool
    argument_evaluation = evaluate_arguments(
        expected=case.get("expected_arguments"),
        actual=effective_arguments,
    )
    argument_json_valid = parse_error is None
    argument_pass = bool(argument_evaluation["passed"]) and argument_json_valid
    passed = tool_pass and argument_pass
    failure_types = classify_failure_types(
        expected_tool=expected_tool,
        actual_tool=actual_tool,
        argument_json_valid=argument_json_valid,
        argument_pass=argument_pass,
        extra_tool_count=max(0, len(called_tools) - 1),
    )

    return {
        "id": case["id"],
        "case_metadata": {
            field: case[field]
            for field in GROUP_FIELDS
            if field in case
        },
        "passed": passed,
        "tool_pass": tool_pass,
        "argument_pass": argument_pass,
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "first_tool": first_tool,
        "first_arguments": first_arguments,
        "parsed_arguments": parsed_arguments,
        "effective_arguments": effective_arguments,
        "argument_checks": argument_evaluation["checks"],
        "argument_json_valid": argument_json_valid,
        "argument_parse_error": parse_error,
        "used_tool_defaults": uses_tool_defaults(first_tool, parsed_arguments, effective_arguments),
        "called_tools": called_tools,
        "extra_tool_count": max(0, len(called_tools) - 1),
        "failure_types": failure_types,
        "finish_reason": getattr(choice, "finish_reason", None),
        "assistant_content": content,
        "latency_seconds": round(latency_seconds, 3),
    }


def classify_failure_types(
    *,
    expected_tool: str,
    actual_tool: str,
    argument_json_valid: bool,
    argument_pass: bool,
    extra_tool_count: int,
) -> list[str]:
    failures: list[str] = []
    if actual_tool != expected_tool:
        if actual_tool == NO_TOOL_LABEL:
            failures.append("no_tool_false_negative")
        elif expected_tool == NO_TOOL_LABEL:
            failures.append("tool_false_positive")
        else:
            failures.append("wrong_tool")
    if not argument_json_valid:
        failures.append("invalid_tool_arguments_json")
    elif not argument_pass:
        failures.append("argument_mismatch")
    if extra_tool_count > 0:
        failures.append("extra_tool_call")
    return failures


def build_report(
    results: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    profile: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    labels = sorted(
        {
            result["expected_tool"]
            for result in results
        }
        | {result["actual_tool"] for result in results}
    )
    total = len(results)
    confusion_matrix = {
        expected: {actual: 0 for actual in labels}
        for expected in labels
    }
    for result in results:
        confusion_matrix[result["expected_tool"]][result["actual_tool"]] += 1

    per_class = {}
    for label in labels:
        true_positive = confusion_matrix[label][label]
        false_positive = sum(
            confusion_matrix[expected][label] for expected in labels if expected != label
        )
        false_negative = sum(
            confusion_matrix[label][actual] for actual in labels if actual != label
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion_matrix[label].values()),
        }

    called_tool_count = sum(1 for result in results if result["actual_tool"] != NO_TOOL_LABEL)
    expected_tool_count = sum(1 for result in results if result["expected_tool"] != NO_TOOL_LABEL)
    wrong_tool_count = sum(1 for result in results if not result["tool_pass"] and result["actual_tool"] != NO_TOOL_LABEL)
    no_tool_false_negative_count = sum(
        1
        for result in results
        if result["expected_tool"] != NO_TOOL_LABEL and result["actual_tool"] == NO_TOOL_LABEL
    )
    tool_false_positive_count = sum(
        1
        for result in results
        if result["expected_tool"] == NO_TOOL_LABEL and result["actual_tool"] != NO_TOOL_LABEL
    )
    default_reliance_count = sum(1 for result in results if result["used_tool_defaults"])
    constrained_results = [result for result in results if result["argument_checks"]]
    latencies = [float(result["latency_seconds"]) for result in results]
    failure_type_counts: dict[str, int] = {}
    for result in results:
        for failure_type in result.get("failure_types", []):
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
    groups = {
        field: build_group_metrics(results, field)
        for field in GROUP_FIELDS
    }
    return {
        "type": "tool_choice_report",
        "profile": profile,
        "provider": provider,
        "model": model,
        "split": split,
        "total_cases": total,
        "passed_cases": sum(1 for result in results if result["passed"]),
        "tool_selection_accuracy": round(
            sum(1 for result in results if result["tool_pass"]) / total,
            4,
        )
        if total
        else 0.0,
        "argument_accuracy": round(
            sum(1 for result in constrained_results if result["argument_pass"]) / len(constrained_results),
            4,
        )
        if constrained_results
        else None,
        "overall_accuracy": round(sum(1 for result in results if result["passed"]) / total, 4)
        if total
        else 0.0,
        "argument_json_validity_rate": round(
            sum(1 for result in results if result["argument_json_valid"]) / total,
            4,
        )
        if total
        else 0.0,
        "call_rate": round(called_tool_count / total, 4) if total else 0.0,
        "expected_call_rate": round(expected_tool_count / total, 4) if total else 0.0,
        "wrong_tool_rate": round(wrong_tool_count / total, 4) if total else 0.0,
        "no_tool_false_negative_rate": round(no_tool_false_negative_count / total, 4) if total else 0.0,
        "tool_false_positive_rate": round(tool_false_positive_count / total, 4) if total else 0.0,
        "extra_tool_rate": round(
            sum(1 for result in results if result["extra_tool_count"] > 0) / total,
            4,
        )
        if total
        else 0.0,
        "default_reliance_rate": round(default_reliance_count / total, 4) if total else 0.0,
        "failure_type_counts": failure_type_counts,
        "groups": groups,
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
        "latency": {
            "average_seconds": round(sum(latencies) / total, 3) if total else 0.0,
            "max_seconds": round(max(latencies), 3) if latencies else 0.0,
            "total_seconds": round(sum(latencies), 3),
        },
        "results": results,
    }


def build_group_metrics(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        value = result.get("case_metadata", {}).get(field, "<missing>")
        grouped.setdefault(str(value), []).append(result)
    return {
        value: summarize_result_subset(subset)
        for value, subset in sorted(grouped.items())
    }


def summarize_result_subset(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    constrained_results = [result for result in results if result["argument_checks"]]
    latencies = [float(result["latency_seconds"]) for result in results]
    failure_type_counts: dict[str, int] = {}
    for result in results:
        for failure_type in result.get("failure_types", []):
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
    return {
        "total_cases": total,
        "passed_cases": sum(1 for result in results if result["passed"]),
        "overall_accuracy": round(sum(1 for result in results if result["passed"]) / total, 4)
        if total
        else 0.0,
        "tool_selection_accuracy": round(sum(1 for result in results if result["tool_pass"]) / total, 4)
        if total
        else 0.0,
        "argument_accuracy": round(
            sum(1 for result in constrained_results if result["argument_pass"]) / len(constrained_results),
            4,
        )
        if constrained_results
        else None,
        "argument_json_validity_rate": round(
            sum(1 for result in results if result["argument_json_valid"]) / total,
            4,
        )
        if total
        else 0.0,
        "tool_false_positive_rate": round(
            sum(
                1
                for result in results
                if result["expected_tool"] == NO_TOOL_LABEL and result["actual_tool"] != NO_TOOL_LABEL
            )
            / total,
            4,
        )
        if total
        else 0.0,
        "no_tool_false_negative_rate": round(
            sum(
                1
                for result in results
                if result["expected_tool"] != NO_TOOL_LABEL and result["actual_tool"] == NO_TOOL_LABEL
            )
            / total,
            4,
        )
        if total
        else 0.0,
        "wrong_tool_rate": round(
            sum(
                1
                for result in results
                if not result["tool_pass"] and result["actual_tool"] != NO_TOOL_LABEL
            )
            / total,
            4,
        )
        if total
        else 0.0,
        "extra_tool_rate": round(
            sum(1 for result in results if result["extra_tool_count"] > 0) / total,
            4,
        )
        if total
        else 0.0,
        "failure_type_counts": failure_type_counts,
        "latency": {
            "average_seconds": round(sum(latencies) / total, 3) if total else 0.0,
            "max_seconds": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a model chooses the expected Scratchpad tool for simple requests."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument(
        "--split",
        choices=SPLIT_LABELS,
        help="Evaluate only cases assigned to this canonical split.",
    )
    parser.add_argument("--profile", help="Named model profile from config/models.json or config/models.local.json.")
    parser.add_argument("--provider", help="Provider for the model under test. Defaults to EVAL_PROVIDER or LLM_PROVIDER.")
    parser.add_argument("--model", help="Model id for the model under test. Defaults to EVAL_MODEL or LLM_MODEL.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for the model under test.")
    parser.add_argument("--api-key", help="API key for the model under test. Prefer env vars for real keys.")
    parser.add_argument("--start-script", help="llama.cpp start script for the model under test.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--report", type=Path, help="Write a JSON report with aggregate metrics.")
    parser.add_argument("--mlflow-experiment", help="Optionally log this report to an MLflow experiment.")
    parser.add_argument("--mlflow-run-name", help="Optional MLflow run name.")
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        default=[],
        help="Additional artifact path to attach when MLflow logging is enabled.",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not auto-start llama.cpp even when provider is llama_cpp.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSONL instead of text.")
    args = parser.parse_args(argv)

    config = prepare_eval_provider(eval_config_from_args(args), auto_start=not args.no_auto_start)
    cases = load_cases(args.cases)
    cases = filter_cases_by_split(cases, args.split)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No tool-choice eval case found with id: {args.case_id}")

    results: list[dict[str, Any]] = []
    for case in cases:
        result = run_case(case, config=config, temperature=args.temperature, top_p=args.top_p)
        results.append(result)
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

    report = build_report(
        results,
        model=config.model_name,
        provider=config.provider,
        profile=args.profile or os.getenv("EVAL_PROFILE"),
        split=args.split,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.mlflow_experiment:
        run_id = mlflow_log_report(
            report,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            report_path=args.report,
            artifacts=args.artifact,
            params={
                "cases": str(args.cases),
                "case": args.case_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        )
        print(f"Logged MLflow run: {run_id}", file=sys.stderr)
    if args.json:
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=True))
    else:
        print(
            "SUMMARY: "
            f"{report['passed_cases']}/{report['total_cases']} passed, "
            f"tool_accuracy {report['tool_selection_accuracy']:.2f}, "
            f"argument_accuracy {report['argument_accuracy']}, "
            f"avg_latency {report['latency']['average_seconds']}s"
        )

    failures = sum(1 for result in results if not result["passed"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
