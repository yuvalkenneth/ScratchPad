from __future__ import annotations

import argparse
import json
import os
import re
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
from app.tools.registry import get_tool_definitions


DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "retention" / "cases.json"
RETENTION_LABELS = ("pass", "degraded", "fail")
RETENTION_KINDS = (
    "conceptual",
    "instruction_following",
    "url_abstention",
    "simple_coding_math",
    "general_assistant",
)


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
        raise ValueError("Expected retention cases file to contain a list.")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    for key in ("id", "retention_kind", "user", "expectations"):
        if key not in case:
            raise ValueError(f"Retention case is missing required field: {key}")
    if case["retention_kind"] not in RETENTION_KINDS:
        raise ValueError(f"Unsupported retention_kind for {case['id']}: {case['retention_kind']}")
    if not isinstance(case["expectations"], dict):
        raise ValueError(f"Retention case {case['id']} expectations must be an object.")


def count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text.strip()) if part.strip()])


def evaluate_text_expectations(content: str, expectations: dict[str, Any]) -> list[dict[str, Any]]:
    lowered = content.lower()
    checks: list[dict[str, Any]] = []

    if expected_any := expectations.get("must_contain_any"):
        matches = [term for term in expected_any if str(term).lower() in lowered]
        checks.append(
            {
                "type": "must_contain_any",
                "expected": expected_any,
                "matched": matches,
                "passed": bool(matches),
            }
        )

    if expected_all := expectations.get("must_contain_all"):
        missing = [term for term in expected_all if str(term).lower() not in lowered]
        checks.append(
            {
                "type": "must_contain_all",
                "expected": expected_all,
                "missing": missing,
                "passed": not missing,
            }
        )

    if forbidden := expectations.get("must_not_contain"):
        present = [term for term in forbidden if str(term).lower() in lowered]
        checks.append(
            {
                "type": "must_not_contain",
                "forbidden": forbidden,
                "present": present,
                "passed": not present,
            }
        )

    if max_words := expectations.get("max_words"):
        word_count = len(re.findall(r"\S+", content))
        checks.append(
            {
                "type": "max_words",
                "max": max_words,
                "actual": word_count,
                "passed": word_count <= int(max_words),
            }
        )

    if max_lines := expectations.get("max_lines"):
        line_count = len([line for line in content.splitlines() if line.strip()])
        checks.append(
            {
                "type": "max_lines",
                "max": max_lines,
                "actual": line_count,
                "passed": line_count <= int(max_lines),
            }
        )

    if max_sentences := expectations.get("max_sentences"):
        sentence_count = count_sentences(content)
        checks.append(
            {
                "type": "max_sentences",
                "max": max_sentences,
                "actual": sentence_count,
                "passed": sentence_count <= int(max_sentences),
            }
        )

    return checks


def retention_label(*, tool_called: bool, content: str, checks: list[dict[str, Any]]) -> str:
    if tool_called:
        return "fail"
    if not content.strip():
        return "fail"
    if checks and all(check["passed"] for check in checks):
        return "pass"
    return "degraded"


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

    started_at = time.perf_counter()
    response = create_completion_with_retries(config, request_kwargs)
    latency_seconds = time.perf_counter() - started_at
    choice = response.choices[0]
    message = choice.message
    tool_calls = getattr(message, "tool_calls", None) or []
    called_tools = [tool_call.function.name for tool_call in tool_calls]
    content = getattr(message, "content", "") or ""
    checks = evaluate_text_expectations(content, case["expectations"])
    label = retention_label(tool_called=bool(called_tools), content=content, checks=checks)

    return {
        "id": case["id"],
        "retention_kind": case["retention_kind"],
        "label": label,
        "passed": label == "pass",
        "tool_called": bool(called_tools),
        "called_tools": called_tools,
        "content_checks": checks,
        "assistant_content": content,
        "finish_reason": getattr(choice, "finish_reason", None),
        "latency_seconds": round(latency_seconds, 3),
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    profile: str | None = None,
) -> dict[str, Any]:
    total = len(results)
    label_counts = {label: 0 for label in RETENTION_LABELS}
    kind_counts: dict[str, dict[str, int]] = {}
    for result in results:
        label_counts[result["label"]] += 1
        kind_counts.setdefault(result["retention_kind"], {label: 0 for label in RETENTION_LABELS})
        kind_counts[result["retention_kind"]][result["label"]] += 1
    latencies = [float(result["latency_seconds"]) for result in results]
    tool_false_positive_count = sum(1 for result in results if result["tool_called"])
    return {
        "type": "retention_report",
        "profile": profile,
        "provider": provider,
        "model": model,
        "total_cases": total,
        "label_counts": label_counts,
        "retention_pass_rate": round(label_counts["pass"] / total, 4) if total else 0.0,
        "tool_false_positive_rate": round(tool_false_positive_count / total, 4) if total else 0.0,
        "by_kind": kind_counts,
        "latency": {
            "average_seconds": round(sum(latencies) / total, 3) if total else 0.0,
            "max_seconds": round(max(latencies), 3) if latencies else 0.0,
            "total_seconds": round(sum(latencies), 3),
        },
        "results": results,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a tuned model retains basic no-tool assistant behavior."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--profile", help="Named model profile from config/models.json or config/models.local.json.")
    parser.add_argument("--provider", help="Provider for the model under test. Defaults to EVAL_PROVIDER or LLM_PROVIDER.")
    parser.add_argument("--model", help="Model id for the model under test. Defaults to EVAL_MODEL or LLM_MODEL.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for the model under test.")
    parser.add_argument("--api-key", help="API key for the model under test. Prefer env vars for real keys.")
    parser.add_argument("--start-script", help="llama.cpp start script for the model under test.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--report", type=Path, help="Write a JSON report with aggregate metrics.")
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
            raise ValueError(f"No retention eval case found with id: {args.case_id}")

    results: list[dict[str, Any]] = []
    for case in cases:
        result = run_case(case, config=config, temperature=args.temperature, top_p=args.top_p)
        results.append(result)
        if args.json:
            print(json.dumps(result, ensure_ascii=True))
        else:
            status = result["label"].upper()
            print(f"CASE {result['id']}: {status} kind={result['retention_kind']}")
            if result["called_tools"]:
                print(f"  called_tools: {result['called_tools']}")
            for check in result["content_checks"]:
                print(f"  content_check: {check}")

    report = build_report(
        results,
        model=config.model_name,
        provider=config.provider,
        profile=args.profile or os.getenv("EVAL_PROFILE"),
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=True))
    else:
        print(
            "SUMMARY: "
            f"{report['label_counts']['pass']}/{report['total_cases']} pass, "
            f"degraded {report['label_counts']['degraded']}, "
            f"fail {report['label_counts']['fail']}, "
            f"tool_false_positive_rate {report['tool_false_positive_rate']:.2f}"
        )

    return 1 if report["label_counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(run())
