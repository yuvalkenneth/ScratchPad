from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.fetchers.common import estimate_time_minutes
from app.llm.config import LLMConfig
from app.llm.openai_compatible import complete_text
from app.llm.profiles import config_from_profile
from app.llm.runtime import ensure_provider_ready
from app.content import build_content_profile_payload
import app.tools.url_analyze_tool as url_analyze_tool
import app.tools.youtube_analyze_tool as youtube_analyze_tool


DEFAULT_FIXTURE_PATH = REPO_ROOT / "evals" / "content_profiles" / "cases.json"
REQUIRED_CASE_KEYS = {"id", "source_type", "url", "title", "input", "expected"}
REQUIRED_INPUT_KEYS = {"text"}
REQUIRED_PROFILE_KEYS = {
    "status",
    "source_type",
    "source_id",
    "url",
    "title",
    "summary",
    "subject",
    "depth_level",
    "categories",
    "estimated_time_minutes",
    "confidence",
}
JUDGE_RESULT_KEYS = {
    "summary_faithful",
    "summary_covers_main_points",
    "subject_reasonable",
    "categories_useful",
    "depth_reasonable",
    "time_reasonable",
    "overall_useful",
    "notes",
}


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


@dataclass
class JudgeConfig:
    provider: str
    model_name: str
    base_url: str
    api_key: str
    temperature: float
    top_p: float

    @classmethod
    def from_args(cls, args: argparse.Namespace, default: LLMConfig) -> "JudgeConfig":
        profile_name = args.judge_profile or os.getenv("EVAL_JUDGE_PROFILE")
        if profile_name:
            profile_config = config_from_profile(
                profile_name,
                provider=args.judge_provider or os.getenv("EVAL_JUDGE_PROVIDER"),
                model=args.judge_model or os.getenv("EVAL_JUDGE_MODEL"),
                base_url=args.judge_base_url or os.getenv("EVAL_JUDGE_BASE_URL"),
                api_key=args.judge_api_key or os.getenv("EVAL_JUDGE_API_KEY"),
            )
            provider = profile_config.provider
            model_name = profile_config.model_name
            base_url = profile_config.base_url
            api_key = profile_config.api_key
        else:
            provider = args.judge_provider or os.getenv("EVAL_JUDGE_PROVIDER") or default.provider
            model_name = args.judge_model or os.getenv("EVAL_JUDGE_MODEL") or default.model_name
            gemini_base_url = os.getenv("GEMINI_BASE_URL") if model_name.startswith("gemini") else None
            gemini_api_key = os.getenv("GEMINI_API_KEY") if model_name.startswith("gemini") else None
            base_url = args.judge_base_url or os.getenv("EVAL_JUDGE_BASE_URL") or gemini_base_url or default.base_url
            api_key = args.judge_api_key or os.getenv("EVAL_JUDGE_API_KEY") or gemini_api_key or default.api_key or "local"
        temperature = float(
            args.judge_temperature
            if args.judge_temperature is not None
            else os.getenv("EVAL_JUDGE_TEMPERATURE", "0.0")
        )
        top_p = float(
            args.judge_top_p
            if args.judge_top_p is not None
            else os.getenv("EVAL_JUDGE_TOP_P", "1.0")
        )
        return cls(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
        )


def load_cases(path: Path = DEFAULT_FIXTURE_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Expected fixture file to contain a list of cases.")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    missing = sorted(key for key in REQUIRED_CASE_KEYS if key not in case)
    if missing:
        raise ValueError(f"Case is missing required fields: {', '.join(missing)}")
    if case["source_type"] not in {"web", "github", "reddit", "youtube"}:
        raise ValueError(f"Unsupported source_type for eval case {case['id']}: {case['source_type']}")
    input_data = case["input"]
    if not isinstance(input_data, dict):
        raise ValueError(f"Case {case['id']} input must be an object.")
    missing_input = sorted(key for key in REQUIRED_INPUT_KEYS if key not in input_data)
    if missing_input:
        raise ValueError(f"Case {case['id']} input is missing: {', '.join(missing_input)}")
    if not str(input_data["text"]).strip():
        raise ValueError(f"Case {case['id']} input.text must not be empty.")
    expected = case["expected"]
    if not isinstance(expected, dict):
        raise ValueError(f"Case {case['id']} expected must be an object.")
    if "depth_level_options" in expected:
        options = expected["depth_level_options"]
        if not isinstance(options, list) or not options:
            raise ValueError(f"Case {case['id']} expected.depth_level_options must be a non-empty list.")
        invalid = [option for option in options if option not in {"light", "medium", "deep"}]
        if invalid:
            raise ValueError(f"Case {case['id']} has invalid depth options: {invalid}")
    elif expected.get("depth_level") not in {"light", "medium", "deep"}:
        raise ValueError(f"Case {case['id']} expected.depth_level must be light, medium, or deep.")


def analyze_case(
    case: dict[str, Any],
    *,
    config: LLMConfig | None = None,
    complete_text: Callable[[LLMConfig, list[dict[str, str]], int], str] | None = None,
) -> dict[str, Any]:
    input_data = case["input"]
    text = str(input_data["text"]).strip()
    title = str(input_data.get("title") or case["title"]).strip()
    source_type = str(case["source_type"])
    estimated_time_minutes = int(
        input_data.get("estimated_time_minutes") or estimate_time_minutes(text)
    )

    config = config or LLMConfig.from_env()
    if complete_text is None:
        complete_text = _complete_text

    if source_type == "youtube":
        messages = [
            {
                "role": "system",
                "content": youtube_analyze_tool._analysis_prompt(
                    "content_profile",
                    question=None,
                    include_timestamps=False,
                ),
            },
            {"role": "user", "content": text},
        ]
        trust_model_time = False
    else:
        system_prompt = url_analyze_tool._analysis_prompt(title, str(case["url"]))
        if source_type == "github":
            system_prompt += (
                "\nFor GitHub repositories, estimate the time needed to understand the "
                "project purpose, setup, architecture, and whether it is worth using. "
                "Choose categories for learning and recommendation usefulness, not only "
                "implementation languages."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        trust_model_time = True

    raw_analysis = complete_text(config, messages, 500)
    return build_content_profile_payload(
        source_type=source_type,
        source_id=case.get("source_id"),
        url=str(case["url"]),
        title=title,
        estimated_time_minutes=estimated_time_minutes,
        raw_analysis=raw_analysis,
        trust_model_time=trust_model_time,
        metadata=input_data.get("metadata"),
        extra_fields={
            "eval_case_id": case["id"],
            "word_count": len(text.split()),
        },
    )


def _complete_text(config: LLMConfig, messages: list[dict[str, str]], max_tokens: int) -> str:
    return url_analyze_tool._complete_text(config, messages, max_tokens=max_tokens)


def judge_case(
    profile: dict[str, Any],
    case: dict[str, Any],
    *,
    config: JudgeConfig | LLMConfig | None = None,
    complete_text: Callable[[Any, list[dict[str, str]], int], str] | None = None,
) -> dict[str, Any]:
    config = config or LLMConfig.from_env()
    if complete_text is None:
        complete_text = _complete_judge_text

    raw_judgment = complete_text(config, build_judge_messages(profile, case), 4096)
    try:
        parsed = _extract_json_object(raw_judgment)
    except ValueError:
        return {
            "status": "error",
            "error": "Judge did not return a JSON object.",
            "raw_judgment": raw_judgment,
        }

    result: dict[str, Any] = {"status": "completed"}
    for key in JUDGE_RESULT_KEYS:
        if key == "notes":
            result[key] = str(parsed.get(key) or "").strip()
        else:
            result[key] = bool(parsed.get(key))
    result["raw_judgment"] = raw_judgment
    result["judge_model"] = getattr(config, "model_name", None)
    result["judge_provider"] = getattr(config, "provider", None)
    return result


def _complete_judge_text(config: JudgeConfig | LLMConfig, messages: list[dict[str, str]], max_tokens: int) -> str:
    return complete_text(
        config,
        messages,
        max_tokens=max_tokens,
        temperature=getattr(config, "temperature", 0.7),
        top_p=getattr(config, "top_p", 1.0),
    )


def build_judge_messages(profile: dict[str, Any], case: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You are evaluating content profiles for Scratchpad, a local-first learning inbox.",
            (
                "A content_profile is saved metadata for a source and is later used for "
                "search, filtering, recommendation, and deciding what to read, watch, or explore next."
            ),
            "Judge whether the actual profile would be useful as saved library metadata for this source.",
            "Criteria:",
            "1. summary_faithful: The summary only claims things supported by the source.",
            (
                "2. summary_covers_main_points: The summary captures the main learning value "
                "or central ideas, not incidental details."
            ),
            "3. subject_reasonable: The subject is a concise primary topic useful for retrieval.",
            (
                "4. categories_useful: Categories are specific topical/domain labels useful for "
                "recommendation, not source format or generic labels."
            ),
            "5. depth_reasonable: Depth matches the conceptual difficulty and prerequisite knowledge.",
            (
                "6. time_reasonable: The time estimate is plausible for consuming and understanding "
                "the source enough to decide whether to revisit or act on it."
            ),
            "7. overall_useful: The profile is good enough to save and use for future recommendation.",
            "Do not require exact wording. Accept reasonable paraphrases and adjacent category labels.",
            "Do not forgive hallucinated claims or source details not supported by the input.",
            "Return JSON only with this exact schema:",
            (
                '{"summary_faithful":true,"summary_covers_main_points":true,'
                '"subject_reasonable":true,"categories_useful":true,'
                '"depth_reasonable":true,"time_reasonable":true,'
                '"overall_useful":true,"notes":"short explanation"}'
            ),
        ]
    )
    payload = {
        "source": {
            "id": case["id"],
            "source_type": case["source_type"],
            "url": case["url"],
            "title": case["title"],
            "text": case["input"]["text"],
            "metadata": case["input"].get("metadata", {}),
        },
        "expected_rubric": case["expected"],
        "actual_profile": profile,
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found.")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON was not an object.")
    return parsed


def evaluate_case(profile: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    checks = {
        "required_fields": _check(
            REQUIRED_PROFILE_KEYS.issubset(profile.keys()),
            severity="hard",
        ),
        "source_type": _check(
            profile.get("source_type") == case.get("source_type"),
            severity="hard",
        ),
        "source_id": _check(
            profile.get("source_id") == case.get("source_id"),
            severity="hard",
        ),
        "url": _check(profile.get("url") == case.get("url"), severity="hard"),
        "title": _check(profile.get("title") == case.get("title"), severity="hard"),
        "subject": _check(
            _matches_any(str(profile.get("subject") or ""), expected.get("subject_options", [])),
            severity="soft",
        ),
        "categories": _check(
            _has_category_match(profile.get("categories", []), expected),
            severity="soft",
        ),
        "depth_level": _check(
            profile.get("depth_level") in expected_depth_options(expected),
            severity="hard",
        ),
        "estimated_time": _check(
            _in_range(
                profile.get("estimated_time_minutes"),
                expected.get("estimated_time_minutes_range"),
            ),
            severity="soft",
        ),
        "summary_should_cover": _check(
            _summary_covers(
                str(profile.get("summary") or ""),
                expected.get("summary_should_cover", []),
            ),
            severity="soft",
        ),
        "summary_concept_groups": _check(
            _summary_covers_concept_groups(
                str(profile.get("summary") or ""),
                expected.get("summary_concept_groups", []),
            ),
            severity="soft",
        ),
        "summary_must_not_claim": _check(
            _summary_avoids(
                str(profile.get("summary") or ""),
                expected.get("summary_must_not_claim", []),
            ),
            severity="hard",
        ),
        "confidence": _check(
            _at_least(profile.get("confidence"), expected.get("min_confidence", 0.0)),
            severity="soft",
        ),
    }
    total = len(checks)
    passed_count = sum(1 for check in checks.values() if check["passed"])
    hard_failures = [
        name for name, check in checks.items() if check["severity"] == "hard" and not check["passed"]
    ]
    soft_failures = [
        name for name, check in checks.items() if check["severity"] == "soft" and not check["passed"]
    ]
    return {
        "passed": not hard_failures,
        "score": passed_count / total if total else 1.0,
        "passed_checks": passed_count,
        "total_checks": total,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "checks": checks,
    }


def score_case(profile: dict[str, Any], case: dict[str, Any]) -> dict[str, bool]:
    """Return legacy boolean checks for unit tests and simple consumers."""
    return {name: check["passed"] for name, check in evaluate_case(profile, case)["checks"].items()}


def expected_depth_options(expected: dict[str, Any]) -> list[str]:
    if "depth_level_options" in expected:
        return [str(option) for option in expected["depth_level_options"]]
    return [str(expected.get("depth_level"))]


def _check(passed: bool, *, severity: str) -> dict[str, Any]:
    if severity not in {"hard", "soft"}:
        raise ValueError(f"Unsupported check severity: {severity}")
    return {"passed": bool(passed), "severity": severity}


def format_failed_checks(evaluation: dict[str, Any]) -> list[str]:
    return [
        f"{name} ({evaluation['checks'][name]['severity']})"
        for name in evaluation["hard_failures"] + evaluation["soft_failures"]
    ]


def _matches_any(value: str, options: list[Any]) -> bool:
    normalized_value = _normalize(value)
    if not normalized_value:
        return False
    for option in options:
        normalized_option = _normalize(str(option))
        if normalized_option and (
            normalized_value == normalized_option
            or normalized_value in normalized_option
            or normalized_option in normalized_value
        ):
            return True
    return False


def _has_overlap(values: Any, options: list[Any]) -> bool:
    if isinstance(values, str):
        values = [values]
    normalized_values = {_normalize(str(value)) for value in values or []}
    normalized_options = {_normalize(str(option)) for option in options or []}
    return bool(normalized_values & normalized_options)


def _has_category_match(values: Any, expected: dict[str, Any]) -> bool:
    category_any = expected.get("category_any", [])
    concept_groups = expected.get("category_concept_groups", [])
    if _has_overlap(values, category_any):
        return True
    if concept_groups:
        return _values_cover_concept_groups(values, concept_groups)
    return not category_any and bool(values)


def _values_cover_concept_groups(values: Any, concept_groups: list[Any]) -> bool:
    if not concept_groups:
        return bool(values)
    if isinstance(values, str):
        values = [values]
    normalized_values = [_normalize(str(value)) for value in values or []]
    for group in concept_groups:
        options = group if isinstance(group, list) else [group]
        normalized_options = [_normalize(str(option)) for option in options]
        if not any(
            option and any(option in value or value in option for value in normalized_values)
            for option in normalized_options
        ):
            return False
    return True


def _in_range(value: Any, expected_range: Any) -> bool:
    if not expected_range:
        return True
    try:
        numeric = int(value)
        lower = int(expected_range[0])
        upper = int(expected_range[1])
    except (TypeError, ValueError, IndexError):
        return False
    return lower <= numeric <= upper


def _summary_covers(summary: str, required_terms: list[Any]) -> bool:
    normalized_summary = _normalize(summary)
    return all(_normalize(str(term)) in normalized_summary for term in required_terms or [])


def _summary_covers_concept_groups(summary: str, concept_groups: list[Any]) -> bool:
    normalized_summary = _normalize(summary)
    for group in concept_groups or []:
        options = group if isinstance(group, list) else [group]
        if not any(_normalize(str(option)) in normalized_summary for option in options):
            return False
    return True


def _summary_avoids(summary: str, forbidden_terms: list[Any]) -> bool:
    normalized_summary = _normalize(summary)
    return all(_normalize(str(term)) not in normalized_summary for term in forbidden_terms or [])


def _at_least(value: Any, minimum: Any) -> bool:
    try:
        return float(value) >= float(minimum)
    except (TypeError, ValueError):
        return False


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", " ").split())


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate content-profile LLM output on frozen inputs.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N loaded cases.")
    parser.add_argument("--profile", help="Named model profile from config/models.json or config/models.local.json.")
    parser.add_argument("--provider", help="Provider for the model under test. Defaults to EVAL_PROVIDER or LLM_PROVIDER.")
    parser.add_argument("--model", help="Model id for the model under test. Defaults to EVAL_MODEL or LLM_MODEL.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for the model under test.")
    parser.add_argument("--api-key", help="API key for the model under test. Prefer env vars for real keys.")
    parser.add_argument("--start-script", help="llama.cpp start script for the model under test.")
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not auto-start llama.cpp even when provider is llama_cpp.",
    )
    parser.add_argument("--judge", action="store_true", help="Use the configured LLM as a qualitative judge.")
    parser.add_argument("--judge-profile", help="Named model profile for the qualitative judge.")
    parser.add_argument("--judge-provider", help="Judge provider. Defaults to EVAL_JUDGE_PROVIDER or LLM_PROVIDER.")
    parser.add_argument("--judge-model", help="Judge model id. Defaults to EVAL_JUDGE_MODEL or LLM_MODEL.")
    parser.add_argument("--judge-base-url", help="Judge OpenAI-compatible base URL.")
    parser.add_argument("--judge-api-key", help="Judge API key. Prefer env vars for real keys.")
    parser.add_argument("--judge-temperature", type=float, help="Judge sampling temperature. Defaults to 0.0.")
    parser.add_argument("--judge-top-p", type=float, help="Judge top_p. Defaults to 1.0.")
    parser.add_argument("--json", action="store_true", help="Emit JSONL results instead of text.")
    args = parser.parse_args(argv)
    eval_profile = args.profile or os.getenv("EVAL_PROFILE")
    judge_profile = args.judge_profile or os.getenv("EVAL_JUDGE_PROFILE")
    eval_config = prepare_eval_provider(eval_config_from_args(args), auto_start=not args.no_auto_start)
    judge_config = JudgeConfig.from_args(args, eval_config) if args.judge else None

    cases = load_cases(args.fixtures)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No eval case found with id: {args.case_id}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[: args.limit]

    failures = 0
    results: list[dict[str, Any]] = []
    total_started_at = time.perf_counter()
    for case in cases:
        case_started_at = time.perf_counter()
        try:
            profile = analyze_case(case, config=eval_config)
            analysis_error = None
        except Exception as exc:
            profile = {}
            analysis_error = str(exc)
        analysis_latency_seconds = time.perf_counter() - case_started_at
        if analysis_error is not None:
            failures += 1
            result = {
                "type": "case",
                "id": case["id"],
                "passed": False,
                "score": 0.0,
                "passed_checks": 0,
                "total_checks": 0,
                "hard_failures": ["analysis_error"],
                "soft_failures": [],
                "checks": {},
                "profile": profile,
                "analysis_error": analysis_error,
                "eval_config": {
                    "profile": eval_profile,
                    "provider": eval_config.provider,
                    "model": eval_config.model_name,
                    "base_url": eval_config.base_url,
                },
                "latency": {
                    "analysis_seconds": round(analysis_latency_seconds, 3),
                    "judge_seconds": None,
                    "total_seconds": round(time.perf_counter() - case_started_at, 3),
                },
            }
            results.append(result)
            if args.json:
                print(json.dumps(result, ensure_ascii=True))
            else:
                print(f"CASE {case['id']}: ERROR")
                print(f"  analysis_latency_seconds: {result['latency']['analysis_seconds']}")
                print(f"  analysis_error: {analysis_error}")
                print()
            continue
        evaluation = evaluate_case(profile, case)
        judge_latency_seconds = None
        if judge_config:
            judge_started_at = time.perf_counter()
            judgment = judge_case(profile, case, config=judge_config)
            judge_latency_seconds = time.perf_counter() - judge_started_at
        else:
            judgment = None
        passed = bool(evaluation["passed"])
        failures += 0 if passed else 1
        result = {
            "type": "case",
            "id": case["id"],
            "passed": passed,
            "score": evaluation["score"],
            "passed_checks": evaluation["passed_checks"],
            "total_checks": evaluation["total_checks"],
            "hard_failures": evaluation["hard_failures"],
            "soft_failures": evaluation["soft_failures"],
            "checks": evaluation["checks"],
            "profile": profile,
            "eval_config": {
                "profile": eval_profile,
                "provider": eval_config.provider,
                "model": eval_config.model_name,
                "base_url": eval_config.base_url,
            },
            "latency": {
                "analysis_seconds": round(analysis_latency_seconds, 3),
                "judge_seconds": round(judge_latency_seconds, 3)
                if judge_latency_seconds is not None
                else None,
                "total_seconds": round(time.perf_counter() - case_started_at, 3),
            },
        }
        if judge_config is not None:
            result["judge_config"] = {
                "profile": judge_profile,
                "provider": judge_config.provider,
                "model": judge_config.model_name,
                "base_url": judge_config.base_url,
                "temperature": judge_config.temperature,
                "top_p": judge_config.top_p,
            }
        if judgment is not None:
            result["judgment"] = judgment
        results.append(result)
        if args.json:
            print(json.dumps(result, ensure_ascii=True))
        else:
            print(
                f"CASE {case['id']}: {'PASS' if passed else 'FAIL'} "
                f"({evaluation['passed_checks']}/{evaluation['total_checks']} checks, "
                f"score {evaluation['score']:.2f})"
            )
            print(f"  subject: {profile.get('subject', '')}")
            print(f"  depth: {profile.get('depth_level', '')}")
            print(f"  estimated_time_minutes: {profile.get('estimated_time_minutes', '')}")
            print(f"  categories: {profile.get('categories', [])}")
            print(f"  analysis_latency_seconds: {result['latency']['analysis_seconds']}")
            print(f"  hard_failures: {evaluation['hard_failures']}")
            print(f"  soft_failures: {evaluation['soft_failures']}")
            if judgment is not None:
                print(
                    "  judge: "
                    f"{judge_config.provider}/{judge_config.model_name} "
                    f"temp={judge_config.temperature} top_p={judge_config.top_p}"
                )
                if judgment.get("status") == "completed":
                    print(f"  judge_overall_useful: {judgment.get('overall_useful')}")
                    print(f"  judge_notes: {judgment.get('notes')}")
                else:
                    print(f"  judge_error: {judgment.get('error')}")

            print()

    total_latency_seconds = time.perf_counter() - total_started_at
    summary = build_run_summary(
        results,
        total_latency_seconds=total_latency_seconds,
        eval_config=eval_config,
        judge_config=judge_config,
        eval_profile=eval_profile,
        judge_profile=judge_profile,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=True))
    else:
        print(
            "SUMMARY: "
            f"{summary['passed_cases']}/{summary['total_cases']} cases passed, "
            f"avg_score {summary['average_score']:.2f}, "
            f"total {summary['latency']['total_seconds']}s, "
            f"avg_analysis {summary['latency']['average_analysis_seconds']}s"
        )

    return 1 if failures else 0


def build_run_summary(
    results: list[dict[str, Any]],
    *,
    total_latency_seconds: float,
    eval_config: LLMConfig,
    judge_config: JudgeConfig | None,
    eval_profile: str | None = None,
    judge_profile: str | None = None,
) -> dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["passed"])
    analysis_latencies = [result["latency"]["analysis_seconds"] for result in results]
    judge_latencies = [
        result["latency"]["judge_seconds"]
        for result in results
        if result["latency"]["judge_seconds"] is not None
    ]
    return {
        "type": "summary",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "average_score": round(
            sum(float(result["score"]) for result in results) / total_cases,
            4,
        )
        if total_cases
        else 0.0,
        "latency": {
            "total_seconds": round(total_latency_seconds, 3),
            "average_analysis_seconds": round(sum(analysis_latencies) / total_cases, 3)
            if total_cases
            else 0.0,
            "max_analysis_seconds": max(analysis_latencies) if analysis_latencies else 0.0,
            "average_judge_seconds": round(sum(judge_latencies) / len(judge_latencies), 3)
            if judge_latencies
            else None,
        },
        "eval_config": {
            "profile": eval_profile,
            "provider": eval_config.provider,
            "model": eval_config.model_name,
            "base_url": eval_config.base_url,
        },
        "judge_config": {
            "profile": judge_profile,
            "provider": judge_config.provider,
            "model": judge_config.model_name,
            "base_url": judge_config.base_url,
            "temperature": judge_config.temperature,
            "top_p": judge_config.top_p,
        }
        if judge_config
        else None,
    }


if __name__ == "__main__":
    raise SystemExit(run())
