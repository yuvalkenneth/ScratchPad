from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.fetchers.common import estimate_time_minutes
from app.llm.config import LLMConfig
from app.tools.content_profile import build_content_profile_payload
import app.tools.url_analyze_tool as url_analyze_tool
import app.tools.youtube_analyze_tool as youtube_analyze_tool


DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "content_profile_eval_cases.json"
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
    if expected.get("depth_level") not in {"light", "medium", "deep"}:
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


def score_case(profile: dict[str, Any], case: dict[str, Any]) -> dict[str, bool]:
    expected = case["expected"]
    checks = {
        "required_fields": REQUIRED_PROFILE_KEYS.issubset(profile.keys()),
        "source_type": profile.get("source_type") == case.get("source_type"),
        "source_id": profile.get("source_id") == case.get("source_id"),
        "url": profile.get("url") == case.get("url"),
        "title": profile.get("title") == case.get("title"),
        "subject": _matches_any(str(profile.get("subject") or ""), expected.get("subject_options", [])),
        "categories": _has_category_match(profile.get("categories", []), expected),
        "depth_level": profile.get("depth_level") == expected.get("depth_level"),
        "estimated_time": _in_range(
            profile.get("estimated_time_minutes"),
            expected.get("estimated_time_minutes_range"),
        ),
        "summary_should_cover": _summary_covers(
            str(profile.get("summary") or ""),
            expected.get("summary_should_cover", []),
        ),
        "summary_concept_groups": _summary_covers_concept_groups(
            str(profile.get("summary") or ""),
            expected.get("summary_concept_groups", []),
        ),
        "summary_must_not_claim": _summary_avoids(
            str(profile.get("summary") or ""),
            expected.get("summary_must_not_claim", []),
        ),
        "confidence": _at_least(profile.get("confidence"), expected.get("min_confidence", 0.0)),
    }
    return checks


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
    if _has_overlap(values, expected.get("category_any", [])):
        return True
    return _values_cover_concept_groups(values, expected.get("category_concept_groups", []))


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
    parser.add_argument("--json", action="store_true", help="Emit JSONL results instead of text.")
    args = parser.parse_args(argv)

    cases = load_cases(args.fixtures)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No eval case found with id: {args.case_id}")

    failures = 0
    for case in cases:
        profile = analyze_case(case)
        checks = score_case(profile, case)
        passed = all(checks.values())
        failures += 0 if passed else 1
        result = {
            "id": case["id"],
            "passed": passed,
            "checks": checks,
            "profile": profile,
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=True))
        else:
            print(f"CASE {case['id']}: {'PASS' if passed else 'FAIL'}")
            print(f"  subject: {profile.get('subject', '')}")
            print(f"  depth: {profile.get('depth_level', '')}")
            print(f"  estimated_time_minutes: {profile.get('estimated_time_minutes', '')}")
            print(f"  categories: {profile.get('categories', [])}")
            print(f"  failed_checks: {[key for key, value in checks.items() if not value]}")
            print()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
