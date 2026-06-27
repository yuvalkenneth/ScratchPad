"""Run deterministic recommendation-ranking checks over fake libraries."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.library.markdown_store import content_list, content_save
from app.library.user_profile import read_user_profile, user_profile_path


DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "recommendations" / "cases.json"


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """Load and validate recommendation eval cases."""
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Expected recommendation cases file to contain a list.")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    """Validate the recommendation eval case schema."""
    for key in ("id", "user_request", "items", "filters", "expected"):
        if key not in case:
            raise ValueError(f"Recommendation case is missing required field: {key}")
    if not isinstance(case["items"], list) or not case["items"]:
        raise ValueError(f"Recommendation case {case['id']} must include items.")


def run_case(case: dict[str, Any], *, library_root: Path) -> dict[str, Any]:
    """Materialize one fake library and evaluate the resulting listing."""
    write_profile(case.get("user_profile") or {}, library_root=library_root)
    profile = read_user_profile(library_root=library_root)["profile"]
    for item in case["items"]:
        content_save(item, library_root=library_root)

    listing = content_list(case["filters"], library_root=library_root)
    evaluation = evaluate_listing(listing, case["expected"])
    return {
        "id": case["id"],
        "user_request": case["user_request"],
        "passed": evaluation["passed"],
        "failures": evaluation["failures"],
        "profile": profile,
        "filters": case["filters"],
        "titles": [item.get("title") for item in listing.get("items", [])],
        "query_policy": listing.get("query_policy"),
    }


def write_profile(profile: dict[str, Any], *, library_root: Path) -> None:
    """Write a fake editable user profile for recommendation tests."""
    path = user_profile_path(library_root=library_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_profile(profile), encoding="utf-8")


def render_profile(profile: dict[str, Any]) -> str:
    """Render a profile dict into the Markdown format the product reads."""
    return "\n".join(
        [
            "# Scratchpad User Profile",
            "",
            "## Current Goals",
            *render_bullets(profile.get("current_goals", [])),
            "",
            "## Interests",
            *render_bullets(profile.get("interests", [])),
            "",
            "## Avoided Topics",
            *render_bullets(profile.get("avoided_topics", [])),
            "",
            "## Preferences",
            *render_preferences(profile.get("preferences", {})),
            "",
        ]
    )


def render_bullets(values: Any) -> list[str]:
    """Render a sequence as Markdown bullets."""
    return [f"- {value}" for value in values or []] or ["- "]


def render_preferences(values: Any) -> list[str]:
    """Render preference key/value pairs as Markdown bullets."""
    if not isinstance(values, dict) or not values:
        return ["- "]
    return [f"- {key}: {value}" for key, value in values.items()]


def evaluate_listing(listing: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Check listing output against deterministic recommendation constraints."""
    failures: list[str] = []
    items = listing.get("items", [])
    titles = [str(item.get("title") or "") for item in items]

    first_title = expected.get("first_title")
    if first_title and (not titles or titles[0] != first_title):
        failures.append(f"first_title expected {first_title!r} got {titles[0] if titles else None!r}")

    missing = [title for title in expected.get("titles_include", []) if title not in titles]
    if missing:
        failures.append(f"missing titles: {missing}")

    forbidden = [title for title in expected.get("titles_exclude", []) if title in titles]
    if forbidden:
        failures.append(f"forbidden titles present: {forbidden}")

    max_time = expected.get("max_estimated_time_minutes")
    if max_time is not None:
        too_long = [
            item.get("title")
            for item in items
            if int(item.get("estimated_time_minutes") or 0) > int(max_time)
        ]
        if too_long:
            failures.append(f"items over max_estimated_time_minutes: {too_long}")

    return {"passed": not failures, "failures": failures}


def run(argv: list[str] | None = None) -> int:
    """Run the recommendation eval CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Scratchpad recommendation ranking constraints."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No recommendation eval case found with id: {args.case_id}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="scratchpad-recommendation-eval-") as temp_dir:
        for case in cases:
            result = run_case(case, library_root=Path(temp_dir) / case["id"])
            results.append(result)
            if args.json:
                print(json.dumps(result, ensure_ascii=True))
            else:
                status = "PASS" if result["passed"] else "FAIL"
                print(f"CASE {result['id']}: {status} titles={result['titles']}")
                for failure in result["failures"]:
                    print(f"  failure: {failure}")

    passed = sum(1 for result in results if result["passed"])
    if args.json:
        print(json.dumps({"type": "recommendation_summary", "passed": passed, "total": len(results)}, ensure_ascii=True))
    else:
        print(f"SUMMARY: {passed}/{len(results)} recommendation cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
