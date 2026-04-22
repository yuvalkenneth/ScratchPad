from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.tools.youtube_analyze_tool as analyze_tool
from app.tools.youtube_analyze_tool import youtube_analyze


FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "youtube_profile_eval_cases.json"
)


def load_cases() -> list[dict[str, Any]]:
    with FIXTURE_PATH.open() as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Expected fixture file to contain a list of cases.")
    return cases


def _score_case(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    profile = result.get("profile") or {}
    subject = str(profile.get("subject") or "").strip().lower()
    allowed_subjects = [str(item).strip().lower() for item in expected.get("subject_options", [])]

    return {
        "subject_match": subject in allowed_subjects,
        "depth_match": profile.get("depth_level") == expected.get("depth_level"),
        "time_match": profile.get("estimated_time_minutes") == expected.get("estimated_time_minutes"),
    }


def run() -> int:
    cases = load_cases()
    original_fetch = analyze_tool.fetch_transcript_segments

    try:
        for case in cases:
            analyze_tool.fetch_transcript_segments = lambda *_args, _case=case, **_kwargs: _case["segments"]
            result = json.loads(
                youtube_analyze(
                    {
                        "url": case["url"],
                        "task": "content_profile",
                    }
                )
            )
            checks = _score_case(result, case["expected"])
            profile = result.get("profile") or {}

            print(f"CASE {case['name']}")
            print(f"  subject: {profile.get('subject', '')}")
            print(f"  depth: {profile.get('depth_level', '')}")
            print(f"  estimated_time_minutes: {profile.get('estimated_time_minutes', '')}")
            print(f"  categories: {profile.get('categories', [])}")
            print(f"  confidence: {profile.get('confidence', '')}")
            print(f"  checks: {checks}")
            print()
    finally:
        analyze_tool.fetch_transcript_segments = original_fetch

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
