from scripts import eval_content_profiles


def eval_case() -> dict[str, object]:
    return {
        "id": "local-first-sync",
        "source_type": "web",
        "source_id": None,
        "url": "https://example.com/local-first-sync",
        "title": "Local-first Sync",
        "input": {"text": "Local-first apps use offline sync and conflict handling."},
        "expected": {
            "subject_options": ["local-first software"],
            "category_any": ["software", "sync"],
            "depth_level": "medium",
            "estimated_time_minutes_range": [3, 10],
            "summary_should_cover": ["offline sync", "conflict handling"],
            "min_confidence": 0.5,
        },
    }


def test_evaluate_case_accepts_expected_profile() -> None:
    case = eval_case()
    profile = {
        "status": "completed",
        "source_type": case["source_type"],
        "source_id": case["source_id"],
        "url": case["url"],
        "title": case["title"],
        "summary": "This covers offline sync and conflict handling for local-first apps.",
        "subject": "local-first software",
        "depth_level": "medium",
        "categories": ["software", "sync"],
        "estimated_time_minutes": 5,
        "confidence": 0.7,
    }

    evaluation = eval_content_profiles.evaluate_case(profile, case)

    assert evaluation["passed"]
    assert evaluation["passed_checks"] == evaluation["total_checks"]


def test_evaluate_case_reports_soft_failures_without_failing_schema() -> None:
    case = eval_case()
    profile = {
        "status": "completed",
        "source_type": case["source_type"],
        "source_id": case["source_id"],
        "url": case["url"],
        "title": case["title"],
        "summary": "This covers offline sync and conflict handling for local-first apps.",
        "subject": "adjacent but not expected",
        "depth_level": "medium",
        "categories": ["unrelated"],
        "estimated_time_minutes": 5,
        "confidence": 0.7,
    }

    evaluation = eval_content_profiles.evaluate_case(profile, case)

    assert evaluation["passed"]
    assert evaluation["hard_failures"] == []
    assert "subject" in evaluation["soft_failures"]
    assert "categories" in evaluation["soft_failures"]
