import json

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


def test_evaluate_case_accepts_depth_options() -> None:
    case = eval_case()
    case["expected"] = {
        **case["expected"],
        "depth_level_options": ["medium", "deep"],
    }
    case["expected"].pop("depth_level")
    profile = {
        "status": "completed",
        "source_type": case["source_type"],
        "source_id": case["source_id"],
        "url": case["url"],
        "title": case["title"],
        "summary": "This covers offline sync and conflict handling for local-first apps.",
        "subject": "local-first software",
        "depth_level": "deep",
        "categories": ["software", "sync"],
        "estimated_time_minutes": 5,
        "confidence": 0.7,
    }

    evaluation = eval_content_profiles.evaluate_case(profile, case)

    assert evaluation["checks"]["depth_level"]["passed"]


def test_run_limit_evaluates_first_n_cases(tmp_path, monkeypatch, capsys) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([eval_case(), {**eval_case(), "id": "second"}]), encoding="utf-8")

    def fake_analyze_case(case, **_kwargs):
        return {
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

    monkeypatch.setattr(eval_content_profiles, "analyze_case", fake_analyze_case)

    exit_code = eval_content_profiles.run(["--fixtures", str(cases_path), "--limit", "1", "--json"])
    lines = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "local-first-sync"
    assert json.loads(lines[1])["total_cases"] == 1
