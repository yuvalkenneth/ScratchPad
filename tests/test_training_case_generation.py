"""Tests for generated tool-choice case coverage, metadata, and splits."""

from scripts.evals.tool_choice import load_cases
from scripts.training.generate_tool_choice_cases import SPLITS, generate_cases, split_for_case_id


def test_generated_tool_choice_cases_are_large_and_cover_all_tools() -> None:
    cases = generate_cases()
    expected_tools = {case["expected_tool"] for case in cases}

    assert len(cases) >= 100
    assert expected_tools == {
        "analyze_source",
        "content_add",
        "content_status_update",
        "content_update",
        "skill_view",
        "content_list",
        "no_tool",
    }
    assert len(cases) >= 240


def test_generated_tool_choice_case_ids_are_unique() -> None:
    cases = generate_cases()
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))


def test_generated_tool_choice_cases_match_eval_schema(tmp_path) -> None:
    path = tmp_path / "cases.json"
    cases = generate_cases()
    path.write_text(__import__("json").dumps(cases), encoding="utf-8")

    loaded_cases = load_cases(path)

    assert len(loaded_cases) == len(cases)


def test_generated_url_action_cases_include_url_argument_expectations() -> None:
    cases = generate_cases()
    url_action_cases = [
        case
        for case in cases
        if case["expected_tool"] in {"analyze_source", "content_add"}
    ]

    assert url_action_cases
    for case in url_action_cases:
        expected_arguments = case.get("expected_arguments") or {}
        assert "url" in expected_arguments.get("must_equal", {})


def test_generated_cases_include_experiment_metadata() -> None:
    cases = generate_cases()

    for case in cases:
        assert case["intent"]
        assert case["category"]
        assert case["difficulty"] in {"easy", "medium", "hard", "ambiguous"}
        assert case["context_kind"] in {"stateless", "contextual"}
        assert case["split"] in SPLITS


def test_generated_case_splits_are_stable_and_cover_all_splits() -> None:
    cases = generate_cases()

    assert {case["split"] for case in cases} == set(SPLITS)
    for case in cases:
        assert case["split"] == split_for_case_id(case["id"])


def test_generated_cases_target_qwen_08b_failures() -> None:
    cases = generate_cases()
    categories = {case["category"] for case in cases}

    assert "known_failure_url_vs_id" in categories
    assert "known_failure_content_add_vs_analyze" in categories
    assert "known_failure_content_update_vs_add" in categories
    assert "known_failure_recommendation_skill_routing" in categories
    assert "known_failure_depth_status_filters" in categories


def test_generated_cases_include_contextual_and_ambiguous_examples() -> None:
    cases = generate_cases()

    contextual = [case for case in cases if case["context_kind"] == "contextual"]
    ambiguous = [case for case in cases if case["difficulty"] == "ambiguous"]

    assert contextual
    assert ambiguous
    assert all("messages" in case for case in contextual)
    assert all(case["expected_tool"] == "no_tool" for case in ambiguous)
