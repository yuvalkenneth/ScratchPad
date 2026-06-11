from pathlib import Path

from scripts.eval_recommendations import load_cases, run, run_case


def test_recommendation_cases_load() -> None:
    cases = load_cases()

    assert [case["id"] for case in cases] == [
        "time_boxed_llm_deployment",
        "security_goal_ignores_done_items",
    ]


def test_recommendation_case_passes(tmp_path: Path) -> None:
    case = next(case for case in load_cases() if case["id"] == "time_boxed_llm_deployment")

    result = run_case(case, library_root=tmp_path)

    assert result["passed"]
    assert result["titles"][0] == "LLM Endpoint Deployment"
    assert result["query_policy"]["mode"] == "frontmatter_body_scan"


def test_eval_recommendations_cli_runs_selected_case() -> None:
    assert run(["--case", "security_goal_ignores_done_items", "--json"]) == 0
