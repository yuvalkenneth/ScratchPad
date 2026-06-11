from pathlib import Path

from scripts.eval_workflows import load_cases, run, run_case


def test_workflow_cases_load() -> None:
    cases = load_cases()

    assert [case["id"] for case in cases] == [
        "time_boxed_recommendation_state_flow",
        "default_recommendation_status_excludes_done",
    ]


def test_time_boxed_workflow_case_passes(tmp_path: Path) -> None:
    case = next(
        case for case in load_cases() if case["id"] == "time_boxed_recommendation_state_flow"
    )

    result = run_case(case, library_root=tmp_path)

    assert result["passed"]
    assert [step["type"] for step in result["steps"]] == [
        "save",
        "save",
        "save",
        "list",
        "status_update",
        "list",
    ]


def test_eval_workflows_cli_runs_selected_case() -> None:
    assert run(["--case", "default_recommendation_status_excludes_done", "--json"]) == 0
