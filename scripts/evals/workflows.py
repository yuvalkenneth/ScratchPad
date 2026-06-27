"""Run deterministic multi-step workflows against a temporary library."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.library.markdown_store import content_list, content_save, content_status_update


DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "workflows" / "cases.json"


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """Load and validate workflow eval cases."""
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Expected workflow cases file to contain a list.")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    """Validate the workflow eval case schema."""
    if not str(case.get("id") or "").strip():
        raise ValueError("Workflow case is missing required field: id")
    steps = case.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Workflow case {case['id']} must contain non-empty steps.")
    for index, step in enumerate(steps, start=1):
        step_type = step.get("type")
        if step_type not in {"save", "list", "status_update"}:
            raise ValueError(f"Workflow case {case['id']} step {index} has invalid type: {step_type}")


def run_case(case: dict[str, Any], *, library_root: Path) -> dict[str, Any]:
    """Execute every step in a workflow case and collect per-step outcomes."""
    step_results: list[dict[str, Any]] = []
    for index, step in enumerate(case["steps"], start=1):
        started_at = time.perf_counter()
        result = run_step(step, library_root=library_root)
        check = evaluate_step(step, result)
        step_results.append(
            {
                "index": index,
                "type": step["type"],
                "passed": check["passed"],
                "failures": check["failures"],
                "result": compact_step_result(step["type"], result),
                "latency_seconds": round(time.perf_counter() - started_at, 3),
            }
        )

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "passed": all(step["passed"] for step in step_results),
        "steps": step_results,
    }


def run_step(step: dict[str, Any], *, library_root: Path) -> dict[str, Any]:
    """Dispatch a single workflow step to the corresponding product tool."""
    step_type = step["type"]
    if step_type == "save":
        return content_save(step["item"], library_root=library_root)
    if step_type == "list":
        return content_list(step.get("filters") or {}, library_root=library_root)
    if step_type == "status_update":
        return content_status_update(
            item_id=step.get("id"),
            url=step.get("url"),
            source_type=step.get("source_type"),
            source_id=step.get("source_id"),
            status=step.get("status"),
            notes=step.get("notes"),
            library_root=library_root,
        )
    raise ValueError(f"Unsupported step type: {step_type}")


def evaluate_step(step: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Compare a workflow step result to its expected fields."""
    expected = step.get("expect") or {}
    failures: list[str] = []
    step_type = step["type"]

    if result.get("status") not in {"saved", "completed", "updated"}:
        failures.append(f"unexpected status: {result.get('status')}")

    if step_type == "save":
        if "created" in expected and bool(result.get("created")) != bool(expected["created"]):
            failures.append(f"created expected {expected['created']} got {result.get('created')}")
        if "duplicate" in expected and bool(result.get("duplicate")) != bool(expected["duplicate"]):
            failures.append(f"duplicate expected {expected['duplicate']} got {result.get('duplicate')}")

    if step_type == "status_update":
        item = result.get("item") or {}
        if expected.get("status") and item.get("status") != expected["status"]:
            failures.append(f"status expected {expected['status']} got {item.get('status')}")

    if step_type == "list":
        titles = [str(item.get("title") or "") for item in result.get("items", [])]
        if "titles_exact" in expected and titles != expected["titles_exact"]:
            failures.append(f"titles expected {expected['titles_exact']} got {titles}")
        missing = [title for title in expected.get("titles_include", []) if title not in titles]
        if missing:
            failures.append(f"missing titles: {missing}")
        present_forbidden = [title for title in expected.get("titles_exclude", []) if title in titles]
        if present_forbidden:
            failures.append(f"forbidden titles present: {present_forbidden}")
        if "max_count" in expected and len(titles) > int(expected["max_count"]):
            failures.append(f"count expected <= {expected['max_count']} got {len(titles)}")

    return {"passed": not failures, "failures": failures}


def compact_step_result(step_type: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep workflow eval output readable by retaining only useful fields."""
    if step_type == "list":
        return {
            "status": result.get("status"),
            "count": result.get("count"),
            "titles": [item.get("title") for item in result.get("items", [])],
        }
    if step_type == "save":
        return {
            "status": result.get("status"),
            "id": result.get("id"),
            "created": result.get("created"),
            "duplicate": result.get("duplicate"),
        }
    if step_type == "status_update":
        item = result.get("item") or {}
        return {
            "status": result.get("status"),
            "id": result.get("id"),
            "item_status": item.get("status"),
        }
    return {"status": result.get("status")}


def run(argv: list[str] | None = None) -> int:
    """Run the workflow eval CLI."""
    parser = argparse.ArgumentParser(
        description="Run deterministic Scratchpad workflow evals against a temporary Markdown library."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--json", action="store_true", help="Emit JSONL instead of text.")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No workflow eval case found with id: {args.case_id}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="scratchpad-workflow-eval-") as temp_dir:
        for case in cases:
            case_root = Path(temp_dir) / case["id"]
            result = run_case(case, library_root=case_root)
            results.append(result)
            if args.json:
                print(json.dumps(result, ensure_ascii=True))
            else:
                status = "PASS" if result["passed"] else "FAIL"
                print(f"CASE {result['id']}: {status} ({len(result['steps'])} steps)")
                for step in result["steps"]:
                    if not step["passed"]:
                        print(f"  step {step['index']} {step['type']}: {step['failures']}")

    passed = sum(1 for result in results if result["passed"])
    if args.json:
        print(json.dumps({"type": "workflow_summary", "passed": passed, "total": len(results)}, ensure_ascii=True))
    else:
        print(f"SUMMARY: {passed}/{len(results)} workflow cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
