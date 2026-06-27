"""Generate deterministic Scratchpad tool-choice cases for evals and SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "evals" / "tool_choice" / "generated_cases.json"


URLS = [
    "https://example.com/article",
    "https://builders.ramp.com/post/stack-benchmarking",
    "https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html",
    "https://www.youtube.com/watch?v=l6DKRf-fAAM",
    "https://github.com/ggml-org/llama.cpp",
    "https://example.com/old-rag-post",
]
TOPICS = ["AI agents", "evals", "local LLMs", "RL", "security", "coding agents"]
STATUSES = ["unread", "started", "done", "archived", "abandoned"]
DEPTHS = ["light", "medium", "deep"]
SPLITS = ("train", "validation", "heldout")


def split_for_case_id(case_id: str) -> str:
    """Assign a stable split from a case id hash."""
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 2:
        return "validation"
    if bucket < 4:
        return "heldout"
    return "train"


def case(
    case_id: str,
    user: str | None,
    expected_tool: str,
    expected_arguments: dict[str, Any] | None = None,
    *,
    intent: str,
    category: str,
    difficulty: str = "easy",
    messages: list[dict[str, str]] | None = None,
    context_kind: str = "stateless",
    retention_kind: str | None = None,
) -> dict[str, Any]:
    """Build one tool-choice case with required experiment metadata."""
    payload: dict[str, Any] = {
        "id": case_id,
        "expected_tool": expected_tool,
        "intent": intent,
        "category": category,
        "difficulty": difficulty,
        "context_kind": context_kind,
    }
    if messages:
        payload["messages"] = messages
    elif user:
        payload["user"] = user
    else:
        raise ValueError(f"Case {case_id} needs user or messages.")
    if expected_arguments:
        payload["expected_arguments"] = expected_arguments
    if retention_kind:
        payload["retention_kind"] = retention_kind
    return payload


def must_equal(**values: Any) -> dict[str, Any]:
    """Build exact-match argument expectations."""
    return {"must_equal": values}


def must_include(**values: list[Any]) -> dict[str, Any]:
    """Build list-inclusion argument expectations."""
    return {"must_include": values}


def list_default_status_args(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build expected default status filters for content-list routing."""
    payload = {
        "must_include": {"status": ["unread", "started"]},
        "must_not_include": {"status": ["done", "archived", "abandoned"]},
    }
    if extra:
        payload.setdefault("must_equal", {}).update(extra.get("must_equal", {}))
        payload.setdefault("must_include", {}).update(extra.get("must_include", {}))
    return payload


def generate_cases() -> list[dict[str, Any]]:
    """Generate the complete tool-choice case set."""
    cases: list[dict[str, Any]] = []

    inspect_templates = [
        "Inspect {url} and tell me if it is worth reading. Do not save it.",
        "What is {url} about? Don't add it to my library yet.",
        "Analyze this source without saving: {url}",
        "Can you preview {url} before I decide whether to keep it?",
    ]
    for index, url in enumerate(URLS):
        for template_index, template in enumerate(inspect_templates):
            cases.append(
                case(
                    f"inspect_{index}_{template_index}",
                    template.format(url=url),
                    "analyze_source",
                    must_equal(url=url),
                    intent="inspect_without_saving",
                    category="source_analysis",
                )
            )

    save_templates = [
        "Save {url} to my library.",
        "Add this to Scratchpad: {url}",
        "Remember {url} for later.",
        "Put {url} in my reading list.",
    ]
    for index, url in enumerate(URLS):
        for template_index, template in enumerate(save_templates):
            cases.append(
                case(
                    f"save_{index}_{template_index}",
                    template.format(url=url),
                    "content_add",
                    must_equal(url=url),
                    intent="save_source",
                    category="library_write",
                )
            )

    save_vs_inspect_templates = [
        "Keep {url} in my library; don't just preview it.",
        "Save {url}; I already know I want it later.",
        "Add {url} as a saved source, not just an analysis.",
        "Store {url} in Scratchpad for future reading.",
    ]
    for index, url in enumerate(URLS):
        for template_index, template in enumerate(save_vs_inspect_templates):
            cases.append(
                case(
                    f"save_disambiguation_{index}_{template_index}",
                    template.format(url=url),
                    "content_add",
                    must_equal(url=url),
                    intent="save_source",
                    category="known_failure_content_add_vs_analyze",
                    difficulty="medium",
                )
            )

    for status in STATUSES:
        for index, url in enumerate(URLS[:4]):
            cases.append(
                case(
                    f"status_{status}_{index}",
                    f"Mark {url} as {status}.",
                    "content_status_update",
                    must_equal(url=url, status=status),
                    intent="status_update",
                    category="library_update",
                    difficulty="easy",
                )
            )

    status_grounding_templates = [
        "Set the saved URL {url} to {status}; use the URL itself as the identifier.",
        "For {url}, update only the reading status to {status}.",
        "Change status for this exact URL to {status}: {url}",
        "Mark the source at {url} {status}; do not invent an item id.",
    ]
    for status in STATUSES:
        for index, url in enumerate(URLS[:4]):
            for template_index, template in enumerate(status_grounding_templates):
                cases.append(
                    case(
                        f"status_grounding_{status}_{index}_{template_index}",
                        template.format(url=url, status=status),
                        "content_status_update",
                        must_equal(url=url, status=status),
                        intent="status_update",
                        category="known_failure_url_vs_id",
                        difficulty="hard",
                    )
                )

    metadata_updates = [
        (
            "title",
            "Correct the saved item for https://example.com/article: set the title to Better Local LLM Evals.",
            must_equal(url="https://example.com/article", title="Better Local LLM Evals"),
        ),
        (
            "summary",
            "Update the summary for https://builders.ramp.com/post/stack-benchmarking to mention stack benchmarking and engineering tradeoffs.",
            {"must_equal": {"url": "https://builders.ramp.com/post/stack-benchmarking"}},
        ),
        (
            "categories",
            "For the saved YouTube item with source_id l6DKRf-fAAM, set categories to AI coding tools and interviews.",
            {
                "must_equal": {"source_id": "l6DKRf-fAAM"},
                "must_include": {"categories": ["AI coding tools", "interviews"]},
            },
        ),
        (
            "time",
            "Change the estimated time for https://example.com/article to 12 minutes.",
            must_equal(url="https://example.com/article", estimated_time_minutes=12),
        ),
        (
            "notes",
            "Add a note to https://github.com/ggml-org/llama.cpp that it may be useful for GGUF serving experiments.",
            {"must_equal": {"url": "https://github.com/ggml-org/llama.cpp"}},
        ),
        (
            "subject",
            "For https://example.com/article, change the subject to local model evaluation.",
            must_equal(url="https://example.com/article", subject="local model evaluation"),
        ),
        (
            "depth",
            "Update https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html so its depth is deep.",
            must_equal(
                url="https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html",
                depth_level="deep",
            ),
        ),
        (
            "confidence",
            "Set confidence for https://example.com/article to 0.7.",
            must_equal(url="https://example.com/article", confidence=0.7),
        ),
        (
            "learning_effort",
            "For https://github.com/ggml-org/llama.cpp, set learning effort to 90 minutes.",
            must_equal(url="https://github.com/ggml-org/llama.cpp", learning_effort_minutes=90),
        ),
        (
            "youtube_title",
            "Fix the saved YouTube item with source_id ILdE7FaAjVA: title should be Practical AI Agents Talk.",
            must_equal(source_id="ILdE7FaAjVA", title="Practical AI Agents Talk"),
        ),
    ]
    for update_type, user, expected_arguments in metadata_updates:
        cases.append(
            case(
                f"update_{update_type}",
                user,
                "content_update",
                expected_arguments,
                intent="metadata_update",
                category="library_update",
                difficulty="medium",
            )
        )

    update_vs_add_templates = [
        (
            "update_notes_existing",
            "This is already saved: add note 'useful for GGUF serving experiments' to https://github.com/ggml-org/llama.cpp.",
            must_equal(url="https://github.com/ggml-org/llama.cpp"),
        ),
        (
            "update_summary_existing",
            "Do not add a new item. Update https://builders.ramp.com/post/stack-benchmarking summary to emphasize engineering tradeoffs.",
            must_equal(url="https://builders.ramp.com/post/stack-benchmarking"),
        ),
        (
            "update_title_existing",
            "The item exists already. Rename https://example.com/article to Better Local LLM Evals.",
            must_equal(url="https://example.com/article", title="Better Local LLM Evals"),
        ),
        (
            "update_depth_existing",
            "For the saved OWASP prompt-injection cheat sheet URL, set depth_level to deep: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html",
            must_equal(
                url="https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html",
                depth_level="deep",
            ),
        ),
    ]
    for update_id, user, expected_arguments in update_vs_add_templates:
        cases.append(
            case(
                update_id,
                user,
                "content_update",
                expected_arguments,
                intent="metadata_update",
                category="known_failure_content_update_vs_add",
                difficulty="hard",
            )
        )

    recommendation_templates = [
        "What should I read in {minutes} minutes?",
        "I have {minutes} minutes, pick something useful from my library.",
        "Recommend something I can learn from in under {minutes} minutes.",
        "What should I watch or read now if I only have {minutes} minutes?",
    ]
    for minutes in [10, 20, 30, 45]:
        for template_index, template in enumerate(recommendation_templates):
            cases.append(
                case(
                    f"recommend_time_{minutes}_{template_index}",
                    template.format(minutes=minutes),
                    "skill_view",
                    must_equal(name="scratchpad-recommendation"),
                    intent="recommendation_policy",
                category="recommendation_skill",
                difficulty="easy",
            )
        )

    recommendation_skill_templates = [
        "Before recommending, load the Scratchpad recommendation policy for a 15 minute session.",
        "Use the recommendation skill to decide what I should read now.",
        "I want a personalized reading recommendation from my library; load the right skill first.",
        "Pick something for me, but follow the Scratchpad recommendation process.",
    ]
    for index, prompt in enumerate(recommendation_skill_templates):
        cases.append(
            case(
                f"recommend_skill_routing_{index}",
                prompt,
                "skill_view",
                must_equal(name="scratchpad-recommendation"),
                intent="recommendation_policy",
                category="known_failure_recommendation_skill_routing",
                difficulty="medium",
            )
        )

    for index, topic in enumerate(TOPICS):
        cases.append(
            case(
                f"list_topic_{index}",
                f"Show me saved items about {topic}.",
                "content_list",
                list_default_status_args(),
                intent="library_query",
                category="content_list",
                difficulty="easy",
            )
        )
        cases.append(
            case(
                f"recommend_topic_{index}",
                f"Find me unread deep dives about {topic} from my library.",
                "content_list",
                {
                    "must_include": {"status": ["unread"], "depth_level": ["deep"]},
                    "must_not_include": {"status": ["done", "archived", "abandoned"]},
                },
                intent="library_query",
                category="known_failure_depth_status_filters",
                difficulty="hard",
            )
        )

    for depth in DEPTHS:
        cases.append(
            case(
                f"list_depth_{depth}",
                f"Show me {depth} unread items from my library.",
                "content_list",
                {
                    "must_include": {"status": ["unread"], "depth_level": [depth]},
                    "must_not_include": {"status": ["done", "archived", "abandoned"]},
                },
                intent="library_query",
                category="known_failure_depth_status_filters",
                difficulty="hard",
            )
        )

    filter_templates = [
        ("medium_unread", "Only show unread medium-depth saved items.", "medium"),
        ("deep_unread", "Find deep unread sources; exclude anything done or archived.", "deep"),
        ("light_unread", "I want light unread items from the library.", "light"),
    ]
    for case_id, prompt, depth in filter_templates:
        cases.append(
            case(
                f"list_filter_{case_id}",
                prompt,
                "content_list",
                {
                    "must_include": {"status": ["unread"], "depth_level": [depth]},
                    "must_not_include": {"status": ["done", "archived", "abandoned"]},
                },
                intent="library_query",
                category="known_failure_depth_status_filters",
                difficulty="hard",
            )
        )

    contextual_cases = [
        case(
            "context_save_previous_url",
            None,
            "content_add",
            must_equal(url="https://builders.ramp.com/post/stack-benchmarking"),
            intent="save_source",
            category="contextual_reference",
            difficulty="hard",
            context_kind="contextual",
            messages=[
                {
                    "role": "user",
                    "content": "I found this engineering article: https://builders.ramp.com/post/stack-benchmarking",
                },
                {"role": "assistant", "content": "I can help inspect it or save it."},
                {"role": "user", "content": "Save that one for later."},
            ],
        ),
        case(
            "context_status_previous_url",
            None,
            "content_status_update",
            must_equal(url="https://github.com/ggml-org/llama.cpp", status="done"),
            intent="status_update",
            category="contextual_reference",
            difficulty="hard",
            context_kind="contextual",
            messages=[
                {"role": "user", "content": "The saved item is https://github.com/ggml-org/llama.cpp."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "Mark it done."},
            ],
        ),
        case(
            "context_update_previous_url",
            None,
            "content_update",
            must_equal(
                url="https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html",
                depth_level="deep",
            ),
            intent="metadata_update",
            category="contextual_reference",
            difficulty="hard",
            context_kind="contextual",
            messages=[
                {
                    "role": "user",
                    "content": "We already saved the OWASP prompt injection cheat sheet: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html",
                },
                {"role": "assistant", "content": "I have the URL in context."},
                {"role": "user", "content": "Update its depth to deep."},
            ],
        ),
        case(
            "context_inspect_then_save_correction",
            None,
            "content_add",
            must_equal(url="https://www.youtube.com/watch?v=ILdE7FaAjVA"),
            intent="save_source",
            category="contextual_correction",
            difficulty="hard",
            context_kind="contextual",
            messages=[
                {"role": "user", "content": "Can you inspect https://www.youtube.com/watch?v=ILdE7FaAjVA?"},
                {"role": "assistant", "content": "I would use analyze_source to preview it."},
                {"role": "user", "content": "Actually save it instead."},
            ],
        ),
        case(
            "ambiguous_status_missing_target",
            "Mark it done.",
            "no_tool",
            intent="clarify_missing_context",
            category="ambiguous_no_tool",
            difficulty="ambiguous",
        ),
        case(
            "ambiguous_save_missing_url",
            "Save the article we talked about.",
            "no_tool",
            intent="clarify_missing_context",
            category="ambiguous_no_tool",
            difficulty="ambiguous",
        ),
        case(
            "ambiguous_update_missing_target",
            "Update the title to Better Local LLM Evals.",
            "no_tool",
            intent="clarify_missing_context",
            category="ambiguous_no_tool",
            difficulty="ambiguous",
        ),
        case(
            "ambiguous_recommend_no_library_query",
            "Should I learn SFT or evals first? Answer from general reasoning, do not inspect my library.",
            "no_tool",
            intent="answer_without_tools",
            category="ambiguous_no_tool",
            difficulty="ambiguous",
        ),
    ]
    cases.extend(contextual_cases)

    no_tool_prompts = [
        "What is local-first software? Give me a short explanation, don't inspect or save anything.",
        "Explain what SFT means in one paragraph.",
        "What is the difference between LoRA and full fine-tuning?",
        "Why are evals useful for local LLM projects?",
        "Give me three reasons small models can still be useful.",
        "What does GGUF mean at a high level?",
        "What is the difference between a training dataset and an eval dataset?",
        "Explain why overfitting is dangerous in one paragraph.",
        "What does heldout mean in model evaluation?",
        "Give me a high-level explanation of DPO.",
        "What is a quantized model?",
        "Why might a 350M model still be worth testing?",
        "Explain tool calling as if I am new to agents.",
        "What is the purpose of a validation split?",
        "Tell me the tradeoff between local and cloud LLMs.",
        "What is a model profile?",
    ]
    for index, prompt in enumerate(no_tool_prompts):
        cases.append(
            case(
                f"no_tool_{index}",
                prompt,
                "no_tool",
                intent="answer_without_tools",
                category="retention_no_tool",
                retention_kind="conceptual",
            )
        )

    for generated_case in cases:
        generated_case["split"] = split_for_case_id(generated_case["id"])
    return cases


def run(argv: list[str] | None = None) -> int:
    """Run the case-generation CLI."""
    parser = argparse.ArgumentParser(description="Generate expanded deterministic tool-choice cases.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    cases = generate_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} generated tool-choice cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
