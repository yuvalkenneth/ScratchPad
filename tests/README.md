# Tests

The test suite should stay deterministic and fast. Real-model quality checks
belong in `scripts/eval.py` commands, not in normal `pytest` runs.

## Product Behavior

* `test_agent_tool_choice.py`: fake-model agent tests for tool routing across
  save, inspect, status update, metadata update, listing, and recommendation
  intents.
* `test_analyzers.py`: source extraction and URL/YouTube content-profile
  analyzer behavior, including fallback cases.
* `test_content_library.py`: Markdown persistence, deduplication, listing,
  metadata updates, status updates, and library Git commits.
* `test_content_profile_prompt.py`: content-profile prompt/schema contract
  checks shared by analyzers.
* `test_registry_policy.py`: LLM-facing tool-surface and routing-policy
  regression tests.
* `test_user_profile.py`: editable user-profile template and parser behavior.

## LLM Configuration

* `test_llm_catalog.py`: model catalog loading, local overrides, env-backed
  values, and unknown-provider errors.
* `test_llm_runtime.py`: runtime prompt policy, client env resolution,
  quota-error handling, `.env` loading, and repeated-tool-call handling.

## Evaluation Logic

* `test_content_profile_eval_scoring.py`: deterministic scoring behavior for
  content-profile eval fixtures.
* `test_eval_benchmark.py`: benchmark command construction and dry-run
  manifest output.
* `test_eval_recommendations.py`: recommendation eval case loading and CLI
  smoke coverage.
* `test_eval_retention.py`: retention case schema, text checks, labels, and
  grouped summaries.
* `test_eval_tool_choice.py`: tool-choice argument checks, split filtering,
  grouped metrics, confusion-style scoring, and failure classification.
* `test_eval_workflows.py`: workflow eval case loading and CLI smoke coverage.

## Training And Experiment Utilities

* `test_training_case_generation.py`: generated tool-choice case coverage,
  metadata, stable splits, and targeted failure cases.
* `test_training_chat_templates.py`: tokenizer-rendered row validation for
  Qwen-style tool-call templates and no-tool examples.
* `test_training_exports.py`: SFT export rows, OpenAI-style tool calls,
  rendered text rows, metadata preservation, and split writing.
* `test_training_reports.py`: base-vs-SFT scorecard construction.

## Observability

* `test_observability_runtime.py`: runtime JSON parsing, missing-RSS handling,
  HTML rendering, and process-stat fallbacks.
* `test_observability_tracking.py`: optional MLflow metric flattening.
