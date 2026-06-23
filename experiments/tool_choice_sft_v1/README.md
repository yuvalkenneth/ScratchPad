# Experiment 1: Qwen 0.8B Tool-Routing SFT

## Goal

Learn whether a small local model can become a more reliable Scratchpad agent
through narrow product-specific SFT.

Target model: `Qwen3.5-0.8B`.

Primary hypothesis: tool-routing SFT should improve Scratchpad tool calls,
especially argument grounding, without causing catastrophic forgetting or
tool-call obsession.

## Baseline

Current base report:

```text
report: evals/tool_choice/reports/qwen08b-generated-tool-choice-report.json
overall_accuracy: 58.4%
tool_selection_accuracy: 79.2%
argument_accuracy: 58.6%
argument_json_validity_rate: 100%
average_latency: 2.02s
```

Observed base failures:

* Uses invented `id` values instead of source `url` for status updates.
* Confuses `content_add`, `content_update`, and `analyze_source`.
* Often skips `skill_view(name="scratchpad-recommendation")` for recommendation requests.
* Misses `depth_level` and status filters in library queries.

## Dataset Policy

Canonical tool-choice data stays model-agnostic: user request plus expected
Scratchpad action. The SFT export writes OpenAI-style `messages` with
`tool_calls`, or tokenizer-rendered `text` for model-specific training.
Cases may be stateless single-turn prompts (`user`) or contextual multi-turn
prompts (`messages`) where the correct tool arguments must be grounded in prior
conversation. Difficulty uses `easy`, `medium`, `hard`, and `ambiguous`;
ambiguous cases expect `no_tool` because there is not enough context to safely
mutate the library.

Each generated case carries an explicit deterministic `split` field based on a
stable hash of the case id:

```text
train: hash bucket 4-9
validation: hash bucket 0-1
heldout: hash bucket 2-3
```

Training can iterate on `train` and `validation`. Do not tune against `heldout`;
use it only for final before/after comparison. Adding new cases should not move
existing cases between splits.

## Reproduce Data

```bash
uv run python scripts/generate_tool_choice_cases.py
uv run python scripts/export_sft_tool_choice.py \
  --cases evals/tool_choice/generated_cases.json \
  --split-dir training/datasets/tool_choice
uv run python scripts/export_sft_tool_choice.py \
  --cases evals/tool_choice/generated_cases.json \
  --split-dir training/datasets/tool_choice-qwen35-text \
  --output-format text \
  --tokenizer models/hf/unsloth--Qwen3.5-0.8B
uv run --with transformers --with jinja2 python scripts/validate_chat_template.py \
  --cases evals/tool_choice/generated_cases.json \
  --tokenizer models/hf/unsloth--Qwen3.5-0.8B \
  --family qwen \
  --limit 240
```

Before training, inspect at least one rendered row for each target:

```text
content_add
content_status_update
content_update
skill_view
content_list
no_tool
```

## Evaluation

Run base and SFT on identical tool-choice and retention suites.

Tool-choice eval:

```bash
uv run python scripts/eval_tool_choice.py \
  --cases evals/tool_choice/generated_cases.json \
  --split heldout \
  --profile qwen-local \
  --temperature 0 \
  --report experiments/tool_choice_sft_v1/reports/base-tool-choice.json
```

Retention eval:

```bash
uv run python scripts/eval_retention.py \
  --cases evals/retention/cases.json \
  --profile qwen-local \
  --temperature 0 \
  --report experiments/tool_choice_sft_v1/reports/base-retention.json
```

After training, run the same commands against the served SFT checkpoint and
write `sft-tool-choice.json` and `sft-retention.json`.

Comparison scorecard:

```bash
uv run python scripts/compare_experiment_reports.py \
  --base-tool-report experiments/tool_choice_sft_v1/reports/base-tool-choice.json \
  --sft-tool-report experiments/tool_choice_sft_v1/reports/sft-tool-choice.json \
  --base-retention-report experiments/tool_choice_sft_v1/reports/base-retention.json \
  --sft-retention-report experiments/tool_choice_sft_v1/reports/sft-retention.json \
  --output experiments/tool_choice_sft_v1/reports/scorecard.json
```

## Metrics

Primary tool metrics:

* `overall_accuracy`
* `tool_selection_accuracy`
* `argument_accuracy`
* `argument_json_validity_rate`
* `tool_false_positive_rate`
* `wrong_tool_rate`
* `extra_tool_rate`
* per-tool F1
* average latency

Retention metrics:

* `retention_pass_rate`
* `tool_false_positive_rate`
* `pass`, `degraded`, and `fail` counts
* grouped counts by retention kind

Retention kinds:

* conceptual no-tool answers
* instruction following
* URL abstention
* simple coding/math
* general assistant behavior

## Success Criteria

Use relative improvement, not a fixed target.

Keep the SFT if:

* `overall_accuracy` improves materially.
* `argument_accuracy` improves materially, especially on status/update cases.
* `tool_false_positive_rate` does not rise meaningfully.
* Retention does not gain many `fail` labels.
* The model remains fast enough to justify using a sub-1B model.

Retry data or prompting if tool accuracy improves but false-positive tool calls
increase. Reduce learning rate or training steps if retention collapses. Do not
move to recommendation preference tuning or RL until tool routing is stable.

## Scope

This experiment does not train content-profile summaries, recommendation
quality, DPO, ORPO, or RL. Those come after tool routing is measurable and
stable.
