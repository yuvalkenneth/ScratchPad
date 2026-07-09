# Scratchpad SLM Training Lessons

This directory is for learning practical SFT, preference tuning, and small
RL-style optimization on bounded Scratchpad workflows.

The observable LoRA/QLoRA learning track is documented in:

```text
training/FT_LABS.md
experiments/ft_learning/
notebooks/ft_labs/
```

Use those labs for experiment manifests, classic training plots, and
blog-style interpretation notes before moving into preference tuning.

## Lesson 1: Baseline Tiny Models

Run the existing evals against 1-5 small models before training anything.

Primary candidate models:

* `unsloth/SmolLM2-135M-Instruct-bnb-4bit` for a very small lower-bound router experiment
* `unsloth/SmolLM2-360M-Instruct-bnb-4bit` for an intentionally tiny baseline
* `unsloth/Qwen3.5-0.8B` as the main sub-1B Qwen SFT candidate
* `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` as a smaller Qwen3 baseline
* `LiquidAI/LFM2.5-350M` as a Liquid architecture 350M baseline
* `LiquidAI/LFM2.5-1.2B-Instruct` as a larger Liquid instruct baseline

Secondary candidates:

* Llama-3.2-1B-Instruct as a stable 1B ecosystem baseline
* Gemma-3-270M or Gemma-3-1B for a second architecture family
* SmolLM2-1.7B only if we decide to allow a slightly larger comparison

Measure tool accuracy, argument accuracy, JSON validity, latency, and failure
types before any tuning.

The first concrete experiment is documented in:

```text
experiments/tool_choice_sft_v1/
```

## Lesson 2: SFT Tool Routing

First supervised task:

```text
Scratchpad system prompt + user request -> {"tool": "...", "arguments": {...}}
```

Export the initial dataset:

```bash
uv run python scripts/training/generate_tool_choice_cases.py
uv run python scripts/training/export_sft_tool_choice.py
uv run python scripts/training/export_sft_tool_choice.py --cases evals/tool_choice/generated_cases.json --split-dir training/datasets/tool_choice
```

This writes:

```text
training/datasets/tool_choice/sft.jsonl
training/datasets/tool_choice/train.jsonl
training/datasets/tool_choice/validation.jsonl
training/datasets/tool_choice/heldout.jsonl
```

The first model should learn routing and structured argument extraction before
we try long summaries or recommendations.

Use `train.jsonl` for SFT, `validation.jsonl` for checking overfitting during
training, and `heldout.jsonl` only for final before/after comparison.

The canonical dataset is model-agnostic: each case stores the user request and
the expected Scratchpad action. The default export writes OpenAI-style
`messages` with `tool_calls`, plus tool schemas for tokenizer rendering. This
keeps the source data close to Hugging Face TRL workflows while letting each
target model's tokenizer render its native chat/tool format.

For trainers that expect a single pre-rendered `text` column, render with the
target tokenizer instead of hand-writing ChatML:

```bash
uv run python scripts/training/export_sft_tool_choice.py \
  --cases evals/tool_choice/generated_cases.json \
  --split-dir training/datasets/tool_choice-qwen35-text \
  --output-format text \
  --tokenizer models/hf/unsloth--Qwen3.5-0.8B
```

Validate rendered examples before training. For Qwen, this checks that the
tokenizer's real template emits Qwen markers such as `<|im_start|>` /
`<|im_end|>`, includes the tokenizer-rendered `<tools>...</tools>` block, and
contains the expected native `<tool_call>` function and parameters:

```bash
uv run --group training python scripts/training/validate_chat_template.py \
  --cases evals/tool_choice/generated_cases.json \
  --tokenizer models/hf/unsloth--Qwen3.5-0.8B \
  --family qwen \
  --limit 240
```

Keep two evaluation lanes separate:

* Primary lane: OpenAI-style `messages` + `tools` + `tool_calls`, rendered through each target tokenizer/chat template.
* Model-specific parsing lane: evaluate the rendered/native output format expected by each served model, such as Qwen's `<tool_call>` XML.

For Qwen, the primary lane should render a `<tools>...</tools>` block and
assistant calls such as `<tool_call><function=content_add>...`.

Also run a retention smoke eval before and after SFT so tool-routing gains do
not hide catastrophic forgetting:

```bash
uv run python scripts/eval.py retention \
  --cases evals/retention/cases.json \
  --model-ref custom:llamacpp:qwen3.5:9b \
  --temperature 0 \
  --report experiments/tool_choice_sft_v1/reports/base-retention.json
```

Compare base and SFT runs with:

```bash
uv run python scripts/eval.py compare \
  --base-tool-report experiments/tool_choice_sft_v1/reports/base-tool-choice.json \
  --sft-tool-report experiments/tool_choice_sft_v1/reports/sft-tool-choice.json \
  --base-retention-report experiments/tool_choice_sft_v1/reports/base-retention.json \
  --sft-retention-report experiments/tool_choice_sft_v1/reports/sft-retention.json \
  --output experiments/tool_choice_sft_v1/reports/scorecard.json
```

## Lesson 3: SFT Content Profiles

Use frozen source fixtures to train:

```text
source text + metadata -> normalized content_profile JSON
```

This is harder than tool routing because summaries, categories, and time
estimates are fuzzy.

## Lesson 4: Preference Tuning

For recommendations, collect chosen/rejected pairs:

```text
user request + saved items -> chosen recommendation vs rejected recommendation
```

Use DPO/ORPO-style tuning before attempting open-ended RL.

## Lesson 5: Small RL-Style Optimization

Only after SFT and preference tuning are working, try a narrow reward function
for tool routing:

* valid JSON
* correct tool
* required arguments present
* no extra tool calls

Keep the reward product-specific and inspectable.
