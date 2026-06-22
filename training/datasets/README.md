# Training Datasets

Generated datasets live under this directory and are ignored by git.

The first dataset is tool-choice SFT:

```bash
uv run python scripts/generate_tool_choice_cases.py
uv run python scripts/export_sft_tool_choice.py --cases evals/tool_choice/generated_cases.json --split-dir training/datasets/tool_choice
```

Expected files:

* `tool_choice/train.jsonl`
* `tool_choice/validation.jsonl`
* `tool_choice/heldout.jsonl`

Default rows contain OpenAI-style `messages` with `tool_calls`. To pre-render a
`text` column for a specific model, use the model tokenizer/chat template:

```bash
uv run python scripts/export_sft_tool_choice.py \
  --cases evals/tool_choice/generated_cases.json \
  --split-dir training/datasets/tool_choice-text \
  --output-format text \
  --tokenizer models/hf/unsloth--Qwen3.5-0.8B
```

Validate model-rendered rows before training:

```bash
uv run --with transformers --with jinja2 python scripts/validate_chat_template.py \
  --cases evals/tool_choice/generated_cases.json \
  --tokenizer models/hf/unsloth--Qwen3.5-0.8B \
  --family qwen \
  --limit 125
```

For Qwen, the default target format passes Scratchpad tool schemas to the
tokenizer as `tools=...` and renders Qwen's native `<tool_call>` XML format.

Do not train on heldout rows. They are for final base-versus-SFT comparison.
