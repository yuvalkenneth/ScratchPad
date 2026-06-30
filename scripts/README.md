# Scripts

The root of `scripts/` should stay small. Keep user-facing entrypoints here and
move implementation helpers into a named subdirectory.

## Root Entry Points

* `eval.py`: single CLI wrapper for eval commands such as `tool-choice`,
  `content-profiles`, `retention`, `recommendations`, `workflows`, `benchmark`,
  and `compare`.
* `__init__.py`: marks `scripts` as an explicit package for test and command
  imports.
* `models.py`: lists, resolves, and checks configured local/API model refs.
* `ensure_local_server.sh`: helper for starting a local llama.cpp-compatible
  server when a model ref or manual run expects one.
* `stop_local_server.sh`: helper for stopping the local server used during
  Scratchpad development.

## `scripts/evals/`

* `__init__.py`: marks eval modules as an importable package.
* `benchmark.py`: orchestrates deterministic evals plus optional model evals
  into a benchmark manifest under `evals/runs/`.
* `content_profiles.py`: evaluates source-to-content-profile analysis against
  frozen fixtures and optional LLM judge output.
* `recommendations.py`: runs deterministic recommendation scenarios over fake
  libraries and user profiles.
* `retention.py`: checks whether a model still handles ordinary no-tool
  assistant behavior after tool-routing SFT.
* `tool_choice.py`: evaluates first-tool selection, argument grounding, JSON
  validity, grouped metrics, and failure types.
* `workflows.py`: runs deterministic multi-step product workflows such as
  save, query, and status update.

## `scripts/training/`

* `__init__.py`: marks training helpers as an importable package.
* `compare_reports.py`: builds base-vs-SFT scorecards from tool-choice and
  retention reports.
* `export_sft_tool_choice.py`: exports canonical tool-choice cases as
  OpenAI-style messages or tokenizer-rendered text for SFT.
* `generate_tool_choice_cases.py`: generates the larger deterministic
  tool-choice dataset with stable splits and metadata.
* `validate_chat_template.py`: validates rendered SFT rows for expected chat
  and tool-call markers before training.

## `scripts/observability/`

* `__init__.py`: marks observability helpers as an importable package.
* `collect_llm_runtime.py`: samples llama.cpp endpoint state and optional
  process memory/CPU into JSON.
* `experiment_tracking.py`: contains optional MLflow logging helpers for eval
  reports.
* `render_runtime_report.py`: renders runtime JSON into an inspectable HTML
  report.
