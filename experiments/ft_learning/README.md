# Fine-Tuning Learning Labs

This experiment area is for observable, interpretable fine-tuning work. The
first track uses Qwen3.5-0.8B on Scratchpad tool routing so we can learn LoRA,
QLoRA, evaluation, and training observability on a bounded task before moving
to preference tuning or RL.

## Learning Contract

Every experiment should answer one question. Do not change rank, alpha,
dropout, target modules, dataset shape, and learning rate in the same run unless
the experiment is explicitly a follow-up bundle.

Each run should save:

* `manifest.json` with config, git SHA, dataset hashes, and artifact paths
* trainer logs with loss, validation loss, LR, gradient norm, step time,
  throughput, and GPU memory when available
* eval metrics for tool choice and retention
* a rendered HTML report with teaching notes
* optional adapter/checkpoint artifacts outside Git

## First Matrix

The starter matrix is:

```text
base-eval
sanity-overfit
lora-r4-alpha8
lora-r8-alpha16
lora-r16-alpha32
lora-r8-alpha8
lora-r8-alpha32
lora-r8-dropout005
lora-r8-attn-only
lora-r8-attn-mlp
qlora-r8-alpha16
```

Validate the matrix without writing artifacts:

```bash
uv run python scripts/training/run_sft_experiment.py --dry-run
```

Create planned run manifests:

```bash
uv run python scripts/training/run_sft_experiment.py --write-manifest
```

Execute one SFT experiment in a GPU environment with the optional training
stack installed:

```bash
uv run --group training python scripts/training/run_sft_experiment.py \
  --execute \
  --experiment-id lora-r8-alpha16
```

Execution writes `trainer_log.json` in the run directory. That file is the
source for loss, validation loss, learning rate, gradient norm, epoch, runtime,
and throughput plots.

Render a report after a run has a trainer log:

```bash
uv run python scripts/training/render_experiment_report.py \
  --manifest experiments/ft_learning/runs/lora-r8-alpha16/manifest.json \
  --trainer-log experiments/ft_learning/runs/lora-r8-alpha16/trainer_log.json \
  --eval-metrics experiments/ft_learning/runs/lora-r8-alpha16/eval_metrics.json \
  --output experiments/ft_learning/runs/lora-r8-alpha16/report.html
```

Render the matrix dashboard after multiple runs have manifests and scorecards:

```bash
uv run python scripts/training/render_experiment_matrix_report.py \
  --runs-root experiments/ft_learning/runs \
  --output experiments/ft_learning/matrix_report.html
```

## How To Read The Plots

Training loss tells us whether the model can fit the supervised examples.
Validation loss tells us whether that fitting generalizes. If train loss keeps
falling while validation loss rises, the run is memorizing.

Learning rate and gradient norm explain training stability. If loss spikes line
up with LR or gradient spikes, change optimizer settings before blaming LoRA.

Runtime plots make QLoRA concrete: if QLoRA saves memory but costs too much
throughput or quality, the tradeoff is visible instead of theoretical.

Tool-choice and retention evals are the decision metrics. We only keep runs
that improve routing without creating tool-call obsession or damaging general
assistant behavior.
