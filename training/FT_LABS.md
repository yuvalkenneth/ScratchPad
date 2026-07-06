# Fine-Tuning Labs

These labs are the teaching layer for Scratchpad fine-tuning. Scripts create
and render the artifacts; notebooks and blog-style notes explain them.

## Lab 1: SFT Sanity

Question: can the stack learn at all?

Run `sanity-overfit` on a tiny slice. Expected result: training loss drops
quickly and validation may not matter yet. If it cannot overfit, inspect the
chat template, labels, target format, masking, and optimizer before trying a
larger sweep.

Classic plots:

* train loss vs step
* learning rate vs step
* gradient norm vs step
* step time and memory

## Lab 2: LoRA Rank And Alpha

Question: how much adapter capacity is useful?

Compare `lora-r4-alpha8`, `lora-r8-alpha16`, `lora-r16-alpha32`,
`lora-r8-alpha8`, and `lora-r8-alpha32`.

Interpretation:

* Rank controls adapter capacity and trainable parameter count.
* Alpha controls update scale, especially relative to rank.
* Higher rank is not automatically better if validation loss or retention gets
  worse.

## Lab 3: Dropout And Target Modules

Question: are we overfitting, and which modules need adaptation?

Compare `lora-r8-alpha16`, `lora-r8-dropout005`, `lora-r8-attn-only`, and
`lora-r8-attn-mlp`.

Interpretation:

* Dropout is useful when train loss improves but validation/eval gets worse.
* Attention-only is cheaper and often enough for routing behavior.
* Attention plus MLP can improve quality but should justify its runtime and
  memory cost.

## Lab 4: QLoRA Tradeoffs

Question: is lower memory worth the tradeoff?

Compare `qlora-r8-alpha16` with `lora-r8-alpha16`.

Interpretation:

* QLoRA should reduce VRAM pressure.
* It may change throughput and training stability.
* Keep it only if memory savings matter and task metrics remain competitive.

## Lab 5: Preference Tuning Preview

Question: when should we move past SFT?

Only start DPO or ORPO after tool-routing SFT has stable evals, visible failure
categories, and retention checks. PPO/GRPO-style work should wait until the
reward function is narrow, inspectable, and already validated against heldout
cases.
