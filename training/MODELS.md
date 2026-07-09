# Local SLM Starter Set

The actual model snapshots live under ignored `models/hf/`.

Primary downloaded starter models use Unsloth repos where available. Prefer
`Qwen3.5-0.8B` as the main Qwen SFT candidate. Do not include older Qwen2.x
models in the starter set.

| Purpose | Hugging Face repo | Local path |
| --- | --- | --- |
| Very small lower-bound router experiment | `unsloth/SmolLM2-135M-Instruct-bnb-4bit` | `models/hf/unsloth--SmolLM2-135M-Instruct-bnb-4bit` |
| Small non-Qwen baseline | `unsloth/SmolLM2-360M-Instruct-bnb-4bit` | `models/hf/unsloth--SmolLM2-360M-Instruct-bnb-4bit` |
| Main sub-1B Qwen SFT candidate | `unsloth/Qwen3.5-0.8B` | `models/hf/unsloth--Qwen3.5-0.8B` |
| Smaller Qwen3 baseline | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` | `models/hf/unsloth--Qwen3-0.6B-unsloth-bnb-4bit` |
| Liquid architecture 350M baseline | `LiquidAI/LFM2.5-350M` | `models/hf/LiquidAI--LFM2.5-350M` |
| Liquid architecture 1.2B instruct baseline | `LiquidAI/LFM2.5-1.2B-Instruct` | `models/hf/LiquidAI--LFM2.5-1.2B-Instruct` |

## llama.cpp Serving Models

Use these GGUF files for `llama-server` / `llama-cli` smoke tests. `Q4_K_M` is
the default local-serving quantization for the first lesson because it is a
reasonable quality/size tradeoff.

| Repo | Local GGUF |
| --- | --- |
| `unsloth/SmolLM2-135M-Instruct-GGUF` | `models/gguf/unsloth--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf` |
| `unsloth/SmolLM2-360M-Instruct-GGUF` | `models/gguf/unsloth--SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q4_K_M.gguf` |
| `unsloth/Qwen3.5-0.8B-GGUF` | `models/gguf/unsloth--Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf` |
| `unsloth/Qwen3-0.6B-GGUF` | `models/gguf/unsloth--Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` |
| `LiquidAI/LFM2.5-350M-GGUF` | `models/gguf/LiquidAI--LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf` |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `models/gguf/LiquidAI--LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` |

## llama.cpp Architecture Support Check

The Hugging Face safetensors snapshots are for training/fine-tuning.
The GGUF files above are for llama.cpp serving.

Local config/model types and llama.cpp converter status:

| Repo | Local `model_type` / architecture | llama.cpp converter support |
| --- | --- | --- |
| `unsloth/SmolLM2-135M-Instruct-bnb-4bit` | `llama` / `LlamaForCausalLM` | Supported through Llama architecture |
| `unsloth/SmolLM2-360M-Instruct-bnb-4bit` | `llama` / `LlamaForCausalLM` | Supported through Llama architecture |
| `unsloth/Qwen3.5-0.8B` | `qwen3_5` / `Qwen3_5ForConditionalGeneration` | Supported by local llama.cpp converter as `QWEN35` |
| `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` | `qwen3` / `Qwen3ForCausalLM` | Supported by local llama.cpp converter as `QWEN3` |
| `LiquidAI/LFM2.5-350M` | `lfm2` / `Lfm2ForCausalLM` | Supported by local llama.cpp converter as `LFM2` |
| `LiquidAI/LFM2.5-1.2B-Instruct` | `lfm2` / `Lfm2ForCausalLM` | Supported by local llama.cpp converter as `LFM2` |

If a model does not have a downloaded GGUF variant, the fallback is to install
the converter Python dependencies for the local llama.cpp
`convert_hf_to_gguf.py`, convert the HF snapshot to GGUF, then optionally
quantize with `llama-quantize`.

Start with baseline evals before training. The first SFT lesson should use the
tool-choice JSONL exported by:

```bash
uv run python scripts/training/export_sft_tool_choice.py
```

If downloads need to be repeated, prefer explicit local directories:

```bash
hf download unsloth/SmolLM2-135M-Instruct-bnb-4bit --local-dir models/hf/unsloth--SmolLM2-135M-Instruct-bnb-4bit
hf download unsloth/SmolLM2-360M-Instruct-bnb-4bit --local-dir models/hf/unsloth--SmolLM2-360M-Instruct-bnb-4bit
hf download unsloth/Qwen3.5-0.8B --local-dir models/hf/unsloth--Qwen3.5-0.8B
hf download unsloth/Qwen3-0.6B-unsloth-bnb-4bit --local-dir models/hf/unsloth--Qwen3-0.6B-unsloth-bnb-4bit
hf download LiquidAI/LFM2.5-350M --local-dir models/hf/LiquidAI--LFM2.5-350M
hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir models/hf/LiquidAI--LFM2.5-1.2B-Instruct
```

Raw upstream snapshots were also downloaded during initial exploration. Prefer
the Unsloth paths above for training lessons.
