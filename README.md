# Scratchpad

A local-first reading assistant for deciding what to read, when, and how.

---

## What it does

Scratchpad helps you manage what to learn when you have limited time.

You can save anything:

* articles
* papers
* videos
* threads
* notes

Then later ask:

* “I have 20 minutes, what should I read?”
* “Give me something light”
* “Show me deep dives on RL”

Each item is automatically classified (e.g. depth and estimated time), so the system can suggest what fits your current context.

---

## Core idea

Scratchpad is not just a place to store links.

It is a system that:

* estimates how much attention something requires
* helps you choose what to consume next
* resurfaces things at the right time

The goal is to make reading and learning more intentional.

---

## Secondary goal (learning & experimentation)

Scratchpad also serves as a compact environment for experimenting with LLM-based systems.

It is used to explore:

* tool-driven agent loops
* structured vs unstructured memory
* retrieval and reasoning over personal data
* small local model capabilities (e.g. Qwen 9b)
* local vs cloud tradeoffs

The reading-focused product provides a realistic and bounded setting for these experiments.

---

## Current state

Early development.

Current focus:

* minimal LLM client
* tool-driven local chat loop with guarded tool execution
* model switching and local server observability
* basic classification (depth + estimated time)
* simple save + retrieve flow

---

## Tech stack (initial)

* Python
* SQLite (planned)
* HTTP-based LLM calls (OpenAI-compatible)
* Local models via llama.cpp

---

## Planned content schema

The first persistence pass is planned around a single `content_items` table.

Proposed v1 fields:

* `id`
* `source_type`
* `source_id`
* `url`
* `title`
* `summary`
* `subject`
* `depth_level`
* `estimated_time_minutes`
* `created_at`
* `updated_at`

Notes:

* `source_id` is intended to hold a stable external identifier such as a YouTube video ID
* `source_type + source_id` should be unique for deduplication
* `url` should also be unique when present
* `subject` is intentionally singular for v1 to keep the schema simple, though multi-topic support may replace it later

For YouTube ingestion, the product goal is not just transcript retrieval. The target output is a DB-ready content profile with fields such as:

* `source_type = "youtube"`
* `source_id = <video_id>`
* `url`
* `title`
* `summary`
* `subject`
* `depth_level`
* `estimated_time_minutes`

---

## Running locally

1. Set environment variables in `.env` or your shell:

```bash
LLM_PROVIDER=llama_cpp
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=Qwen3.5-0.8B-BF16
LLM_API_KEY=
LLM_START_SCRIPT=/path/to/run-model-server.sh
```

2. Run the chat app:

```bash
uv run python main.py
```

3. The app will start the local server automatically for `llama_cpp` if needed.

### Chat commands

The REPL in [main.py](main.py) supports a few built-in commands:

* `/reset` resets the conversation but keeps the current client and model
* `/reload` reloads `.env`, rebuilds the client, and resets the conversation
* `/model <model_name> <start_script>` stops the current local server, starts the requested model server, rebuilds the client, and resets the conversation
* `/server-status` shows server health plus the latest parsed timing block from `llama-server.log`

Example model switch:

```bash
/model Qwen3.5-0.8B-BF16 /Users/yuvalkenneth/Desktop/local-llms/scripts/run-qwen-0.8b-server.sh
```

---

## Project structure (initial)

```text
app/
  llm/        # model client + classifier
  agent/      # loop + parsing (later)
  tools/      # tool implementations (later)
  db/         # persistence (later)
  memory/     # user memory (later)

skills/       # markdown skill definitions
memory/       # memory.md

scripts/      # dev scripts
tests/        # basic tests
```

---

## Philosophy

* focus on usefulness first
* start simple, refine over time
* use rough estimates, then learn from behavior
* keep systems observable and debuggable
* avoid premature abstraction

---

## Status

Work in progress. Expect breaking changes.

The local chat runtime currently supports:

* OpenAI-compatible chat completions
* tool execution with a small local registry
* loop protection for repeated or excessive tool rounds
* local `llama.cpp` server startup and shutdown
* log-based server timing inspection for prompt/output token counts and speed

---

## Command executor

The repo includes a simple local executor in [app/tools/executor.py](app/tools/executor.py).

It currently supports:

* `Executor.run_shell(cmd, cwd=None)` via `bash -lc`
* `Executor.run_python(code, cwd=None)` via `uv run python -c`
* fixed workspace scoping rooted at this repository
* permission checks before execution
* timeout support
* stdout/stderr truncation
* a stripped environment allowlist to avoid leaking secrets

Permission behavior is intentionally simple:

* deny `sudo`, privilege escalation, destructive commands, and sensitive paths
* ask for approval on networked commands, background processes, and paths outside the workspace
* otherwise allow

There is no sandbox backend yet. Commands run directly on the host process with the checks above.
