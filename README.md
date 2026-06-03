# Scratchpad

A local-first learning inbox and testbed for small/local LLM agents.

---

## What it does

Scratchpad helps you manage what to learn when you have limited time, while providing a realistic environment for testing whether local and small LLMs can handle useful personal-knowledge workflows.

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
* creates repeatable tasks for evaluating local LLM behavior

The product goal is to make reading and learning more intentional. The engineering goal is to use that bounded product loop to test agentic LLM systems under realistic constraints.

---

## Local LLM testbed

Scratchpad is also a compact environment for experimenting with LLM-based systems.

It is used to explore:

* tool-driven agent loops
* structured vs unstructured memory
* retrieval and reasoning over personal data
* small local model capabilities
* local vs cloud tradeoffs
* model switching, latency, output quality, and failure modes
* whether small models can reliably classify, save, retrieve, and recommend useful learning material

The learning-inbox product provides a realistic and bounded setting for these experiments. Instead of testing models on isolated prompts, Scratchpad tests whether they can complete a real workflow: ingest a source, produce a useful profile, store it, and later retrieve or recommend it in context.

---

## Evaluation goal

Over time, Scratchpad should include a small eval set built from valuable, recent links that are less likely to be memorized by model training data.

The intended eval loop:

* collect fresh URLs across articles, repos, Reddit threads, videos, and papers
* freeze fetched source text/transcripts/metadata into fixtures
* run different local models against the same content-profile tasks
* judge outputs for usefulness, faithfulness, topic accuracy, depth, and time estimates
* optionally use a larger/better local model as an LLM judge for qualitative scoring

The unit tests should stay deterministic. Model-quality evals should live in explicit scripts so they can be run manually when comparing prompts or models.

---

## Current state

Early development.

Current focus:

* minimal LLM client
* tool-driven local chat loop with a small product-focused tool surface
* model switching and local server observability
* normalized content profiles across web, GitHub, Reddit, and YouTube
* Markdown-backed content saving, deduplication, listing, status updates, and Git-backed history
* content-profile eval fixtures for comparing small/local model behavior

---

## Tech stack (initial)

* Python
* Markdown files as the likely first persistence layer
* SQLite later if querying and ranking outgrow file-based storage
* HTTP-based LLM calls (OpenAI-compatible)
* Local models via llama.cpp

---

## Content profile contract

Analyzer tools return a normalized `content_profile` shape so different sources can be stored and ranked through one contract.

Current v1 fields:

* `source_type`
* `source_id`
* `url`
* `title`
* `summary`
* `subject`
* `depth_level`
* `categories`
* `estimated_time_minutes`
* `confidence`
* `metadata`

Notes:

* `source_id` is intended to hold a stable external identifier such as a YouTube video ID
* `source_type + source_id` should be unique for deduplication once persistence exists
* `url` should also be unique when present once persistence exists
* `subject` is intentionally singular for v1 to keep the schema simple, though multi-topic support may replace it later
* `estimated_time_minutes` means consumption time in v1, not broader learning time

For YouTube ingestion, the product goal is not just transcript retrieval. The target output is a save-ready content profile with fields such as:

* `source_type = "youtube"`
* `source_id = <video_id>`
* `url`
* `title`
* `summary`
* `subject`
* `depth_level`
* `estimated_time_minutes`

The likely first storage format is one Markdown file per content item with YAML frontmatter for the normalized fields and a Markdown body for notes, excerpts, and richer analysis.

The Markdown library is intentionally not separated by source. Files live under:

```text
library/
  .git/
  items/
    <source-type>-<topic-slug>-<stable-hash>.md
```

Retrieval should be by metadata and text, such as subject, category, depth, available time, status, and free-text query. `source_type` stays in frontmatter for deduplication and source-specific rendering.

Use `content_add` for the normal ingestion path: it analyzes a URL, converts the analyzer result into the normalized profile contract, and writes the corresponding Markdown item. If the same source is added again, Scratchpad updates the existing Markdown file instead of creating a duplicate.

### Library history

The content library is versioned separately from the application code. On the first content mutation, Scratchpad initializes a Git repository under `library/.git` and commits the changed Markdown item.

This is not just backup. Git history makes local-model experiments easier to inspect:

* every saved or updated item has an operation-level commit
* analyzer and recommendation behavior can be audited through diffs
* failed Git commits are reported in tool output without blocking the content write
* the app repo and the personal library history stay separate

Current commit messages are intentionally simple:

```text
Add content: <title>
Update content: <title>
Update status: <title> -> <status>
Update notes: <title>
```

History viewing, diffs, and restore commands are planned later. For now, Git is used to make library mutations observable while keeping Markdown as the source of truth.

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

### Tests and evals

Run deterministic unit tests with pytest:

```bash
uv run pytest
```

Run the content-profile model eval manually when comparing prompts or models:

```bash
uv run python scripts/eval_content_profiles.py
```

The eval script reads frozen cases from:

```text
evals/content_profiles/cases.json
```

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
  content.py  # normalized content profile/item contract
  fetchers/   # source-specific fetching and extraction
  library/    # Markdown-backed content storage
  llm/        # model client, runtime, and prompting
  tools/      # tool implementations and registry

skills/       # markdown skill definitions
library/      # local Markdown content items and their separate Git history, created as needed

scripts/      # dev scripts
evals/        # frozen model-eval fixtures
tests/        # deterministic pytest unit tests
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
* Markdown-backed content ingestion, saving, status updates, listing, and Git-backed mutation history through `content_add`, `content_save`, `content_status_update`, and `content_list`

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

These executor tools are not exposed to the app LLM by default. Enable them only for development/debugging:

```bash
SCRATCHPAD_ENABLE_EXECUTOR_TOOLS=1
```

Permission behavior is intentionally simple:

* deny `sudo`, privilege escalation, destructive commands, and sensitive paths
* ask for approval on networked commands, background processes, and paths outside the workspace
* otherwise allow

There is no sandbox backend yet. Commands run directly on the host process with the checks above.
