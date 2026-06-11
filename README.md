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

The key loop is:

```text
capture source -> analyze into a content profile -> save as Markdown -> query/recommend later -> update status -> evaluate what the model did
```

This loop matters because it is small enough to run locally, but real enough to expose local-model failures that isolated prompts hide: wrong tool choice, weak extraction, bad time estimates, hallucinated summaries, poor ranking, missing persistence, and unhelpful recommendations.

### Product success hypotheses

Scratchpad should be judged by a few concrete hypotheses:

* Given a time budget and a rough goal, the assistant can recommend 1-3 saved items that are actually worth reading or watching now.
* Saved metadata should make future retrieval and recommendation better than searching raw links alone.
* User-visible explanations should cite concrete item metadata such as status, estimated time, depth, subject, categories, and match reason.
* Small/local models should be able to complete the core workflow reliably enough to be useful, even if larger models remain better judges.
* Every important workflow should leave inspectable artifacts: Markdown files, Git commits, eval fixtures, reports, or test output.

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

The evaluation ladder is:

* deterministic unit tests for storage, parsing, query behavior, safety checks, and fake-model tool routing
* tool-choice evals for whether a real local model chooses the right first tool and required arguments
* content-profile evals for whether a model produces useful, faithful, normalized metadata from frozen source text
* deterministic workflow evals for whether save, query, recommendation constraints, and status updates compose correctly
* future recommendation evals with fake libraries, fake user profiles, user requests, and expected ranking constraints

The project should avoid treating a green eval as a vague "model is good" claim. Each eval should name the behavior it measures and preserve enough output to compare local models over time.

### Failure taxonomy

Local-model failures should be tracked in product terms, not just pass/fail:

* schema failure: invalid JSON, missing fields, invalid enum values
* source-identity failure: wrong URL, title, source type, or source ID
* faithfulness failure: summary or metadata claims unsupported by the source
* usefulness failure: subject/categories are too generic to help retrieval
* time/depth failure: estimates do not match how hard the item is to consume
* tool-choice failure: wrong tool, no tool, extra tool, or missing required arguments
* persistence failure: user asked to save/update, but the library state did not change correctly
* recommendation failure: ignores status, time budget, user goals, or cannot explain why an item fits

---

## Current state

Early development.

Current focus:

* minimal LLM client
* tool-driven local chat loop with a small product-focused tool surface
* model switching and local server observability
* normalized content profiles across web, GitHub, Reddit, and YouTube
* Markdown-backed content saving, deduplication, listing, metadata updates, status updates, and Git-backed history
* content-profile eval fixtures for comparing small/local model behavior

Already implemented foundation:

* flat Markdown persistence under `library/items/`
* deterministic deduplication by source identity or URL
* a normalized `ContentProfile` / `ContentItem` contract in code
* content add, list, update, and status-update tools
* a `scratchpad-recommendation` skill that requires querying saved items before recommending
* a local editable user profile at `library/user/profile.md` exposed through `user_profile_get`
* basic metadata and free-text query over Markdown frontmatter/body
* separate Git history for personal library mutations
* deterministic pytest coverage for analyzers, tool policy, executor safety, library behavior, and eval scoring
* manual model eval scripts for content profiles and tool choice
* deterministic workflow evals for save -> list/recommend -> status-update product flows

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

One open product decision is whether v1 should keep only consumption time or add a separate learning-effort estimate. Recommendation requests such as "I have 20 minutes" probably need consumption time as a hard constraint, while learning plans may need a separate estimate for understanding, practice, or follow-up work.

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

Use `analyze_source` when the user wants to inspect a source before saving it. Use `content_update` for corrections to saved item details such as title, summary, subject, categories, depth, time, confidence, metadata, or notes. Use `content_status_update` only for reading state changes such as `unread`, `started`, `done`, `archived`, and `abandoned`.

The LLM-facing `content_list` tool defaults to `status=["unread", "started"]` when no status is supplied, so recommendation requests do not surface completed, archived, or abandoned items unless the user explicitly asks for them.

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

The fixture includes recent real-source cases for agent/coding-agent evaluation papers and repositories. Eval output reports per-case analysis latency plus an aggregate summary with pass counts, average score, total runtime, and average analysis latency.

Run the tool-choice eval manually to check whether a model chooses the expected tool and required arguments for simple requests:

```bash
uv run python scripts/eval_tool_choice.py
```

This eval does not execute tools. It only inspects the first tool call selected by the model and applies simple argument constraints, so it can safely test save/update/status intents without mutating the Markdown library.

Use `--report <path>` to write aggregate classification metrics, including first-tool accuracy, argument accuracy, confusion matrix, per-class precision/recall/F1, default reliance rate, and latency.

Run deterministic product workflow evals to check whether library operations compose into useful flows:

```bash
uv run python scripts/eval_workflows.py
```

Workflow fixtures live in:

```text
evals/workflows/cases.json
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
* prefer transparent scoring before semantic search or embeddings
* keep local-model constraints visible in tool and prompt design

---

## Improvement roadmap

These are the main improvement ideas after rereading the repo against the project objective.

### Architecture

* Narrow the public tool surface further. Small models are sensitive to adjacent tools with overlapping meanings, so the visible interface should stay centered on `analyze_source`, `content_add`, `content_list`, `content_update`, and `content_status_update`; lower-level analyzer/save tools should remain internal.
* Split Markdown storage responsibilities once the file grows painful. `app/library/markdown_store.py` currently handles normalization, serialization, lookup, mutation, notes, querying, and Git commits. The likely split is serializer, repository, mutations, and history.
* Replace hand-rolled frontmatter parsing when human editing becomes common. The current JSON-in-frontmatter approach is simple and testable, but real YAML frontmatter or an explicit JSON metadata block would be safer for multiline fields and manual edits.
* Move shared content-profile prompt/schema logic into one profiler module. URL and YouTube analyzers should share the same core profile contract with source-specific context hooks.
* Move runtime/server lifecycle logic out of the CLI over time. `main.py` should mostly orchestrate the REPL; provider startup, shutdown, health, model listing, and timing should live under `app/llm/runtime.py`.
* Avoid eval scripts depending on private underscore functions from tool modules. Export stable profile-message builders so evals and tools test the same behavior through an intentional interface.

### Product and Recommendation

* Build the recommendation policy before adding heavy ranking infrastructure. The first version uses a `scratchpad-recommendation` skill plus `content_list`; add a `content_recommend` tool only when repeated behavior shows the policy is stable.
* Add behavioral signals after the explicit profile. The readable `library/user/profile.md` now provides transparent context; append-only signals can later record saved, recommended, started, done, and accepted events.
* Keep recommendation output explainable. Recommendations should show why each item fits: time, status, depth, topic match, preference match, and whether it is a stretch.
* Decide how to model topic multiplicity. A singular `subject` keeps v1 simple, but real recommendations likely need multi-topic support or weighted tags.
* Treat semantic search as an optional later layer, not the core design. For a small library, transparent Markdown scanning and simple scoring are easier to debug and better for local-model evals.

### Evaluation

* Add recommendation evals as the next major eval type. Use fake libraries and fake user profiles so expected ranking constraints can be deterministic.
* Extend workflow evals toward real model-in-the-loop runs. The deterministic base now checks whether save, query, recommend constraints, and status update compose correctly.
* Diversify content-profile fixtures beyond LLM-agent material. Keep adding recent non-arXiv examples across security, systems, product, design, finance, research, videos, and repos.
* Track failure categories in reports. Reports should make it easy to compare models by wrong tool, no tool, invalid arguments, weak summary, generic categories, hallucination, bad time estimate, and latency.
* Run repeated generations for unstable local models. A `--runs N` mode would expose nondeterminism that a single pass hides.

---

## Status

Work in progress. Expect breaking changes.

The local chat runtime currently supports:

* OpenAI-compatible chat completions
* tool execution with a small local registry
* current local time included directly in the system prompt, without a separate time tool
* loop protection for repeated or excessive tool rounds
* local `llama.cpp` server startup and shutdown
* log-based server timing inspection for prompt/output token counts and speed
* Markdown-backed content analysis, ingestion, metadata updates, status updates, listing, and Git-backed mutation history through `analyze_source`, `content_add`, `content_update`, `content_status_update`, and `content_list`

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
