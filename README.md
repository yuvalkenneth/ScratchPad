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
* basic classification (depth + estimated time)
* simple save + retrieve flow

---

## Tech stack (initial)

* Python
* SQLite (planned)
* HTTP-based LLM calls (OpenAI-compatible)
* Local models via llama.cpp

---

## Running locally

1. Start a model server (e.g. llama.cpp with an OpenAI-compatible API)

2. Set environment variables:

```bash
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=qwen
LLM_API_KEY=
```

3. Run the dev script:

```bash
python scripts/dev_chat.py
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
