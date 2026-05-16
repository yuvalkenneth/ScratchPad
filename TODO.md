# TODO

## Unified Analysis Contract

* Keep one unified analysis schema for all sources/fetchers/analyzers so GitHub, Reddit, web articles, YouTube, and future X/browser paths all converge to the same LLM output and persistence shape
* Keep fetchers source-specific, but require analyzers to emit the same contract and persistence to accept only that contract
* Treat `estimated_time_minutes` as consumption time in v1
* Decide later whether a separate `time_to_learn_minutes` field is needed

## Executor

* Add a clearer public result type instead of raw `dict[str, Any]`
* Add tests for timeout behavior and stdout/stderr truncation
* Improve path detection for quoted shell fragments and command substitutions
* Decide whether outside-workspace references should be denied by default instead of approval-gated
* Add explicit tests for denied sensitive paths such as `/etc` and `~/.ssh`

## Markdown Persistence

* Store saved items as flat Markdown files under `library/items/` with YAML frontmatter:
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
  * `created_at`
  * `updated_at`
* Use the Markdown body for personal notes, excerpts, and richer analysis
* Enforce stable deduplication by deriving deterministic filenames from `source_type + source_id`, falling back to URL slug/hash
* Add update/status flows for marking items as started, done, archived, or abandoned
* Add ranking logic for "what should I read now?" using time, depth, status, and topic/query match
* Keep a future SQLite index optional if file scanning becomes too slow or ranking queries become painful
* Decide when to replace singular `subject` with multi-topic support

## YouTube Profiling

* Keep transcript retrieval separate from profiling/classification
* Add title extraction or title input so YouTube entries can be stored cleanly
* Build a YouTube profiling eval set from around 10 recent, non-canonical videos to reduce train-set contamination risk
* Store tagged transcript fixtures for those eval cases so classification can be tested without depending on live YouTube fetches

## Cross-Source Profiling

* Make all analysis paths estimate `time_to_learn` for the main topic `Y`, not just generic reading or viewing time
* Define whether `estimated_time_minutes` should mean consumption time, learning time, or whether both fields are needed
* For GitHub repos, estimate learning time based on repo complexity, codebase size/signals, and whether the repo is educational/tutorial-oriented versus a regular production library or app
* For Reddit posts, estimate learning time based on whether the post is a deep dive, experiment, walkthrough, or a short discussion/update
* Extend other source types similarly so learning-time estimates reflect depth and effort, not only source length
* Add eval cases that check learning-time estimates across GitHub, Reddit, articles, and YouTube

## Agent Loop

* Truncate or summarize large tool outputs before appending them back into model context
* Add request-level logging for tool name, output size, and message growth per round
* Add configurable `max_tokens` defaults for local small models
* Add tests for repeated tool calls and no-final-answer tool flows beyond the current basic coverage

## Runtime

* Persist model aliases so `/model <alias>` can work without typing full script paths
* Improve `/server-status` to query live server metrics endpoints when available instead of relying mostly on log parsing
* Decide whether model switching should preserve conversation optionally instead of always resetting
