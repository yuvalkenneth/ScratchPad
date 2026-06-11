# TODO

## Completed Foundation

* Unified content profile contract exists for product-facing analyzers
* Source-specific fetchers exist for generic web pages, GitHub, and Reddit
* YouTube analyzer can emit the normalized `content_profile` shape
* Markdown library writes flat item files under `library/items/`
* Saved item frontmatter includes source, title, summary, subject, depth, categories, estimated time, confidence, status, and timestamps
* Markdown body stores the summary plus optional personal notes
* Deterministic deduplication updates an existing item when the same source or URL is added again
* `content_add` analyzes a URL and saves it in one step
* `content_save` saves an already-normalized profile
* `content_list` lists saved items with basic filters for subject, category, depth, status, max estimated time, and query
* `content_list` now uses a lightweight in-memory Markdown query layer with multi-status filters, status exclusions, min/max time windows, sorting, relevance scores, and match reasons
* LLM-facing `content_list` defaults to `unread` and `started` when status is omitted
* `content_update` corrects saved item metadata/profile fields and notes while preserving item identity
* `content_status_update` marks saved items `unread`, `started`, `done`, `archived`, or `abandoned`
* Deterministic fake-model tool-choice tests cover source analysis, saving, metadata updates, status updates, and recommendation/listing requests
* Manual tool-choice eval exists for simple real-model requests without executing tools
* Tool-choice evals can score required argument constraints such as status updates and recommendation time/status filters
* Tool-choice evals report first-tool multiclass classification metrics, per-class precision/recall/F1, confusion matrix, default reliance, and latency
* Current local time is injected into the system prompt instead of exposed through a separate tool

## Priority 1: Content-Profile Evaluation Calibration

* Make `scripts/eval_content_profiles.py` report partial scores such as `passed_checks / total_checks`, not only binary pass/fail
* Split checks into severity levels, such as hard failures for schema/source identity and soft warnings for fuzzy subject/category matches
* Let the model under test use explicit provider, model id, base URL, API key, and llama.cpp start-script arguments
* Auto-start the llama.cpp server for eval runs unless explicitly disabled
* Add `--runs N` so local model nondeterminism is visible across repeated generations
* Use optional `--judge` scoring for qualitative summary, subject, category, depth, and time usefulness
* Let the judge use a separate provider, model id, base URL, temperature, and top-p from the model being evaluated
* Keep LLM judge results separate from deterministic checks so invalid schema/source identity cannot be hidden by a permissive judge
* Add per-case timing and model metadata to the eval output so model comparisons are traceable
* Expand and keep curating the recent real-source content-profile fixture; it currently includes agent/coding-agent papers and repositories from February-June 2026
* Rerun `uv run python scripts/eval_content_profiles.py --provider gemini --model gemini-3.5-flash --limit 3 --json` after Gemini free-tier daily quota resets; the latest attempt failed with `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
* Improve semantic matching for subjects and categories without hiding genuinely poor classifications
* Keep content-profile LLM eval inputs frozen so prompt/model changes are compared against identical source text
* Allow eval expectations to express semantic alternatives instead of exact strings, especially for subjects, categories, and summary coverage
* Use source-specific eval rubrics where needed, especially GitHub repository understanding time versus article reading time
* Add a few real frozen fixtures from valuable URLs once candidate links are collected
* Keep eval scripts separate from unit tests so normal tests remain deterministic and fast

## Priority 2: Recommendation Skill

* Add a `scratchpad-recommendation` skill that teaches the assistant how to answer "what should I read/learn now?"
* Require the skill to call `content_list` before recommending saved items
* Prefer `unread` and `started` items; exclude `done`, `archived`, and `abandoned` unless explicitly requested
* Treat available time as a hard constraint by default, with explicit "stretch" alternatives only when useful
* Recommend 1-3 items and explain each recommendation using concrete metadata: title, subject, categories, depth, estimated time, status, and match reason
* If the user chooses an item, use `content_status_update` to mark it `started`
* If time, topic, or goal is missing and materially affects the recommendation, ask one short clarification instead of guessing
* Use the skill as the first recommendation policy before hardcoding ranking behavior into `content_recommend`

## Priority 3: Recommendation Tool Primitives

* Keep the in-memory Markdown query layer simple and transparent; do not add SQLite until file scanning becomes painful
* Keep tool output compact enough for small local models while preserving fields needed for recommendation
* Tune relevance scoring as real saved libraries expose weak ranking behavior
* Add agent behavior tests that verify recommendation requests load the skill, call `content_list`, obey status/time constraints, and use returned metadata in the answer

## Priority 4: User Context

* Add a lightweight editable user profile at `library/user/profile.md`
* Store explicit preferences such as interests, avoided topics, preferred depth, preferred session length, and current goals
* Keep behavioral signals separate from the human-editable profile, likely as append-only `library/user/signals.jsonl`
* Record useful events such as saved, recommended, started, done, abandoned, and accepted recommendation
* Use user context as transparent recommendation input, not hidden personalization magic
* Add tests showing explicit user goals can influence recommendation queries without overriding hard constraints like status and available time

## Priority 5: Recommendation Scoring

* Add ranking logic for "what should I read now?" using time, depth, status, topic/query match, explicit preferences, and current goals
* Start with simple transparent scoring over Markdown frontmatter and body text before adding embeddings
* Decide how recommendations should trade off shorter items, confidence, freshness, depth, and exploration
* Add a `content_recommend` tool only after the skill + `content_list` flow stabilizes
* Keep `content_recommend` explainable by returning score components or reason fields
* Add tests covering empty library, no matching items, time-constrained recommendations, status filtering, user-profile relevance, and query/topic relevance

## Evaluation Follow-ups

* Extend deterministic tool-choice evals with harder ambiguous requests, such as "fix this title and mark it done", to verify tool sequencing or clarification behavior
* Add more real-model tool-choice cases for simple natural-language requests as new user workflows appear
* Build recommendation eval fixtures with fake libraries, fake user profiles, user requests, and expected ranking constraints
* Evaluate hard rules deterministically: status exclusions, time limits, query match, and required tool use
* Evaluate qualitative recommendation usefulness separately with an optional stronger local model as judge
* Build a cross-source content-profile eval set from recent, valuable, non-canonical URLs to reduce train-set contamination risk
* Freeze fetched source text/transcripts/metadata into fixtures so eval runs do not depend on live network access
* Later add `scripts/eval_recommendations.py` to compare recommendation behavior across local models

## Markdown Persistence Follow-ups

The core Markdown persistence path is already implemented. Remaining work here is cleanup and schema evolution, not a blocker for recommendation work.

* Keep saved items as flat Markdown files under `library/items/` with YAML frontmatter:
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
* Keep a future SQLite index optional if file scanning becomes too slow or ranking queries become painful
* Decide when to replace singular `subject` with multi-topic support

## Unified Analysis Contract

* Keep one unified analysis schema for all sources/fetchers/analyzers so GitHub, Reddit, web articles, YouTube, and future X/browser paths all converge to the same LLM output and persistence shape
* Keep fetchers source-specific, but require analyzers to emit the same contract and persistence to accept only that contract
* Treat `estimated_time_minutes` as consumption time in v1
* Decide later whether a separate `time_to_learn_minutes` field is needed
* Later add prompt-injection hardening to analyzer prompts so scraped pages and transcripts are treated as untrusted source data

## YouTube Profiling

* Keep transcript retrieval separate from profiling/classification
* Add title extraction or title input so YouTube entries can be stored cleanly

## Browser-Backed Fetching

* Evaluate `browse.sh` / Browserbase CLI as an optional rendered-page fetch backend for sites where static HTTP extraction fails
* Keep existing source fetchers as the first path; use browser-backed fetching only as a configured fallback
* Treat X/Twitter support as experimental because rendering, login state, rate limits, and scraping restrictions may make it unreliable
* Cache or save extracted content immediately so repeated analysis does not depend on repeatedly scraping fragile pages
* Add a small `app/fetchers/browser_cli.py` adapter only after confirming the local CLI command surface and install path
* Mock browser CLI output in unit tests; do not require the CLI or network access for the default test suite

## Cross-Source Profiling

* Make all analysis paths estimate `time_to_learn` for the main topic `Y`, not just generic reading or viewing time
* Define whether `estimated_time_minutes` should mean consumption time, learning time, or whether both fields are needed
* For GitHub repos, estimate learning time based on repo complexity, codebase size/signals, and whether the repo is educational/tutorial-oriented versus a regular production library or app
* For Reddit posts, estimate learning time based on whether the post is a deep dive, experiment, walkthrough, or a short discussion/update
* Extend other source types similarly so learning-time estimates reflect depth and effort, not only source length
* Add eval cases that check learning-time estimates across GitHub, Reddit, articles, and YouTube

## Executor

* Add a clearer public result type instead of raw `dict[str, Any]`
* Add tests for timeout behavior and stdout/stderr truncation
* Improve path detection for quoted shell fragments and command substitutions
* Decide whether outside-workspace references should be denied by default instead of approval-gated
* Add explicit tests for denied sensitive paths such as `/etc` and `~/.ssh`

## Agent Loop

* Truncate or summarize large tool outputs before appending them back into model context
* Add request-level logging for tool name, output size, and message growth per round
* Add configurable `max_tokens` defaults for local small models
* Add tests for repeated tool calls and no-final-answer tool flows beyond the current basic coverage

## Runtime

* Persist model aliases so `/model <alias>` can work without typing full script paths
* Improve `/server-status` to query live server metrics endpoints when available instead of relying mostly on log parsing
* Decide whether model switching should preserve conversation optionally instead of always resetting
