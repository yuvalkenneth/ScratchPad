---
name: scratchpad-recommendation
description: >
  Recommend saved Scratchpad library items for what to read, watch, or learn
  next using content_list results, time/status constraints, and transparent
  metadata-based explanations.
tags: [recommendation, learning, library]
---

# Scratchpad Recommendation

Use this skill when the user asks what to read, watch, revisit, learn, study,
or pick next from the saved Scratchpad library.

The goal is not to invent a reading list. The goal is to query saved items,
respect the user's context, and recommend a small number of concrete choices
with clear reasons.

## Required Workflow

1. Call `user_profile_get` to load explicit goals, interests, avoided topics,
   preferred depth, and preferred session length from `library/user/profile.md`.
2. Call `content_list` before recommending saved items.
3. If the user gives a time budget, pass it as `max_estimated_time_minutes`.
4. If the user gives a topic, goal, or free-text intent, pass it as `query`
   unless a narrower structured filter is clearly better.
5. Prefer unread and started items. If the user does not explicitly ask for
   done, archived, or abandoned items, rely on the default status filter or pass
   `status=["unread","started"]`.
6. Recommend 1-3 items from the returned results.
7. Explain each recommendation using concrete returned metadata: title, subject,
   categories, depth, estimated time, status, and match reasons when available.
8. If the user chooses an item, call `content_status_update` with
   `status="started"` unless it is already started.

## Recommendation Rules

- Treat available time as a hard constraint by default.
- If no item fits the time budget, say that directly and optionally offer the
  closest stretch item.
- Do not recommend completed, archived, or abandoned items unless the user asks
  for them explicitly.
- Prefer already-started items when the user asks to continue or resume.
- Prefer unread items when the user asks for something new.
- Prefer practical/light items when the user asks for quick or low-effort work.
- Prefer deep items when the user asks for depth, study, research, or a serious
  learning session.
- Use explicit profile interests/goals to break ties, but do not override hard
  user constraints such as time budget, requested topic, or excluded statuses.
- If profile data is only the default template, treat it as weak context and say
  recommendations are mainly based on the saved item metadata.
- If time, topic, or goal is missing and materially changes the answer, ask one
  short clarification instead of guessing.

## Output Shape

Keep the answer compact:

- Start with the best recommendation.
- Include why it fits now.
- Mention estimated time and depth.
- Include one or two alternatives only when useful.
- If there are no matches, say what constraint caused the miss and suggest a
  next query.

Example:

```text
Pick "Stack Benchmarking" first. It is unread, 12 minutes, medium depth, and
matches your "systems" query through the title and categories.

Two alternatives:
- "LLM endpoint deployment" — 5 minutes, light, practical.
- "Prompt injection prevention" — 18 minutes, medium, security-focused.
```

## Guardrails

- Do not invent saved items.
- Do not infer private user preferences that are not in the request or explicit
  user profile data.
- Do not claim an item is a perfect fit if the match reasons are weak.
- If `content_list` returns no items, do not answer from memory; explain that no
  saved items matched.
