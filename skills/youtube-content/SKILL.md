---
name: youtube-content
description: >
  Analyze and transform already-fetched YouTube transcript data into structured
  content such as chapters, summaries, threads, blog posts, quotes, and
  project-ready notes.
---

# YouTube Transcript Analysis

Use this skill after transcript data has already been fetched through a tool.
This skill is for analysis and transformation, not retrieval.

## Inputs

Expected transcript inputs:

- `preview_text` or `full_text`
- `preview_segments` or timestamped transcript text
- metadata such as `video_id`, `duration`, `segment_count`, and language if available

## Output Formats

After transcript data is available, format it based on what the user asks for:

- **Chapters**: Group by topic shifts, output timestamped chapter list
- **Summary**: Concise 5-10 sentence overview of the entire video
- **Chapter summaries**: Chapters with a short paragraph summary for each
- **Thread**: Twitter/X thread format — numbered posts, each under 280 chars
- **Blog post**: Full article with title, sections, and key takeaways
- **Quotes**: Notable quotes with timestamps

### Example — Chapters Output

```
00:00 Introduction — host opens with the problem statement
03:45 Background — prior work and why existing solutions fall short
12:20 Core method — walkthrough of the proposed approach
24:10 Results — benchmark comparisons and key takeaways
31:55 Q&A — audience questions on scalability and next steps
```

## Workflow

1. **Validate inputs**: confirm transcript data is non-empty and note whether you only have a preview or the full transcript.
2. **Choose output**: if the user did not specify a format, default to a summary.
3. **Chunk if needed**: if full transcript text exceeds ~50K characters, split into overlapping chunks (~40K with 2K overlap) and summarize each chunk before merging.
4. **Transform** the transcript into the requested output format.
5. **Project-normalize when useful**: infer depth, likely time-to-consume, save-worthy takeaways, and whether the video should become a scratchpad entry.
6. **Verify**: re-read the transformed output to check for coherence, correct timestamps, and completeness before presenting.

## Error Handling

- **Preview only**: note when the output is based on a partial transcript and avoid overclaiming completeness.
- **No timestamps**: provide untimestamped summaries or thematic sections instead of fabricated chapter times.
- **Incomplete transcript**: say so explicitly and limit claims to the available content.
