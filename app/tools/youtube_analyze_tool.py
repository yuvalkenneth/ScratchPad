from __future__ import annotations

import json
import os
from typing import Any, Optional

from app.llm.config import LLMConfig
from openai import OpenAI
from app.tools.content_profile import build_content_profile_payload
from app.tools.youtube_tool import extract_video_id, fetch_transcript_segments, format_timestamp


DEFAULT_CHUNK_CHARS = 12_000
CHUNK_OVERLAP_CHARS = 1_000
ANALYSIS_TASKS = {
    "content_profile",
    "summary",
    "detailed_summary",
    "chapters",
    "key_points",
    "study_notes",
    "explain",
    "quotes",
}

PROVIDER_TO_API_KEY_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "llama_cpp": "LLAMA_CPP_API_KEY",
}

YOUTUBE_ANALYZE_SCHEMA = {
    "name": "youtube_analyze",
    "description": (
        "Analyze a YouTube video without exposing the full transcript to the main chat. "
        "This tool fetches the transcript internally, chunks it when needed, runs "
        "task-specific LLM analysis, and returns compact structured results. "
        "It uses the same active LLM provider, server, and model as the main chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "A YouTube URL or raw 11-character video ID.",
            },
            "task": {
                "type": "string",
                "description": (
                    "Analysis task to perform. Use 'content_profile' for product-facing "
                    "classification such as difficulty, subject, categories, and estimated time. "
                    "Use 'detailed_summary' when the user asks for a substantial summary."
                ),
                "enum": [
                    "content_profile",
                    "summary",
                    "detailed_summary",
                    "chapters",
                    "key_points",
                    "study_notes",
                    "explain",
                    "quotes",
                ],
            },
            "question": {
                "type": "string",
                "description": (
                    "Optional user question or focus area to steer the analysis."
                ),
            },
            "language": {
                "type": "string",
                "description": (
                    "Optional comma-separated language fallback chain for transcript retrieval."
                ),
            },
            "include_timestamps": {
                "type": "boolean",
                "description": "Whether to ask for timestamps in the analysis output. Defaults to true.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}

def _transcript_to_text(segments: list[dict[str, Any]], include_timestamps: bool) -> str:
    if include_timestamps:
        return "\n".join(
            f"{format_timestamp(segment['start'])} {segment['text']}" for segment in segments
        )
    return "\n".join(segment["text"] for segment in segments)


def _chunk_text(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_chars, text_length)
        if end < text_length:
            split_at = text.rfind("\n", start, end)
            if split_at > start + (chunk_chars // 2):
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def _analysis_prompt(task: str, question: Optional[str], include_timestamps: bool) -> str:
    lines = [
        "You analyze YouTube transcripts.",
        "Read the provided transcript carefully and cover the whole material, not just the opening.",
        "Do not mention missing context unless the transcript text is obviously incomplete.",
    ]
    if include_timestamps:
        lines.append("Preserve or reference timestamps when they materially help the answer.")

    task_instructions = {
        "content_profile": (
            "Return a compact JSON object only. Use this exact schema: "
            '{"summary":"string","subject":"string","depth_level":"light|medium|deep",'
            '"categories":["string"],"estimated_time_minutes":0,"confidence":0.0}. '
            "Use 1-4 short categories. `subject` should be the single best primary subject. "
            "Base `estimated_time_minutes` on likely total viewing time from the transcript, "
            "rounded to an integer. Keep the summary to 1-2 sentences."
        ),
        "summary": "Produce a concise but faithful summary of the whole video in one short paragraph.",
        "detailed_summary": (
            "Produce a comprehensive summary of the whole video. Cover the main progression "
            "of ideas, the core arguments, important examples, and the final conclusions."
        ),
        "chapters": (
            "Produce a chapter breakdown. Group the transcript into major topic shifts and "
            "give each chapter a timestamp and a one-sentence description."
        ),
        "key_points": (
            "Extract the most important ideas, claims, techniques, and takeaways from the video."
        ),
        "study_notes": (
            "Turn the video into study notes with sections, main ideas, definitions, and takeaways."
        ),
        "explain": (
            "Explain the video clearly for a smart reader who wants the ideas simplified without losing meaning."
        ),
        "quotes": (
            "Extract the most notable quotes or near-quotes from the transcript with timestamps when available."
        ),
    }
    lines.append(task_instructions[task])
    if question:
        lines.append(f"Pay extra attention to this user request: {question}")
    if task == "content_profile":
        lines.append("Output JSON only. Do not wrap it in markdown fences.")
    else:
        lines.append("Be concrete. Avoid generic filler.")
    return "\n".join(lines)


def _complete_text(
    config: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int = 1400,
) -> str:
    api_key_env_var = PROVIDER_TO_API_KEY_ENV_VAR.get(config.provider)
    api_key = os.getenv(api_key_env_var) if api_key_env_var else None
    if not api_key:
        api_key = config.api_key or "local"
    client = OpenAI(api_key=api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _analyze_chunks(
    config: LLMConfig,
    transcript_text: str,
    *,
    task: str,
    question: Optional[str],
    include_timestamps: bool,
) -> dict[str, Any]:
    chunks = _chunk_text(transcript_text)
    chunk_summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_messages = [
            {"role": "system", "content": _analysis_prompt(task, question, include_timestamps)},
            {
                "role": "user",
                "content": (
                    f"Transcript chunk {index} of {len(chunks)}:\n\n{chunk}\n\n"
                    + (
                        "Extract compact notes for later profiling. Focus on subject matter, "
                        "difficulty, audience assumptions, and major topics."
                        if task == "content_profile"
                        else "Summarize this chunk faithfully for later merging."
                    )
                ),
            },
        ]
        chunk_summaries.append(
            _complete_text(
                config,
                chunk_messages,
                max_tokens=450 if task == "content_profile" else 900,
            )
        )

    if len(chunk_summaries) == 1:
        return {
            "analysis": chunk_summaries[0],
            "chunk_count": 1,
            "summary_strategy": "single_pass",
        }

    merge_messages = [
        {"role": "system", "content": _analysis_prompt(task, question, include_timestamps)},
        {
            "role": "user",
            "content": (
                (
                    "Merge these per-chunk notes into one final JSON content profile for the whole video.\n\n"
                    if task == "content_profile"
                    else "Merge these per-chunk notes into one coherent final answer that covers the entire video.\n\n"
                )
                + "\n\n".join(
                    f"Chunk {index} notes:\n{summary}"
                    for index, summary in enumerate(chunk_summaries, start=1)
                )
            ),
        },
    ]
    return {
        "analysis": _complete_text(
            config,
            merge_messages,
            max_tokens=500 if task == "content_profile" else 1600,
        ),
        "chunk_count": len(chunk_summaries),
        "summary_strategy": "map_reduce",
    }


def youtube_analyze(arguments: dict[str, Any]) -> str:
    config = LLMConfig.from_env()
    video_id = extract_video_id(arguments["url"])
    task = arguments.get("task") or "detailed_summary"
    if task not in ANALYSIS_TASKS:
        return json.dumps(
            {"status": "error", "error": f"Unsupported analysis task: {task}"},
            ensure_ascii=True,
        )

    languages = None
    if arguments.get("language"):
        languages = [item.strip() for item in arguments["language"].split(",") if item.strip()]
    include_timestamps = True
    if "include_timestamps" in arguments:
        include_timestamps = bool(arguments["include_timestamps"])
    question = arguments.get("question")

    try:
        segments = fetch_transcript_segments(video_id, languages)
    except Exception as exc:
        error_text = str(exc)
        lower = error_text.lower()
        if "disabled" in lower:
            payload = {"status": "error", "error": "Transcripts are disabled for this video."}
        elif "no transcript" in lower:
            payload = {
                "status": "error",
                "error": "No transcript found. Try specifying a language with --language.",
            }
        else:
            payload = {"status": "error", "error": error_text}
        return json.dumps(payload, ensure_ascii=True)

    transcript_text = _transcript_to_text(segments, include_timestamps)
    analysis_result = _analyze_chunks(
        config,
        transcript_text,
        task=task,
        question=question,
        include_timestamps=include_timestamps,
    )

    duration_minutes = max(
        1,
        round((segments[-1]["start"] + segments[-1]["duration"]) / 60) if segments else 1,
    )

    payload = {
        "status": "completed",
        "video_id": video_id,
        "task": task,
        "requested_language": languages,
        "segment_count": len(segments),
        "duration": (
            format_timestamp(segments[-1]["start"] + segments[-1]["duration"])
            if segments
            else "0:00"
        ),
        "estimated_time_minutes": duration_minutes,
        "chunk_count": analysis_result["chunk_count"],
        "summary_strategy": analysis_result["summary_strategy"],
    }

    if task == "content_profile":
        payload = build_content_profile_payload(
            source_type="youtube",
            source_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=video_id,
            estimated_time_minutes=duration_minutes,
            raw_analysis=analysis_result["analysis"],
            trust_model_time=False,
            extra_fields={
                "requested_language": languages,
                "segment_count": len(segments),
                "duration": payload["duration"],
                "chunk_count": analysis_result["chunk_count"],
                "summary_strategy": analysis_result["summary_strategy"],
            },
        )
    else:
        payload["analysis"] = analysis_result["analysis"]

    return json.dumps(payload, ensure_ascii=True)
