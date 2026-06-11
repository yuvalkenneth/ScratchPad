CONTENT_PROFILE_SCHEMA_TEXT = (
    '{"summary":"string","subject":"string","depth_level":"light|medium|deep",'
    '"categories":["string"],"estimated_time_minutes":0,'
    '"learning_effort_minutes":0,"confidence":0.0}'
)


def content_profile_schema_instruction() -> str:
    return f"Use this exact schema: {CONTENT_PROFILE_SCHEMA_TEXT}."


def common_content_profile_field_guidance(*, consumption_time_label: str) -> list[str]:
    return [
        "summary: 1-2 decision-useful sentences about what the item teaches or argues.",
        "subject: the single best primary topic, not the title, channel, site name, or content format.",
        (
            "depth_level: light for overview/introduction, medium for practical explanation "
            "with some detail, deep for advanced, dense, or prerequisite-heavy material."
        ),
        (
            "categories: 1-4 short topical/domain tags that help search and recommendations; "
            "avoid generic labels and avoid source format unless central."
        ),
        consumption_time_label,
        (
            "learning_effort_minutes: optional broader time needed to practice, follow up, "
            "or understand the topic beyond basic consumption; use null if not applicable."
        ),
        "confidence: 0.0-1.0 based on how clear and complete the source evidence is.",
    ]
