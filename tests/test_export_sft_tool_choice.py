import json
from pathlib import Path

from scripts.export_sft_tool_choice import (
    export_rows,
    export_split_rows,
    messages_from_case,
    render_messages_as_text,
    sft_row_from_case,
    split_rows,
    target_from_case,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        if "tools" in kwargs:
            return "<tools></tools>\n" + "\n".join(
                f"<{message['role']}>{message.get('content', '')}" for message in conversation
            )
        return "\n".join(f"<{message['role']}>{message['content']}" for message in conversation)


def test_target_from_case_uses_expected_tool_and_argument_constraints() -> None:
    target = target_from_case(
        {
            "id": "mark_done",
            "user": "mark this done",
            "expected_tool": "content_status_update",
            "expected_arguments": {
                "must_equal": {
                    "url": "https://example.com",
                    "status": "done",
                }
            },
        }
    )

    assert target == {
        "tool": "content_status_update",
        "arguments": {
            "url": "https://example.com",
            "status": "done",
        },
    }


def test_export_rows_writes_chat_jsonl(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    output_path = tmp_path / "sft.jsonl"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "id": "save_url",
                    "user": "Save https://example.com",
                    "expected_tool": "content_add",
                }
            ]
        ),
        encoding="utf-8",
    )

    row_count = export_rows(cases_path, output_path)

    assert row_count == 1
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["id"] == "save_url"
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][-1]["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "content_add",
                "arguments": {},
            },
        }
    ]
    assert row["metadata"]["output_format"] == "messages"
    assert row["metadata"]["target_format"] == "openai-tools"
    assert row["metadata"]["target"] == {
        "tool": "content_add",
        "arguments": {},
    }
    assert row["metadata"]["native_tools"] is True


def test_messages_from_case_uses_openai_tool_calls() -> None:
    messages = messages_from_case(
        {
            "id": "analyze_url",
            "user": "Should I read https://example.com?",
            "expected_tool": "analyze_source",
            "expected_arguments": {"must_equal": {"url": "https://example.com"}},
        },
        system_prompt="system",
    )

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[-1]["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "analyze_source",
                "arguments": {"url": "https://example.com"},
            },
        }
    ]


def test_text_row_renders_with_supplied_chat_template_tokenizer() -> None:
    row = sft_row_from_case(
        {
            "id": "list_unread",
            "user": "What should I read?",
            "expected_tool": "content_list",
        },
        system_prompt="system",
        source="cases.json",
        output_format="text",
        tokenizer=FakeTokenizer(),
    )

    assert "messages" not in row
    assert row["text"].startswith("<tools></tools>")
    assert row["metadata"]["output_format"] == "text"
    assert row["metadata"]["target_format"] == "openai-tools"


def test_openai_tools_row_uses_tool_call_message_and_tools() -> None:
    row = sft_row_from_case(
        {
            "id": "save_url",
            "user": "Save https://example.com",
            "expected_tool": "content_add",
            "expected_arguments": {"must_equal": {"url": "https://example.com"}},
        },
        system_prompt="system",
        source="cases.json",
        output_format="messages",
    )

    assistant = row["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "content_add",
                "arguments": {"url": "https://example.com"},
            },
        }
    ]
    assert row["metadata"]["native_tools"] is True


def test_openai_tools_text_row_passes_tools_to_tokenizer() -> None:
    row = sft_row_from_case(
        {
            "id": "save_url",
            "user": "Save https://example.com",
            "expected_tool": "content_add",
            "expected_arguments": {"must_equal": {"url": "https://example.com"}},
        },
        system_prompt="system",
        source="cases.json",
        output_format="text",
        tokenizer=FakeTokenizer(),
    )

    assert row["text"].startswith("<tools></tools>")
    assert row["metadata"]["native_tools"] is True


def test_render_messages_as_text_uses_tokenizer_chat_template() -> None:
    rendered = render_messages_as_text(
        [{"role": "user", "content": "Save https://example.com"}],
        tokenizer=FakeTokenizer(),
    )

    assert rendered == "<user>Save https://example.com"


def test_split_rows_is_deterministic_and_keeps_heldout() -> None:
    rows = [{"id": str(index)} for index in range(10)]

    splits = split_rows(rows)

    assert [row["id"] for row in splits["train"]] == ["0", "1", "2", "5", "6", "7"]
    assert [row["id"] for row in splits["validation"]] == ["3", "8"]
    assert [row["id"] for row in splits["heldout"]] == ["4", "9"]


def test_export_split_rows_writes_three_files(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    output_dir = tmp_path / "dataset"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "id": f"case_{index}",
                    "user": f"Save https://example.com/{index}",
                    "expected_tool": "content_add",
                }
                for index in range(5)
            ]
        ),
        encoding="utf-8",
    )

    counts = export_split_rows(cases_path, output_dir)

    assert counts == {"train": 3, "validation": 1, "heldout": 1}
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "heldout.jsonl").exists()
