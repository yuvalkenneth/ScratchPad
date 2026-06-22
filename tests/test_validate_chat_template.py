import pytest

from scripts.validate_chat_template import (
    infer_family,
    validate_rendered_row,
)


def test_infer_family_detects_qwen_paths() -> None:
    assert infer_family("models/hf/unsloth--Qwen3.5-0.8B") == "qwen"
    assert infer_family("models/hf/SmolLM2") == "generic"


def test_validate_qwen_rendered_row_rejects_missing_tool_markers() -> None:
    with pytest.raises(ValueError, match="missing required template markers"):
        validate_rendered_row(
            {
                "id": "save_url",
                "text": (
                    "<|im_start|>system\nsys<|im_end|>\n"
                    "<|im_start|>user\nSave https://example.com<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    '<think>\n\n</think>\n\n{"arguments": {}, "tool": "content_add"}<|im_end|>'
                ),
                "metadata": {
                    "expected_tool": "content_add",
                    "target": {"arguments": {}, "tool": "content_add"},
                    "target_format": "openai-tools",
                },
            },
            family="qwen",
        )


def test_validate_qwen_no_tool_row_accepts_normal_answer() -> None:
    result = validate_rendered_row(
        {
            "id": "no_tool",
            "text": (
                "<|im_start|>system\nsys<|im_end|>\n"
                "<|im_start|>user\nExplain SFT<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n\n</think>\n\nNo tool call is needed.<|im_end|>"
            ),
            "metadata": {
                "expected_tool": "no_tool",
                "target": {"arguments": {}, "tool": "no_tool"},
                "target_format": "openai-tools",
            },
        },
        family="qwen",
    )

    assert result["id"] == "no_tool"
    assert result["expected_tool"] == "no_tool"


def test_validate_qwen_rendered_row_rejects_missing_template_markers() -> None:
    with pytest.raises(ValueError, match="missing required template markers"):
        validate_rendered_row(
            {
                "id": "save_url",
                "text": '{"arguments": {}, "tool": "content_add"}',
                "metadata": {
                    "expected_tool": "content_add",
                    "target": {"arguments": {}, "tool": "content_add"},
                    "target_format": "openai-tools",
                },
            },
            family="qwen",
        )


def test_validate_qwen_native_tool_call_row_accepts_tool_markers() -> None:
    result = validate_rendered_row(
        {
            "id": "save_url",
            "text": (
                "<|im_start|>system\n<tools>{}</tools>sys<|im_end|>\n"
                "<|im_start|>user\nSave https://example.com<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<tool_call>\n<function=content_add>\n<parameter=url>\n"
                "https://example.com\n</parameter>\n</function>\n</tool_call><|im_end|>"
            ),
            "metadata": {
                "expected_tool": "content_add",
                "target": {
                    "tool": "content_add",
                    "arguments": {"url": "https://example.com"},
                },
                "target_format": "openai-tools",
            },
        },
        family="qwen",
    )

    assert result["expected_tool"] == "content_add"


def test_validate_qwen_native_tool_call_row_accepts_json_array_parameters() -> None:
    result = validate_rendered_row(
        {
            "id": "list_unread",
            "text": (
                "<|im_start|>system\n<tools>{}</tools>sys<|im_end|>\n"
                "<|im_start|>user\nFind unread deep dives<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<tool_call>\n<function=content_list>\n<parameter=status>\n"
                '["unread"]\n</parameter>\n<parameter=depth_level>\n'
                '["deep"]\n</parameter>\n</function>\n</tool_call><|im_end|>'
            ),
            "metadata": {
                "expected_tool": "content_list",
                "target": {
                    "tool": "content_list",
                    "arguments": {"status": ["unread"], "depth_level": ["deep"]},
                },
                "target_format": "openai-tools",
            },
        },
        family="qwen",
    )

    assert result["expected_tool"] == "content_list"
