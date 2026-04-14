"""Tests for the training data exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from godspeed.training.exporter import (
    ExportFilters,
    TrainingExporter,
    _count_tool_calls,
    _count_user_turns,
    _has_errors,
    _passes_filters,
    _to_chatml,
    _to_openai,
    _to_sharegpt,
    _tools_used,
    _truncate_content,
)

# -- Fixtures ---------------------------------------------------------------


def _make_session_file(tmp_path: Path, name: str, messages: list[dict]) -> Path:
    """Write a list of message dicts as a JSONL file."""
    path = tmp_path / f"{name}.conversation.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


_TS = "2026-01-01T00:00:00"
_SID = "test-1"

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are Godspeed.", "timestamp": _TS, "session_id": _SID},
    {"role": "user", "content": "Fix the bug in auth.py", "timestamp": _TS, "session_id": _SID},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "grep_search",
                "arguments": {"pattern": "auth", "glob": "*.py"},
            },
        ],
        "timestamp": _TS,
        "session_id": _SID,
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "grep_search",
        "content": "auth.py:15: def authenticate(...)",
        "is_error": False,
        "step": 1,
        "timestamp": _TS,
        "session_id": _SID,
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_2",
                "name": "file_read",
                "arguments": {"file_path": "auth.py"},
            },
        ],
        "timestamp": _TS,
        "session_id": _SID,
    },
    {
        "role": "tool",
        "tool_call_id": "call_2",
        "name": "file_read",
        "content": "1: import hashlib\n2: ...",
        "is_error": False,
        "step": 2,
        "timestamp": _TS,
        "session_id": _SID,
    },
    {
        "role": "assistant",
        "content": "I found the bug. Let me fix it.",
        "timestamp": _TS,
        "session_id": _SID,
    },
    {"role": "user", "content": "Thanks, looks good", "timestamp": _TS, "session_id": _SID},
    {
        "role": "assistant",
        "content": "You're welcome!",
        "timestamp": _TS,
        "session_id": _SID,
    },
]


# -- Helper tests -----------------------------------------------------------


class TestHelpers:
    def test_count_tool_calls(self) -> None:
        assert _count_tool_calls(SAMPLE_MESSAGES) == 2

    def test_count_user_turns(self) -> None:
        assert _count_user_turns(SAMPLE_MESSAGES) == 2

    def test_has_errors_false(self) -> None:
        assert _has_errors(SAMPLE_MESSAGES) is False

    def test_has_errors_true(self) -> None:
        msgs = [{"role": "tool", "is_error": True, "content": "fail"}]
        assert _has_errors(msgs) is True

    def test_tools_used(self) -> None:
        assert _tools_used(SAMPLE_MESSAGES) == {"grep_search", "file_read"}

    def test_truncate_short(self) -> None:
        assert _truncate_content("short", 100) == "short"

    def test_truncate_long(self) -> None:
        result = _truncate_content("x" * 200, 50)
        assert len(result) < 200
        assert "truncated" in result


# -- Filter tests -----------------------------------------------------------


class TestFilters:
    def test_passes_default(self) -> None:
        assert _passes_filters(SAMPLE_MESSAGES, ExportFilters()) is True

    def test_fails_min_turns(self) -> None:
        assert _passes_filters(SAMPLE_MESSAGES, ExportFilters(min_turns=5)) is False

    def test_fails_min_tool_calls(self) -> None:
        assert _passes_filters(SAMPLE_MESSAGES, ExportFilters(min_tool_calls=10)) is False

    def test_fails_success_only(self) -> None:
        msgs = [*SAMPLE_MESSAGES, {"role": "tool", "is_error": True, "content": "err"}]
        assert _passes_filters(msgs, ExportFilters(success_only=True)) is False

    def test_passes_success_only(self) -> None:
        assert _passes_filters(SAMPLE_MESSAGES, ExportFilters(success_only=True)) is True

    def test_fails_tools_filter(self) -> None:
        assert _passes_filters(SAMPLE_MESSAGES, ExportFilters(tools=["shell"])) is False

    def test_passes_tools_filter(self) -> None:
        assert _passes_filters(SAMPLE_MESSAGES, ExportFilters(tools=["grep_search"])) is True


# -- OpenAI format tests ----------------------------------------------------


class TestOpenAIFormat:
    def test_basic_structure(self) -> None:
        result = _to_openai(SAMPLE_MESSAGES)
        assert "messages" in result
        msgs = result["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_tool_calls_format(self) -> None:
        result = _to_openai(SAMPLE_MESSAGES)
        assistant_msg = result["messages"][2]
        assert "tool_calls" in assistant_msg
        tc = assistant_msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "grep_search"
        # Arguments must be a JSON string
        assert isinstance(tc["function"]["arguments"], str)

    def test_tool_result_format(self) -> None:
        result = _to_openai(SAMPLE_MESSAGES)
        tool_msg = result["messages"][3]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"

    def test_includes_tool_schemas(self) -> None:
        schemas = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        result = _to_openai(SAMPLE_MESSAGES, tool_schemas=schemas)
        assert result["tools"] == schemas

    def test_skips_meta_events(self) -> None:
        msgs = [*SAMPLE_MESSAGES, {"role": "meta", "event": "compaction"}]
        result = _to_openai(msgs)
        roles = [m["role"] for m in result["messages"]]
        assert "meta" not in roles

    def test_multimodal_user_flattened(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "data": "..."},
                    {"type": "text", "text": "World"},
                ],
            },
        ]
        result = _to_openai(msgs)
        assert result["messages"][0]["content"] == "Hello\nWorld"

    def test_truncates_tool_output(self) -> None:
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": "a" * 5000},
        ]
        result = _to_openai(msgs, max_tool_output=100)
        assert len(result["messages"][0]["content"]) < 5000


# -- ChatML format tests ----------------------------------------------------


class TestChatMLFormat:
    def test_has_im_tokens(self) -> None:
        result = _to_chatml(SAMPLE_MESSAGES)
        text = result["text"]
        assert "<|im_start|>system" in text
        assert "<|im_end|>" in text

    def test_tool_call_block(self) -> None:
        result = _to_chatml(SAMPLE_MESSAGES)
        text = result["text"]
        assert "<tool_call>" in text
        assert "</tool_call>" in text

    def test_tool_response_block(self) -> None:
        result = _to_chatml(SAMPLE_MESSAGES)
        text = result["text"]
        assert '<tool_response name="grep_search">' in text

    def test_skips_meta(self) -> None:
        msgs = [{"role": "meta", "event": "compaction"}]
        result = _to_chatml(msgs)
        assert result["text"] == ""


# -- ShareGPT format tests --------------------------------------------------


class TestShareGPTFormat:
    def test_basic_structure(self) -> None:
        result = _to_sharegpt(SAMPLE_MESSAGES)
        assert "conversations" in result
        convos = result["conversations"]
        assert convos[0]["from"] == "system"
        assert convos[1]["from"] == "human"

    def test_assistant_is_gpt(self) -> None:
        result = _to_sharegpt(SAMPLE_MESSAGES)
        gpt_msgs = [c for c in result["conversations"] if c["from"] == "gpt"]
        assert len(gpt_msgs) >= 1

    def test_tool_calls_in_value(self) -> None:
        result = _to_sharegpt(SAMPLE_MESSAGES)
        gpt_msg = result["conversations"][2]
        assert "[Tool Calls]" in gpt_msg["value"]

    def test_tool_result_labeled(self) -> None:
        result = _to_sharegpt(SAMPLE_MESSAGES)
        tool_msgs = [c for c in result["conversations"] if c["from"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "[grep_search]" in tool_msgs[0]["value"]


# -- Exporter integration tests ---------------------------------------------


class TestTrainingExporter:
    def test_export_session_openai(self, tmp_path: Path) -> None:
        session_file = _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        exporter = TrainingExporter()
        result = exporter.export_session(session_file, fmt="openai")
        assert result is not None
        assert "messages" in result

    def test_export_session_chatml(self, tmp_path: Path) -> None:
        session_file = _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        exporter = TrainingExporter()
        result = exporter.export_session(session_file, fmt="chatml")
        assert result is not None
        assert "text" in result

    def test_export_session_sharegpt(self, tmp_path: Path) -> None:
        session_file = _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        exporter = TrainingExporter()
        result = exporter.export_session(session_file, fmt="sharegpt")
        assert result is not None
        assert "conversations" in result

    def test_export_session_empty_file(self, tmp_path: Path) -> None:
        session_file = tmp_path / "empty.conversation.jsonl"
        session_file.write_text("", encoding="utf-8")
        exporter = TrainingExporter()
        assert exporter.export_session(session_file, fmt="openai") is None

    def test_export_session_invalid_format(self, tmp_path: Path) -> None:
        session_file = _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        exporter = TrainingExporter()
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_session(session_file, fmt="invalid")

    def test_export_all_basic(self, tmp_path: Path) -> None:
        _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        _make_session_file(tmp_path, "sess-2", SAMPLE_MESSAGES)
        output = tmp_path / "output.jsonl"

        exporter = TrainingExporter()
        stats = exporter.export_all(tmp_path, output, fmt="openai")

        assert stats.sessions_scanned == 2
        assert stats.sessions_exported == 2
        assert output.exists()

        # Verify each line is valid JSON
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "messages" in parsed

    def test_export_all_with_filters(self, tmp_path: Path) -> None:
        _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        # Session with no tool calls
        no_tools = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        _make_session_file(tmp_path, "sess-2", no_tools)
        output = tmp_path / "output.jsonl"

        exporter = TrainingExporter()
        stats = exporter.export_all(
            tmp_path, output, fmt="openai", filters=ExportFilters(min_tool_calls=1)
        )

        assert stats.sessions_exported == 1
        assert stats.sessions_filtered == 1

    def test_export_all_max_sessions(self, tmp_path: Path) -> None:
        for i in range(5):
            _make_session_file(tmp_path, f"sess-{i}", SAMPLE_MESSAGES)
        output = tmp_path / "output.jsonl"

        exporter = TrainingExporter()
        stats = exporter.export_all(
            tmp_path, output, fmt="openai", filters=ExportFilters(max_sessions=2)
        )

        assert stats.sessions_exported == 2

    def test_export_all_sharegpt(self, tmp_path: Path) -> None:
        _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        output = tmp_path / "output.jsonl"

        exporter = TrainingExporter()
        stats = exporter.export_all(tmp_path, output, fmt="sharegpt")

        assert stats.sessions_exported == 1
        line = json.loads(output.read_text(encoding="utf-8").strip())
        assert "conversations" in line

    def test_export_stats_counts(self, tmp_path: Path) -> None:
        _make_session_file(tmp_path, "sess-1", SAMPLE_MESSAGES)
        output = tmp_path / "output.jsonl"

        exporter = TrainingExporter()
        stats = exporter.export_all(tmp_path, output, fmt="openai")

        assert stats.total_messages == len(SAMPLE_MESSAGES)
        assert stats.total_tool_calls == 2
        assert stats.errors == []
