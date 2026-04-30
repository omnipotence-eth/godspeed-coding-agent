"""Stress tests for conversation compaction — the most dangerous state mutation."""

from __future__ import annotations

from godspeed.agent.conversation import Conversation

_SYSTEM = "You are a helpful coding assistant."


class TestCompactionPreservesPairs:
    """Property: compact() must preserve every tool_call / tool_result pair."""

    def test_single_pair_preserved(self) -> None:
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("hello")
        conv.add_assistant_message(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "arguments": '{"file_path": "a.py"}',
                    },
                }
            ],
        )
        conv.add_tool_result(tool_call_id="call_1", content="content of a.py")

        conv.compact("Summary of work so far.")

        # After compaction there should be exactly 2 messages: system + summary
        assert len(conv.messages) == 2
        assert conv.messages[1]["role"] == "user"
        assert "Summary" in conv.messages[1]["content"]

    def test_multiple_pairs_preserved(self) -> None:
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("task")
        for i in range(5):
            conv.add_assistant_message(
                content=f"turn {i}",
                tool_calls=[
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": '{"file_path": "x.py"}',
                        },
                    }
                ],
            )
            conv.add_tool_result(tool_call_id=f"call_{i}", content=f"result {i}")

        conv.compact("All 5 turns summarized.")
        assert len(conv.messages) == 2

    def test_empty_conversation_compact(self) -> None:
        """Compacting an empty conversation should not crash."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.compact("Nothing happened.")
        assert len(conv.messages) == 2  # system + summary

    def test_token_cache_invalidated_after_compact(self) -> None:
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("hello")
        _ = conv.token_count  # warm cache
        conv.compact("Summary.")
        # Token count should reflect the compacted state, not the old cache
        assert conv.token_count < 100  # summary is short


class TestCompactionAdversarial:
    """Adversarial: garbage summaries must not corrupt conversation state."""

    def test_garbage_summary_still_valid_message(self) -> None:
        """Even a nonsensical summary must produce a valid message list."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("hello")
        conv.add_assistant_message(content="hi")

        conv.compact("🤖💥🔥" * 1000)  # garbage but valid unicode
        # Must not raise and must have exactly system + user messages
        assert len(conv.messages) == 2
        assert all("role" in m for m in conv.messages)

    def test_empty_string_summary(self) -> None:
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("hello")
        conv.compact("")
        assert len(conv.messages) == 2
        # compact() wraps the summary in a formatted string
        assert "[Conversation compacted" in conv.messages[1]["content"]

    def test_very_long_summary_truncated_gracefully(self) -> None:
        """Summary exceeding context should not OOM or crash."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=1_000)
        conv.add_user_message("hello")
        huge = "word " * 50_000
        conv.compact(huge)
        assert len(conv.messages) == 2
        # Content is wrapped, so length is summary + wrapper text
        assert huge in conv.messages[1]["content"]

    def test_html_injection_in_summary(self) -> None:
        """Summary containing HTML/JS must not break rendering."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("hello")
        conv.compact("<script>alert('xss')</script>")
        assert len(conv.messages) == 2
        assert "<script>" in conv.messages[1]["content"]


class TestCompactionDuringActivity:
    """Simulate compaction triggered while the conversation is mutating."""

    def test_compact_after_tool_result(self) -> None:
        """Compact immediately after adding a tool result — pair must be gone but state valid."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("task")
        conv.add_assistant_message(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ],
        )
        conv.add_tool_result(tool_call_id="call_1", content="file1.py\nfile2.py")

        conv.compact("Listed files.")
        # After compaction, adding another turn should work normally
        conv.add_user_message("next task")
        assert len(conv.messages) == 3  # system + summary + new user

    def test_double_compact(self) -> None:
        """Compacting twice should be idempotent-ish (only one summary remains)."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("task 1")
        conv.add_assistant_message(content="done")
        conv.compact("First summary.")

        conv.add_user_message("task 2")
        conv.add_assistant_message(content="done again")
        conv.compact("Second summary.")

        # Should have system + second summary
        assert len(conv.messages) == 2
        assert "Second summary" in conv.messages[1]["content"]

    def test_compact_then_add_tool_call(self) -> None:
        """After compaction, new tool calls should work normally."""
        conv = Conversation(system_prompt=_SYSTEM, max_tokens=100_000)
        conv.add_user_message("task")
        conv.compact("Summary.")

        conv.add_assistant_message(
            content="",
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "arguments": '{"file_path": "b.py"}',
                    },
                }
            ],
        )
        conv.add_tool_result(tool_call_id="call_2", content="content")
        assert len(conv.messages) == 4  # system + summary + assistant + tool
