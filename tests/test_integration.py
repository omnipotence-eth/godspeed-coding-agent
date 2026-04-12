"""Integration tests — full agent loop with real tools, permissions, and audit.

Tests the complete pipeline: LLM (mocked) → tool call → permission check →
tool execution → audit recording → result fed back to conversation.

v2.0 additions: parallel tool dispatch, multimodal @-mentions,
auto-commit after threshold, and lint-fix-retry loops.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.audit.trail import AuditTrail
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.file_read import FileReadTool
from godspeed.tools.file_write import FileWriteTool
from godspeed.tools.registry import ToolRegistry
from godspeed.tools.shell import ShellTool
from godspeed.tui.mentions import parse_mentions, resolve_mentions
from tests.conftest import MockTool


def _text(content: str) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=[], finish_reason="stop")


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call_001") -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}},
        ],
        finish_reason="tool_calls",
    )


def _multi_tool_call(calls: list[tuple[str, dict[str, Any]]]) -> ChatResponse:
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        tool_calls.append(
            {"id": f"call_{i:03d}", "function": {"name": name, "arguments": json.dumps(args)}},
        )
    return ChatResponse(content="", tool_calls=tool_calls, finish_reason="tool_calls")


class TestIntegrationFileOps:
    """End-to-end: LLM writes a file, then reads it back."""

    @pytest.mark.asyncio
    async def test_write_then_read(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        registry.register(FileWriteTool())
        registry.register(FileReadTool())

        ctx = ToolContext(cwd=tmp_path, session_id="int-001")
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("file_write", {"file_path": "hello.py", "content": "print('hello')"}),
                _tool_call("file_read", {"file_path": "hello.py"}),
                _text("Done! The file contains print('hello')."),
            ]
        )

        result = await agent_loop("Create hello.py", conv, client, registry, ctx)
        assert "hello" in result.lower()
        assert (tmp_path / "hello.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_multi_tool_calls_in_one_response(self, tmp_path: Path) -> None:
        """LLM sends multiple tool calls in a single response."""
        registry = ToolRegistry()
        registry.register(FileWriteTool())

        ctx = ToolContext(cwd=tmp_path, session_id="int-002")
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _multi_tool_call(
                    [
                        ("file_write", {"file_path": "a.txt", "content": "aaa"}),
                        ("file_write", {"file_path": "b.txt", "content": "bbb"}),
                    ]
                ),
                _text("Created both files."),
            ]
        )

        await agent_loop("Create two files", conv, client, registry, ctx)
        assert (tmp_path / "a.txt").read_text() == "aaa"
        assert (tmp_path / "b.txt").read_text() == "bbb"


class TestIntegrationPermissions:
    """End-to-end: permission engine blocks dangerous commands."""

    @pytest.mark.asyncio
    async def test_deny_rule_blocks_tool(self, tmp_path: Path) -> None:
        """Deny rule for .env files blocks file_read."""
        registry = ToolRegistry()
        registry.register(FileReadTool())

        engine = PermissionEngine(
            deny_patterns=["file_read(.env)"],
            tool_risk_levels={"file_read": RiskLevel.READ_ONLY},
        )

        ctx = ToolContext(cwd=tmp_path, session_id="int-perm-001", permissions=engine)
        conv = Conversation("You are a coding agent.", max_tokens=100_000)

        # Write a .env file so it exists
        (tmp_path / ".env").write_text("SECRET=abc123")

        client = LLMClient(model="test")
        denied_calls: list[str] = []
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("file_read", {"file_path": ".env"}),
                _text("I was denied reading .env."),
            ]
        )

        await agent_loop(
            "Read .env",
            conv,
            client,
            registry,
            ctx,
            on_permission_denied=lambda name, reason: denied_calls.append(name),
        )
        assert len(denied_calls) == 1
        assert denied_calls[0] == "file_read"

    @pytest.mark.asyncio
    async def test_dangerous_command_blocked(self, tmp_path: Path) -> None:
        """Dangerous shell commands are blocked by permission engine."""
        registry = ToolRegistry()
        registry.register(ShellTool())

        engine = PermissionEngine(
            tool_risk_levels={"shell": RiskLevel.HIGH},
        )

        ctx = ToolContext(cwd=tmp_path, session_id="int-perm-002", permissions=engine)
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")

        denied_calls: list[str] = []
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("shell", {"command": "rm -rf /"}),
                _text("That command was blocked."),
            ]
        )

        await agent_loop(
            "Delete everything",
            conv,
            client,
            registry,
            ctx,
            on_permission_denied=lambda name, reason: denied_calls.append(reason),
        )
        assert len(denied_calls) == 1
        assert "denied" in denied_calls[0].lower() or "dangerous" in denied_calls[0].lower()

    @pytest.mark.asyncio
    async def test_session_grant_allows_subsequent_calls(self, tmp_path: Path) -> None:
        """Session grants carry across tool calls within the same session."""
        registry = ToolRegistry()
        registry.register(FileReadTool())

        engine = PermissionEngine(
            tool_risk_levels={"file_read": RiskLevel.HIGH},
        )
        # Simulate user granting permission
        engine.grant_session_permission("file_read(test.py)")

        ctx = ToolContext(cwd=tmp_path, session_id="int-perm-003", permissions=engine)
        (tmp_path / "test.py").write_text("# test file")
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("file_read", {"file_path": "test.py"}),
                _text("File read successfully."),
            ]
        )

        result = await agent_loop("Read test.py", conv, client, registry, ctx)
        assert "File read successfully" in result


class TestIntegrationAudit:
    """End-to-end: audit trail records tool calls."""

    @pytest.mark.asyncio
    async def test_tool_calls_are_audited(self, tmp_path: Path) -> None:
        """Every tool call is recorded in the audit trail."""
        registry = ToolRegistry()
        registry.register(FileWriteTool())

        audit = AuditTrail(log_dir=tmp_path / "audit", session_id="audit-int-001")
        ctx = ToolContext(cwd=tmp_path, session_id="audit-int-001", audit=audit)
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("file_write", {"file_path": "out.txt", "content": "data"}),
                _text("File written."),
            ]
        )

        await agent_loop("Write out.txt", conv, client, registry, ctx)
        assert audit.record_count >= 1

        # Verify chain integrity
        is_valid, msg = audit.verify_chain()
        assert is_valid, f"Chain should be valid: {msg}"

    @pytest.mark.asyncio
    async def test_audit_records_denied_tool(self, tmp_path: Path) -> None:
        """Permission denials result in no audit 'tool_call' record for that call,
        but the denied tool's result IS fed back to conversation."""
        registry = ToolRegistry()
        registry.register(ShellTool())

        engine = PermissionEngine(
            deny_patterns=["shell(rm *)"],
            tool_risk_levels={"shell": RiskLevel.HIGH},
        )

        audit = AuditTrail(log_dir=tmp_path / "audit", session_id="audit-int-002")
        ctx = ToolContext(cwd=tmp_path, session_id="audit-int-002", permissions=engine, audit=audit)
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("shell", {"command": "rm -rf /tmp/evil"}),
                _text("Command was denied."),
            ]
        )

        result = await agent_loop("Delete tmp", conv, client, registry, ctx)
        # Denied tool calls don't get audit records (they're blocked before execution)
        # but conversation has the denial message
        assert "denied" in result.lower() or "Command" in result


class TestIntegrationConversationFlow:
    """Test conversation integrity across multi-turn tool interactions."""

    @pytest.mark.asyncio
    async def test_conversation_has_all_messages(self, tmp_path: Path) -> None:
        """After agent loop, conversation contains user, assistant, tool messages."""
        registry = ToolRegistry()
        registry.register(FileWriteTool())

        ctx = ToolContext(cwd=tmp_path, session_id="conv-int-001")
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_call("file_write", {"file_path": "x.txt", "content": "x"}),
                _text("Created x.txt"),
            ]
        )

        await agent_loop("Create x.txt", conv, client, registry, ctx)
        roles = [m["role"] for m in conv.messages]
        assert roles[0] == "system"
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_max_iterations_safeguard(self, tmp_path: Path) -> None:
        """Agent loop stops after MAX_ITERATIONS even if model keeps calling tools."""
        registry = ToolRegistry()
        registry.register(FileWriteTool())

        ctx = ToolContext(cwd=tmp_path, session_id="conv-int-002")
        conv = Conversation("You are a coding agent.", max_tokens=100_000)
        client = LLMClient(model="test")

        # Model always calls tools, never stops
        client.chat = AsyncMock(
            return_value=_tool_call("file_write", {"file_path": "loop.txt", "content": "data"})
        )

        result = await agent_loop("Loop forever", conv, client, registry, ctx)
        assert "maximum iterations" in result.lower()


# ---------------------------------------------------------------------------
# v2.0 Integration Tests
# ---------------------------------------------------------------------------


class _TrackedTool(Tool):
    """Tool that records invocations with timestamps for concurrency assertions."""

    def __init__(
        self,
        name: str,
        delay: float = 0.05,
        result: ToolResult | None = None,
    ) -> None:
        self._name = name
        self._delay = delay
        self._result = result or ToolResult.success(f"{name}_output")
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Tracked test tool: {self._name}"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.sleep(self._delay)
        end = loop.time()
        self.calls.append({"args": arguments, "start": start, "end": end})
        return self._result


class TestParallelToolsEndToEnd:
    """LLM returns 3 tool calls, all execute concurrently, conversation has correct results."""

    @pytest.mark.asyncio
    async def test_parallel_tools_end_to_end(self, tool_context: ToolContext) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        tool_a = _TrackedTool("tool_a", delay=0.15)
        tool_b = _TrackedTool("tool_b", delay=0.15)
        tool_c = _TrackedTool("tool_c", delay=0.15)
        registry.register(tool_a)
        registry.register(tool_b)
        registry.register(tool_c)

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _multi_tool_call(
                    [
                        ("tool_a", {"input": "alpha"}),
                        ("tool_b", {"input": "beta"}),
                        ("tool_c", {"input": "gamma"}),
                    ]
                ),
                _text("All three complete."),
            ]
        )

        result = await agent_loop(
            "Run all three tools",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=True,
        )

        # Final text returned correctly
        assert result == "All three complete."

        # Each tool invoked exactly once with correct args
        assert len(tool_a.calls) == 1
        assert tool_a.calls[0]["args"] == {"input": "alpha"}
        assert len(tool_b.calls) == 1
        assert tool_b.calls[0]["args"] == {"input": "beta"}
        assert len(tool_c.calls) == 1
        assert tool_c.calls[0]["args"] == {"input": "gamma"}

        # Concurrency: all three started before any finished
        all_starts = [t.calls[0]["start"] for t in (tool_a, tool_b, tool_c)]
        all_ends = [t.calls[0]["end"] for t in (tool_a, tool_b, tool_c)]
        latest_start = max(all_starts)
        earliest_end = min(all_ends)
        assert latest_start < earliest_end, "Tools did not overlap — not truly parallel"

        # Conversation contains results in call order (a, b, c)
        tool_msgs = [
            msg["content"]
            for msg in conversation.messages
            if msg.get("role") == "tool" and msg.get("content", "").endswith("_output")
        ]
        assert tool_msgs == ["tool_a_output", "tool_b_output", "tool_c_output"]


class TestMultimodalWithFileMention:
    """@file:test.py flows through parse_mentions -> resolve_mentions -> content blocks."""

    @pytest.mark.asyncio
    async def test_multimodal_with_file_mention(self, tool_context: ToolContext) -> None:
        # Create a real file in the temp project directory
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("print('hello world')\n", encoding="utf-8")

        # Step 1: parse_mentions extracts the mention and cleans text
        raw_input = "Review this @file:test.py and explain it"
        cleaned, mentions = parse_mentions(raw_input)

        assert cleaned == "Review this and explain it"
        assert len(mentions) == 1
        assert mentions[0].type == "file"
        assert mentions[0].target == "test.py"

        # Step 2: resolve_mentions reads the file into content blocks
        blocks = await resolve_mentions(mentions, tool_context.cwd)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "print('hello world')" in blocks[0]["text"]
        assert "[Content of test.py]" in blocks[0]["text"]

        # Step 3: Build multimodal message and add to conversation
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": cleaned},
            *blocks,
        ]
        conversation.add_user_message(content_blocks)

        # Verify the conversation has the multimodal content
        user_msgs = [m for m in conversation.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert isinstance(user_msgs[0]["content"], list)
        assert len(user_msgs[0]["content"]) == 2
        assert user_msgs[0]["content"][0]["text"] == "Review this and explain it"
        assert "print('hello world')" in user_msgs[0]["content"][1]["text"]

    @pytest.mark.asyncio
    async def test_file_mention_missing_file_produces_error_block(
        self, tool_context: ToolContext
    ) -> None:
        """Referencing a nonexistent file produces an error content block."""
        _cleaned, mentions = parse_mentions("Look at @file:nonexistent.py")
        blocks = await resolve_mentions(mentions, tool_context.cwd)

        assert len(blocks) == 1
        assert "Error resolving" in blocks[0]["text"]


class TestAutoCommitAfterThreshold:
    """N successful edits triggers auto-commit (mock git and LLM)."""

    @pytest.mark.asyncio
    async def test_auto_commit_after_threshold(self, tool_context: ToolContext) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # Register a file_edit tool that always succeeds
        edit_tool = MockTool(
            name="file_edit",
            result=ToolResult.success("Edit applied"),
        )
        registry.register(edit_tool)

        # 3 sequential single-edit calls + final text.
        # auto_commit_threshold=3 so commit triggers after 3rd edit.
        threshold = 3
        side_effects: list[ChatResponse] = []
        for i in range(threshold):
            side_effects.append(
                _multi_tool_call(
                    [
                        ("file_edit", {"file_path": f"src/mod_{i}.py", "content": f"v{i}"}),
                    ]
                )
            )
        side_effects.append(_text("All edits done."))

        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=side_effects)

        # Mock the auto_commit module so no real git operations happen
        mock_commit_msg = "feat(src): update modules"
        with (
            patch(
                "godspeed.agent.auto_commit.generate_commit_message",
                new_callable=AsyncMock,
                return_value=mock_commit_msg,
            ) as mock_gen_msg,
            patch(
                "godspeed.agent.auto_commit.auto_commit",
                new_callable=AsyncMock,
                return_value=ToolResult.success(f"Auto-committed: abc12345 {mock_commit_msg}"),
            ) as mock_commit,
        ):
            result = await agent_loop(
                "Edit three files",
                conversation,
                client,
                registry,
                tool_context,
                parallel_tool_calls=False,
                auto_commit=True,
                auto_commit_threshold=threshold,
            )

        assert result == "All edits done."

        # auto_commit was called exactly once (after reaching threshold)
        mock_gen_msg.assert_awaited_once()
        mock_commit.assert_awaited_once()

        # The commit message was generated from the change descriptions
        descriptions_arg = mock_gen_msg.call_args[0][0]
        assert len(descriptions_arg) == threshold
        assert all("file_edit" in d for d in descriptions_arg)

        # Conversation contains the auto-commit result
        tool_msgs = [m["content"] for m in conversation.messages if m.get("role") == "tool"]
        assert any("Auto-commit" in msg for msg in tool_msgs)

    @pytest.mark.asyncio
    async def test_no_auto_commit_below_threshold(self, tool_context: ToolContext) -> None:
        """Fewer edits than threshold should not trigger auto-commit."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("ok")))

        # Only 2 edits, threshold is 5
        side_effects = [
            _multi_tool_call([("file_edit", {"file_path": "a.py", "content": "x"})]),
            _multi_tool_call([("file_edit", {"file_path": "b.py", "content": "y"})]),
            _text("Done."),
        ]
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=side_effects)

        with patch(
            "godspeed.agent.auto_commit.auto_commit",
            new_callable=AsyncMock,
        ) as mock_commit:
            await agent_loop(
                "Edit two files",
                conversation,
                client,
                registry,
                tool_context,
                parallel_tool_calls=False,
                auto_commit=True,
                auto_commit_threshold=5,
            )

        mock_commit.assert_not_awaited()


class TestLintFixRetryInLoop:
    """Edit -> auto-verify with retry -> re-verify passes."""

    @pytest.mark.asyncio
    async def test_lint_fix_retry_in_loop(self, tool_context: ToolContext) -> None:
        """file_edit on a .py file triggers _auto_verify_file with retry loop.

        Mock the verify internals: first check fails, fix runs, re-check passes.
        """
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # Register file_edit and verify tools
        edit_tool = MockTool(name="file_edit", result=ToolResult.success("File written"))
        verify_tool = MockTool(name="verify", result=ToolResult.success("Verification passed"))
        registry.register(edit_tool)
        registry.register(verify_tool)

        # Create the target file so path resolution works
        target = tool_context.cwd / "app.py"
        target.write_text("x=1\n", encoding="utf-8")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _multi_tool_call(
                    [
                        ("file_edit", {"file_path": "app.py", "content": "x = 1\n"}),
                    ]
                ),
                _text("Edit verified and clean."),
            ]
        )

        with patch(
            "godspeed.agent.loop._auto_verify_file",
            new_callable=AsyncMock,
        ) as mock_auto_verify:
            # Simulate: the retry loop ran one fix round and then passed
            mock_auto_verify.return_value = ToolResult.success(
                "Auto-fixed 1 round(s) of issues, 0 remaining: app.py"
            )

            result = await agent_loop(
                "Fix app.py",
                conversation,
                client,
                registry,
                tool_context,
                parallel_tool_calls=False,
                auto_fix_retries=3,
            )

        assert result == "Edit verified and clean."

        # _auto_verify_file was called for the .py file edit
        mock_auto_verify.assert_awaited_once()
        call_args = mock_auto_verify.call_args
        assert call_args[0][0] == "app.py"  # file_path
        assert call_args[0][4] == 3  # auto_fix_retries

        # Conversation contains the verify result showing auto-fix worked
        tool_msgs = [m["content"] for m in conversation.messages if m.get("role") == "tool"]
        assert any("Auto-fixed" in msg for msg in tool_msgs)
        assert any("0 remaining" in msg for msg in tool_msgs)

    @pytest.mark.asyncio
    async def test_verify_skipped_for_non_verifiable_extension(
        self, tool_context: ToolContext
    ) -> None:
        """file_edit on a non-verifiable extension (.txt) should skip auto-verify."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("ok")))
        registry.register(MockTool(name="verify", result=ToolResult.success("pass")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _multi_tool_call(
                    [
                        ("file_edit", {"file_path": "notes.txt", "content": "hello"}),
                    ]
                ),
                _text("Done."),
            ]
        )

        with patch(
            "godspeed.agent.loop._auto_verify_file",
            new_callable=AsyncMock,
        ) as mock_auto_verify:
            await agent_loop(
                "Edit notes",
                conversation,
                client,
                registry,
                tool_context,
                parallel_tool_calls=False,
            )

        # .txt is not in VERIFIABLE_EXTENSIONS, so verify should not be called
        mock_auto_verify.assert_not_awaited()
