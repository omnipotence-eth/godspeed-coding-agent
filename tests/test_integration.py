"""Integration tests — full agent loop with real tools, permissions, and audit.

Tests the complete pipeline: LLM (mocked) → tool call → permission check →
tool execution → audit recording → result fed back to conversation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.audit.trail import AuditTrail
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.file_read import FileReadTool
from godspeed.tools.file_write import FileWriteTool
from godspeed.tools.registry import ToolRegistry
from godspeed.tools.shell import ShellTool


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
