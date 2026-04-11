"""Tests for the verify tool."""

from __future__ import annotations

import shutil

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.verify import VerifyTool


@pytest.fixture
def tool() -> VerifyTool:
    return VerifyTool()


class TestVerifyTool:
    """Test linter verification on files."""

    def test_metadata(self, tool: VerifyTool) -> None:
        assert tool.name == "verify"
        assert tool.risk_level == "read_only"

    @pytest.mark.asyncio
    async def test_clean_python_file(self, tool: VerifyTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "clean.py"
        f.write_text("x = 1\n")
        result = await tool.execute({"file_path": "clean.py"}, tool_context)
        assert not result.is_error
        # Should pass or skip if ruff not available
        assert "clean.py" in result.output or "ruff not found" in result.output

    @pytest.mark.asyncio
    async def test_python_file_with_errors(
        self, tool: VerifyTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "bad.py"
        f.write_text("import os\nimport sys\n")  # unused imports
        result = await tool.execute({"file_path": "bad.py"}, tool_context)
        assert not result.is_error
        if shutil.which("ruff"):
            assert "F401" in result.output  # unused import

    @pytest.mark.asyncio
    async def test_non_python_file_skipped(
        self, tool: VerifyTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "data.json"
        f.write_text('{"key": "value"}\n')
        result = await tool.execute({"file_path": "data.json"}, tool_context)
        assert not result.is_error
        assert "no linter" in result.output.lower() or "skipping" in result.output.lower()

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool: VerifyTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": "missing.py"}, tool_context)
        assert result.is_error
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_file_path(self, tool: VerifyTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": ""}, tool_context)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_pyi_file_supported(self, tool: VerifyTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "stub.pyi"
        f.write_text("def foo(x: int) -> str: ...\n")
        result = await tool.execute({"file_path": "stub.pyi"}, tool_context)
        assert not result.is_error


class TestAutoVerifyInLoop:
    """Test that auto-verification triggers after file edits in agent loop."""

    @pytest.mark.asyncio
    async def test_verify_called_after_file_edit(self, tool_context: ToolContext) -> None:
        """After a successful file_edit on .py, verify result is in conversation."""
        import json
        from unittest.mock import AsyncMock

        from godspeed.agent.conversation import Conversation
        from godspeed.agent.loop import agent_loop
        from godspeed.llm.client import ChatResponse, LLMClient
        from godspeed.tools.file_edit import FileEditTool
        from godspeed.tools.registry import ToolRegistry

        # Set up a real Python file
        f = tool_context.cwd / "test.py"
        f.write_text("x = 1\n")

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(FileEditTool())
        registry.register(VerifyTool())

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_001",
                            "function": {
                                "name": "file_edit",
                                "arguments": json.dumps(
                                    {
                                        "file_path": "test.py",
                                        "old_string": "x = 1",
                                        "new_string": "x = 2",
                                    }
                                ),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="Done editing.", tool_calls=[], finish_reason="stop"),
            ]
        )

        result = await agent_loop("Edit test.py", conversation, client, registry, tool_context)
        assert "Done editing" in result

        # Check that a verify result was injected into conversation
        messages = conversation.messages
        verify_found = any(
            msg.get("role") == "tool" and "call_001_verify" in msg.get("tool_call_id", "")
            for msg in messages
        )
        assert verify_found, "Auto-verify result should be in conversation"

    @pytest.mark.asyncio
    async def test_no_verify_for_non_python(self, tool_context: ToolContext) -> None:
        """Non-Python file edits should not trigger auto-verify."""
        import json
        from unittest.mock import AsyncMock

        from godspeed.agent.conversation import Conversation
        from godspeed.agent.loop import agent_loop
        from godspeed.llm.client import ChatResponse, LLMClient
        from godspeed.tools.file_edit import FileEditTool
        from godspeed.tools.registry import ToolRegistry

        f = tool_context.cwd / "data.txt"
        f.write_text("old content\n")

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(FileEditTool())
        registry.register(VerifyTool())

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_002",
                            "function": {
                                "name": "file_edit",
                                "arguments": json.dumps(
                                    {
                                        "file_path": "data.txt",
                                        "old_string": "old content",
                                        "new_string": "new content",
                                    }
                                ),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="Done.", tool_calls=[], finish_reason="stop"),
            ]
        )

        result = await agent_loop("Edit data.txt", conversation, client, registry, tool_context)
        assert "Done" in result

        # No verify result should be present
        messages = conversation.messages
        verify_found = any(
            msg.get("role") == "tool" and "verify" in msg.get("tool_call_id", "")
            for msg in messages
        )
        assert not verify_found, "No auto-verify for non-Python files"
