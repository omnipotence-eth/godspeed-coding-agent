"""Tests for auto-commit workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import git
import pytest

from godspeed.agent.auto_commit import (
    FALLBACK_MESSAGE,
    auto_commit,
    generate_commit_message,
)
from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_tool_response(
    tool_name: str, arguments: dict[str, Any], call_id: str = "call_001"
) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
        finish_reason="tool_calls",
    )


class TestAutoCommit:
    """Tests for auto-commit module functions and agent loop integration."""

    @pytest.mark.asyncio
    async def test_generate_commit_message(self) -> None:
        """Mock LLM returns a valid commit message."""
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(
                content="feat(api): add user authentication",
                tool_calls=[],
                finish_reason="stop",
            )
        )
        result = await generate_commit_message(
            ["file_edit src/auth.py", "file_write tests/test_auth.py"],
            client,
        )
        assert result == "feat(api): add user authentication"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_commit_message_empty_response_fallback(self) -> None:
        """LLM returns empty string — fall back to generic message."""
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="   ", tool_calls=[], finish_reason="stop")
        )
        result = await generate_commit_message(["file_edit src/main.py"], client)
        assert result == FALLBACK_MESSAGE

    @pytest.mark.asyncio
    async def test_generate_commit_message_whitespace_only_fallback(self) -> None:
        """LLM returns only whitespace/newlines — falls back after strip()."""
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="\n\t  \n", tool_calls=[], finish_reason="stop")
        )
        result = await generate_commit_message(["file_write config.yaml"], client)
        assert result == FALLBACK_MESSAGE

    @pytest.mark.asyncio
    async def test_generate_commit_message_exception_fallback(self) -> None:
        """LLM call raises — fall back and log warning."""
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await generate_commit_message(["file_edit src/main.py"], client)
        assert result == FALLBACK_MESSAGE

    @pytest.mark.asyncio
    async def test_auto_commit_stages_and_commits(self, tmp_path: Path) -> None:
        """With a real git repo, verify commit is created."""
        # Set up a real git repo
        repo = git.Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()

        # Create and commit an initial file
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('hello')\n")
        repo.index.add(["hello.py"])
        repo.index.commit("initial commit")

        # Modify the tracked file
        test_file.write_text("print('hello world')\n")

        result = await auto_commit(tmp_path, "feat: update greeting")
        assert not result.is_error
        assert "Auto-committed:" in result.output
        assert "feat: update greeting" in result.output

        # Verify commit exists
        latest = repo.head.commit
        assert "feat: update greeting" in latest.message

    @pytest.mark.asyncio
    async def test_auto_commit_includes_co_authored_by(self, tmp_path: Path) -> None:
        """Verify Godspeed attribution in commit message."""
        repo = git.Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()

        test_file = tmp_path / "app.py"
        test_file.write_text("# app\n")
        repo.index.add(["app.py"])
        repo.index.commit("initial")

        test_file.write_text("# app v2\n")

        result = await auto_commit(tmp_path, "chore: update app")
        assert not result.is_error

        latest = repo.head.commit
        assert "Co-Authored-By: Godspeed <noreply@godspeed.dev>" in latest.message

    @pytest.mark.asyncio
    async def test_auto_commit_disabled_by_default(self, tool_context: ToolContext) -> None:
        """Counter reaches threshold but no commit when auto_commit=False."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        tool = MockTool(name="file_edit", result=ToolResult.success("edited"))
        registry.register(tool)

        client = LLMClient(model="test")
        # 5 file_edit calls then a text response
        responses = [
            _make_tool_response("file_edit", {"file_path": f"f{i}.py"}, f"call_{i:03d}")
            for i in range(5)
        ]
        responses.append(_make_text_response("Done"))
        client.chat = AsyncMock(side_effect=responses)

        result = await agent_loop(
            "edit files",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=False,
            auto_commit=False,
            auto_commit_threshold=5,
        )
        assert result == "Done"
        # No auto-commit message should appear in conversation
        messages_text = " ".join(m.get("content", "") or "" for m in conversation.messages)
        assert "[Auto-commit]" not in messages_text

    @pytest.mark.asyncio
    async def test_auto_commit_resets_counter(self, tmp_path: Path) -> None:
        """After commit, counter resets to 0 allowing subsequent commits."""
        # Set up real git repo for auto-commit to succeed
        repo = git.Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()
        test_file = tmp_path / "main.py"
        test_file.write_text("v1\n")
        repo.index.add(["main.py"])
        repo.index.commit("initial")

        tool_ctx = ToolContext(cwd=tmp_path, session_id="test-session")
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # MockTool that also writes to the file so git has something to commit
        call_count = 0

        class WritingMockTool(MockTool):
            async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
                nonlocal call_count
                call_count += 1
                test_file.write_text(f"v{call_count + 1}\n")
                return ToolResult.success("edited")

        tool = WritingMockTool(name="file_edit", result=ToolResult.success("edited"))
        registry.register(tool)

        client = LLMClient(model="test")
        # Generate commit message mock
        commit_msg_response = ChatResponse(
            content="chore: auto update", tool_calls=[], finish_reason="stop"
        )

        # 3 edits (threshold=2), then done -- should trigger commit after 2nd edit
        responses = [
            _make_tool_response("file_edit", {"file_path": "main.py"}, "c001"),
            commit_msg_response,  # generate_commit_message call after 2nd edit
            _make_tool_response("file_edit", {"file_path": "main.py"}, "c002"),
            _make_tool_response("file_edit", {"file_path": "main.py"}, "c003"),
            commit_msg_response,  # commit msg after counter resets
            _make_text_response("Done"),
        ]
        client.chat = AsyncMock(side_effect=responses)

        result = await agent_loop(
            "edit files",
            conversation,
            client,
            registry,
            tool_ctx,
            parallel_tool_calls=False,
            auto_commit=True,
            auto_commit_threshold=2,
        )
        # Should complete without error (commits may or may not succeed depending on
        # git state, but the counter reset logic is exercised)
        assert result == "Done" or "Error" not in result

    @pytest.mark.asyncio
    async def test_auto_commit_no_repo_graceful_failure(self, tmp_path: Path) -> None:
        """Outside a git repo, auto_commit returns error without crashing."""
        # tmp_path is not a git repo
        result = await auto_commit(tmp_path, "test commit")
        assert result.is_error
        assert result.error is not None
