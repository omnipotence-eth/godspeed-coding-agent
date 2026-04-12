"""Integration tests for @-mention wiring: completion → parsing → resolution → conversation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from godspeed.agent.conversation import Conversation
from godspeed.tui.completions import MENTION_TYPES, GodspeedCompleter
from godspeed.tui.mentions import parse_mentions, resolve_mentions

# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------


class TestMentionTabCompletion:
    """Tab completion for @-mentions in the TUI."""

    def test_bare_at_shows_three_mention_types(self) -> None:
        """Typing bare `@` should suggest file, folder, and web."""
        completer = GodspeedCompleter(cwd=Path("."))
        doc = Document("@", cursor_position=1)
        completions = list(completer.get_completions(doc, CompleteEvent()))

        labels = [c.text for c in completions]
        assert len(labels) == len(MENTION_TYPES)
        assert "@file:" in labels
        assert "@folder:" in labels
        assert "@web:" in labels

    def test_file_mention_completes_paths(self, tmp_path: Path) -> None:
        """@file:src/ should return child paths under that directory."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("# main", encoding="utf-8")
        (src_dir / "utils.py").write_text("# utils", encoding="utf-8")
        (src_dir / "subpkg").mkdir()

        completer = GodspeedCompleter(cwd=tmp_path)
        doc = Document("@file:src/", cursor_position=len("@file:src/"))
        completions = list(completer.get_completions(doc, CompleteEvent()))

        texts = [c.text for c in completions]
        assert any("main.py" in t for t in texts)
        assert any("utils.py" in t for t in texts)
        assert any("subpkg" in t for t in texts)

    def test_folder_mention_completes_paths(self, tmp_path: Path) -> None:
        """@folder: should also yield directory path completions."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "tests").mkdir()

        completer = GodspeedCompleter(cwd=tmp_path)
        doc = Document("@folder:", cursor_position=len("@folder:"))
        completions = list(completer.get_completions(doc, CompleteEvent()))

        texts = [c.text for c in completions]
        assert any("docs" in t for t in texts)
        assert any("tests" in t for t in texts)

    def test_partial_type_filters(self) -> None:
        """@fi should narrow completions to @file: only."""
        completer = GodspeedCompleter(cwd=Path("."))
        doc = Document("@fi", cursor_position=3)
        completions = list(completer.get_completions(doc, CompleteEvent()))

        assert len(completions) == 1
        assert completions[0].text == "@file:"


# ---------------------------------------------------------------------------
# skip_user_message prevents double-add
# ---------------------------------------------------------------------------


class TestSkipUserMessage:
    """Verify that skip_user_message=True prevents duplicate user messages."""

    @pytest.mark.asyncio
    async def test_skip_user_message_does_not_add(self) -> None:
        """When skip_user_message=True, user_input must NOT be added again."""
        conversation = Conversation(system_prompt="test", model="gpt-4")
        # Pre-add the user message (simulating TUI pre-adding with content blocks)
        conversation.add_user_message("Hello with @file:test.py context")

        assert len(conversation._messages) == 1

        llm_client = AsyncMock()
        llm_client.chat.return_value = MagicMock(text="response", tool_calls=[], refusal=None)

        tool_registry = MagicMock()
        tool_registry.get_schemas.return_value = []
        tool_context = MagicMock()

        from godspeed.agent.loop import agent_loop

        await agent_loop(
            user_input="Hello with @file:test.py context",
            conversation=conversation,
            llm_client=llm_client,
            tool_registry=tool_registry,
            tool_context=tool_context,
            skip_user_message=True,
            max_iterations=1,
        )

        # Should still be 1 user message, not 2
        user_msgs = [m for m in conversation._messages if m["role"] == "user"]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_without_skip_adds_user_message(self) -> None:
        """Default (skip_user_message=False) adds the user message normally."""
        conversation = Conversation(system_prompt="test", model="gpt-4")

        llm_client = AsyncMock()
        llm_client.chat.return_value = MagicMock(text="ok", tool_calls=[], refusal=None)

        tool_registry = MagicMock()
        tool_registry.get_schemas.return_value = []
        tool_context = MagicMock()

        from godspeed.agent.loop import agent_loop

        await agent_loop(
            user_input="plain message",
            conversation=conversation,
            llm_client=llm_client,
            tool_registry=tool_registry,
            tool_context=tool_context,
            skip_user_message=False,
            max_iterations=1,
        )

        user_msgs = [m for m in conversation._messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "plain message"


# ---------------------------------------------------------------------------
# End-to-end: parse → resolve → content block in conversation
# ---------------------------------------------------------------------------


class TestEndToEndMentionFlow:
    """Full pipeline: user input with mention → parsed → resolved → conversation."""

    @pytest.mark.asyncio
    async def test_file_mention_to_content_block(self, tmp_path: Path) -> None:
        """A file mention should be parsed, resolved, and added as a content block."""
        # Setup: create a real file
        test_file = tmp_path / "config.yaml"
        test_file.write_text("key: value\nport: 8080", encoding="utf-8")

        # Step 1: Parse
        user_text = "Explain @file:config.yaml please"
        cleaned, mentions = parse_mentions(user_text)

        assert cleaned == "Explain please"
        assert len(mentions) == 1
        assert mentions[0].type == "file"
        assert mentions[0].target == "config.yaml"

        # Step 2: Resolve
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "key: value" in blocks[0]["text"]
        assert "port: 8080" in blocks[0]["text"]

        # Step 3: Build multimodal content and add to conversation
        content_parts: list[dict[str, str]] = [{"type": "text", "text": cleaned}]
        content_parts.extend(blocks)

        conversation = Conversation(system_prompt="You are a helper.", model="gpt-4")
        conversation.add_user_message(content_parts)

        msgs = conversation.messages
        # messages[0] = system, messages[1] = user
        assert len(msgs) == 2
        user_content = msgs[1]["content"]

        assert isinstance(user_content, list)
        assert len(user_content) == 2
        assert user_content[0]["text"] == "Explain please"
        assert "config.yaml" in user_content[1]["text"]
        assert "key: value" in user_content[1]["text"]

    @pytest.mark.asyncio
    async def test_multiple_mentions_end_to_end(self, tmp_path: Path) -> None:
        """Multiple mentions resolve to multiple content blocks."""
        (tmp_path / "a.py").write_text("def hello(): ...", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "b.txt").touch()

        user_text = "Compare @file:a.py and @folder:subdir"
        cleaned, mentions = parse_mentions(user_text)

        assert len(mentions) == 2

        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 2
        assert "def hello()" in blocks[0]["text"]
        assert "b.txt" in blocks[1]["text"]

        # Wire into conversation
        content_parts: list[dict[str, str]] = [{"type": "text", "text": cleaned}]
        content_parts.extend(blocks)

        conversation = Conversation(system_prompt="sys", model="gpt-4")
        conversation.add_user_message(content_parts)

        user_content = conversation.messages[1]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 3  # cleaned text + 2 resolved blocks

    @pytest.mark.asyncio
    async def test_failed_mention_still_produces_error_block(self, tmp_path: Path) -> None:
        """A mention pointing to a missing file still adds an error block."""
        user_text = "Read @file:nonexistent.py"
        cleaned, mentions = parse_mentions(user_text)

        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert "Error" in blocks[0]["text"]

        # Error block still gets wired into conversation
        content_parts: list[dict[str, str]] = [{"type": "text", "text": cleaned}]
        content_parts.extend(blocks)

        conversation = Conversation(system_prompt="sys", model="gpt-4")
        conversation.add_user_message(content_parts)

        user_content = conversation.messages[1]["content"]
        assert isinstance(user_content, list)
        assert any("Error" in block["text"] for block in user_content if "text" in block)
