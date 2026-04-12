"""Tests for @-mention parsing and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tui.mentions import Mention, parse_mentions, resolve_mentions


class TestParseMentions:
    """Test mention extraction from user input."""

    def test_parse_file_mention(self) -> None:
        cleaned, mentions = parse_mentions("Read @file:src/main.py please")
        assert cleaned == "Read please"
        assert len(mentions) == 1
        assert mentions[0].type == "file"
        assert mentions[0].target == "src/main.py"
        assert mentions[0].raw == "@file:src/main.py"

    def test_parse_folder_mention(self) -> None:
        cleaned, mentions = parse_mentions("List @folder:src/")
        assert cleaned == "List"
        assert len(mentions) == 1
        assert mentions[0].type == "folder"
        assert mentions[0].target == "src/"

    def test_parse_web_mention(self) -> None:
        cleaned, mentions = parse_mentions("Check @web:https://example.com/docs")
        assert cleaned == "Check"
        assert len(mentions) == 1
        assert mentions[0].type == "web"
        assert mentions[0].target == "https://example.com/docs"

    def test_multiple_mentions(self) -> None:
        text = "Compare @file:a.py with @file:b.py and @folder:tests/"
        cleaned, mentions = parse_mentions(text)
        assert cleaned == "Compare with and"
        assert len(mentions) == 3
        assert mentions[0].target == "a.py"
        assert mentions[1].target == "b.py"
        assert mentions[2].target == "tests/"

    def test_no_mentions_passthrough(self) -> None:
        cleaned, mentions = parse_mentions("Just a normal message")
        assert cleaned == "Just a normal message"
        assert mentions == []

    def test_empty_string(self) -> None:
        cleaned, mentions = parse_mentions("")
        assert cleaned == ""
        assert mentions == []

    def test_mention_at_start(self) -> None:
        cleaned, mentions = parse_mentions("@file:test.py what is this?")
        assert cleaned == "what is this?"
        assert len(mentions) == 1

    def test_mention_at_end(self) -> None:
        cleaned, mentions = parse_mentions("Read this @file:main.py")
        assert cleaned == "Read this"
        assert len(mentions) == 1

    def test_invalid_mention_type_ignored(self) -> None:
        """@unknown:foo should not be parsed as a mention."""
        cleaned, mentions = parse_mentions("Try @unknown:foo bar")
        assert cleaned == "Try @unknown:foo bar"
        assert mentions == []


class TestResolveMentions:
    """Test mention resolution to content blocks."""

    @pytest.mark.asyncio
    async def test_resolve_file_mention(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')", encoding="utf-8")

        mentions = [Mention(type="file", raw="@file:test.py", target="test.py")]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "print('hello')" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_resolve_folder_mention(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "subdir").mkdir()

        mentions = [Mention(type="folder", raw="@folder:.", target=".")]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert "a.py" in blocks[0]["text"]
        assert "b.py" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_resolve_web_https_only(self, tmp_path: Path) -> None:
        """HTTP URLs should be rejected."""
        mentions = [Mention(type="web", raw="@web:http://evil.com", target="http://evil.com")]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert "Error" in blocks[0]["text"]
        assert "HTTPS" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Paths that escape the project directory should be blocked."""
        mentions = [
            Mention(
                type="file",
                raw="@file:../../etc/passwd",
                target="../../etc/passwd",
            )
        ]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert "Error" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path: Path) -> None:
        mentions = [Mention(type="file", raw="@file:missing.py", target="missing.py")]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert "Error" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_nonexistent_folder(self, tmp_path: Path) -> None:
        mentions = [Mention(type="folder", raw="@folder:missing/", target="missing/")]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 1
        assert "Error" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_multiple_mentions_resolved(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("file_a", encoding="utf-8")
        (tmp_path / "b.py").write_text("file_b", encoding="utf-8")

        mentions = [
            Mention(type="file", raw="@file:a.py", target="a.py"),
            Mention(type="file", raw="@file:b.py", target="b.py"),
        ]
        blocks = await resolve_mentions(mentions, tmp_path)

        assert len(blocks) == 2
        assert "file_a" in blocks[0]["text"]
        assert "file_b" in blocks[1]["text"]
