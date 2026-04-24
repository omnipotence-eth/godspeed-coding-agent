"""Tests for file write tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from godspeed.tools.base import ToolContext
from godspeed.tools.file_write import FileWriteTool


def _ctx(cwd: Path) -> ToolContext:
    return ToolContext(cwd=cwd, session_id="test")


class TestFileWriteTool:
    """Test FileWriteTool."""

    def test_metadata(self) -> None:
        tool = FileWriteTool()
        assert tool.name == "file_write"
        assert tool.risk_level == "low"
        assert "file_path" in tool.get_schema()["required"]

    def test_write_new_file(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "hello.txt", "content": "Hello, world!"}, _ctx(tmp_path))
        )
        assert result.success
        assert (tmp_path / "hello.txt").read_text() == "Hello, world!"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute(
                {"file_path": "a/b/c/deep.txt", "content": "nested"},
                _ctx(tmp_path),
            )
        )
        assert result.success
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text() == "nested"

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        (tmp_path / "existing.txt").write_text("old content")
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute(
                {"file_path": "existing.txt", "content": "new content"},
                _ctx(tmp_path),
            )
        )
        assert result.success
        assert (tmp_path / "existing.txt").read_text() == "new content"

    def test_empty_path_fails(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "", "content": "x"}, _ctx(tmp_path))
        )
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    def test_empty_content_writes_empty(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "empty.txt", "content": ""}, _ctx(tmp_path))
        )
        assert result.success
        assert (tmp_path / "empty.txt").read_text() == ""

    def test_reports_bytes_written(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "size.txt", "content": "12345"}, _ctx(tmp_path))
        )
        assert "5 bytes" in result.output
