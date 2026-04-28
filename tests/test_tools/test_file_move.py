"""Tests for the file_move tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.file_move import FileMoveTool


@pytest.fixture
def ctx_with_dir(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.mark.asyncio
async def test_missing_source(ctx_with_dir: ToolContext) -> None:
    tool = FileMoveTool()
    result = await tool.execute(
        {"source": "nonexistent.txt", "destination": "new.txt"},
        ctx_with_dir,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_empty_source(ctx_with_dir: ToolContext) -> None:
    tool = FileMoveTool()
    result = await tool.execute(
        {"source": "", "destination": "new.txt"},
        ctx_with_dir,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_move_file(ctx_with_dir: ToolContext) -> None:
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")
    tool = FileMoveTool()
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest.txt"},
        ctx_with_dir,
    )
    assert not result.is_error
    assert not src.exists()
    assert (ctx_with_dir.cwd / "dest.txt").read_text() == "hello"


class TestFileMoveToolMetadata:
    def test_name(self) -> None:
        tool = FileMoveTool()
        assert tool.name == "file_move"

    def test_risk_level(self) -> None:
        tool = FileMoveTool()
        assert tool.risk_level.value == "low"

    def test_schema(self) -> None:
        tool = FileMoveTool()
        schema = tool.get_schema()
        assert "source" in schema["properties"]
        assert "destination" in schema["properties"]
        assert "force" in schema["properties"]
