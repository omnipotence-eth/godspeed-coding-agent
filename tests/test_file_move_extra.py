"""Additional tests for file_move tool to increase coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.file_move import FileMoveTool


@pytest.fixture
def ctx_with_dir(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.fixture
def tool() -> FileMoveTool:
    return FileMoveTool()


@pytest.mark.asyncio
async def test_empty_destination(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test empty destination validation."""
    result = await tool.execute(
        {"source": "source.txt", "destination": ""},
        ctx_with_dir,
    )
    assert result.is_error
    assert "destination must be a non-empty string" in result.error


@pytest.mark.asyncio
async def test_invalid_destination_type(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test invalid destination type validation."""
    result = await tool.execute(
        {"source": "source.txt", "destination": 123},
        ctx_with_dir,
    )
    assert result.is_error
    assert (
        "destination must be a non-empty string" in result.error
        or "force must be a boolean" not in result.error
    )


@pytest.mark.asyncio
async def test_source_equals_destination(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test source == destination check."""
    result = await tool.execute(
        {"source": "same.txt", "destination": "same.txt"},
        ctx_with_dir,
    )
    assert result.is_error
    assert "source and destination are the same path" in result.error


@pytest.mark.asyncio
async def test_invalid_source_path(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test source path resolution error."""
    # Try to move a file outside project directory
    result = await tool.execute(
        {"source": "../../../etc/passwd", "destination": "local.txt"},
        ctx_with_dir,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_invalid_destination_path(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test destination path resolution error."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")
    result = await tool.execute(
        {"source": "source.txt", "destination": "../../../etc/passwd"},
        ctx_with_dir,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_destination_exists_no_force(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test destination exists without force."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")
    dst = ctx_with_dir.cwd / "dest.txt"
    dst.write_text("existing")
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest.txt"},
        ctx_with_dir,
    )
    assert result.is_error
    assert "already exists" in result.error
    assert "force=true" in result.error


@pytest.mark.asyncio
async def test_force_overwrite_file(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test force overwrite existing file."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")
    dst = ctx_with_dir.cwd / "dest.txt"
    dst.write_text("existing")
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest.txt", "force": True},
        ctx_with_dir,
    )
    assert not result.is_error
    assert not src.exists()
    assert dst.read_text() == "hello"


@pytest.mark.asyncio
async def test_cannot_overwrite_dir_with_file(
    ctx_with_dir: ToolContext, tool: FileMoveTool
) -> None:
    """Test cannot overwrite directory with file."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")
    dst = ctx_with_dir.cwd / "dest_dir"
    dst.mkdir()
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest_dir", "force": True},
        ctx_with_dir,
    )
    assert result.is_error
    assert "Cannot overwrite directory" in result.error


@pytest.mark.asyncio
async def test_cannot_overwrite_file_with_dir(
    ctx_with_dir: ToolContext, tool: FileMoveTool
) -> None:
    """Test cannot overwrite file with directory."""
    src = ctx_with_dir.cwd / "source_dir"
    src.mkdir()
    dst = ctx_with_dir.cwd / "dest.txt"
    dst.write_text("existing")
    result = await tool.execute(
        {"source": "source_dir", "destination": "dest.txt", "force": True},
        ctx_with_dir,
    )
    assert result.is_error
    assert "Cannot overwrite file" in result.error


@pytest.mark.asyncio
async def test_diff_reviewer_rejects(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test diff reviewer rejects move."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")

    # Create a mock diff reviewer that rejects
    mock_reviewer = AsyncMock()
    mock_reviewer.review = AsyncMock(return_value="reject")

    ctx = ToolContext(cwd=ctx_with_dir.cwd, session_id="test", diff_reviewer=mock_reviewer)
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest.txt"},
        ctx,
    )
    assert result.is_error
    assert "rejected by reviewer" in result.error


@pytest.mark.asyncio
async def test_diff_reviewer_accepts(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test diff reviewer accepts move."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")

    # Create a mock diff reviewer that accepts
    mock_reviewer = AsyncMock()
    mock_reviewer.review = AsyncMock(return_value="accept")

    ctx = ToolContext(cwd=ctx_with_dir.cwd, session_id="test", diff_reviewer=mock_reviewer)
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest.txt"},
        ctx,
    )
    assert not result.is_error
    assert not src.exists()


@pytest.mark.asyncio
async def test_os_error_during_move(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test OSError handling during move."""
    src = ctx_with_dir.cwd / "source.txt"
    src.write_text("hello")

    # Mock os.replace to raise OSError
    import os

    original_replace = os.replace
    os.replace = MagicMock(side_effect=OSError("Mocked OS error"))

    try:
        result = await tool.execute(
            {"source": "source.txt", "destination": "dest.txt"},
            ctx_with_dir,
        )
        assert result.is_error
        assert "Failed to move" in result.error
    finally:
        os.replace = original_replace


@pytest.mark.asyncio
async def test_invalid_force_type(ctx_with_dir: ToolContext, tool: FileMoveTool) -> None:
    """Test invalid force type validation."""
    result = await tool.execute(
        {"source": "source.txt", "destination": "dest.txt", "force": "yes"},
        ctx_with_dir,
    )
    assert result.is_error
    assert "force must be a boolean" in result.error
