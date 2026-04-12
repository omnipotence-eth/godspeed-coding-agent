"""Tests for the image_read tool."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.image_read import ImageReadTool

# Minimal valid PNG: 1x1 pixel, RGB
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx"
    b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal JPEG (SOI + APP0 marker + EOI)
MINIMAL_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"


def _make_context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.fixture
def tool() -> ImageReadTool:
    return ImageReadTool()


@pytest.mark.asyncio
async def test_read_valid_png(tool: ImageReadTool, tmp_path: Path) -> None:
    img = tmp_path / "test.png"
    img.write_bytes(MINIMAL_PNG)
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "test.png"}, ctx)

    assert not result.is_error
    assert result.output.startswith("[Image: test.png] (image/png,")
    assert "data:image/png;base64," in result.output
    # Verify round-trip decoding
    b64_part = result.output.split("data:image/png;base64,", 1)[1]
    assert base64.b64decode(b64_part) == MINIMAL_PNG


@pytest.mark.asyncio
async def test_read_valid_jpg(tool: ImageReadTool, tmp_path: Path) -> None:
    img = tmp_path / "photo.jpg"
    img.write_bytes(MINIMAL_JPEG)
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "photo.jpg"}, ctx)

    assert not result.is_error
    assert "[Image: photo.jpg] (image/jpeg," in result.output
    assert "data:image/jpeg;base64," in result.output


@pytest.mark.asyncio
async def test_read_jpeg_uppercase_ext(tool: ImageReadTool, tmp_path: Path) -> None:
    img = tmp_path / "photo.JPEG"
    img.write_bytes(MINIMAL_JPEG)
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "photo.JPEG"}, ctx)

    assert not result.is_error
    assert "image/jpeg" in result.output


@pytest.mark.asyncio
async def test_reject_unsupported_format_txt(tool: ImageReadTool, tmp_path: Path) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("not an image")
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "notes.txt"}, ctx)

    assert result.is_error
    assert "Unsupported image format" in result.error
    assert ".txt" in result.error


@pytest.mark.asyncio
async def test_reject_unsupported_format_bmp(tool: ImageReadTool, tmp_path: Path) -> None:
    bmp = tmp_path / "image.bmp"
    bmp.write_bytes(b"\x42\x4d" + b"\x00" * 100)
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "image.bmp"}, ctx)

    assert result.is_error
    assert "Unsupported image format" in result.error
    assert ".bmp" in result.error


@pytest.mark.asyncio
async def test_file_not_found(tool: ImageReadTool, tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "nonexistent.png"}, ctx)

    assert result.is_error
    assert "File not found" in result.error


@pytest.mark.asyncio
async def test_reject_file_over_20mb(tool: ImageReadTool, tmp_path: Path) -> None:
    img = tmp_path / "huge.png"
    img.write_bytes(MINIMAL_PNG)
    ctx = _make_context(tmp_path)

    fake_size = 21 * 1024 * 1024
    real_stat = img.stat()

    class FakeStat:
        st_mode = real_stat.st_mode
        st_size = fake_size

    with patch("pathlib.Path.stat", return_value=FakeStat()):
        result = await tool.execute({"file_path": "huge.png"}, ctx)

    assert result.is_error
    assert "too large" in result.error
    assert "20MB" in result.error


@pytest.mark.asyncio
async def test_warn_file_over_5mb(tool: ImageReadTool, tmp_path: Path) -> None:
    img = tmp_path / "big.png"
    # Create a file just over 5MB
    data = MINIMAL_PNG + b"\x00" * (5 * 1024 * 1024 + 1)
    img.write_bytes(data)
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "big.png"}, ctx)

    assert not result.is_error
    assert "WARNING" in result.output
    assert "[Image: big.png]" in result.output


@pytest.mark.asyncio
async def test_path_traversal_blocked(tool: ImageReadTool, tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "../../etc/passwd"}, ctx)

    assert result.is_error
    assert "Access denied" in result.error


@pytest.mark.asyncio
async def test_directory_rejected(tool: ImageReadTool, tmp_path: Path) -> None:
    subdir = tmp_path / "images"
    subdir.mkdir()
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "images"}, ctx)

    assert result.is_error
    assert "Not a file" in result.error


@pytest.mark.asyncio
async def test_empty_file_path(tool: ImageReadTool, tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": ""}, ctx)

    assert result.is_error
    assert "non-empty string" in result.error


@pytest.mark.asyncio
async def test_missing_file_path(tool: ImageReadTool, tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)

    result = await tool.execute({}, ctx)

    assert result.is_error
    assert "non-empty string" in result.error


@pytest.mark.asyncio
async def test_output_contains_size_kb(tool: ImageReadTool, tmp_path: Path) -> None:
    img = tmp_path / "sized.png"
    img.write_bytes(MINIMAL_PNG)
    ctx = _make_context(tmp_path)

    result = await tool.execute({"file_path": "sized.png"}, ctx)

    assert not result.is_error
    # Output should contain size in KB format like "0.1KB"
    assert "KB)" in result.output
