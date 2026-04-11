"""Tests for multi-language verify tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.verify import _EXTENSION_MAP, VerifyTool


@pytest.fixture
def verify_tool() -> VerifyTool:
    return VerifyTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


class TestVerifyToolSchema:
    """Test tool metadata."""

    def test_name(self, verify_tool: VerifyTool) -> None:
        assert verify_tool.name == "verify"

    def test_description_mentions_languages(self, verify_tool: VerifyTool) -> None:
        desc = verify_tool.description
        assert "Python" in desc
        assert "JS/TS" in desc
        assert "Go" in desc
        assert "Rust" in desc


class TestExtensionMap:
    """Test extension → language mapping."""

    def test_python_extensions(self) -> None:
        assert _EXTENSION_MAP[".py"] == "python"
        assert _EXTENSION_MAP[".pyi"] == "python"

    def test_js_extensions(self) -> None:
        assert _EXTENSION_MAP[".js"] == "javascript"
        assert _EXTENSION_MAP[".jsx"] == "javascript"

    def test_ts_extensions(self) -> None:
        assert _EXTENSION_MAP[".ts"] == "typescript"
        assert _EXTENSION_MAP[".tsx"] == "typescript"

    def test_go_extension(self) -> None:
        assert _EXTENSION_MAP[".go"] == "go"

    def test_rust_extension(self) -> None:
        assert _EXTENSION_MAP[".rs"] == "rust"

    def test_c_cpp_extensions(self) -> None:
        assert _EXTENSION_MAP[".c"] == "c_cpp"
        assert _EXTENSION_MAP[".cpp"] == "c_cpp"
        assert _EXTENSION_MAP[".h"] == "c_cpp"


class TestVerifyExecution:
    """Test verify tool execution."""

    @pytest.mark.asyncio
    async def test_missing_file(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        result = await verify_tool.execute({"file_path": "nonexistent.py"}, ctx)
        assert result.is_error
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_path(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        result = await verify_tool.execute({"file_path": ""}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_unsupported_extension(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "file.txt").write_text("hello", encoding="utf-8")
        result = await verify_tool.execute({"file_path": "file.txt"}, ctx)
        assert not result.is_error
        assert "No linter configured" in result.output

    @pytest.mark.asyncio
    async def test_python_file_with_ruff(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        """Test Python verification with ruff available."""
        (ctx.cwd / "good.py").write_text("x = 1\n", encoding="utf-8")
        result = await verify_tool.execute({"file_path": "good.py"}, ctx)
        assert not result.is_error
        # Either passes or reports ruff not found
        assert "passed" in result.output.lower() or "ruff not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_js_file_no_linter(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        """JS files without biome/eslint should report no linter found."""
        (ctx.cwd / "app.js").write_text("const x = 1;", encoding="utf-8")
        with patch("godspeed.tools.verify.shutil.which", return_value=None):
            result = await verify_tool.execute({"file_path": "app.js"}, ctx)
        assert not result.is_error
        assert "No JS/TS linter found" in result.output

    @pytest.mark.asyncio
    async def test_go_file_no_go(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "main.go").write_text("package main", encoding="utf-8")
        with patch("godspeed.tools.verify.shutil.which", return_value=None):
            result = await verify_tool.execute({"file_path": "main.go"}, ctx)
        assert not result.is_error
        assert "go not found" in result.output

    @pytest.mark.asyncio
    async def test_rust_file_no_cargo(self, verify_tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "main.rs").write_text("fn main() {}", encoding="utf-8")
        with patch("godspeed.tools.verify.shutil.which", return_value=None):
            result = await verify_tool.execute({"file_path": "main.rs"}, ctx)
        assert not result.is_error
        assert "cargo not found" in result.output
