"""Tests for godspeed.tools.verify."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.verify import _EXTENSION_MAP, _FIXABLE_LANGUAGES, VerifyTool, _run_linter


class TestConstants:
    def test_extension_map_not_empty(self):
        assert len(_EXTENSION_MAP) > 0
        assert ".py" in _EXTENSION_MAP
        assert ".js" in _EXTENSION_MAP
        assert ".go" in _EXTENSION_MAP

    def test_fixable_languages(self):
        assert len(_FIXABLE_LANGUAGES) > 0
        assert "python" in _FIXABLE_LANGUAGES
        assert "javascript" in _FIXABLE_LANGUAGES


class TestRunLinter:
    def test_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK\n"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_linter(["ruff", "check", "test.py"], "test.py", "ruff")
            assert result.is_error is False
            assert "passed" in result.output.lower()

    def test_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error\n"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_linter(["ruff", "check", "test.py"], "test.py", "ruff")
            assert result.is_error is False  # Verify returns success with issues
            assert "issues" in result.output.lower() or "error" in result.output.lower()

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("timed out", 30)):
            result = _run_linter(["ruff", "check", "test.py"], "test.py", "ruff")
            assert result.is_error is True
            assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_not_found(self):
        with patch("subprocess.run", side_effect=OSError("not found")):
            result = _run_linter(["ruff", "check", "test.py"], "test.py", "ruff")
            assert result.is_error is True
            assert "not found" in result.error.lower() or "failed" in result.error.lower()


class TestVerifyToolMetadata:
    def test_name(self):
        tool = VerifyTool()
        assert tool.name == "verify"

    def test_risk_level(self):
        tool = VerifyTool()
        assert tool.risk_level.value == "read_only"

    def test_description_contains_keywords(self):
        tool = VerifyTool()
        desc = tool.description.lower()
        assert "verify" in desc or "lint" in desc

    def test_get_schema(self):
        tool = VerifyTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"]
        assert "file_path" in schema["required"]


class TestVerifyToolExecute:
    @pytest.mark.asyncio
    async def test_missing_file_path(self):
        tool = VerifyTool()
        result = await tool.execute({}, MagicMock())
        assert result.is_error is True
        assert "file_path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_file_path(self):
        tool = VerifyTool()
        result = await tool.execute({"file_path": ""}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path):
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        result = await tool.execute({"file_path": "nonexistent.py"}, context)
        assert result.is_error is True
        assert "not found" in result.error.lower() or "exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unsupported_extension(self, tmp_path):
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        result = await tool.execute({"file_path": "test.xyz"}, context)
        assert result.is_error is False  # Returns success with skip message
        assert "no linter" in result.output.lower() or "skipping" in result.output.lower()

    @pytest.mark.asyncio
    async def test_python_verify_success(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        with patch(
            "godspeed.tools.verify._run_linter",
            return_value=MagicMock(is_error=False, output="passed"),
        ):
            result = await tool.execute({"file_path": "test.py"}, context)
            assert result.is_error is False

    @pytest.mark.asyncio
    async def test_python_verify_with_errors(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("import os\nx = 1\n")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        with patch("godspeed.tools.verify._run_linter") as mock_lint:
            mock_lint.return_value = MagicMock(is_error=False, output="F401 imported but unused")
            result = await tool.execute({"file_path": "test.py"}, context)
            assert result.is_error is False
            assert "F401" in result.output or "unused" in result.output.lower()

    @pytest.mark.asyncio
    async def test_javascript_verify(self, tmp_path):
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        with patch(
            "godspeed.tools.verify._run_linter",
            return_value=MagicMock(is_error=False, output="passed"),
        ):
            result = await tool.execute({"file_path": "test.js"}, context)
            assert result.is_error is False

    @pytest.mark.asyncio
    async def test_go_verify(self, tmp_path):
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        with patch(
            "godspeed.tools.verify._run_linter",
            return_value=MagicMock(is_error=False, output="passed"),
        ):
            result = await tool.execute({"file_path": "test.go"}, context)
            assert result.is_error is False

    @pytest.mark.asyncio
    async def test_rust_verify(self, tmp_path):
        test_file = tmp_path / "test.rs"
        test_file.write_text("fn main() {}\n")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        with patch(
            "godspeed.tools.verify._run_linter",
            return_value=MagicMock(is_error=False, output="passed"),
        ):
            result = await tool.execute({"file_path": "test.rs"}, context)
            assert result.is_error is False

    @pytest.mark.asyncio
    async def test_c_cpp_verify(self, tmp_path):
        test_file = tmp_path / "test.c"
        test_file.write_text("#include <stdio.h>\n")
        tool = VerifyTool()
        context = MagicMock()
        context.cwd = tmp_path
        with patch(
            "godspeed.tools.verify._run_linter",
            return_value=MagicMock(is_error=False, output="passed"),
        ):
            result = await tool.execute({"file_path": "test.c"}, context)
            assert result.is_error is False
