"""Additional tests for verify tool to increase coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.verify import (
    VerifyTool,
    _has_lint_issues,
    _one_shot_verify,
    _run_fix,
    _verify_js_ts,
    _verify_with_retry,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.fixture
def tool() -> VerifyTool:
    return VerifyTool()


class TestExecuteInvalidInput:
    @pytest.mark.asyncio
    async def test_file_path_not_string(self, tool: VerifyTool, ctx: ToolContext) -> None:
        """Test execute with non-string file_path."""
        result = await tool.execute({"file_path": 123}, ctx)
        assert result.is_error
        assert "must be a non-empty string" in result.error

    @pytest.mark.asyncio
    async def test_file_path_none(self, tool: VerifyTool, ctx: ToolContext) -> None:
        """Test execute with None file_path."""
        result = await tool.execute({"file_path": None}, ctx)
        assert result.is_error


class TestVerifyPython:
    @patch("shutil.which", return_value=None)
    @pytest.mark.asyncio
    async def test_python_ruff_not_found(self, mock_which, tmp_path: Path) -> None:
        """Test Python verify when ruff is not found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        tool = VerifyTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute({"file_path": "test.py"}, ctx)
        assert not result.is_error  # Returns success with skip message
        assert "ruff not found" in result.output


class TestVerifyJsTs:
    @patch("shutil.which", side_effect=[None, None, None])  # biome, eslint, npx all not found
    @pytest.mark.asyncio
    async def test_js_no_linters(self, mock_which, tmp_path: Path) -> None:
        """Test JS verify when no linters are found."""
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        tool = VerifyTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute({"file_path": "test.js"}, ctx)
        assert not result.is_error
        assert "No JS/TS linter found" in result.output

    @patch("shutil.which", side_effect=["biome", None, None])
    @pytest.mark.asyncio
    async def test_js_biome_success(self, mock_which, tmp_path: Path) -> None:
        """Test JS verify with biome."""
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_js_ts(test_file, "test.js", tmp_path)
            assert not result.is_error


class TestVerifyGo:
    @patch("shutil.which", return_value=None)
    @pytest.mark.asyncio
    async def test_go_no_binary(self, mock_which, tmp_path: Path) -> None:
        """Test Go verify when go is not found."""
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        tool = VerifyTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute({"file_path": "test.go"}, ctx)
        assert not result.is_error
        assert "go not found" in result.output


class TestVerifyRust:
    @patch("shutil.which", return_value=None)
    @pytest.mark.asyncio
    async def test_rust_no_cargo(self, mock_which, tmp_path: Path) -> None:
        """Test Rust verify when cargo is not found."""
        test_file = tmp_path / "test.rs"
        test_file.write_text("fn main() {}\n")
        tool = VerifyTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute({"file_path": "test.rs"}, ctx)
        assert not result.is_error
        assert "cargo not found" in result.output

    @patch("shutil.which", return_value="/usr/bin/cargo")
    @pytest.mark.asyncio
    async def test_rust_no_cargo_toml(self, mock_which, tmp_path: Path) -> None:
        """Test Rust verify when Cargo.toml is not found."""
        test_file = tmp_path / "test.rs"
        test_file.write_text("fn main() {}\n")
        tool = VerifyTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute({"file_path": "test.rs"}, ctx)
        assert not result.is_error
        assert "No Cargo.toml found" in result.output


class TestVerifyCCpp:
    @patch("shutil.which", return_value=None)
    @pytest.mark.asyncio
    async def test_c_cpp_no_clang_tidy(self, mock_which, tmp_path: Path) -> None:
        """Test C/C++ verify when clang-tidy is not found."""
        test_file = tmp_path / "test.c"
        test_file.write_text("#include <stdio.h>\n")
        tool = VerifyTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute({"file_path": "test.c"}, ctx)
        assert not result.is_error
        assert "clang-tidy not found" in result.output


class TestHasLintIssues:
    def test_no_issues(self) -> None:
        """Test _has_lint_issues with clean result."""
        result = MagicMock(is_error=False, output="Verification passed")
        assert not _has_lint_issues(result)

    def test_has_issues_in_output(self) -> None:
        """Test _has_lint_issues with lint issues."""
        result = MagicMock(is_error=False, output="Lint issues found")
        assert _has_lint_issues(result)

    def test_has_build_issues(self) -> None:
        """Test _has_lint_issues with build issues."""
        result = MagicMock(is_error=False, output="Build issues in file")
        assert _has_lint_issues(result)

    def test_is_error(self) -> None:
        """Test _has_lint_issues with error result."""
        result = MagicMock(is_error=True)
        assert _has_lint_issues(result)


class TestOneShotVerify:
    def test_python(self, tmp_path: Path) -> None:
        """Test _one_shot_verify for Python."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        with patch("godspeed.tools.verify._verify_python") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            _one_shot_verify(test_file, "test.py", "python", tmp_path)
            mock_verify.assert_called_once()

    def test_unsupported_lang(self, tmp_path: Path) -> None:
        """Test _one_shot_verify for unsupported language."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        result = _one_shot_verify(test_file, "test.xyz", "unsupported", tmp_path)
        assert not result.is_error
        assert "No linter configured" in result.output


class TestRunFix:
    def test_python_fix(self, tmp_path: Path) -> None:
        """Test _run_fix for Python."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        with patch("godspeed.tools.verify._verify_python") as mock_verify:
            _run_fix(test_file, "test.py", "python", tmp_path)
            mock_verify.assert_called_once_with(test_file, "test.py", fix_mode=True)

    def test_js_fix_biome(self, tmp_path: Path) -> None:
        """Test _run_fix for JS with biome."""
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("shutil.which", return_value="/usr/bin/biome"):
            with patch("subprocess.run"):
                _run_fix(test_file, "test.js", "javascript", tmp_path)

    def test_js_fix_eslint(self, tmp_path: Path) -> None:
        """Test _run_fix for JS with eslint."""
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("shutil.which", side_effect=[None, "/usr/bin/eslint"]):
            with patch("subprocess.run"):
                _run_fix(test_file, "test.js", "javascript", tmp_path)

    def test_unsupported_lang(self, tmp_path: Path) -> None:
        """Test _run_fix for unsupported language (no-op)."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        # Should not raise
        _run_fix(test_file, "test.xyz", "unsupported", tmp_path)


class TestVerifyWithRetry:
    def test_non_fixable_lang(self, tmp_path: Path) -> None:
        """Test _verify_with_retry for non-fixable language."""
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        with patch("godspeed.tools.verify._one_shot_verify") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_with_retry(test_file, "test.go", "go", tmp_path, max_retries=3)
            assert not result.is_error

    def test_fixable_lang_no_issues(self, tmp_path: Path) -> None:
        """Test _verify_with_retry when no issues found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        with patch("godspeed.tools.verify._one_shot_verify") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_with_retry(test_file, "test.py", "python", tmp_path, max_retries=3)
            assert not result.is_error
            assert "passed" in result.output

    def test_fixable_lang_with_retries(self, tmp_path: Path) -> None:
        """Test _verify_with_retry with retries."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        with patch("godspeed.tools.verify._one_shot_verify") as mock_verify:
            # First call has issues, second call passes
            mock_verify.side_effect = [
                MagicMock(is_error=False, output="Lint issues"),
                MagicMock(is_error=False, output="passed"),
            ]
            with patch("godspeed.tools.verify._run_fix"):
                result = _verify_with_retry(test_file, "test.py", "python", tmp_path, max_retries=3)
                assert not result.is_error

    def test_max_retries_exhausted(self, tmp_path: Path) -> None:
        """Test _verify_with_retry when retries are exhausted."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        with patch("godspeed.tools.verify._one_shot_verify") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="Lint issues")
            with patch("godspeed.tools.verify._run_fix"):
                result = _verify_with_retry(test_file, "test.py", "python", tmp_path, max_retries=2)
                assert result.is_error
                assert "remaining" in result.error.lower()
