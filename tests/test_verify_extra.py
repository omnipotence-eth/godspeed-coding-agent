"""Additional tests for verify tool to increase coverage."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.verify import (
    VerifyTool,
    _has_lint_issues,
    _one_shot_verify,
    _run_fix,
    _run_linter,
    _verify_c_cpp,
    _verify_go,
    _verify_js_ts,
    _verify_python,
    _verify_rust,
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


class TestResolveToolPathError:
    """Cover lines 101-102: ValueError from resolve_tool_path."""

    @patch("godspeed.tools.verify.resolve_tool_path")
    @pytest.mark.asyncio
    async def test_resolve_tool_path_raises_value_error(
        self, mock_resolve: MagicMock, tool: VerifyTool, ctx: ToolContext
    ) -> None:
        mock_resolve.side_effect = ValueError("Path is outside the project directory")
        result = await tool.execute({"file_path": "../outside.py"}, ctx)
        assert result.is_error
        assert "outside" in result.error.lower() or "path" in result.error.lower()


class TestRunLinterStderrFallback:
    """Cover stderr fallback in _run_linter when stdout is empty."""

    def test_stderr_used_when_stdout_empty(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error on stderr"
        with patch("subprocess.run", return_value=mock_result):
            result = _run_linter(["ruff", "check", "test.py"], "test.py", "ruff")
            assert "error on stderr" in result.output

    def test_stdout_and_stderr_both_empty(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_linter(["ruff", "check", "test.py"], "test.py", "ruff")
            assert result.is_error is False


class TestVerifyPythonFixMode:
    """Cover fix_mode parameter in _verify_python."""

    @patch("shutil.which", return_value="/bin/ruff")
    def test_fix_mode_true(self, mock_which: MagicMock) -> None:
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            _verify_python(Path("test.py"), "test.py", fix_mode=True)
            cmd = mock_run.call_args[0][0]
            assert "--fix" in cmd

    @patch("shutil.which", return_value="/bin/ruff")
    def test_fix_mode_false(self, mock_which: MagicMock) -> None:
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            _verify_python(Path("test.py"), "test.py", fix_mode=False)
            cmd = mock_run.call_args[0][0]
            assert "--no-fix" in cmd


class TestVerifyJsTsEslint:
    """Cover line 182: eslint found when biome is not."""

    @patch("shutil.which", side_effect=[None, "/usr/bin/eslint"])
    def test_eslint_fallback(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_js_ts(test_file, "test.js", tmp_path)
            assert not result.is_error
            mock_run.assert_called_once()

    @patch("shutil.which", side_effect=[None, None, "/usr/bin/npx"])
    def test_npx_eslint_fallback_with_node_modules(
        self, mock_which: MagicMock, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x = 1;\n")
        (tmp_path / "node_modules").mkdir()
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_js_ts(test_file, "test.ts", tmp_path)
            assert not result.is_error
            mock_run.assert_called_once()

    @patch("shutil.which", side_effect=[None, None, "/usr/bin/npx"])
    def test_npx_eslint_no_node_modules(
        self, mock_which: MagicMock, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x = 1;\n")
        result = _verify_js_ts(test_file, "test.ts", tmp_path)
        assert "No JS/TS linter found" in result.output


class TestVerifyGo:
    """Cover line 205: go vet with go binary found."""

    @patch("shutil.which", return_value="/usr/bin/go")
    def test_go_vet_success(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_go(test_file, "test.go")
            assert not result.is_error

    @patch("shutil.which", return_value="/usr/bin/go")
    def test_go_vet_has_issues(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="issues found")
            result = _verify_go(test_file, "test.go")
            assert "issues" in result.output.lower()


class TestVerifyRustCargo:
    """Cover lines 226, 231-248: Rust cargo check success/failure/timeout/OSError."""

    @patch("shutil.which", return_value="/usr/bin/cargo")
    def test_cargo_check_success(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "src" / "main.rs"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("fn main() {}\n")
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"\n")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "pass"
        with patch("subprocess.run", return_value=mock_result):
            result = _verify_rust(test_file, "src/main.rs")
            assert not result.is_error
            assert "passed" in result.output.lower() or "cargo" in result.output.lower()

    @patch("shutil.which", return_value="/usr/bin/cargo")
    def test_cargo_check_failure(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "src" / "main.rs"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("fn main() {}\n")
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"\n")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "build error in main.rs"
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _verify_rust(test_file, "src/main.rs")
            assert "build error" in result.output

    @patch("shutil.which", return_value="/usr/bin/cargo")
    def test_cargo_check_stderr_empty_fallback_stdout(
        self, mock_which: MagicMock, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "src" / "main.rs"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("fn main() {}\n")
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"\n")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = ""
        mock_result.stdout = "stdout error"
        with patch("subprocess.run", return_value=mock_result):
            result = _verify_rust(test_file, "src/main.rs")
            assert "stdout error" in result.output

    @patch("shutil.which", return_value="/usr/bin/cargo")
    def test_cargo_check_timeout(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "src" / "main.rs"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("fn main() {}\n")
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"\n")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cargo", 30)):
            result = _verify_rust(test_file, "src/main.rs")
            assert result.is_error
            assert "timed out" in result.error.lower()

    @patch("shutil.which", return_value="/usr/bin/cargo")
    def test_cargo_check_oserror(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "src" / "main.rs"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("fn main() {}\n")
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"\n")
        with patch("subprocess.run", side_effect=OSError("cannot execute")):
            result = _verify_rust(test_file, "src/main.rs")
            assert result.is_error
            assert "cannot execute" in result.error

    @patch("shutil.which", return_value="/usr/bin/cargo")
    def test_cargo_toml_in_parent_dir(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "deep" / "nested" / "main.rs"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("fn main() {}\n")
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"\n")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _verify_rust(test_file, "deep/nested/main.rs")
            assert not result.is_error


class TestVerifyCCpp:
    """Cover line 256: clang-tidy found and used."""

    @patch("shutil.which", return_value="/usr/bin/clang-tidy")
    def test_clang_tidy_success(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "test.c"
        test_file.write_text("#include <stdio.h>\n")
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="passed")
            result = _verify_c_cpp(test_file, "test.c")
            assert not result.is_error

    @patch("shutil.which", return_value="/usr/bin/clang-tidy")
    def test_clang_tidy_has_issues(self, mock_which: MagicMock, tmp_path: Path) -> None:
        test_file = tmp_path / "test.cpp"
        test_file.write_text("int main() { return 0; }\n")
        with patch("godspeed.tools.verify._run_linter") as mock_run:
            mock_run.return_value = MagicMock(is_error=False, output="warnings found")
            result = _verify_c_cpp(test_file, "test.cpp")
            assert "warnings" in result.output.lower()


class TestOneShotVerifyAllBranches:
    """Cover lines 333, 335, 337, 339: _one_shot_verify for all languages."""

    def test_one_shot_javascript(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("godspeed.tools.verify._verify_js_ts") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            _one_shot_verify(test_file, "test.js", "javascript", tmp_path)
            mock_verify.assert_called_once()

    def test_one_shot_typescript(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x = 1;\n")
        with patch("godspeed.tools.verify._verify_js_ts") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            _one_shot_verify(test_file, "test.ts", "typescript", tmp_path)
            mock_verify.assert_called_once()

    def test_one_shot_go(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        with patch("godspeed.tools.verify._verify_go") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            _one_shot_verify(test_file, "test.go", "go", tmp_path)
            mock_verify.assert_called_once()

    def test_one_shot_rust(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.rs"
        test_file.write_text("fn main() {}\n")
        with patch("godspeed.tools.verify._verify_rust") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            _one_shot_verify(test_file, "test.rs", "rust", tmp_path)
            mock_verify.assert_called_once()

    def test_one_shot_c_cpp(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.c"
        test_file.write_text("int main() { return 0; }\n")
        with patch("godspeed.tools.verify._verify_c_cpp") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            _one_shot_verify(test_file, "test.c", "c_cpp", tmp_path)
            mock_verify.assert_called_once()


class TestRunFixAllBranches:
    """Cover line 362: eslint fix path in _run_fix."""

    def test_js_fix_eslint_no_biome(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("shutil.which", side_effect=[None, "/usr/bin/eslint"]):
            with patch("subprocess.run") as mock_run:
                _run_fix(test_file, "test.js", "javascript", tmp_path)
                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]
                assert "--fix" in cmd

    def test_js_fix_biome_oserror_suppressed(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("shutil.which", return_value="/usr/bin/biome"):
            with patch("subprocess.run", side_effect=OSError("bad")):
                _run_fix(test_file, "test.js", "javascript", tmp_path)

    def test_ts_fix_biome(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x = 1;\n")
        with patch("shutil.which", return_value="/usr/bin/biome"):
            with patch("subprocess.run") as mock_run:
                _run_fix(test_file, "test.ts", "typescript", tmp_path)
                mock_run.assert_called_once()

    def test_ts_fix_eslint_no_biome(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x = 1;\n")
        with patch("shutil.which", side_effect=[None, "/usr/bin/eslint"]):
            with patch("subprocess.run") as mock_run:
                _run_fix(test_file, "test.ts", "typescript", tmp_path)
                mock_run.assert_called_once()

    def test_go_noop_fix(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n")
        _run_fix(test_file, "test.go", "go", tmp_path)

    def test_fix_no_linter_available(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.js"
        test_file.write_text("var x = 1;\n")
        with patch("shutil.which", side_effect=[None, None]):
            _run_fix(test_file, "test.js", "javascript", tmp_path)


class TestVerifyWithRetryAdditional:
    """Additional retry edge cases."""

    def test_retry_error_result_in_final(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        with patch("godspeed.tools.verify._one_shot_verify") as mock_verify:
            mock_verify.return_value = MagicMock(
                is_error=False, output="Lint issues: E999"
            )
            with patch("godspeed.tools.verify._run_fix"):
                result = _verify_with_retry(
                    test_file, "test.py", "python", tmp_path, max_retries=1
                )
                assert result.is_error
                assert "some remaining" in result.error

    def test_retry_final_error_from_toolresult(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")

        def fake_verify(*args, **kwargs):
            return MagicMock(is_error=True, error="Lint issues found")

        with patch("godspeed.tools.verify._one_shot_verify", side_effect=fake_verify):
            with patch("godspeed.tools.verify._run_fix"):
                result = _verify_with_retry(
                    test_file, "test.py", "python", tmp_path, max_retries=2
                )
                assert result.is_error

    def test_retry_exhausted_error_final_non_error_result(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        call_count = 0

        def fake_verify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(is_error=False, output="Lint issues: E999")
            return MagicMock(is_error=False, output="Lint issues still present: E999")

        with patch("godspeed.tools.verify._one_shot_verify", side_effect=fake_verify):
            with patch("godspeed.tools.verify._run_fix"):
                result = _verify_with_retry(
                    test_file, "test.py", "python", tmp_path, max_retries=1
                )
                assert result.is_error


class TestVerifyToolMultiLangExtensions:
    """Test language routing for all extension types through execute."""

    @pytest.mark.asyncio
    async def test_pyi_extension(self, tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "types.pyi").write_text("x: int\n", encoding="utf-8")
        with patch("godspeed.tools.verify._verify_python") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            await tool.execute({"file_path": "types.pyi"}, ctx)
            mock_verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_jsx_extension(self, tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "app.jsx").write_text("const App = () => null;\n", encoding="utf-8")
        with patch("godspeed.tools.verify._verify_js_ts") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            await tool.execute({"file_path": "app.jsx"}, ctx)
            mock_verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_tsx_extension(self, tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "app.tsx").write_text("const App: React.FC = () => null;\n", encoding="utf-8")
        with patch("godspeed.tools.verify._verify_js_ts") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            await tool.execute({"file_path": "app.tsx"}, ctx)
            mock_verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_hpp_extension(self, tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "header.hpp").write_text("#pragma once\n", encoding="utf-8")
        with patch("godspeed.tools.verify._verify_c_cpp") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            await tool.execute({"file_path": "header.hpp"}, ctx)
            mock_verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_h_extension(self, tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "header.h").write_text("int x;\n", encoding="utf-8")
        with patch("godspeed.tools.verify._verify_c_cpp") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            await tool.execute({"file_path": "header.h"}, ctx)
            mock_verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_cpp_extension(self, tool: VerifyTool, ctx: ToolContext) -> None:
        (ctx.cwd / "main.cpp").write_text("int main() {}\n", encoding="utf-8")
        with patch("godspeed.tools.verify._verify_c_cpp") as mock_verify:
            mock_verify.return_value = MagicMock(is_error=False, output="passed")
            await tool.execute({"file_path": "main.cpp"}, ctx)
            mock_verify.assert_called_once()


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
