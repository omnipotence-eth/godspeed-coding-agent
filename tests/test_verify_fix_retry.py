"""Tests for lint-fix retry loop (_verify_with_retry)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolResult
from godspeed.tools.verify import (
    _FIXABLE_LANGUAGES,
    _verify_with_retry,
)


@pytest.fixture
def tmp_py_file(tmp_path: Path) -> Path:
    """Create a temporary Python file for testing."""
    f = tmp_path / "example.py"
    f.write_text("x = 1\n")
    return f


class TestVerifyFixRetry:
    """Tests for the _verify_with_retry function."""

    def test_clean_file_no_retry(self, tmp_py_file: Path) -> None:
        """Clean file: verify runs once, returns pass — no fix needed."""
        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                return_value=ToolResult.success("Verification passed (ruff): example.py"),
            ) as mock_verify,
            patch(
                "godspeed.tools.verify._run_fix",
            ) as mock_fix,
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="example.py",
                lang="python",
                cwd=tmp_py_file.parent,
                max_retries=3,
            )

        assert "Verification passed" in result.output
        assert not result.is_error
        mock_verify.assert_called_once()
        mock_fix.assert_not_called()

    def test_fixable_issues_auto_fixed(self, tmp_py_file: Path) -> None:
        """Mock ruff finding fixable issues: verify --fix runs, re-check passes."""
        call_count = 0

        def fake_verify(resolved, display_path, lang, cwd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First check finds issues
                return ToolResult.success("Lint issues in example.py (ruff):\nexample.py:1:1: F401")
            # After fix, clean
            return ToolResult.success("Verification passed (ruff): example.py")

        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                side_effect=fake_verify,
            ),
            patch(
                "godspeed.tools.verify._run_fix",
            ) as mock_fix,
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="example.py",
                lang="python",
                cwd=tmp_py_file.parent,
                max_retries=3,
            )

        assert "Auto-fixed" in result.output
        assert "0 remaining" in result.output
        assert not result.is_error
        mock_fix.assert_called_once()

    def test_unfixable_issues_stop_after_max_retries(self, tmp_py_file: Path) -> None:
        """Unfixable issues: verify stops at max_retries."""
        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                return_value=ToolResult.success(
                    "Lint issues in example.py (ruff):\nexample.py:1:1: E999 SyntaxError"
                ),
            ),
            patch(
                "godspeed.tools.verify._run_fix",
            ) as mock_fix,
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="example.py",
                lang="python",
                cwd=tmp_py_file.parent,
                max_retries=3,
            )

        # After v2.8.0 the retry path returns a failure result when issues
        # remain, so is_error=True and the fingerprint lives in result.error.
        assert "some remaining" in (result.error or "")
        assert result.is_error
        # Fix was attempted max_retries times
        assert mock_fix.call_count == 3

    def test_retry_count_configurable(self, tmp_py_file: Path) -> None:
        """Set max_retries=1: verify only retries once."""
        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                return_value=ToolResult.success(
                    "Lint issues in example.py (ruff):\nexample.py:1:1: E999"
                ),
            ),
            patch(
                "godspeed.tools.verify._run_fix",
            ) as mock_fix,
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="example.py",
                lang="python",
                cwd=tmp_py_file.parent,
                max_retries=1,
            )

        # Fingerprint is in the error field post-v2.8.0 failure-semantic fix.
        assert "some remaining" in (result.error or "")
        assert result.is_error
        assert mock_fix.call_count == 1

    def test_retry_disabled_when_zero(self, tmp_py_file: Path) -> None:
        """max_retries=0: one-shot behavior, no fix attempted."""
        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                return_value=ToolResult.success(
                    "Lint issues in example.py (ruff):\nexample.py:1:1: F401"
                ),
            ) as mock_verify,
            patch(
                "godspeed.tools.verify._run_fix",
            ) as mock_fix,
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="example.py",
                lang="python",
                cwd=tmp_py_file.parent,
                max_retries=0,
            )

        # One-shot: returns the check result directly, no fix
        assert "Lint issues" in result.output
        mock_verify.assert_called_once()
        mock_fix.assert_not_called()

    def test_fix_summary_in_output(self, tmp_py_file: Path) -> None:
        """Result includes 'Auto-fixed' message after successful fix."""
        call_count = 0

        def fake_verify(resolved, display_path, lang, cwd):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two checks find issues
                return ToolResult.success("Lint issues in example.py (ruff):\nexample.py:1:1: W291")
            # Third check is clean
            return ToolResult.success("Verification passed (ruff): example.py")

        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                side_effect=fake_verify,
            ),
            patch(
                "godspeed.tools.verify._run_fix",
            ),
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="example.py",
                lang="python",
                cwd=tmp_py_file.parent,
                max_retries=3,
            )

        assert "Auto-fixed" in result.output
        assert "0 remaining" in result.output

    def test_non_fixable_language_skips_retry(self, tmp_py_file: Path) -> None:
        """Go/Rust/C++ files use one-shot verify, no retry loop."""
        with (
            patch(
                "godspeed.tools.verify._one_shot_verify",
                return_value=ToolResult.success("Build issues in main.go (go vet):\nerror"),
            ) as mock_verify,
            patch(
                "godspeed.tools.verify._run_fix",
            ) as mock_fix,
        ):
            result = _verify_with_retry(
                resolved=tmp_py_file,
                display_path="main.go",
                lang="go",
                cwd=tmp_py_file.parent,
                max_retries=3,
            )

        assert "Build issues" in result.output
        mock_verify.assert_called_once()
        mock_fix.assert_not_called()

    def test_fixable_languages_constant(self) -> None:
        """Verify that only Python and JS/TS are in the fixable set."""
        assert "python" in _FIXABLE_LANGUAGES
        assert "javascript" in _FIXABLE_LANGUAGES
        assert "typescript" in _FIXABLE_LANGUAGES
        assert "go" not in _FIXABLE_LANGUAGES
        assert "rust" not in _FIXABLE_LANGUAGES
        assert "c_cpp" not in _FIXABLE_LANGUAGES
