"""Tests for the complexity tool (v2.7.0)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.complexity import (
    COMPLEXITY_TIMEOUT,
    ComplexityTool,
    _cc_to_radon_grade,
    _detect_target,
)


class TestCcToRadonGrade:
    @pytest.mark.parametrize(
        ("cc", "expected"),
        [
            (1, "A"),
            (5, "A"),
            (6, "B"),
            (10, "B"),
            (11, "C"),
            (20, "C"),
            (21, "D"),
            (30, "D"),
            (31, "E"),
            (40, "E"),
            (41, "F"),
            (100, "F"),
        ],
    )
    def test_grade_boundaries(self, cc: int, expected: str) -> None:
        assert _cc_to_radon_grade(cc) == expected


class TestDetectTarget:
    def test_returns_src_when_present(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_target(ctx) == "src"

    def test_returns_dot_when_no_src(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_target(ctx) == "."


class TestComplexityTool:
    @pytest.mark.asyncio
    async def test_no_tools_installed_returns_error(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        with patch("godspeed.tools.complexity.shutil.which", return_value=None):
            result = await tool.execute({}, ctx)
        assert result.is_error
        err = (result.error or "").lower()
        assert "radon" in err
        assert "lizard" in err

    @pytest.mark.asyncio
    async def test_invalid_max_cc_rejected(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        result = await tool.execute({"max_cc": 0}, ctx)
        assert result.is_error
        assert "max_cc" in (result.error or "")

    @pytest.mark.asyncio
    async def test_clean_radon_run(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        clean = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.complexity.shutil.which",
                side_effect=lambda name: "/usr/bin/radon" if name == "radon" else None,
            ),
            patch("godspeed.tools.complexity.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({"target": "src", "max_cc": 10}, ctx)
        assert not result.is_error
        assert "radon" in result.output
        assert "All functions within CC <= 10" in result.output

    @pytest.mark.asyncio
    async def test_radon_findings_produce_error(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        found = CompletedProcess(
            args=[],
            returncode=0,
            stdout="src/app.py\n  F 24:0 bad_func - C (15)",
            stderr="",
        )
        with (
            patch(
                "godspeed.tools.complexity.shutil.which",
                side_effect=lambda name: "/usr/bin/radon" if name == "radon" else None,
            ),
            patch("godspeed.tools.complexity.subprocess.run", return_value=found),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert result.is_error
        assert "bad_func" in (result.error or "")

    @pytest.mark.asyncio
    async def test_lizard_nonzero_exit_is_breach(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        breach = CompletedProcess(
            args=[], returncode=1, stdout="src/a.py:12: complex_func CCN=15", stderr=""
        )
        with (
            patch(
                "godspeed.tools.complexity.shutil.which",
                side_effect=lambda name: "/usr/bin/lizard" if name == "lizard" else None,
            ),
            patch("godspeed.tools.complexity.subprocess.run", return_value=breach),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_radon_timeout_expired(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        with (
            patch(
                "godspeed.tools.complexity.shutil.which",
                side_effect=lambda name: "/usr/bin/radon" if name == "radon" else None,
            ),
            patch(
                "godspeed.tools.complexity.subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["radon", "cc"], timeout=COMPLEXITY_TIMEOUT
                ),
            ),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_lizard_timeout_expired(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        with (
            patch(
                "godspeed.tools.complexity.shutil.which",
                side_effect=lambda name: "/usr/bin/lizard" if name == "lizard" else None,
            ),
            patch(
                "godspeed.tools.complexity.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["lizard"], timeout=COMPLEXITY_TIMEOUT),
            ),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_lizard_clean_with_empty_output(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        clean = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.complexity.shutil.which",
                side_effect=lambda name: "/usr/bin/lizard" if name == "lizard" else None,
            ),
            patch("godspeed.tools.complexity.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({"target": "src", "max_cc": 10}, ctx)
        assert not result.is_error
        assert "lizard" in result.output
        assert "All functions within CC <= 10" in result.output

    @pytest.mark.asyncio
    async def test_detect_target_no_src(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = ComplexityTool()
        with patch("godspeed.tools.complexity.shutil.which", return_value="/usr/bin/radon"):
            with patch("godspeed.tools.complexity.subprocess.run") as mock_run:
                mock_run.return_value = CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                )
                result = await tool.execute({}, ctx)
        assert not result.is_error
        assert "Complexity scan of ." in result.output
