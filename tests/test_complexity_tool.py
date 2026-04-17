"""Tests for the complexity tool (v2.7.0)."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.complexity import ComplexityTool, _cc_to_radon_grade


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
        # lizard exits non-zero when functions exceed the threshold
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
