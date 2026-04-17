"""Tests for the coverage tool (v2.6.0)."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.coverage import CoverageTool, _parse_overall_percent


class TestParseOverallPercent:
    def test_parses_standard_total_line(self) -> None:
        report = """Name                     Stmts   Miss  Cover
src/app.py                  50     4    92%
TOTAL                      100     8    92%
"""
        assert _parse_overall_percent(report) == 92.0

    def test_parses_with_branch_column(self) -> None:
        report = "TOTAL    100    4    40    2    95%\n"
        assert _parse_overall_percent(report) == 95.0

    def test_none_when_no_total_line(self) -> None:
        assert _parse_overall_percent("something random") is None

    def test_handles_decimal_percentage(self) -> None:
        report = "TOTAL    100    3    97.5%\n"
        assert _parse_overall_percent(report) == 97.5


class TestCoverageTool:
    @pytest.mark.asyncio
    async def test_missing_coverage_binary_returns_clear_error(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        with patch("godspeed.tools.coverage.shutil.which", return_value=None):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "coverage" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_success_path_returns_report(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()

        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "Name            Stmts   Miss  Cover\n"
                "src/app.py         10      1    90%\n"
                "TOTAL              10      1    90%\n"
            ),
            stderr="",
        )
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=[run_result, report_result],
            ),
        ):
            result = await tool.execute({}, ctx)

        assert not result.is_error
        assert "Coverage: 90.0%" in result.output
        assert "src/app.py" in result.output

    @pytest.mark.asyncio
    async def test_min_percent_fails_when_below_threshold(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[],
            returncode=0,
            stdout="TOTAL    10    5    50%\n",
            stderr="",
        )
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=[run_result, report_result],
            ),
        ):
            result = await tool.execute({"min_percent": 80}, ctx)
        assert result.is_error
        assert "50" in (result.error or "")
        assert "80" in (result.error or "")

    @pytest.mark.asyncio
    async def test_pytest_failure_propagates(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        # pytest returncode 1 = tests failed
        run_result = CompletedProcess(args=[], returncode=1, stdout="fail", stderr="")
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch("godspeed.tools.coverage.subprocess.run", side_effect=[run_result]),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "pytest failed" in (result.error or "")
