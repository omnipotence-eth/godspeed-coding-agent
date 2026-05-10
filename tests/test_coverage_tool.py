"""Tests for the coverage tool (v2.6.0)."""

from __future__ import annotations

import subprocess
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

    def test_total_line_without_percentage(self) -> None:
        """TOTAL line exists but no % token — returns None."""
        report = "TOTAL    100    0\n"
        assert _parse_overall_percent(report) is None

    def test_total_line_with_unparseable_percentage(self) -> None:
        """Token ends with % but float() fails — returns None."""
        report = "TOTAL    100    0    N/A%\n"
        assert _parse_overall_percent(report) is None

    def test_percentage_not_last_token(self) -> None:
        """Percentage token is not the last one, still parsed correctly."""
        report = "src/app.py   50   4    92%    missing\nTOTAL    100   8    92%    extra\n"
        assert _parse_overall_percent(report) == 92.0

    def test_no_lines(self) -> None:
        """Empty report returns None."""
        assert _parse_overall_percent("") is None


class TestCoverageTool:
    def test_name(self) -> None:
        tool = CoverageTool()
        assert tool.name == "coverage"

    def test_description(self) -> None:
        tool = CoverageTool()
        desc = tool.description
        assert "coverage" in desc.lower()
        assert "pytest" in desc.lower()

    def test_risk_level(self) -> None:
        tool = CoverageTool()
        assert tool.risk_level.value == "read_only"

    def test_get_schema(self) -> None:
        tool = CoverageTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "target" in schema["properties"]
        assert "source" in schema["properties"]
        assert "min_percent" in schema["properties"]

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

    @pytest.mark.asyncio
    async def test_no_tests_returncode_5_not_error(self, tmp_path: Path) -> None:
        """pytest returncode 5 (no tests collected) should not be treated as failure."""
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=5, stdout="no tests", stderr="")
        report_result = CompletedProcess(
            args=[], returncode=0, stdout="TOTAL    0    0    100%\n", stderr=""
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

    @pytest.mark.asyncio
    async def test_with_target_and_source_args(self, tmp_path: Path) -> None:
        """Verify --source and target are passed to coverage run."""
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[], returncode=0, stdout="TOTAL    10    1    90%\n", stderr=""
        )
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=[run_result, report_result],
            ) as mock_run,
        ):
            result = await tool.execute(
                {"target": "tests/test_api.py", "source": "src/mypkg"}, ctx
            )
        assert not result.is_error
        run_call_args = mock_run.call_args_list[0][0][0]
        assert "--source" in run_call_args
        assert "src/mypkg" in run_call_args
        assert "tests/test_api.py" in run_call_args

    @pytest.mark.asyncio
    async def test_coverage_run_timeout(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="coverage run", timeout=180),
            ),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_coverage_binary_file_not_found(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=FileNotFoundError("coverage binary not found"),
            ),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "not executable" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_coverage_report_timeout(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=[
                    run_result,
                    subprocess.TimeoutExpired(cmd="coverage report", timeout=180),
                ],
            ),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_coverage_report_nonzero_exit(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[], returncode=1, stdout="", stderr="report generation failed"
        )
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=[run_result, report_result],
            ),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "report failed" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_min_percent_invalid_type(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[], returncode=0, stdout="TOTAL    10    1    90%\n", stderr=""
        )
        with (
            patch("godspeed.tools.coverage.shutil.which", return_value="/usr/bin/coverage"),
            patch(
                "godspeed.tools.coverage.subprocess.run",
                side_effect=[run_result, report_result],
            ),
        ):
            result = await tool.execute({"min_percent": "eighty"}, ctx)
        assert result.is_error
        assert "numeric" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_min_percent_overall_none(self, tmp_path: Path) -> None:
        """When min_percent is set but overall can't be parsed, return error."""
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[], returncode=0, stdout="No data to report.\n", stderr=""
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
        assert "Could not parse" in (result.error or "")

    @pytest.mark.asyncio
    async def test_min_percent_met_returns_success(self, tmp_path: Path) -> None:
        """When coverage meets the threshold, the tool succeeds with header."""
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = CoverageTool()
        run_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        report_result = CompletedProcess(
            args=[],
            returncode=0,
            stdout="TOTAL    10    1    90%\n",
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
        assert not result.is_error
        assert "Coverage: 90.0%" in result.output

    @pytest.mark.asyncio
    async def test_detect_source_falls_back_to_dot(self, tmp_path: Path) -> None:
        """When src/ directory doesn't exist, _detect_source returns '.'."""
        from godspeed.tools.coverage import _detect_source

        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert not (ctx.cwd / "src").is_dir()
        assert _detect_source(ctx) == "."

    def test_detect_source_returns_src_when_present(self, tmp_path: Path) -> None:
        """When src/ directory exists, _detect_source returns 'src'."""
        from godspeed.tools.coverage import _detect_source

        (tmp_path / "src").mkdir()
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_source(ctx) == "src"
