"""Tests for the security_scan tool (v2.6.0)."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.security_scan import SecurityScanTool


class TestSecurityScanTool:
    @pytest.mark.asyncio
    async def test_no_scanners_installed_returns_clear_error(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = SecurityScanTool()
        with patch("godspeed.tools.security_scan.shutil.which", return_value=None):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "bandit" in (result.error or "").lower()
        assert "semgrep" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_clean_bandit_run_reports_no_issues(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = SecurityScanTool()
        # returncode 0 = no issues for bandit
        clean = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.security_scan.shutil.which",
                side_effect=lambda name: "/usr/bin/bandit" if name == "bandit" else None,
            ),
            patch("godspeed.tools.security_scan.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert not result.is_error
        assert "bandit" in result.output

    @pytest.mark.asyncio
    async def test_bandit_findings_produce_error_result(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = SecurityScanTool()
        issue = CompletedProcess(args=[], returncode=1, stdout="Issue: [B102] exec_used", stderr="")
        with (
            patch(
                "godspeed.tools.security_scan.shutil.which",
                side_effect=lambda name: "/usr/bin/bandit" if name == "bandit" else None,
            ),
            patch("godspeed.tools.security_scan.subprocess.run", return_value=issue),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert result.is_error
        assert "B102" in (result.error or "")

    @pytest.mark.asyncio
    async def test_invalid_severity_rejected(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = SecurityScanTool()
        result = await tool.execute({"severity": "critical"}, ctx)
        assert result.is_error
        assert "severity" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_both_scanners_run_when_available(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = SecurityScanTool()
        clean = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.security_scan.shutil.which",
                side_effect=lambda name: f"/usr/bin/{name}",
            ),
            patch("godspeed.tools.security_scan.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({"target": "src"}, ctx)
        assert not result.is_error
        assert "bandit" in result.output
        assert "semgrep" in result.output
