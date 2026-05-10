"""Tests for the dep_audit tool (v2.7.0)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.dep_audit import (
    DepAuditTool,
    _detect_managers,
    _run_one,
    _run_pip_audit,
    _run_npm_audit,
    _run_cargo_audit,
)


# ---------------------------------------------------------------------------
# Static shape
# ---------------------------------------------------------------------------


def test_tool_name() -> None:
    assert DepAuditTool().name == "dep_audit"


def test_tool_is_read_only() -> None:
    assert DepAuditTool().risk_level == RiskLevel.READ_ONLY


def test_tool_description() -> None:
    desc = DepAuditTool().description.lower()
    assert "scan" in desc
    assert "pip" in desc or "npm" in desc or "cargo" in desc


def test_tool_schema() -> None:
    schema = DepAuditTool().get_schema()
    assert "manager" in schema["properties"]
    assert schema["properties"]["manager"]["enum"] == ["pip", "npm", "cargo", "auto"]


# ---------------------------------------------------------------------------
# _detect_managers
# ---------------------------------------------------------------------------


class TestDetectManagers:
    def test_pyproject_detected_as_pip(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_managers(ctx) == ["pip"]

    def test_requirements_txt_detected_as_pip(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_managers(ctx) == ["pip"]

    def test_package_json_detected_as_npm(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_managers(ctx) == ["npm"]

    def test_cargo_detected(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_managers(ctx) == ["cargo"]

    def test_multi_language_monorepo(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_managers(ctx) == ["pip", "npm"]

    def test_empty_project(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        assert _detect_managers(ctx) == []


# ---------------------------------------------------------------------------
# DepAuditTool — execute
# ---------------------------------------------------------------------------


class TestDepAuditTool:
    @pytest.mark.asyncio
    async def test_no_manifests_returns_clear_error(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        result = await tool.execute({}, ctx)
        assert result.is_error
        assert "manifest" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_invalid_manager_rejected(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        result = await tool.execute({"manager": "yarn"}, ctx)
        assert result.is_error
        assert "manager" in (result.error or "")

    @pytest.mark.asyncio
    async def test_pip_audit_clean(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        clean = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/pip-audit" if name == "pip-audit" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({}, ctx)
        assert not result.is_error
        assert "pip-audit" in result.output

    @pytest.mark.asyncio
    async def test_pip_audit_vulns_are_gating(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        vuln = CompletedProcess(
            args=[],
            returncode=1,
            stdout="requests 2.28.0 has CVE-2023-32681",
            stderr="",
        )
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/pip-audit" if name == "pip-audit" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=vuln),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "CVE-2023-32681" in (result.error or "")

    @pytest.mark.asyncio
    async def test_no_auditor_installed_despite_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "pip-audit" in (result.error or "")

    # -----------------------------------------------------------------------
    # NEW TESTS — cover missed branches (33 statements)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pip_audit_timeout(self, tmp_path: Path) -> None:
        """Cover lines 156-157: pip-audit TimeoutExpired."""
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/pip-audit" if name == "pip-audit" else None,
            ),
            patch(
                "godspeed.tools.dep_audit.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=[], timeout=120),
            ),
        ):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_npm_audit_clean(self, tmp_path: Path) -> None:
        """Cover lines 164-181: npm audit path with clean result."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        clean = CompletedProcess(args=[], returncode=0, stdout="found 0 vulnerabilities", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({"manager": "npm"}, ctx)
        assert not result.is_error
        assert "npm audit" in result.output

    @pytest.mark.asyncio
    async def test_npm_audit_vulns_are_gating(self, tmp_path: Path) -> None:
        """Cover npm audit with vulnerabilities (non-zero returncode)."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        vuln = CompletedProcess(
            args=[],
            returncode=1,
            stdout="lodash 4.17.21 has CVE-2021-23337",
            stderr="",
        )
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=vuln),
        ):
            result = await tool.execute({"manager": "npm"}, ctx)
        assert result.is_error
        assert "CVE-2021-23337" in (result.error or "")

    @pytest.mark.asyncio
    async def test_npm_audit_not_installed(self, tmp_path: Path) -> None:
        """Cover npm audit path when npm binary is missing."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            result = await tool.execute({"manager": "npm"}, ctx)
        assert result.is_error
        assert "npm" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_npm_audit_timeout(self, tmp_path: Path) -> None:
        """Cover lines 177-178: npm audit timeout."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ),
            patch(
                "godspeed.tools.dep_audit.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=[], timeout=120),
            ),
        ):
            result = await tool.execute({"manager": "npm"}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cargo_audit_clean(self, tmp_path: Path) -> None:
        """Cover lines 185-206: cargo audit path with clean result."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        clean = CompletedProcess(
            args=[], returncode=0, stdout="Success no vulnerabilities", stderr=""
        )
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=clean),
        ):
            result = await tool.execute({"manager": "cargo"}, ctx)
        assert not result.is_error
        assert "cargo audit" in result.output

    @pytest.mark.asyncio
    async def test_cargo_audit_vulns_are_gating(self, tmp_path: Path) -> None:
        """Cover cargo audit with vulnerabilities."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        vuln = CompletedProcess(
            args=[],
            returncode=1,
            stdout="CVE: RUSTSEC-2024-0336",
            stderr="",
        )
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=vuln),
        ):
            result = await tool.execute({"manager": "cargo"}, ctx)
        assert result.is_error
        assert "CVE" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cargo_audit_not_installed(self, tmp_path: Path) -> None:
        """Cover cargo audit path when cargo binary is missing."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            result = await tool.execute({"manager": "cargo"}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_cargo_audit_subcommand_not_installed(self, tmp_path: Path) -> None:
        """Cover lines 202-203: cargo returns exit 101 'no such command'."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        not_installed = CompletedProcess(
            args=[],
            returncode=101,
            stdout="",
            stderr="error: no such command: `audit`",
        )
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=not_installed),
        ):
            result = await tool.execute({"manager": "cargo"}, ctx)
        assert result.is_error
        assert "cargo-audit" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_cargo_audit_timeout(self, tmp_path: Path) -> None:
        """Cover lines 198-199: cargo audit timeout."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None,
            ),
            patch(
                "godspeed.tools.dep_audit.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=[], timeout=120),
            ),
        ):
            result = await tool.execute({"manager": "cargo"}, ctx)
        assert result.is_error
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_all_auditors_uninstalled_clean_error(self, tmp_path: Path) -> None:
        """Cover the 'ran_any == False' path — all auditors missing."""
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            result = await tool.execute({}, ctx)
        assert result.is_error
        assert "pip-audit" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_auto_detection_runs_both_pip_and_npm(self, tmp_path: Path) -> None:
        """When pyproject.toml + package.json exist, both auditors run."""
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        pip_clean = CompletedProcess(args=[], returncode=0, stdout="pip clean", stderr="")
        npm_clean = CompletedProcess(args=[], returncode=0, stdout="npm clean", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: {
                    "pip-audit": "/usr/bin/pip-audit",
                    "npm": "/usr/bin/npm",
                }.get(name),
            ),
            patch(
                "godspeed.tools.dep_audit.subprocess.run",
                side_effect=[pip_clean, npm_clean],
            ),
        ):
            result = await tool.execute({}, ctx)
        assert not result.is_error
        assert "pip clean" in result.output
        assert "npm clean" in result.output

    @pytest.mark.asyncio
    async def test_explicit_manager_overrides_auto(self, tmp_path: Path) -> None:
        """Explicit manager='npm' when pyproject.toml exists runs only npm."""
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        npm_clean = CompletedProcess(args=[], returncode=0, stdout="npm result", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=npm_clean),
        ):
            result = await tool.execute({"manager": "npm"}, ctx)
        assert not result.is_error
        assert "npm audit" in result.output

    @pytest.mark.asyncio
    async def test_npm_audit_no_output_placeholder(self, tmp_path: Path) -> None:
        """When npm audit produces empty stdout+stderr, placeholder appears."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        empty = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=empty),
        ):
            result = await tool.execute({"manager": "npm"}, ctx)
        assert not result.is_error
        assert "(no vulnerabilities)" in result.output

    @pytest.mark.asyncio
    async def test_cargo_audit_no_output_placeholder(self, tmp_path: Path) -> None:
        """When cargo audit produces empty output, placeholder appears."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        empty = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=empty),
        ):
            result = await tool.execute({"manager": "cargo"}, ctx)
        assert not result.is_error
        assert "(no vulnerabilities)" in result.output

    @pytest.mark.asyncio
    async def test_import_vulns_from_ppm_json_in_stderr(self, tmp_path: Path) -> None:
        """Vulnerability output in stderr is still captured."""
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t")
        tool = DepAuditTool()
        vuln = CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="pydantic 2.10.6 has CVE-2026-28684",
        )
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/pip-audit" if name == "pip-audit" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=vuln),
        ):
            result = await tool.execute({}, ctx)
        assert not result.is_error
        assert "CVE-2026-28684" in result.output


# ---------------------------------------------------------------------------
# _run_one — direct unit tests
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="t")


class TestRunOne:
    def test_unknown_manager_returns_placeholder(self, tmp_path: Path) -> None:
        """Cover line 139: _run_one with unknown manager."""
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            output, vuln, ran = _run_one("unknown_mgr", _ctx(tmp_path))
        assert "[dep_audit]" in output
        assert not vuln
        assert not ran

    def test_run_pip_audit_direct_not_installed(self, tmp_path: Path) -> None:
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            output, vuln, ran = _run_pip_audit(_ctx(tmp_path))
        assert output == ""
        assert not vuln
        assert not ran

    def test_run_npm_audit_direct_not_installed(self, tmp_path: Path) -> None:
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            output, vuln, ran = _run_npm_audit(_ctx(tmp_path))
        assert output == ""
        assert not vuln
        assert not ran

    def test_run_cargo_audit_direct_not_installed(self, tmp_path: Path) -> None:
        with patch("godspeed.tools.dep_audit.shutil.which", return_value=None):
            output, vuln, ran = _run_cargo_audit(_ctx(tmp_path))
        assert output == ""
        assert not vuln
        assert not ran

    def test_run_pip_audit_direct_clean(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        clean = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/pip-audit" if name == "pip-audit" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=clean),
        ):
            output, vuln, ran = _run_pip_audit(ctx)
        assert ran
        assert not vuln
        assert "pip-audit" in output

    def test_run_npm_audit_direct_clean(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        clean = CompletedProcess(args=[], returncode=0, stdout="found 0", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=clean),
        ):
            output, vuln, ran = _run_npm_audit(ctx)
        assert ran
        assert not vuln
        assert "npm audit" in output

    def test_run_cargo_audit_direct_clean(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        clean = CompletedProcess(args=[], returncode=0, stdout="Success", stderr="")
        with (
            patch(
                "godspeed.tools.dep_audit.shutil.which",
                side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None,
            ),
            patch("godspeed.tools.dep_audit.subprocess.run", return_value=clean),
        ):
            output, vuln, ran = _run_cargo_audit(ctx)
        assert ran
        assert not vuln
        assert "cargo audit" in output
