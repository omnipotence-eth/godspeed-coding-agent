"""Tests for the dep_audit tool (v2.7.0)."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.dep_audit import DepAuditTool, _detect_managers


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
