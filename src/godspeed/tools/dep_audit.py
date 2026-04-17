"""Dep-audit tool — scan project dependencies for known CVEs.

Auto-detects the project's package manager from the working directory:
- ``pyproject.toml`` / ``requirements.txt`` → ``pip-audit``
- ``package.json`` → ``npm audit``
- ``Cargo.toml`` → ``cargo audit`` (when installed)

All scanners optional; if none installed and a relevant manifest is
present, returns a clear error asking the agent to install the right one.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

AUDIT_TIMEOUT = 120
MAX_OUTPUT_CHARS = 5000


class DepAuditTool(Tool):
    """Scan project dependencies for known CVEs.

    Detects the package manager from files in the project root and runs
    the matching auditor. Non-zero vulnerabilities produce an error
    result so the agent treats them as gating.
    """

    @property
    def name(self) -> str:
        return "dep_audit"

    @property
    def description(self) -> str:
        return (
            "Scan project dependencies for known CVEs. Auto-detects the "
            "package manager (pip / npm / cargo) from project files. "
            "Returns an error result when vulnerabilities are found so "
            "the agent treats them as gating.\n\n"
            "Example: dep_audit()\n"
            "Example: dep_audit(manager='pip')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "manager": {
                    "type": "string",
                    "enum": ["pip", "npm", "cargo", "auto"],
                    "description": (
                        "Package manager to audit. Defaults to 'auto' "
                        "which detects from project files. Explicit "
                        "values override detection."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        requested = arguments.get("manager", "auto").lower()
        if requested not in ("pip", "npm", "cargo", "auto"):
            return ToolResult.failure(
                f"manager must be one of pip/npm/cargo/auto, got {requested!r}"
            )

        managers = _detect_managers(context) if requested == "auto" else [requested]
        if not managers:
            return ToolResult.failure(
                "No supported package manifest found in project "
                "(looked for pyproject.toml/requirements.txt, package.json, Cargo.toml)."
            )

        sections: list[str] = []
        any_vulns = False
        ran_any = False

        for mgr in managers:
            out, vuln, ran = _run_one(mgr, context)
            if ran:
                ran_any = True
                sections.append(out)
                any_vulns = any_vulns or vuln

        if not ran_any:
            # Every detected manager had its auditor uninstalled.
            detected = ", ".join(managers)
            return ToolResult.failure(
                f"Detected package manager(s) [{detected}] but none of the "
                "corresponding auditors are installed. Install with one of "
                "`pip install pip-audit`, `npm install -g npm`, or "
                "`cargo install cargo-audit`."
            )

        body = "\n\n".join(sections)
        body = body[-MAX_OUTPUT_CHARS:] if len(body) > MAX_OUTPUT_CHARS else body

        if any_vulns:
            return ToolResult.failure(body)
        return ToolResult.success(body)


def _detect_managers(context: ToolContext) -> list[str]:
    """Return the ordered list of package managers detected from files."""
    managers: list[str] = []
    cwd = context.cwd
    if (cwd / "pyproject.toml").exists() or (cwd / "requirements.txt").exists():
        managers.append("pip")
    if (cwd / "package.json").exists():
        managers.append("npm")
    if (cwd / "Cargo.toml").exists():
        managers.append("cargo")
    return managers


def _run_one(manager: str, context: ToolContext) -> tuple[str, bool, bool]:
    """Run the auditor for one manager.

    Returns (section_text, has_vulns, did_run). When did_run is False the
    auditor binary is missing and the tool call hasn't consumed that slot.
    """
    if manager == "pip":
        return _run_pip_audit(context)
    if manager == "npm":
        return _run_npm_audit(context)
    if manager == "cargo":
        return _run_cargo_audit(context)
    return (f"[dep_audit] unknown manager: {manager}", False, False)


def _run_pip_audit(context: ToolContext) -> tuple[str, bool, bool]:
    pip_audit_bin = shutil.which("pip-audit")
    if pip_audit_bin is None:
        return ("", False, False)
    cmd = [pip_audit_bin, "--strict"]
    logger.info("pip-audit cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIT_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return (f"[pip-audit] timed out after {AUDIT_TIMEOUT}s", True, True)
    output = (proc.stdout + proc.stderr).strip()
    vuln = proc.returncode != 0
    return (f"## pip-audit\n{output or '(no vulnerabilities)'}", vuln, True)


def _run_npm_audit(context: ToolContext) -> tuple[str, bool, bool]:
    npm_bin = shutil.which("npm")
    if npm_bin is None:
        return ("", False, False)
    cmd = [npm_bin, "audit", "--audit-level=low"]
    logger.info("npm-audit cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIT_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return (f"[npm audit] timed out after {AUDIT_TIMEOUT}s", True, True)
    output = (proc.stdout + proc.stderr).strip()
    vuln = proc.returncode != 0
    return (f"## npm audit\n{output or '(no vulnerabilities)'}", vuln, True)


def _run_cargo_audit(context: ToolContext) -> tuple[str, bool, bool]:
    cargo_bin = shutil.which("cargo")
    if cargo_bin is None:
        return ("", False, False)
    cmd = [cargo_bin, "audit", "--quiet"]
    logger.info("cargo-audit cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIT_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return (f"[cargo-audit] timed out after {AUDIT_TIMEOUT}s", True, True)
    # `cargo audit` is a subcommand — if cargo-audit isn't installed, cargo
    # returns a usage error. Treat exit 101 (no such command) as "not installed".
    if proc.returncode == 101 and "no such command" in (proc.stderr or "").lower():
        return ("", False, False)
    output = (proc.stdout + proc.stderr).strip()
    vuln = proc.returncode != 0
    return (f"## cargo audit\n{output or '(no vulnerabilities)'}", vuln, True)
