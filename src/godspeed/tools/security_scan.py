"""Security scan tool — run SAST on the project or a single file.

Wraps ``bandit`` (Python SAST) and optionally ``semgrep`` (polyglot, rule-based).
Both optional: the tool reports which scanners ran and which were skipped.
If neither is available, returns a clear error so the agent can suggest install.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

SCAN_TIMEOUT = 120
MAX_OUTPUT_CHARS = 5000


class SecurityScanTool(Tool):
    """Run static security analysis on a target path.

    - ``bandit`` for Python (Ruby on imports, AST-level issues)
    - ``semgrep`` for polyglot rule-based scanning (when installed)

    Returns a consolidated report with severity counts and per-finding detail.
    Non-zero findings mark the result as an error so the agent can treat
    security issues as gating rather than advisory.
    """

    @property
    def name(self) -> str:
        return "security_scan"

    @property
    def description(self) -> str:
        return (
            "Run static security analysis (SAST) on a file or directory. "
            "Uses bandit (Python) and semgrep (polyglot) when available. "
            "Returns an error result when issues at severity >= low are found, "
            "so callers can treat security findings as gating.\n\n"
            "Example: security_scan(target='src/myapp')\n"
            "Example: security_scan(target='src/myapp/auth.py', severity='medium')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "File or directory to scan (relative to project root). "
                        "Defaults to 'src' if present, else the cwd."
                    ),
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": (
                        "Minimum severity level to report and to fail the "
                        "tool call on. Defaults to 'low' (report everything)."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        target = arguments.get("target") or _detect_target(context)
        severity = arguments.get("severity", "low").lower()
        if severity not in ("low", "medium", "high"):
            return ToolResult.failure(f"severity must be one of low/medium/high, got {severity!r}")

        bandit_bin = shutil.which("bandit")
        semgrep_bin = shutil.which("semgrep")
        if bandit_bin is None and semgrep_bin is None:
            return ToolResult.failure(
                "Neither `bandit` nor `semgrep` is installed. Install one with "
                "`pip install bandit` or `pip install semgrep`."
            )

        sections: list[str] = []
        any_findings = False

        if bandit_bin is not None:
            out, found = _run_bandit(bandit_bin, target, severity, context)
            sections.append(out)
            any_findings = any_findings or found

        if semgrep_bin is not None:
            out, found = _run_semgrep(semgrep_bin, target, context)
            sections.append(out)
            any_findings = any_findings or found

        header = f"Security scan of {target} (min severity: {severity})\n"
        body = header + "\n\n".join(sections)
        body = body[-MAX_OUTPUT_CHARS:] if len(body) > MAX_OUTPUT_CHARS else body

        if any_findings:
            return ToolResult.failure(body)
        return ToolResult.success(body)


def _detect_target(context: ToolContext) -> str:
    src = context.cwd / "src"
    if src.is_dir():
        return "src"
    return "."


_BANDIT_SEVERITY_FLAGS = {
    "low": "-ll",
    "medium": "-lll",
    "high": "-llll",
}


def _run_bandit(
    bandit_bin: str, target: str, severity: str, context: ToolContext
) -> tuple[str, bool]:
    flag = _BANDIT_SEVERITY_FLAGS.get(severity, "-ll")
    cmd = [bandit_bin, "-r", target, flag, "-q"]
    logger.info("bandit cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SCAN_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return f"[bandit] timed out after {SCAN_TIMEOUT}s", True
    output = proc.stdout + proc.stderr
    # bandit returns non-zero when issues are found
    found = proc.returncode != 0
    return f"## bandit\n{output.strip() or '(no issues)'}", found


def _run_semgrep(semgrep_bin: str, target: str, context: ToolContext) -> tuple[str, bool]:
    cmd = [semgrep_bin, "--config=auto", "--error", "--quiet", target]
    logger.info("semgrep cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SCAN_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return f"[semgrep] timed out after {SCAN_TIMEOUT}s", True
    output = proc.stdout + proc.stderr
    # semgrep with --error returns non-zero when rules match
    found = proc.returncode != 0
    return f"## semgrep\n{output.strip() or '(no issues)'}", found
