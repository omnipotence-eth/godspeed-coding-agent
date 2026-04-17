"""Complexity tool — surface functions likely to need refactoring.

Wraps ``radon`` (cyclomatic + maintainability) for Python and ``lizard``
(polyglot cyclomatic + length) when installed. Both optional. Returns an
error result when any function exceeds the configured cyclomatic threshold
so the agent can treat complexity as a gate.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

COMPLEXITY_TIMEOUT = 60
MAX_OUTPUT_CHARS = 5000

# Default cyclomatic-complexity ceiling: anything above this is flagged.
# 10 is the commonly cited "avoid unless necessary" threshold (McCabe 1976).
DEFAULT_MAX_CC = 10


class ComplexityTool(Tool):
    """Report cyclomatic complexity + flag functions above a threshold.

    - ``radon`` for Python (cyclomatic + maintainability index)
    - ``lizard`` for polyglot (cyclomatic + function length, when installed)

    Fails the tool call when any function's cyclomatic complexity exceeds
    ``max_cc`` (default 10). Lets the agent enforce a complexity budget per
    edit rather than catching violations in code review.
    """

    @property
    def name(self) -> str:
        return "complexity"

    @property
    def description(self) -> str:
        return (
            "Report cyclomatic complexity on a file or directory. "
            "Uses radon (Python) and lizard (polyglot) when available. "
            "Fails the call when any function exceeds max_cc (default 10), "
            "so callers can treat complexity as a budget.\n\n"
            "Example: complexity(target='src/myapp')\n"
            "Example: complexity(target='src/myapp/core.py', max_cc=15)"
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
                        "File or directory to analyze (relative to project root). "
                        "Defaults to 'src' if present, else the cwd."
                    ),
                },
                "max_cc": {
                    "type": "integer",
                    "description": (
                        f"Maximum cyclomatic complexity per function "
                        f"(default: {DEFAULT_MAX_CC}). Findings above this "
                        "threshold fail the tool call."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        target = arguments.get("target") or _detect_target(context)
        max_cc = int(arguments.get("max_cc", DEFAULT_MAX_CC))
        if max_cc <= 0:
            return ToolResult.failure(f"max_cc must be positive, got {max_cc}")

        radon_bin = shutil.which("radon")
        lizard_bin = shutil.which("lizard")
        if radon_bin is None and lizard_bin is None:
            return ToolResult.failure(
                "Neither `radon` nor `lizard` is installed. Install with "
                "`pip install radon` or `pip install lizard`."
            )

        sections: list[str] = []
        any_breach = False

        if radon_bin is not None:
            out, breach = _run_radon(radon_bin, target, max_cc, context)
            sections.append(out)
            any_breach = any_breach or breach

        if lizard_bin is not None:
            out, breach = _run_lizard(lizard_bin, target, max_cc, context)
            sections.append(out)
            any_breach = any_breach or breach

        header = f"Complexity scan of {target} (max_cc: {max_cc})\n"
        body = header + "\n\n".join(sections)
        body = body[-MAX_OUTPUT_CHARS:] if len(body) > MAX_OUTPUT_CHARS else body

        if any_breach:
            return ToolResult.failure(body)
        return ToolResult.success(body)


def _detect_target(context: ToolContext) -> str:
    src = context.cwd / "src"
    if src.is_dir():
        return "src"
    return "."


def _run_radon(radon_bin: str, target: str, max_cc: int, context: ToolContext) -> tuple[str, bool]:
    """Run ``radon cc -s -n <letter>`` to report functions above the threshold.

    radon maps grades to ranges: A=1-5, B=6-10, C=11-20, D=21-30, E=31-40, F=41+.
    We map max_cc to the minimum grade to show and use that as the breach signal.
    """
    min_grade = _cc_to_radon_grade(max_cc + 1)
    cmd = [radon_bin, "cc", target, "-s", "-n", min_grade]
    logger.info("radon cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=COMPLEXITY_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return f"[radon] timed out after {COMPLEXITY_TIMEOUT}s", True

    output = proc.stdout.strip()
    # radon prints nothing when no function exceeds the threshold.
    if not output:
        return f"## radon\nAll functions within CC <= {max_cc}.", False
    return f"## radon (functions above max_cc={max_cc})\n{output}", True


def _cc_to_radon_grade(cc: int) -> str:
    """Map a cyclomatic complexity value to its radon grade letter."""
    if cc <= 5:
        return "A"
    if cc <= 10:
        return "B"
    if cc <= 20:
        return "C"
    if cc <= 30:
        return "D"
    if cc <= 40:
        return "E"
    return "F"


def _run_lizard(
    lizard_bin: str, target: str, max_cc: int, context: ToolContext
) -> tuple[str, bool]:
    """Run ``lizard --CCN <max_cc> -w <target>`` to report over-threshold functions.

    ``-w`` prints warnings only (one line per over-threshold function).
    Exit code is non-zero when any function exceeds the threshold.
    """
    cmd = [lizard_bin, "--CCN", str(max_cc), "-w", target]
    logger.info("lizard cmd=%r cwd=%s", cmd, context.cwd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=COMPLEXITY_TIMEOUT,
            cwd=str(context.cwd),
        )
    except subprocess.TimeoutExpired:
        return f"[lizard] timed out after {COMPLEXITY_TIMEOUT}s", True

    output = (proc.stdout + proc.stderr).strip()
    breach = proc.returncode != 0
    if not output:
        return f"## lizard\nAll functions within CC <= {max_cc}.", False
    return f"## lizard (functions above max_cc={max_cc})\n{output}", breach
