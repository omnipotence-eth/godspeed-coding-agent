"""Coverage tool — run pytest under coverage and report results.

Wraps ``coverage run pytest`` + ``coverage report``. Optional dependency:
if ``coverage`` or ``pytest`` is not installed, returns a clear error
instead of crashing. Designed as a follow-up to ``test_runner`` for when
the agent needs a quality signal beyond pass/fail.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

COVERAGE_TIMEOUT = 180
MAX_OUTPUT_CHARS = 5000


class CoverageTool(Tool):
    """Run pytest under ``coverage`` and report line/branch coverage.

    Calls ``coverage run --source=<src_dir> -m pytest <target>`` then
    ``coverage report -m``. Fails the tool call if coverage falls below
    ``min_percent`` — lets the agent treat coverage as a gate, not a hint.

    Requires the project to have both ``coverage`` and ``pytest`` installed.
    Returns a clear error when either is missing (no attempt to install).
    """

    @property
    def name(self) -> str:
        return "coverage"

    @property
    def description(self) -> str:
        return (
            "Run pytest under coverage and report line coverage per module. "
            "Optional `min_percent` argument fails the call if overall "
            "coverage is below that threshold — treat as a quality gate. "
            "Requires `coverage` and `pytest` to be installed in the project.\n\n"
            "Example: coverage()\n"
            "Example: coverage(target='tests/test_api.py', min_percent=80)"
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
                        "Optional pytest target (file or directory). "
                        "Defaults to the project's configured test path."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Optional --source argument for coverage "
                        "(directory to measure, e.g. 'src/myproject'). "
                        "Defaults to auto-detect: 'src' if present, else the cwd."
                    ),
                },
                "min_percent": {
                    "type": "number",
                    "description": (
                        "Optional minimum overall coverage percentage (0-100). "
                        "If set, the tool call fails when coverage is below this."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        coverage_bin = shutil.which("coverage")
        if coverage_bin is None:
            return ToolResult.failure(
                "`coverage` is not installed in this environment. "
                "Install with `pip install coverage` or `uv pip install coverage`."
            )

        target = arguments.get("target", "")
        source = arguments.get("source") or _detect_source(context)
        min_percent = arguments.get("min_percent")

        run_cmd = [coverage_bin, "run"]
        if source:
            run_cmd.extend(["--source", source])
        run_cmd.extend(["-m", "pytest"])
        if target:
            run_cmd.append(str(target))

        logger.info("coverage run cmd=%r cwd=%s", run_cmd, context.cwd)
        try:
            run = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=COVERAGE_TIMEOUT,
                cwd=str(context.cwd),
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"coverage run timed out after {COVERAGE_TIMEOUT}s")
        except FileNotFoundError as exc:
            return ToolResult.failure(f"coverage binary not executable: {exc}")

        # pytest exit codes: 0=pass, 1=fail, 5=no tests. Non-zero means tests failed.
        if run.returncode not in (0, 5):
            out_tail = (run.stdout + run.stderr)[-MAX_OUTPUT_CHARS:]
            return ToolResult.failure(f"pytest failed with exit code {run.returncode}\n{out_tail}")

        # Now render the report.
        try:
            report = subprocess.run(
                [coverage_bin, "report", "-m"],
                capture_output=True,
                text=True,
                timeout=COVERAGE_TIMEOUT,
                cwd=str(context.cwd),
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"coverage report timed out after {COVERAGE_TIMEOUT}s")

        if report.returncode != 0:
            tail = (report.stdout + report.stderr)[-MAX_OUTPUT_CHARS:]
            return ToolResult.failure(f"coverage report failed\n{tail}")

        report_text = report.stdout[-MAX_OUTPUT_CHARS:]
        overall = _parse_overall_percent(report.stdout)

        if min_percent is not None:
            try:
                threshold = float(min_percent)
            except (TypeError, ValueError):
                return ToolResult.failure(f"min_percent must be numeric, got {min_percent!r}")
            if overall is None:
                return ToolResult.failure(
                    "Could not parse overall coverage percentage from report:\n" + report_text
                )
            if overall < threshold:
                return ToolResult.failure(
                    f"Coverage {overall:.1f}% is below min_percent={threshold:.1f}\n\n{report_text}"
                )

        header = f"Coverage: {overall:.1f}%\n\n" if overall is not None else ""
        return ToolResult.success(header + report_text)


def _detect_source(context: ToolContext) -> str:
    """Pick a sensible --source default: prefer ./src, else cwd."""
    src = context.cwd / "src"
    if src.is_dir():
        return "src"
    return "."


def _parse_overall_percent(report_output: str) -> float | None:
    """Extract the overall percentage from ``coverage report`` output.

    The last line of a standard report is like:
        TOTAL                     1234    100   92%
    with an optional 4th column for branch coverage. Returns None when
    the TOTAL line is absent or unparseable.
    """
    for line in reversed(report_output.splitlines()):
        if line.strip().startswith("TOTAL"):
            tokens = line.split()
            for tok in reversed(tokens):
                if tok.endswith("%"):
                    try:
                        return float(tok.rstrip("%"))
                    except ValueError:
                        return None
    return None
