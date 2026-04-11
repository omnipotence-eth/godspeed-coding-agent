"""Verify tool — run linter checks on files after edits."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

# File extensions that support verification
PYTHON_EXTENSIONS = {".py", ".pyi"}

# Timeout for linter subprocess
VERIFY_TIMEOUT = 15


class VerifyTool(Tool):
    """Run linter verification on a file.

    Currently supports Python files via ruff. Returns lint errors/warnings
    so the agent can self-correct. Gracefully returns success for
    unsupported file types.
    """

    @property
    def name(self) -> str:
        return "verify"

    @property
    def description(self) -> str:
        return (
            "Run linter checks on a file to catch syntax errors and style issues. "
            "Currently supports Python files (via ruff). Returns clean or error details."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to verify (relative to project root)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")

        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")

        try:
            resolved = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not resolved.exists():
            return ToolResult.failure(f"File not found: {file_path_str}")

        suffix = resolved.suffix.lower()

        if suffix in PYTHON_EXTENSIONS:
            return _verify_python(resolved, file_path_str)

        return ToolResult.success(
            f"No linter configured for {suffix} files. Skipping verification."
        )


def _verify_python(resolved: Any, display_path: str) -> ToolResult:
    """Run ruff check on a Python file."""
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return ToolResult.success(
            "ruff not found — skipping verification. Install with: pip install ruff"
        )

    try:
        result = subprocess.run(
            [ruff_bin, "check", "--select=E,W,F", "--no-fix", str(resolved)],
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(f"Verification timed out after {VERIFY_TIMEOUT}s: {display_path}")
    except OSError as exc:
        return ToolResult.failure(f"Failed to run ruff: {exc}")

    if result.returncode == 0:
        return ToolResult.success(f"Verification passed: {display_path}")

    # ruff returns non-zero when there are findings
    output = result.stdout.strip() or result.stderr.strip()
    return ToolResult.success(f"Lint issues in {display_path}:\n{output}")
