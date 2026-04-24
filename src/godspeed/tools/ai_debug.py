"""AI Debugger - intelligent debugging assistance."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class AIDebugTool(Tool):
    """AI-powered debugging assistant.

    Similar to Cursor's Debug mode and Claude's debug capabilities.
    Analyzes errors, suggests fixes, and walks through code.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "debug"

    @property
    def description(self) -> str:
        return (
            "AI debugger for analyzing errors and bugs. "
            "Provides root cause analysis, fix suggestions, and "
            "step-by-step debugging guidance."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "error": {"type": "string", "description": "Error message or traceback"},
                "file_path": {"type": "string", "description": "File with the error"},
                "action": {
                    "type": "string",
                    "enum": ["analyze", "explain", "fix", "trace"],
                    "description": "Debug action",
                },
            },
        }

    async def execute(
        self,
        tool_context: ToolContext,
        error: str = "",
        file_path: str | None = None,
        action: str = "analyze",
    ) -> ToolResult:
        """Execute debug action."""
        if action == "analyze":
            return self._analyze_error(error, file_path)
        elif action == "explain":
            return self._explain_error(error)
        elif action == "fix":
            return self._suggest_fix(error, file_path)
        elif action == "trace":
            return self._trace_error(error)
        return ToolResult.failure(f"Unknown action: {action}")

    def _analyze_error(self, error: str, file_path: str | None) -> ToolResult:
        """Analyze error and provide root cause."""
        lines = ["## Error Analysis\n"]

        if not error:
            return ToolResult.failure("No error provided")

        lines.append(f"**Error**: {error[:200]}")

        # Parse error type
        error_type = self._extract_error_type(error)
        if error_type:
            lines.append(f"\n**Error Type**: {error_type}")

        # Common patterns
        patterns = self._get_common_patterns(error_type)
        if patterns:
            lines.append(f"\n### Likely Causes")
            for p in patterns:
                lines.append(f"- {p}")

        # File location
        if file_path:
            lines.append(f"\n**File**: {file_path}")

        return ToolResult.ok("\n".join(lines))

    def _extract_error_type(self, error: str) -> str | None:
        """Extract the type of error from message."""
        patterns = [
            (r"(\w+Error):", 1),
            (r"(\w+Exception):", 1),
            (r"(\w+Failed):", 1),
        ]

        for pattern, group in patterns:
            match = re.search(pattern, error)
            if match:
                return match.group(group)
        return None

    def _get_common_patterns(self, error_type: str | None) -> list[str]:
        """Get common patterns for error types."""
        patterns = {
            "TypeError": [
                "Passed wrong type to function",
                "Called function with wrong number of arguments",
                "Tried to call non-callable",
                "Accessed property of None",
            ],
            "ValueError": [
                "Passed invalid value to function",
                "Tried to access invalid index",
                "Failed validation check",
            ],
            "AttributeError": [
                "Object doesn't have that attribute",
                "Typo in attribute name",
                "Module not imported",
            ],
            "ImportError": [
                "Module not installed",
                "Circular import",
                "Missing dependencies",
            ],
            "SyntaxError": [
                "Invalid Python syntax",
                "Missing colon or parenthesis",
                "Indentation error",
            ],
            "NameError": [
                "Used undefined variable",
                "Typo in variable name",
                "Import not found",
            ],
        }
        return patterns.get(error_type or "", [])

    def _explain_error(self, error: str) -> ToolResult:
        """Explain error in plain language."""
        error_type = self._extract_error_type(error)

        explanations = {
            "TypeError": "A function received data it didn't expect - the type was wrong.",
            "ValueError": "A function received data of the right type but the value was wrong.",
            "AttributeError": "Tried to access something that doesn't exist on an object.",
            "ImportError": "Python couldn't find a module or library.",
            "SyntaxError": "The code doesn't follow Python's rules.",
            "NameError": "Used a name (variable/function) that wasn't defined.",
        }

        explanation = explanations.get(error_type or "", "An error occurred.")

        lines = [
            "## Error Explanation\n",
            f"**Brief**: {explanation}",
            f"\n**Technical**: {error[:500]}",
        ]

        return ToolResult.ok("\n".join(lines))

    def _suggest_fix(self, error: str, file_path: str | None) -> ToolResult:
        """Suggest a fix for the error."""
        error_type = self._extract_error_type(error)

        fixes = {
            "TypeError": [
                "Check the types of arguments passed to functions",
                "Add type conversions (int(), str(), list())",
                "Verify the function signature expects your type",
            ],
            "ValueError": [
                "Check valid values for the function",
                "Add validation before calling",
                "Look at the function documentation",
            ],
            "AttributeError": [
                "Check the object type has that attribute",
                "Verify the module is imported",
                "Check for typos",
            ],
            "ImportError": [
                "Install the required package",
                "Check PYTHONPATH",
                "Verify the module name",
            ],
            "NameError": [
                "Define the variable before use",
                "Check for typos",
                "Add import statement",
            ],
        }

        suggested_fixes = fixes.get(error_type or "", ["Review the error message"])

        lines = ["## Suggested Fixes\n"]
        for i, fix in enumerate(suggested_fixes, 1):
            lines.append(f"{i}. {fix}")

        if file_path:
            lines.append(f"\n**File**: {file_path}")

        return ToolResult.ok("\n".join(lines))

    def _trace_error(self, error: str) -> ToolResult:
        """Trace through the error to find the source."""
        lines = ["## Error Trace\n"]

        # Extract file and line numbers
        file_match = re.search(r'File "([^"]+)", line (\d+)', error)
        if file_match:
            lines.append(f"**File**: {file_match.group(1)}")
            lines.append(f"**Line**: {file_match.group(2)}")

        # Extract stack trace
        if "Traceback" in error:
            lines.append("\n### Stack Trace")
            stack = error.split("Traceback")[-1][:500]
            lines.append(f"```\n{stack}\n```")

        return ToolResult.ok("\n".join(lines))


class StackTraceTool(Tool):
    """Interactive stack trace navigator."""

    produces_diff = False

    @property
    def name(self) -> str:
        return "stack_trace"

    @property
    def description(self) -> str:
        return "Navigate through stack traces interactively."

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "traceback": {"type": "string", "description": "Stack trace to analyze"},
                "frame": {"type": "integer", "description": "Frame number to inspect"},
            },
            "required": ["traceback"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        traceback: str,
        frame: int = 0,
    ) -> ToolResult:
        """Navigate stack trace frames."""
        import re

        # Parse frames
        frame_pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        frames = re.findall(frame_pattern, traceback)

        if not frames:
            return ToolResult.ok("No frames found in traceback")

        if frame < 0 or frame >= len(frames):
            return ToolResult.failure(f"Invalid frame: {frame}")

        file_path, line_num, func_name = frames[frame]

        lines = [
            f"## Frame {frame} of {len(frames) - 1}",
            f"**File**: {file_path}",
            f"**Line**: {line_num}",
            f"**Function**: {func_name}",
        ]

        # Navigate
        if frame < len(frames) - 1:
            lines.append(f"\nNext frame: use frame={frame + 1}")
        if frame > 0:
            lines.append(f"Previous frame: use frame={frame - 1}")

        return ToolResult.ok("\n".join(lines))