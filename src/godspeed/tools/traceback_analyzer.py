"""Traceback analyzer tool — parse and explain error tracebacks.

Supports Python tracebacks and JS/TS stack traces. For each frame, extracts the
file path, line number, function name, and surrounding source code context.
Highlights the root-cause frame and provides a structured error summary.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

_CONTEXT_RADIUS = 3

_PY_FRAME_RE = re.compile(
    r'^\s*File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<function>\S+)'
)

_JS_FRAME_RE = re.compile(
    r"^\s*at\s+(?:(?P<function>[^\s(]+)\s+\()?"
    r"(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+)"
    r"(?:\))?"
)

_PYTHON_TRACEBACK_START = re.compile(r"^Traceback\s+\(most recent call last\):\s*$")

_PYTHON_ERROR_LINE_RE = re.compile(
    r"^(\w+(?:\.\w+)*(?:Error|Exception|Warning|Exit|Interrupt"
    r"|KeyboardInterrupt|SystemExit|StopIteration|GeneratorExit"
    r"|NotImplementedError))(?:\s*:\s*(.*))?$"
)

_KNOWN_ERROR_TYPES: dict[str, str] = {
    "SyntaxError": "The Python parser could not understand the code — often a missing colon, "
    "unmatched parenthesis, or incorrect indentation.",
    "IndentationError": "Inconsistent use of tabs and spaces, or wrong indentation level.",
    "ImportError": "A module or name could not be imported — check the module name, "
    "install missing dependencies, or fix circular imports.",
    "ModuleNotFoundError": "The module is not installed or not available in the current "
    "environment — try `pip install <module>`.",
    "AttributeError": "An object does not have the requested attribute or method — "
    "check for typos in the attribute name, or that the object type is what you expect.",
    "TypeError": "An operation was applied to an object of the wrong type — "
    "check argument types and expected signatures.",
    "ValueError": "An operation received an argument with the right type but an "
    "invalid value — check the value against expected constraints.",
    "KeyError": "A dictionary key was not found — verify the key exists or use `.get()`.",
    "IndexError": "A sequence index is out of range — check the length before indexing.",
    "FileNotFoundError": "The specified file or directory does not exist — "
    "verify the path and working directory.",
    "PermissionError": "Insufficient permissions to access a file or directory.",
    "NameError": "A variable or function name is not defined — check for typos "
    "or verify the name is in scope.",
    "ZeroDivisionError": "Division by zero — ensure the denominator is non-zero "
    "before performing division.",
    "RuntimeError": "A generic runtime error — check the error message for details.",
    "NotImplementedError": "An abstract method or interface has not been implemented.",
    "ConnectionError": "A network connection failed — check the URL, network, or service status.",
    "TimeoutError": "An operation timed out — consider increasing the timeout or "
    "checking the remote service.",
    "OSError": "An operating-system-level error — check file paths, permissions, and disk space.",
    "AssertionError": "An `assert` statement failed — review the condition being asserted.",
    "MemoryError": "The operation ran out of memory — reduce data size or free up memory.",
    "RecursionError": "Maximum recursion depth exceeded — check for infinite recursion.",
}


class TracebackAnalyzerTool(Tool):
    """Parse an error traceback and produce a structured analysis.

    Identifies each frame in the call stack, extracts source code context
    around the error lines, and highlights the root-cause frame. Supports
    Python tracebacks and JavaScript/TypeScript stack traces.
    """

    @property
    def name(self) -> str:
        return "traceback_analyzer"

    @property
    def description(self) -> str:
        return (
            "Parse a Python or JS/TS error traceback and produce a structured analysis. "
            "For each frame, shows the file path, line number, function name, and "
            "surrounding source code (+/- 3 lines). Highlights the root-cause frame.\n\n"
            'Example: traceback_analyzer(traceback="Traceback (most recent call last):\\n'
            '  File \\"app.py\\", line 10, in main\\n    ...")'
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "traceback": {
                    "type": "string",
                    "description": (
                        "The full error traceback text to analyze. "
                        "Supports Python tracebacks and JS/TS stack traces."
                    ),
                },
            },
            "required": ["traceback"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        traceback_text = arguments.get("traceback", "")
        if not isinstance(traceback_text, str) or not traceback_text.strip():
            return ToolResult.failure(
                "traceback must be a non-empty string containing the error traceback"
            )

        lines = traceback_text.splitlines()

        is_python = any(_PYTHON_TRACEBACK_START.match(line) for line in lines)

        if is_python:
            frames = _parse_python_traceback(lines)
            error_type, error_message = _extract_python_error(lines)
        else:
            frames = _parse_js_stack(lines)
            error_type, error_message = _extract_js_error(lines)

        if not frames:
            return ToolResult.failure(
                "Could not parse any stack frames from the provided traceback. "
                "Ensure you pass the full error output including file and line references."
            )

        sections: list[str] = []
        sections.append(_format_summary(error_type, error_message))

        for idx, frame in enumerate(frames):
            is_root_cause = idx == len(frames) - 1
            source_context = _read_source_context(frame["file"], frame["line"], context)
            sections.append(_format_frame(idx, len(frames), frame, source_context, is_root_cause))

        return ToolResult.success("\n\n".join(sections))


# -- Parsing helpers ----------------------------------------------------------


def _parse_python_traceback(lines: list[str]) -> list[dict[str, Any]]:
    """Parse Python traceback lines into structured frame dicts."""
    frames: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        m = _PY_FRAME_RE.match(line)
        if not m:
            continue
        code_context = ""
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if (
                next_line
                and not next_line.startswith("  File ")
                and not next_line.startswith("Traceback")
            ):
                code_context = next_line.strip()
        frames.append(
            {
                "file": m.group("file"),
                "line": int(m.group("line")),
                "function": m.group("function"),
                "code_context": code_context[:200] if code_context else "",
            }
        )
    return frames


def _extract_python_error(lines: list[str]) -> tuple[str, str]:
    """Extract the exception type and message from Python traceback lines."""
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("Traceback") or _PY_FRAME_RE.match(line):
            continue
        m = _PYTHON_ERROR_LINE_RE.match(line)
        if m:
            return m.group(1), m.group(2) or ""
        # Also catch lines like just "ErrorType" or "ErrorType: message" without
        # the standard Python exception naming convention
        simple = re.match(r"^(\w[\w.]*)\s*(?::\s*(.*))?$", line)
        if simple:
            return simple.group(1), simple.group(2) or ""
    return "Unknown", ""


def _parse_js_stack(lines: list[str]) -> list[dict[str, Any]]:
    """Parse JS/TS stack trace lines into structured frame dicts."""
    frames: list[dict[str, Any]] = []
    for line in lines:
        m = _JS_FRAME_RE.match(line)
        if m:
            frames.append(
                {
                    "file": m.group("file").strip("()"),
                    "line": int(m.group("line")),
                    "function": m.group("function") or "<anonymous>",
                    "code_context": "",
                }
            )
    return frames


def _extract_js_error(lines: list[str]) -> tuple[str, str]:
    """Extract the error type and message from JS/TS stack trace lines."""
    if lines:
        first = lines[0].strip()
        m = re.match(r"^(\w[\w.]*(?:Error|Exception))\s*:\s*(.*)", first)
        if m:
            return m.group(1), m.group(2)
        m2 = re.match(r"^(\w[\w.]*)\s*:\s*(.*)", first)
        if m2:
            return m2.group(1), m2.group(2)
        # Could be just the message
        return "Error", first
    return "Error", ""


# -- Formatting helpers -------------------------------------------------------


def _format_summary(error_type: str, error_message: str) -> str:
    """Build the error summary header."""
    parts = [
        "=" * 60,
        "TRACEBACK ANALYSIS",
        "=" * 60,
    ]
    emoji = "\u274c"  # red X
    if error_type == "SyntaxError":
        hint = _KNOWN_ERROR_TYPES.get(error_type, "")
        parts.append(f"{emoji}  SYNTAX ERROR")
        parts.append("    The Python parser can't understand this code.")
        if hint:
            parts.append(f"    \u2192 {hint}")
    elif error_type in _KNOWN_ERROR_TYPES:
        parts.append(f"{emoji}  {error_type}")
        parts.append(f"    {_KNOWN_ERROR_TYPES[error_type]}")
    elif error_type:
        parts.append(f"{emoji}  {error_type}")
    if error_message:
        parts.append(f"    Message: {error_message}")
    parts.append("")
    return "\n".join(parts)


def _format_frame(
    idx: int,
    total: int,
    frame: dict[str, Any],
    source_context: str,
    is_root_cause: bool,
) -> str:
    """Format a single stack frame."""
    marker = "\u25b6  ROOT CAUSE" if is_root_cause else f"Frame {idx + 1}/{total}"
    suffix = " (root cause — last frame before crash)" if is_root_cause else ""

    parts = [
        f"{'-' * 60}",
        f"{marker}: {frame['function']}{suffix}",
        f"  File: {frame['file']}:{frame['line']}",
    ]
    if frame.get("code_context"):
        parts.append(f"  Code: {frame['code_context']}")
    if source_context:
        parts.append(f"  Source context (+/- {_CONTEXT_RADIUS} lines):")
        parts.append(source_context)
    return "\n".join(parts)


def _read_source_context(file_path: str, line_no: int, context: ToolContext) -> str:
    """Read +/- CONTEXT_RADIUS lines around *line_no* from *file_path*.

    Resolves paths relative to the project cwd. Gracefully handles missing
    files, out-of-project paths, and binary files.
    """
    try:
        resolved = resolve_tool_path(file_path, context.cwd)
    except ValueError:
        return f"    (file is outside project directory: {file_path})"

    if not resolved.exists():
        return f"    (file not found: {resolved})"

    if not resolved.is_file():
        return f"    (not a regular file: {resolved})"

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "    (cannot display — binary or non-UTF-8 file)"
    except OSError:
        return f"    (cannot read file: {resolved})"

    file_lines = content.splitlines()
    total_lines = len(file_lines)

    start = max(0, line_no - _CONTEXT_RADIUS - 1)
    end = min(total_lines, line_no + _CONTEXT_RADIUS)

    out_lines: list[str] = []
    for i in range(start, end):
        current_line = i + 1
        prefix = ">>>" if current_line == line_no else "   "
        out_lines.append(f"    {prefix} {current_line:>4} | {file_lines[i]}")

    if start > 0:
        out_lines.insert(0, f"    ... ({start} lines above)")
    if end < total_lines:
        out_lines.append(f"    ... ({total_lines - end} more lines)")

    return "\n".join(out_lines)
