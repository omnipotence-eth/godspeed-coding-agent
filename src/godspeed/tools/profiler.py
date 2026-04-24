"""Performance profiler tool."""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ProfileTool(Tool):
    """Profile Python code performance.

    Uses cProfile to analyze code execution and identify
    performance bottlenecks.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "profile"

    @property
    def description(self) -> str:
        return (
            "Profile Python code performance. "
            "Identifies slow functions and bottlenecks."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File to profile"},
                "function": {"type": "string", "description": "Function to run"},
                "limit": {"type": "integer", "description": "Number of results"},
            },
            "required": ["file_path", "function"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        function: str = "main",
        limit: int = 20,
    ) -> ToolResult:
        """Profile a function."""
        import importlib.util
        import os

        if not os.path.exists(file_path):
            return ToolResult.failure(f"File not found: {file_path}")

        module_name = Path(file_path).stem

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                return ToolResult.failure("Cannot load module")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, function):
                return ToolResult.failure(f"Function not found: {function}")

            func = getattr(module, function)

            # Profile
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func()
            finally:
                profiler.disable()

            # Get stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(limit)

            output = s.getvalue()

            lines = [
                f"## Profile Results for {function}()",
                f"File: {file_path}",
                "",
                "### Top Functions by Cumulative Time",
                "",
            ]

            for line in output.split("\n")[:limit + 5]:
                lines.append(line)

            return ToolResult.ok("\n".join(lines))

        except Exception as exc:
            return ToolResult.failure(f"Profile failed: {exc}")


class MemoryProfileTool(Tool):
    """Profile memory usage."""

    produces_diff = False

    @property
    def name(self) -> str:
        return "memory_profile"

    @property
    def description(self) -> str:
        return "Profile memory usage of Python code."

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "function": {"type": "string"},
            },
            "required": ["file_path", "function"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        function: str = "main",
    ) -> ToolResult:
        """Profile memory usage."""
        import tracemalloc

        if not Path(file_path).exists():
            return ToolResult.failure(f"File not found: {file_path}")

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("module", file_path)
            if not spec or not spec.loader:
                return ToolResult.failure("Cannot load")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, function, None)

            if not func:
                return ToolResult.failure(f"Function not found: {function}")

            tracemalloc.start()
            result = func()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            lines = [
                f"## Memory Profile: {function}",
                "",
                f"**Current**: {current / 1024 / 1024:.2f} MB",
                f"**Peak**: {peak / 1024 / 1024:.2f} MB",
            ]

            return ToolResult.ok("\n".join(lines))

        except ImportError:
            return ToolResult.failure("tracemalloc not available (Python 3.4+)")
        except Exception as exc:
            return ToolResult.failure(f"Failed: {exc}")