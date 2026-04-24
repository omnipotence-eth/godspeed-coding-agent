"""Tool usage analytics tool."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ToolAnalyticsTool(Tool):
    """Analyze tool usage patterns and performance.

    Provides statistics on tool usage frequency, success rates,
    timing, and error patterns.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "tool_analytics"

    @property
    def description(self) -> str:
        return (
            "Analyze tool usage patterns and performance. "
            "Shows tool frequency, success rates, timing, and error patterns. "
            "Use to understand agent behavior and identify issues."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "top", "errors", "timing", "slow"],
                    "description": "Analytics action",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Filter by tool name",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        action: str = "summary",
        tool_name: str | None = None,
        limit: int = 10,
    ) -> ToolResult:
        """Execute analytics action."""

        analytics = _get_analytics()

        if action == "summary":
            return self._summary(analytics)
        elif action == "top":
            return self._top_tools(analytics, limit)
        elif action == "errors":
            return self._error_stats(analytics, limit)
        elif action == "timing":
            return self._timing_stats(analytics, tool_name)
        elif action == "slow":
            return self._slowest_tools(analytics, limit)
        else:
            return ToolResult.failure(f"Unknown action: {action}")

    def _summary(self, analytics: dict[str, Any]) -> ToolResult:
        """Show overall summary."""
        total = analytics.get("total_calls", 0)
        success = analytics.get("successful_calls", 0)
        errors = analytics.get("error_calls", 0)

        lines = ["## Tool Analytics Summary\n"]
        lines.append(f"- Total tool calls: {total}")
        lines.append(f"- Successful: {success}")
        lines.append(f"- Errors: {errors}")

        if total > 0:
            success_rate = success / total * 100
            lines.append(f"- Success rate: {success_rate:.1f}%")

        return ToolResult.ok("\n".join(lines))

    def _top_tools(self, analytics: dict[str, Any], limit: int) -> ToolResult:
        """Show most-used tools."""
        tool_counts = analytics.get("tool_counts", {})
        if not tool_counts:
            return ToolResult.ok("No tool usage data")

        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

        lines = ["## Top Tools\n"]
        for name, count in sorted_tools:
            pct = count / analytics.get("total_calls", 1) * 100
            lines.append(f"- {name}: {count} ({pct:.1f}%)")

        return ToolResult.ok("\n".join(lines))

    def _error_stats(self, analytics: dict[str, Any], limit: int) -> ToolResult:
        """Show error statistics by tool."""
        error_counts = analytics.get("error_counts", {})
        if not error_counts:
            return ToolResult.ok("No error data")

        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

        lines = ["## Tool Errors\n"]
        for name, count in sorted_errors:
            tool_total = analytics.get("tool_counts", {}).get(name, 1)
            rate = count / tool_total * 100
            lines.append(f"- {name}: {count} errors ({rate:.1f}% of calls)")

        return ToolResult.ok("\n".join(lines))

    def _timing_stats(
        self, analytics: dict[str, Any], tool_name: str | None
    ) -> ToolResult:
        """Show timing statistics."""
        timing = analytics.get("timing", {})
        if not timing:
            return ToolResult.ok("No timing data")

        if tool_name:
            if tool_name not in timing:
                return ToolResult.failure(f"No timing data for {tool_name}")
            t = timing[tool_name]
            lines = [f"## Timing for {tool_name}\n"]
            lines.append(f"- Calls: {t.get('count', 0)}")
            lines.append(f"- Total: {t.get('total', 0):.2f}s")
            lines.append(f"- Avg: {t.get('avg', 0):.2f}s")
            lines.append(f"- Min: {t.get('min', 0):.2f}s")
            lines.append(f"- Max: {t.get('max', 0):.2f}s")
            return ToolResult.ok("\n".join(lines))

        sorted_timing = sorted(
            timing.items(), key=lambda x: x[1].get("avg", 0), reverse=True
        )[:10]

        lines = ["## Tool Timing (slowest)\n"]
        for name, t in sorted_timing:
            lines.append(f"- {name}: {t.get('avg', 0):.2f}s avg ({t.get('count', 0)} calls)")

        return ToolResult.ok("\n".join(lines))

    def _slowest_tools(self, analytics: dict[str, Any], limit: int) -> ToolResult:
        """Show slowest tools by average time."""
        return self._timing_stats(analytics, None)


_analytics: dict[str, Any] = {
    "total_calls": 0,
    "successful_calls": 0,
    "error_calls": 0,
    "tool_counts": {},
    "error_counts": {},
    "timing": {},
}


def _get_analytics() -> dict[str, Any]:
    """Get shared analytics dict (in-memory for current session)."""
    return _analytics


def record_tool_call(
    tool_name: str, success: bool, duration: float, error: str | None = None
) -> None:
    """Record a tool call for analytics."""
    global _analytics

    _analytics["total_calls"] = _analytics.get("total_calls", 0) + 1
    if success:
        _analytics["successful_calls"] = _analytics.get("successful_calls", 0) + 1
    else:
        _analytics["error_calls"] = _analytics.get("error_calls", 0) + 1
        ec = _analytics.get("error_counts", {})
        ec[tool_name] = ec.get(tool_name, 0) + 1
        _analytics["error_counts"] = ec

    tc = _analytics.get("tool_counts", {})
    tc[tool_name] = tc.get(tool_name, 0) + 1
    _analytics["tool_counts"] = tc

    if duration > 0:
        timing = _analytics.get("timing", {})
        if tool_name not in timing:
            timing[tool_name] = {"count": 0, "total": 0, "min": float("inf"), "max": 0}

        t = timing[tool_name]
        t["count"] = t.get("count", 0) + 1
        t["total"] = t.get("total", 0) + duration
        t["avg"] = t["total"] / t["count"]
        t["min"] = min(t.get("min", float("inf")), duration)
        t["max"] = max(t.get("max", 0), duration)
        _analytics["timing"] = timing
