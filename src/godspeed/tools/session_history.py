"""Session history browser tool."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SessionHistoryTool(Tool):
    """Browse and search session history.

    Shows past sessions, events, and tool calls for continuity
    and learning from past work.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "session_history"

    @property
    def description(self) -> str:
        return (
            "Browse session history and past events. "
            "Shows recent sessions, session events, tool calls, and statistics. "
            "Use to recall what was done in previous sessions."
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
                    "enum": ["list", "events", "search", "stats", "show"],
                    "description": "Action to perform",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID (optional for list/stats)",
                },
                "event_type": {
                    "type": "string",
                    "description": "Filter events by type (tool_call, tool_error, user_correction)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                },
                "query": {
                    "type": "string",
                    "description": "Search query for content search",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        action: str,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 10,
        query: str | None = None,
    ) -> ToolResult:
        """Execute session history action."""
        from godspeed.memory.session import SessionMemory

        memory = SessionMemory()

        try:
            if action == "list":
                return await self._list_sessions(memory, limit)
            elif action == "stats":
                return await self._show_stats(memory)
            elif action == "events":
                if not session_id:
                    return ToolResult.failure("session_id required for events action")
                return await self._show_events(memory, session_id, event_type, limit)
            elif action == "show":
                if not session_id:
                    return ToolResult.failure("session_id required for show action")
                return await self._show_session(memory, session_id)
            elif action == "search":
                return await self._search_events(memory, query or "", limit)
            else:
                return ToolResult.failure(f"Unknown action: {action}")
        finally:
            memory.close()

    async def _list_sessions(self, memory: SessionMemory, limit: int) -> ToolResult:
        """List recent sessions."""
        sessions = memory.list_sessions(limit)
        if not sessions:
            return ToolResult.ok("No sessions found")

        lines = ["## Recent Sessions\n"]
        for s in sessions:
            sid = s.get("id", "?")[:8]
            model = s.get("model", "?")
            started = s.get("started_at", 0)
            import time

            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(started))
            summary = s.get("summary", "")[:50] or "(no summary)"
            lines.append(f"- `{sid}` | {model} | {ts} | {summary}")

        return ToolResult.ok("\n".join(lines))

    async def _show_stats(self, memory: SessionMemory) -> ToolResult:
        """Show session statistics."""
        sessions = memory.list_sessions(100)
        total_events = sum(memory.event_count(s["id"]) for s in sessions)

        lines = ["## Session Statistics\n"]
        lines.append(f"- Total sessions: {len(sessions)}")
        lines.append(f"- Total events: {total_events}")

        from collections import Counter

        event_types: Counter[str] = Counter()
        for s in sessions:
            events = memory.get_events(s["id"], limit=100)
            for e in events:
                event_types[e.get("event_type", "?")] += 1

        if event_types:
            lines.append("\n### Event Types")
            for etype, count in event_types.most_common(10):
                lines.append(f"- {etype}: {count}")

        return ToolResult.ok("\n".join(lines))

    async def _show_events(
        self, memory: SessionMemory, session_id: str, event_type: str | None, limit: int
    ) -> ToolResult:
        """Show events for a session."""
        events = memory.get_events(session_id, event_type, limit)
        if not events:
            return ToolResult.ok(f"No events found for session {session_id[:8]}")

        lines = [f"## Events for {session_id[:8]}\n"]
        for e in events:
            etype = e.get("event_type", "?")
            detail = e.get("detail", "")[:80]
            import time

            ts = time.strftime("%H:%M", time.localtime(e.get("created_at", 0)))
            lines.append(f"- [{ts}] {etype}: {detail}")

        return ToolResult.ok("\n".join(lines))

    async def _show_session(self, memory: SessionMemory, session_id: str) -> ToolResult:
        """Show full session details."""
        session = memory.get_session(session_id)
        if not session:
            return ToolResult.failure(f"Session not found: {session_id}")

        lines = [f"## Session {session_id[:8]}\n"]
        lines.append(f"- Model: {session.get('model', '?')}")
        lines.append(f"- Project: {session.get('project_dir', '?')}")

        import time

        started = session.get("started_at", 0)
        ended = session.get("ended_at", 0)
        lines.append(f"- Started: {time.strftime('%Y-%m-%d %H:%M', time.localtime(started))}")
        if ended:
            lines.append(f"- Duration: {ended - started:.1f}s")

        summary = session.get("summary", "")
        if summary:
            lines.append(f"\n### Summary\n{summary}")

        events = memory.get_events(session_id, limit=50)
        if events:
            lines.append(f"\n### Events ({len(events)})")
            event_types: dict[str, int] = {}
            for e in events:
                t = e.get("event_type", "?")
                event_types[t] = event_types.get(t, 0) + 1
            for t, c in event_types.items():
                lines.append(f"- {t}: {c}")

        return ToolResult.ok("\n".join(lines))

    async def _search_events(
        self, memory: SessionMemory, query: str, limit: int
    ) -> ToolResult:
        """Search events across all sessions."""
        if not query:
            return ToolResult.failure("query required for search")

        sessions = memory.list_sessions(20)
        results: list[dict[str, Any]] = []

        for s in sessions:
            events = memory.get_events(s["id"], limit=20)
            for e in events:
                detail = e.get("detail", "") or ""
                if query.lower() in detail.lower():
                    results.append(
                        {
                            "session": s["id"][:8],
                            "type": e.get("event_type"),
                            "detail": detail[:100],
                            "time": e.get("created_at"),
                        }
                    )

        if not results:
            return ToolResult.ok(f"No matches for: {query}")

        lines = [f"## Search Results for '{query}'\n"]
        for r in results[:limit]:
            lines.append(
                f"- [{r['session']}] {r['type']}: {r['detail'][:60]}"
            )

        return ToolResult.ok("\n".join(lines))
