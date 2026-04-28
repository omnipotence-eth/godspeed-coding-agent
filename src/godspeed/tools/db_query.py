"""Database query tool — read-only SQL queries against SQLite databases."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

_MAX_ROWS = 50
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "REINDEX",
        "VACUUM",
    }
)

_SELECT_RE = __import__("re").compile(r"^\s*SELECT\b", __import__("re").IGNORECASE)


def _validate_select(query: str) -> str | None:
    """Return an error message if *query* is not a SELECT, otherwise None."""
    cleaned = query.strip()
    if not cleaned:
        return "Query is empty"

    if not _SELECT_RE.match(cleaned):
        return "Only SELECT queries are allowed"

    upper = cleaned.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        if keyword in upper:
            return f"Query contains forbidden keyword: {keyword}"

    return None


def _ensure_limit(query: str) -> str:
    """Add ``LIMIT {_MAX_ROWS}`` if the query lacks a LIMIT clause."""
    cleaned = query.rstrip(";").strip()
    upper = cleaned.upper()
    if "LIMIT" in upper:
        return cleaned
    return f"{cleaned} LIMIT {_MAX_ROWS}"


def _format_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Format query results as a readable ASCII table."""
    if not headers:
        return "(empty result set — no columns)"

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    parts: list[str] = []

    # Header
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
    parts.append(" | ".join(header_cells))

    # Separator
    sep_parts = ["-" * col_widths[i] for i in range(len(headers))]
    parts.append("-+-".join(sep_parts))

    # Rows
    for row in rows:
        row_cells = [str(val).ljust(col_widths[i]) for i, val in enumerate(row)]
        parts.append(" | ".join(row_cells))

    return "\n".join(parts)


class DbQueryTool(Tool):
    """Run read-only SQL queries against a SQLite database file.

    Only SELECT statements are permitted. Results are limited to 50 rows.
    The database file must be within the project directory.
    """

    @property
    def name(self) -> str:
        return "db_query"

    @property
    def description(self) -> str:
        return (
            "Run read-only SQL queries against a SQLite database file. "
            "Only SELECT statements are allowed (no INSERT, UPDATE, DELETE, "
            "DROP, ALTER, CREATE). Results are limited to 50 rows.\n\n"
            "Example: db_query(path='data/app.db', query='SELECT * FROM users')\n"
            "Example: db_query(path='data/app.db', query='SELECT name, email FROM users "
            "WHERE active = 1 LIMIT 10')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the SQLite database file (relative to project root)",
                    "examples": ["data/app.db", "db/production.sqlite"],
                },
                "query": {
                    "type": "string",
                    "description": (
                        "SQL query to execute. Only SELECT statements are allowed. "
                        "Results are automatically limited to 50 rows if no LIMIT is specified."
                    ),
                    "examples": [
                        "SELECT * FROM users",
                        "SELECT name, email FROM users WHERE active = 1",
                    ],
                },
            },
            "required": ["path", "query"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = arguments.get("path", "")
        query = arguments.get("query", "")

        if not isinstance(path_str, str) or not path_str.strip():
            return ToolResult.failure("path must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.failure("query must be a non-empty string")

        # Path containment check
        try:
            db_path = resolve_tool_path(path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not db_path.exists():
            return ToolResult.failure(f"Database file not found: {db_path}")
        if not db_path.is_file():
            return ToolResult.failure(f"Not a file: {db_path}")

        # Validate SELECT-only
        validation_error = _validate_select(query)
        if validation_error:
            return ToolResult.failure(validation_error)

        # Ensure row limit
        safe_query = _ensure_limit(query)

        return await self._run_query(db_path, safe_query)

    @staticmethod
    async def _run_query(db_path: Any, query: str) -> ToolResult:
        import asyncio

        def _query_sync() -> ToolResult:
            conn = sqlite3.connect(str(db_path))
            try:
                # Verify it's a valid SQLite DB
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 0")

                cursor = conn.execute(query)
                headers = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchmany(_MAX_ROWS + 1)
                truncated = len(rows) > _MAX_ROWS
                if truncated:
                    rows = rows[:_MAX_ROWS]

                if not headers:
                    return ToolResult.success("Query executed successfully (no rows returned)")

                table = _format_table(headers, rows)
                if truncated:
                    table += f"\n\n(Results truncated at {_MAX_ROWS} rows)"
                table += f"\n\n({len(rows)} row{'s' if len(rows) != 1 else ''})"
                return ToolResult.success(table)

            except sqlite3.DatabaseError as exc:
                return ToolResult.failure(f"Database error: {exc}")
            except Exception as exc:
                logger.error("Unexpected SQLite error: %s", exc, exc_info=True)
                return ToolResult.failure(f"SQLite error: {exc}")
            finally:
                conn.close()

        return await asyncio.to_thread(_query_sync)
