"""Session memory — persistent session event logging.

Records session events (start, end, tool calls, errors) to SQLite for
cross-session learning and context resumption.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    project_dir TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON session_events(event_type);
"""


class SessionMemory:
    """Persistent session event store backed by SQLite.

    Records session lifecycle events for cross-session learning:
    - session_start / session_end
    - tool_call, tool_error
    - user_correction
    - compaction

    Shares the same database as UserMemory (WAL mode, safe concurrent access).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".godspeed" / "memory.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_INIT_SQL)
        self._conn: sqlite3.Connection | None = conn
        logger.info("session_memory.init db_path=%s", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Return the active connection, raising if closed."""
        if self._conn is None:
            msg = "Database connection is closed"
            raise RuntimeError(msg)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Session lifecycle -----------------------------------------------------

    def start_session(self, session_id: str, model: str, project_dir: str = "") -> None:
        """Record a new session start."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, model, started_at, project_dir) "
            "VALUES (?, ?, ?, ?)",
            (session_id, model, time.time(), project_dir),
        )
        conn.commit()
        logger.info("session_memory.start session_id=%s model=%s", session_id, model)

    def end_session(self, session_id: str, summary: str = "") -> None:
        """Record session end with optional summary."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (time.time(), summary, session_id),
        )
        conn.commit()
        logger.info("session_memory.end session_id=%s", session_id)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent sessions, newest first."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    # -- Events ----------------------------------------------------------------

    def record_event(
        self,
        session_id: str,
        event_type: str,
        detail: str = "",
    ) -> int:
        """Record a session event. Returns the event ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO session_events (session_id, event_type, detail, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, event_type, detail, time.time()),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get events for a session, optionally filtered by type."""
        conn = self._get_conn()
        if event_type:
            cursor = conn.execute(
                "SELECT * FROM session_events "
                "WHERE session_id = ? AND event_type = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, event_type, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        return [dict(row) for row in cursor.fetchall()]

    def event_count(self, session_id: str) -> int:
        """Count events for a session."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0
