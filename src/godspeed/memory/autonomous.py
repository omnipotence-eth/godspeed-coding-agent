"""Memory system for autonomous context persistence between sessions."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DB_PATH = Path.home() / ".godspeed" / "memories.db"


class MemoryStore:
    """Autonomous memory generation and retrieval.

    Stores learnings about the codebase that persist between sessions:
    - Architecture patterns
    - Naming conventions
    - Build commands
    - Configuration quirks
    - Dependency relationships
    """

    def __init__(self, db_path: Path | None = None) -> None:
        import sqlite3

        self._db_path = db_path or MEMORY_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                project_dir TEXT DEFAULT '',
                created_at REAL,
                last_accessed REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                interaction_type TEXT,
                timestamp REAL,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the connection."""
        if self._conn:
            self._conn.close()

    def store_memory(
        self,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        project_dir: str = "",
    ) -> int:
        """Store a new memory."""
        import time

        cursor = self._conn.execute(
            "INSERT INTO memories (memory_type, content, importance, project_dir, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_type, content, importance, project_dir, time.time(), time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_memories(
        self,
        project_dir: str = "",
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories."""
        import time

        if memory_type:
            cursor = self._conn.execute(
                "SELECT * FROM memories WHERE project_dir = ? AND memory_type = ? "
                "ORDER BY importance DESC LIMIT ?",
                (project_dir, memory_type, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM memories WHERE project_dir = ? "
                "ORDER BY importance DESC, last_accessed DESC LIMIT ?",
                (project_dir, limit),
            )

        memories = [dict(row) for row in cursor.fetchall()]

        # Update last accessed
        now = time.time()
        for m in memories:
            self._conn.execute(
                "UPDATE memories SET last_accessed = ? WHERE id = ?",
                (now, m["id"]),
            )
        self._conn.commit()

        return memories

    def update_importance(self, memory_id: int, importance: float) -> None:
        """Update memory importance based on usage."""
        self._conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (importance, memory_id),
        )
        self._conn.commit()

    def delete_memory(self, memory_id: int) -> None:
        """Delete a memory."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def auto_generate_memory(
        self,
        project_dir: str,
        recent_actions: list[dict[str, Any]],
    ) -> list[int]:
        """Automatically generate memories from recent actions.

        Analyzes recent actions and stores important patterns.
        Returns list of created memory IDs.
        """
        import time

        created = []

        # Analyze file edits for patterns
        edits_by_ext: dict[str, int] = {}
        for action in recent_actions:
            if action.get("type") == "file_edit":
                path = action.get("file_path", "")
                ext = os.path.splitext(path)[1]
                edits_by_ext[ext] = edits_by_ext.get(ext, 0) + 1

        # Store file extension patterns
        if edits_by_ext:
            top_exts = sorted(edits_by_ext.items(), key=lambda x: x[1], reverse=True)[:3]
            content = f"Primary file types: {', '.join(f'{e}[{c}]' for e, c in top_exts)}"
            mid = self.store_memory(
                memory_type="architecture",
                content=content,
                importance=0.6,
                project_dir=project_dir,
            )
            created.append(mid)

        # Analyze terminal commands
        commands: dict[str, int] = {}
        for action in recent_actions:
            if action.get("type") == "shell":
                cmd = action.get("command", "")
                if cmd:
                    base = cmd.split()[0] if " " in cmd else cmd
                    commands[base] = commands.get(base, 0) + 1

        if commands:
            top_cmds = sorted(commands.items(), key=lambda x: x[1], reverse=True)[:5]
            content = f"Common commands: {', '.join(f'{c}[{n}]' for c, n in top_cmds)}"
            mid = self.store_memory(
                memory_type="workflow",
                content=content,
                importance=0.7,
                project_dir=project_dir,
            )
            created.append(mid)

        return created


# Global store instance
_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def close_memory_store() -> None:
    """Close the global memory store."""
    global _store
    if _store:
        _store.close()
        _store = None