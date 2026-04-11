"""Task tracking tool — in-memory task management for agent self-organization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult


@dataclass
class Task:
    """A tracked task."""

    id: int
    title: str
    description: str = ""
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


class TaskStore:
    """In-memory task store with sequential IDs."""

    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    def create(self, title: str, description: str = "") -> Task:
        """Create a new task."""
        task = Task(id=self._next_id, title=title, description=description)
        self._tasks[self._next_id] = task
        self._next_id += 1
        return task

    def get(self, task_id: int) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update(self, task_id: int, status: str) -> Task | None:
        """Update a task's status. Returns None if not found."""
        task = self._tasks.get(task_id)
        if task is not None:
            task.status = status
        return task

    def complete(self, task_id: int) -> Task | None:
        """Mark a task as completed."""
        return self.update(task_id, "completed")

    def list_active(self) -> list[Task]:
        """List tasks that are not completed."""
        return [t for t in self._tasks.values() if t.status != "completed"]

    def list_all(self) -> list[Task]:
        """List all tasks."""
        return list(self._tasks.values())

    def format_active(self) -> str | None:
        """Format active tasks for system prompt injection.

        Returns None if no active tasks.
        """
        active = self.list_active()
        if not active:
            return None

        lines = []
        for t in active:
            status_icon = "🔄" if t.status == "in_progress" else "⏳"
            lines.append(f"  {status_icon} [{t.id}] {t.title} ({t.status})")
            if t.description:
                lines.append(f"      {t.description}")

        return "\n".join(lines)


class TaskTool(Tool):
    """Tool for creating and managing tasks during an agent session."""

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "tasks"

    @property
    def description(self) -> str:
        return (
            "Create, update, list, and complete tasks to track work progress. "
            "Use this to break complex work into trackable steps."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "list", "complete"],
                        "description": "Action to perform.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title (for create).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description (for create).",
                    },
                    "task_id": {
                        "type": "integer",
                        "description": "Task ID (for update/complete).",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "New status (for update).",
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a task action."""
        action = arguments.get("action", "")

        if action == "create":
            title = arguments.get("title", "")
            if not title:
                return ToolResult.failure("title is required for create action")
            desc = arguments.get("description", "")
            task = self._store.create(title, desc)
            return ToolResult.ok(f"Created task [{task.id}]: {task.title}")

        if action == "list":
            tasks = self._store.list_all()
            if not tasks:
                return ToolResult.ok("No tasks.")
            lines = []
            for t in tasks:
                lines.append(f"[{t.id}] {t.title} — {t.status}")
                if t.description:
                    lines.append(f"    {t.description}")
            return ToolResult.ok("\n".join(lines))

        if action == "update":
            task_id = arguments.get("task_id")
            status = arguments.get("status")
            if task_id is None or status is None:
                return ToolResult.failure("task_id and status are required for update action")
            task = self._store.update(int(task_id), status)
            if task is None:
                return ToolResult.failure(f"Task {task_id} not found")
            return ToolResult.ok(f"Updated task [{task.id}]: {task.title} → {task.status}")

        if action == "complete":
            task_id = arguments.get("task_id")
            if task_id is None:
                return ToolResult.failure("task_id is required for complete action")
            task = self._store.complete(int(task_id))
            if task is None:
                return ToolResult.failure(f"Task {task_id} not found")
            return ToolResult.ok(f"Completed task [{task.id}]: {task.title}")

        return ToolResult.failure(
            f"Unknown action: {action}. Use create, update, list, or complete."
        )
