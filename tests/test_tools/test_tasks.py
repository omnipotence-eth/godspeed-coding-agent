"""Tests for task tracking tool and store."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.tasks import TaskStore, TaskTool


@pytest.fixture()
def store() -> TaskStore:
    return TaskStore()


@pytest.fixture()
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


class TestTaskStore:
    """Test TaskStore."""

    def test_create_task(self, store: TaskStore) -> None:
        task = store.create("Fix bug", "The login page crashes")
        assert task.id == 1
        assert task.title == "Fix bug"
        assert task.description == "The login page crashes"
        assert task.status == "pending"

    def test_sequential_ids(self, store: TaskStore) -> None:
        t1 = store.create("Task 1")
        t2 = store.create("Task 2")
        t3 = store.create("Task 3")
        assert t1.id == 1
        assert t2.id == 2
        assert t3.id == 3

    def test_get_task(self, store: TaskStore) -> None:
        store.create("Find me")
        task = store.get(1)
        assert task is not None
        assert task.title == "Find me"

    def test_get_nonexistent(self, store: TaskStore) -> None:
        assert store.get(999) is None

    def test_update_status(self, store: TaskStore) -> None:
        store.create("Update me")
        task = store.update(1, "in_progress")
        assert task is not None
        assert task.status == "in_progress"

    def test_update_nonexistent(self, store: TaskStore) -> None:
        assert store.update(999, "in_progress") is None

    def test_complete(self, store: TaskStore) -> None:
        store.create("Complete me")
        task = store.complete(1)
        assert task is not None
        assert task.status == "completed"

    def test_list_active(self, store: TaskStore) -> None:
        store.create("Active 1")
        store.create("Active 2")
        store.create("Done")
        store.complete(3)
        active = store.list_active()
        assert len(active) == 2
        assert all(t.status != "completed" for t in active)

    def test_list_all(self, store: TaskStore) -> None:
        store.create("Task 1")
        store.create("Task 2")
        store.complete(1)
        assert len(store.list_all()) == 2

    def test_format_active_empty(self, store: TaskStore) -> None:
        assert store.format_active() is None

    def test_format_active_with_tasks(self, store: TaskStore) -> None:
        store.create("Fix bug", "Important bug")
        store.create("Write tests")
        store.update(1, "in_progress")
        result = store.format_active()
        assert result is not None
        assert "Fix bug" in result
        assert "Write tests" in result
        assert "in_progress" in result

    def test_format_active_excludes_completed(self, store: TaskStore) -> None:
        store.create("Done task")
        store.complete(1)
        assert store.format_active() is None


class TestTaskTool:
    """Test TaskTool execution."""

    @pytest.fixture()
    def tool(self, store: TaskStore) -> TaskTool:
        return TaskTool(store)

    def test_name_and_risk(self, tool: TaskTool) -> None:
        assert tool.name == "tasks"
        assert tool.risk_level == RiskLevel.LOW

    def test_schema_has_action(self, tool: TaskTool) -> None:
        schema = tool.get_schema()
        assert "action" in schema["parameters"]["properties"]

    @pytest.mark.asyncio()
    async def test_create(self, tool: TaskTool, context: ToolContext) -> None:
        result = await tool.execute(
            {"action": "create", "title": "New task", "description": "Do it"},
            context,
        )
        assert not result.is_error
        assert "New task" in result.output
        assert "[1]" in result.output

    @pytest.mark.asyncio()
    async def test_create_without_title(self, tool: TaskTool, context: ToolContext) -> None:
        result = await tool.execute({"action": "create"}, context)
        assert result.is_error

    @pytest.mark.asyncio()
    async def test_list_empty(self, tool: TaskTool, context: ToolContext) -> None:
        result = await tool.execute({"action": "list"}, context)
        assert "No tasks" in result.output

    @pytest.mark.asyncio()
    async def test_list_with_tasks(self, tool: TaskTool, context: ToolContext) -> None:
        await tool.execute({"action": "create", "title": "Task A"}, context)
        await tool.execute({"action": "create", "title": "Task B"}, context)
        result = await tool.execute({"action": "list"}, context)
        assert "Task A" in result.output
        assert "Task B" in result.output

    @pytest.mark.asyncio()
    async def test_update(self, tool: TaskTool, context: ToolContext) -> None:
        await tool.execute({"action": "create", "title": "Update me"}, context)
        result = await tool.execute(
            {"action": "update", "task_id": 1, "status": "in_progress"},
            context,
        )
        assert not result.is_error
        assert "in_progress" in result.output

    @pytest.mark.asyncio()
    async def test_update_missing_args(self, tool: TaskTool, context: ToolContext) -> None:
        result = await tool.execute({"action": "update", "task_id": 1}, context)
        assert result.is_error

    @pytest.mark.asyncio()
    async def test_complete(self, tool: TaskTool, context: ToolContext) -> None:
        await tool.execute({"action": "create", "title": "Complete me"}, context)
        result = await tool.execute({"action": "complete", "task_id": 1}, context)
        assert not result.is_error
        assert "Completed" in result.output

    @pytest.mark.asyncio()
    async def test_complete_nonexistent(self, tool: TaskTool, context: ToolContext) -> None:
        result = await tool.execute({"action": "complete", "task_id": 999}, context)
        assert result.is_error

    @pytest.mark.asyncio()
    async def test_unknown_action(self, tool: TaskTool, context: ToolContext) -> None:
        result = await tool.execute({"action": "invalid"}, context)
        assert result.is_error
        assert "Unknown action" in result.output or "Unknown action" in (result.error or "")
