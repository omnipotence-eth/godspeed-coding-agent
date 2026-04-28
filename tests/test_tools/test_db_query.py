"""Tests for the db_query tool."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from godspeed.tools.base import ToolContext
from godspeed.tools.db_query import DbQueryTool


@pytest.fixture
def db_context(tmp_path: Path) -> ToolContext:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    conn.commit()
    conn.close()
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.mark.asyncio
async def test_select_all(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "SELECT * FROM users"},
        db_context,
    )
    assert not result.is_error
    assert "Alice" in result.output
    assert "Bob" in result.output


@pytest.mark.asyncio
async def test_blocks_insert(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "INSERT INTO users VALUES (3, 'Eve')"},
        db_context,
    )
    assert result.is_error
    assert "select" in result.error.lower()


@pytest.mark.asyncio
async def test_blocks_drop(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "DROP TABLE users"},
        db_context,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_missing_file(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "missing.db", "query": "SELECT 1"},
        db_context,
    )
    assert result.is_error


class TestDbQueryToolMetadata:
    def test_name(self) -> None:
        tool = DbQueryTool()
        assert tool.name == "db_query"

    def test_risk_level(self) -> None:
        tool = DbQueryTool()
        assert tool.risk_level.value == "read_only"

    def test_schema(self) -> None:
        tool = DbQueryTool()
        schema = tool.get_schema()
        assert "path" in schema["properties"]
        assert "query" in schema["properties"]
