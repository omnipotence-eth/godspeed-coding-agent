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


@pytest.mark.asyncio
async def test_path_is_directory(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    subdir = db_context.cwd / "adir"
    subdir.mkdir()
    result = await tool.execute(
        {"path": "adir", "query": "SELECT 1"},
        db_context,
    )
    assert result.is_error
    assert "Not a file" in result.error


@pytest.mark.asyncio
async def test_path_traversal(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "../../etc/passwd", "query": "SELECT 1"},
        db_context,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_query_with_limit(db_context: ToolContext) -> None:
    conn = sqlite3.connect(str(db_context.cwd / "test.db"))
    for i in range(60):
        conn.execute("INSERT INTO users VALUES (?, ?)", (i + 3, f"User{i}"))
    conn.commit()
    conn.close()
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "SELECT * FROM users LIMIT 60"},
        db_context,
    )
    assert not result.is_error
    assert "truncated" in result.output.lower()


@pytest.mark.asyncio
async def test_empty_result_set(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "SELECT * FROM users WHERE id = 999"},
        db_context,
    )
    assert not result.is_error
    assert "(0 rows)" in result.output


@pytest.mark.asyncio
async def test_single_row_grammar(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "SELECT * FROM users WHERE id = 1"},
        db_context,
    )
    assert not result.is_error
    assert "(1 row)" in result.output


@pytest.mark.asyncio
async def test_empty_query_rejected(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": ""},
        db_context,
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_non_select_query_rejected(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "EXPLAIN SELECT 1"},
        db_context,
    )
    assert result.is_error
    assert "SELECT" in result.error


@pytest.mark.asyncio
async def test_forbidden_keyword_in_select(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": "SELECT * FROM users; DROP TABLE users"},
        db_context,
    )
    assert result.is_error
    assert "forbidden" in result.error.lower()


@pytest.mark.asyncio
async def test_invalid_path_type(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": 123, "query": "SELECT 1"},
        db_context,
    )
    assert result.is_error
    assert "path" in result.error.lower()


@pytest.mark.asyncio
async def test_invalid_query_type(db_context: ToolContext) -> None:
    tool = DbQueryTool()
    result = await tool.execute(
        {"path": "test.db", "query": None},
        db_context,
    )
    assert result.is_error
    assert "query" in result.error.lower()


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
