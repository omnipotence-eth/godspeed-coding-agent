"""Tests for godspeed.tools.db_query."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.tools.db_query import (
    _MAX_ROWS,
    DbQueryTool,
    _ensure_limit,
    _format_table,
    _validate_select,
)


class TestValidateSelect:
    def test_empty_query(self):
        result = _validate_select("")
        assert result == "Query is empty"

    def test_non_select_query(self):
        result = _validate_select("INSERT INTO users VALUES (1, 'test')")
        assert "Only SELECT queries are allowed" in result

    def test_select_allowed(self):
        result = _validate_select("SELECT * FROM users")
        assert result is None

    def test_select_with_whitespace(self):
        result = _validate_select("   SELECT * FROM users   ")
        assert result is None

    def test_forbidden_keyword_insert(self):
        result = _validate_select("SELECT * FROM users; INSERT INTO users VALUES (1)")
        assert "forbidden keyword" in result.lower()

    def test_forbidden_keyword_drop(self):
        result = _validate_select("SELECT * FROM users; DROP TABLE users")
        assert "forbidden keyword" in result.lower()

    def test_forbidden_keyword_update(self):
        # UPDATE is caught when it appears as a separate statement
        result = _validate_select("SELECT * FROM users; UPDATE users SET name='test'")
        assert "forbidden keyword" in result.lower()

    def test_forbidden_keyword_delete(self):
        # DELETE is caught when it appears as a separate statement
        result = _validate_select("SELECT * FROM users; DELETE FROM users")
        assert "forbidden keyword" in result.lower()


class TestEnsureLimit:
    def test_adds_limit(self):
        query = "SELECT * FROM users"
        result = _ensure_limit(query)
        assert f"LIMIT {_MAX_ROWS}" in result

    def test_preserves_existing_limit(self):
        query = "SELECT * FROM users LIMIT 10"
        result = _ensure_limit(query)
        assert result == query

    def test_removes_trailing_semicolon(self):
        query = "SELECT * FROM users;"
        result = _ensure_limit(query)
        assert result == f"SELECT * FROM users LIMIT {_MAX_ROWS}"  # noqa: S608


class TestFormatTable:
    def test_empty_headers(self):
        result = _format_table([], [])
        assert "empty result set" in result

    def test_single_row(self):
        headers = ["id", "name"]
        rows = [(1, "Alice")]
        result = _format_table(headers, rows)
        assert "id" in result
        assert "name" in result
        assert "Alice" in result

    def test_multiple_rows(self):
        headers = ["id", "name"]
        rows = [(1, "Alice"), (2, "Bob")]
        result = _format_table(headers, rows)
        assert "Alice" in result
        assert "Bob" in result

    def test_column_width_adjusts(self):
        headers = ["id", "name"]
        rows = [(1, "A" * 50)]
        result = _format_table(headers, rows)
        # Check that the wide column is formatted
        assert "A" * 50 in result


class TestDbQueryToolMetadata:
    def test_name(self):
        tool = DbQueryTool()
        assert tool.name == "db_query"

    def test_risk_level(self):
        tool = DbQueryTool()
        assert tool.risk_level.value == "read_only"

    def test_description_contains_keywords(self):
        tool = DbQueryTool()
        desc = tool.description.lower()
        assert "sql" in desc or "query" in desc

    def test_get_schema(self):
        tool = DbQueryTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "query" in schema["properties"]
        assert "path" in schema["required"]
        assert "query" in schema["required"]


class TestDbQueryToolExecute:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Create a test SQLite database."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.execute("INSERT INTO users VALUES (2, 'Bob')")
        conn.commit()
        conn.close()
        return db

    @pytest.mark.asyncio
    async def test_missing_path(self):
        tool = DbQueryTool()
        result = await tool.execute({"query": "SELECT * FROM users"}, MagicMock())
        assert result.is_error is True
        assert "path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_path(self):
        tool = DbQueryTool()
        result = await tool.execute({"path": "", "query": "SELECT 1"}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_missing_query(self, db_path):
        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = db_path.parent
        result = await tool.execute({"path": db_path.name}, context)
        assert result.is_error is True
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_db(self, tmp_path):
        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = tmp_path
        result = await tool.execute({"path": "nonexistent.db", "query": "SELECT 1"}, context)
        assert result.is_error is True
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_successful_query(self, db_path):
        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = db_path.parent
        result = await tool.execute({"path": db_path.name, "query": "SELECT * FROM users"}, context)
        assert result.is_error is False
        assert "Alice" in result.output
        assert "Bob" in result.output

    @pytest.mark.asyncio
    async def test_select_only_enforced(self, db_path):
        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = db_path.parent
        result = await tool.execute(
            {"path": db_path.name, "query": "INSERT INTO users VALUES (3, 'Charlie')"}, context
        )
        assert result.is_error is True
        assert "only select" in result.error.lower() or "forbidden" in result.error.lower()

    @pytest.mark.asyncio
    async def test_forbidden_keyword(self, db_path):
        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = db_path.parent
        result = await tool.execute(
            {"path": db_path.name, "query": "SELECT * FROM users; DROP TABLE users"}, context
        )
        assert result.is_error is True
        assert "forbidden" in result.error.lower()

    @pytest.mark.asyncio
    async def test_query_results_truncated(self, db_path):
        # Insert more than _MAX_ROWS rows
        conn = sqlite3.connect(str(db_path))
        for i in range(_MAX_ROWS + 10):
            conn.execute(f"INSERT INTO users VALUES ({i + 100}, 'User{i}')")  # noqa: S608
        conn.commit()
        conn.close()

        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = db_path.parent
        result = await tool.execute({"path": db_path.name, "query": "SELECT * FROM users"}, context)
        assert result.is_error is False
        # Check that results are limited to _MAX_ROWS
        assert f"({_MAX_ROWS} row" in result.output

    @pytest.mark.asyncio
    async def test_empty_result_set(self, db_path):
        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = db_path.parent
        result = await tool.execute(
            {"path": db_path.name, "query": "SELECT * FROM users WHERE 1=0"}, context
        )
        assert result.is_error is False
        # Should return success with empty result message
        assert (
            "0 row" in result.output
            or "no rows" in result.output.lower()
            or "successfully" in result.output.lower()
        )

    @pytest.mark.asyncio
    async def test_database_error(self, tmp_path):
        # Create a file that's not a valid SQLite DB
        not_db = tmp_path / "not_a_db.db"
        not_db.write_text("This is not a database")

        tool = DbQueryTool()
        context = MagicMock()
        context.cwd = tmp_path
        result = await tool.execute({"path": not_db.name, "query": "SELECT * FROM users"}, context)
        assert result.is_error is True
        assert "database error" in result.error.lower() or "sqlite" in result.error.lower()
