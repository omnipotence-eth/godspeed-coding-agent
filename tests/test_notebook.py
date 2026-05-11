"""Tests for notebook (.ipynb) read/edit support (Unit 2)."""

from __future__ import annotations

import json

import pytest

from godspeed.tools.base import ToolContext


def _make_notebook(cells: list[dict] | None = None) -> dict:
    """Create a minimal valid notebook structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python"}},
        "cells": cells or [],
    }


def _code_cell(source: str, outputs: list | None = None) -> dict:
    return {
        "cell_type": "code",
        "source": source.splitlines(keepends=True),
        "metadata": {},
        "execution_count": 1,
        "outputs": outputs or [],
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "source": source.splitlines(keepends=True),
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# FileReadTool: notebook rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_notebook_code_cells(tmp_path):
    """Reading a .ipynb renders code cells with [Cell N: code] headers."""
    from godspeed.tools.file_read import FileReadTool

    nb = _make_notebook(
        [
            _code_cell("print('hello')"),
            _code_cell("x = 42"),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = FileReadTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"file_path": "test.ipynb"}, ctx)

    assert not result.is_error
    assert "[Cell 0: code]" in result.output
    assert "print('hello')" in result.output
    assert "[Cell 1: code]" in result.output
    assert "x = 42" in result.output


@pytest.mark.asyncio
async def test_read_notebook_markdown_cells(tmp_path):
    """Reading a .ipynb renders markdown cells."""
    nb = _make_notebook([_markdown_cell("# Hello World")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    from godspeed.tools.file_read import FileReadTool

    tool = FileReadTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"file_path": "test.ipynb"}, ctx)

    assert "[Cell 0: markdown]" in result.output
    assert "# Hello World" in result.output


@pytest.mark.asyncio
async def test_read_notebook_with_outputs(tmp_path):
    """Reading a .ipynb renders cell outputs."""
    nb = _make_notebook(
        [
            _code_cell(
                "print('hello')",
                outputs=[
                    {"output_type": "stream", "name": "stdout", "text": ["hello\n"]},
                ],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    from godspeed.tools.file_read import FileReadTool

    tool = FileReadTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"file_path": "test.ipynb"}, ctx)

    assert "[Output: stream]" in result.output
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_read_notebook_with_error_output(tmp_path):
    """Reading a .ipynb renders error outputs."""
    nb = _make_notebook(
        [
            _code_cell(
                "1/0",
                outputs=[
                    {
                        "output_type": "error",
                        "ename": "ZeroDivisionError",
                        "evalue": "division by zero",
                        "traceback": [],
                    },
                ],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    from godspeed.tools.file_read import FileReadTool

    tool = FileReadTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"file_path": "test.ipynb"}, ctx)

    assert "ZeroDivisionError" in result.output


@pytest.mark.asyncio
async def test_read_empty_notebook(tmp_path):
    """Reading an empty notebook returns informative message."""
    nb = _make_notebook([])
    path = tmp_path / "empty.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    from godspeed.tools.file_read import FileReadTool

    tool = FileReadTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"file_path": "empty.ipynb"}, ctx)

    assert "empty notebook" in result.output.lower()


@pytest.mark.asyncio
async def test_read_malformed_notebook(tmp_path):
    """Reading a malformed .ipynb returns error."""
    path = tmp_path / "bad.ipynb"
    path.write_text("{invalid json", encoding="utf-8")

    from godspeed.tools.file_read import FileReadTool

    tool = FileReadTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"file_path": "bad.ipynb"}, ctx)

    assert result.is_error
    assert "parse" in result.error.lower() or "Failed" in result.error


# ---------------------------------------------------------------------------
# NotebookEditTool: cell operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_cell(tmp_path):
    """edit_cell updates cell source."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("old code")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "edit_cell",
            "cell_index": 0,
            "content": "new code",
        },
        ctx,
    )

    assert not result.is_error
    assert "Updated cell 0" in result.output

    # Verify file was updated
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert "".join(updated["cells"][0]["source"]) == "new code"


@pytest.mark.asyncio
async def test_add_cell(tmp_path):
    """add_cell inserts a new cell."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("first")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "add_cell",
            "cell_index": 0,
            "content": "# Header",
            "cell_type": "markdown",
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert len(updated["cells"]) == 2
    assert updated["cells"][0]["cell_type"] == "markdown"


@pytest.mark.asyncio
async def test_add_cell_append(tmp_path):
    """add_cell without index appends to end."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("first")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "add_cell",
            "content": "second",
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert len(updated["cells"]) == 2
    assert "".join(updated["cells"][1]["source"]) == "second"


@pytest.mark.asyncio
async def test_delete_cell(tmp_path):
    """delete_cell removes the specified cell."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("keep"), _code_cell("delete")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "delete_cell",
            "cell_index": 1,
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert len(updated["cells"]) == 1
    assert "".join(updated["cells"][0]["source"]) == "keep"


@pytest.mark.asyncio
async def test_move_cell(tmp_path):
    """move_cell repositions a cell."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a"), _code_cell("b"), _code_cell("c")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "move_cell",
            "cell_index": 2,
            "target_index": 0,
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in updated["cells"]]
    assert sources == ["c", "a", "b"]


@pytest.mark.asyncio
async def test_edit_cell_out_of_range(tmp_path):
    """edit_cell with invalid index returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("only")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "edit_cell",
            "cell_index": 5,
            "content": "nope",
        },
        ctx,
    )

    assert result.is_error
    assert "out of range" in result.error


@pytest.mark.asyncio
async def test_notebook_edit_non_ipynb(tmp_path):
    """NotebookEditTool rejects non-.ipynb files."""
    from godspeed.tools.notebook import NotebookEditTool

    (tmp_path / "test.py").write_text("x = 1")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.py",
            "action": "edit_cell",
            "cell_index": 0,
        },
        ctx,
    )

    assert result.is_error
    assert ".ipynb" in result.error


@pytest.mark.asyncio
async def test_notebook_edit_not_found(tmp_path):
    """NotebookEditTool returns error for missing file."""
    from godspeed.tools.notebook import NotebookEditTool

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "missing.ipynb",
            "action": "edit_cell",
            "cell_index": 0,
        },
        ctx,
    )

    assert result.is_error
    assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# NotebookEditTool: additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_cell_raw_type(tmp_path):
    """add_cell with cell_type='raw' creates a raw cell."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("first")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "add_cell",
            "cell_index": 1,
            "content": "raw content",
            "cell_type": "raw",
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["cells"][1]["cell_type"] == "raw"
    assert "execution_count" not in updated["cells"][1]


@pytest.mark.asyncio
async def test_invalid_action(tmp_path):
    """Invalid action returns an error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("first")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "invalid_action",
        },
        ctx,
    )

    assert result.is_error
    assert "Invalid action" in result.error


@pytest.mark.asyncio
async def test_invalid_notebook_json(tmp_path):
    """Invalid notebook JSON returns parse error."""
    from godspeed.tools.notebook import NotebookEditTool

    path = tmp_path / "bad.ipynb"
    path.write_text("not valid json", encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "bad.ipynb",
            "action": "edit_cell",
            "cell_index": 0,
        },
        ctx,
    )

    assert result.is_error
    assert "parse" in result.error.lower() or "Failed" in result.error


@pytest.mark.asyncio
async def test_empty_file_path(tmp_path):
    """Empty file_path returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "",
            "action": "edit_cell",
        },
        ctx,
    )

    assert result.is_error
    assert "file_path" in result.error.lower()


@pytest.mark.asyncio
async def test_move_cell_missing_indices(tmp_path):
    """move_cell missing cell_index or target_index returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a"), _code_cell("b")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "move_cell",
            "cell_index": 0,
        },
        ctx,
    )
    assert result.is_error
    assert "required" in result.error.lower()

    result2 = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "move_cell",
            "target_index": 0,
        },
        ctx,
    )
    assert result2.is_error
    assert "required" in result2.error.lower()


@pytest.mark.asyncio
async def test_move_cell_target_out_of_range(tmp_path):
    """move_cell with target_index out of range returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a"), _code_cell("b")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "move_cell",
            "cell_index": 0,
            "target_index": 99,
        },
        ctx,
    )

    assert result.is_error
    assert "out of range" in result.error


@pytest.mark.asyncio
async def test_delete_cell_missing_index(tmp_path):
    """delete_cell without cell_index returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "delete_cell",
        },
        ctx,
    )

    assert result.is_error
    assert "required" in result.error.lower()


@pytest.mark.asyncio
async def test_delete_cell_out_of_range(tmp_path):
    """delete_cell with out-of-range index returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "delete_cell",
            "cell_index": 10,
        },
        ctx,
    )

    assert result.is_error
    assert "out of range" in result.error


@pytest.mark.asyncio
async def test_edit_cell_missing_index(tmp_path):
    """edit_cell without cell_index returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "edit_cell",
            "content": "new",
        },
        ctx,
    )

    assert result.is_error
    assert "required" in result.error.lower()


@pytest.mark.asyncio
async def test_add_cell_code_with_empty_content(tmp_path):
    """add_cell with code type and no content creates valid code cell."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "add_cell",
            "cell_type": "code",
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["cells"][0]["cell_type"] == "code"
    assert updated["cells"][0]["execution_count"] is None
    assert updated["cells"][0]["outputs"] == []


@pytest.mark.asyncio
async def test_resolve_tool_path_value_error(tmp_path):
    """resolve_tool_path ValueError propagates as ToolResult failure."""
    from unittest.mock import patch

    from godspeed.tools.notebook import NotebookEditTool

    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(_make_notebook([_code_cell("x")])), encoding="utf-8")

    with patch(
        "godspeed.tools.notebook.resolve_tool_path",
        side_effect=ValueError("Access denied"),
    ):
        tool = NotebookEditTool()
        ctx = ToolContext(cwd=tmp_path, session_id="test")
        result = await tool.execute(
            {"file_path": "test.ipynb", "action": "edit_cell", "cell_index": 0},
            ctx,
        )

    assert result.is_error
    assert "Access denied" in result.error


@pytest.mark.asyncio
async def test_notebook_metadata_and_tool_schema(tmp_path):
    """Invalid notebook structure with missing cells key works gracefully."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}}
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "add_cell",
            "content": "new cell",
        },
        ctx,
    )

    assert not result.is_error
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert len(updated["cells"]) == 1


@pytest.mark.asyncio
async def test_notebook_edit_tool_has_correct_metadata(tmp_path):
    """NotebookEditTool has correct name, description, risk_level, and schema."""
    from godspeed.tools.notebook import NotebookEditTool

    tool = NotebookEditTool()
    assert tool.name == "notebook_edit"
    assert tool.risk_level == "low"
    schema = tool.get_schema()
    assert "file_path" in schema["properties"]
    assert "action" in schema["properties"]
    assert "cell_index" in schema["properties"]
    assert schema["required"] == ["file_path", "action"]

    # Test description property exists
    assert isinstance(tool.description, str)
    assert len(tool.description) > 0


@pytest.mark.asyncio
async def test_move_cell_source_out_of_range(tmp_path):
    """move_cell with cell_index out of range returns error."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([_code_cell("a"), _code_cell("b")])
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "test.ipynb",
            "action": "move_cell",
            "cell_index": 99,
            "target_index": 0,
        },
        ctx,
    )

    assert result.is_error
    assert "out of range" in result.error


@pytest.mark.asyncio
async def test_edit_cell_on_empty_notebook(tmp_path):
    """edit_cell on empty notebook returns out of range."""
    from godspeed.tools.notebook import NotebookEditTool

    nb = _make_notebook([])
    path = tmp_path / "empty.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "empty.ipynb",
            "action": "edit_cell",
            "cell_index": 0,
            "content": "x",
        },
        ctx,
    )

    assert result.is_error
    assert "out of range" in result.error


@pytest.mark.asyncio
async def test_unicode_decode_error(tmp_path):
    """UnicodeDecodeError during notebook parse returns clear error."""
    from godspeed.tools.notebook import NotebookEditTool

    path = tmp_path / "bad.ipynb"
    # Write invalid UTF-8
    with open(str(path), "wb") as f:
        f.write(b"\x80\x81\x82")

    tool = NotebookEditTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute(
        {
            "file_path": "bad.ipynb",
            "action": "edit_cell",
            "cell_index": 0,
        },
        ctx,
    )

    assert result.is_error
