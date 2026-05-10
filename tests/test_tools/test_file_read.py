"""Tests for file_read tool."""

from __future__ import annotations

import json

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.file_read import MAX_FILE_SIZE_KB, FileReadTool


@pytest.fixture
def tool() -> FileReadTool:
    return FileReadTool()


class TestFileReadTool:
    """Test file reading."""

    def test_metadata(self, tool: FileReadTool) -> None:
        assert tool.name == "file_read"
        assert tool.risk_level == "read_only"
        assert "Read a file's contents with line numbers" in tool.description
        assert "offset" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_read_file(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")
        result = await tool.execute({"file_path": "test.py"}, tool_context)
        assert not result.is_error
        assert "1\tline 1" in result.output
        assert "2\tline 2" in result.output
        assert "3\tline 3" in result.output

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": "missing.py"}, tool_context)
        assert result.is_error
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines))
        result = await tool.execute({"file_path": "test.py", "offset": 5}, tool_context)
        assert not result.is_error
        assert "5\tline 5" in result.output
        assert "1\tline 1" not in result.output

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines))
        result = await tool.execute({"file_path": "test.py", "limit": 3}, tool_context)
        assert not result.is_error
        assert "3\tline 3" in result.output
        assert "more lines" in result.output

    @pytest.mark.asyncio
    async def test_directory_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        subdir = tool_context.cwd / "subdir"
        subdir.mkdir()
        result = await tool.execute({"file_path": "subdir"}, tool_context)
        assert result.is_error
        assert "Not a file" in result.error

    @pytest.mark.asyncio
    async def test_empty_file(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "empty.py"
        test_file.write_text("")
        result = await tool.execute({"file_path": "empty.py"}, tool_context)
        assert result.is_error
        assert "No content" in result.error

    @pytest.mark.asyncio
    async def test_empty_file_path(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": ""}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    @pytest.mark.asyncio
    async def test_offset_as_convertible_float(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\nc\nd\ne\n")
        result = await tool.execute({"file_path": "test.py", "offset": 3.0}, tool_context)
        assert not result.is_error
        assert "3\ta" in result.output or "3\t" in result.output

    @pytest.mark.asyncio
    async def test_offset_invalid_type(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "offset": "abc"}, tool_context)
        assert result.is_error
        assert "offset must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_limit_as_convertible_float(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\nc\nd\n")
        result = await tool.execute({"file_path": "test.py", "limit": 2.0}, tool_context)
        assert not result.is_error
        assert "more lines" in result.output

    @pytest.mark.asyncio
    async def test_limit_invalid_type(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "limit": "xyz"}, tool_context)
        assert result.is_error
        assert "limit must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_file_path_not_string(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": 123}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    @pytest.mark.asyncio
    async def test_offset_below_one_clamped(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("line 1\nline 2\n")
        result = await tool.execute({"file_path": "test.py", "offset": 0}, tool_context)
        assert not result.is_error
        assert "1\tline 1" in result.output

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "limit": 99999}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_offset_beyond_file(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "offset": 100}, tool_context)
        assert result.is_error
        assert "No content" in result.error

    @pytest.mark.asyncio
    async def test_file_too_large(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "big.bin"
        content = "x" * int((MAX_FILE_SIZE_KB + 1) * 1024)
        test_file.write_text(content)
        result = await tool.execute({"file_path": "big.bin"}, tool_context)
        assert result.is_error
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_binary_file_detection(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "data.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
        result = await tool.execute({"file_path": "data.bin"}, tool_context)
        assert result.is_error
        assert "binary file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_notebook_empty(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "empty.ipynb"
        nb = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "empty.ipynb"}, tool_context)
        assert not result.is_error
        assert "empty notebook" in result.output.lower()

    @pytest.mark.asyncio
    async def test_notebook_with_code_cell(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "test.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "outputs": [],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "test.ipynb"}, tool_context)
        assert not result.is_error
        assert "print('hello')" in result.output
        assert "[Cell 0: code]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_with_stream_output(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "outputs.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hi')"],
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["hi\n"],
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "outputs.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: stream]" in result.output
        assert "hi" in result.output

    @pytest.mark.asyncio
    async def test_notebook_with_execute_result_output(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "result.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["42"],
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": ["42"],
                            },
                            "execution_count": 1,
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "result.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: execute_result]" in result.output
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_notebook_with_error_output(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "error.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["1/0"],
                    "outputs": [
                        {
                            "output_type": "error",
                            "ename": "ZeroDivisionError",
                            "evalue": "division by zero",
                            "traceback": ["Traceback..."],
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "error.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: error]" in result.output
        assert "ZeroDivisionError" in result.output

    @pytest.mark.asyncio
    async def test_notebook_invalid_json(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "broken.ipynb"
        nb_path.write_text("not valid json")
        result = await tool.execute({"file_path": "broken.ipynb"}, tool_context)
        assert result.is_error
        assert "Failed to parse notebook" in result.error

    @pytest.mark.asyncio
    async def test_notebook_with_display_data(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "display.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import matplotlib"],
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/plain": ["<Figure>"],
                            },
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "display.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: display_data]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_cell_without_source(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "nosource.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "nosource.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Cell 0: markdown]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_unknown_output_type(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "unknown.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('test')"],
                    "outputs": [
                        {
                            "output_type": "unknown_type_xyz",
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "unknown.ipynb"}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_path_outside_cwd(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": "../outside.py"}, tool_context)
        assert result.is_error
        assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_notebook_binary_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "bin.ipynb"
        nb_path.write_bytes(b"\x00\x01\x02\xff\xfe")
        result = await tool.execute({"file_path": "bin.ipynb"}, tool_context)
        assert result.is_error

    # --- Offset edge cases ---

    @pytest.mark.asyncio
    async def test_offset_none_type_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "offset": None}, tool_context)
        assert result.is_error
        assert "offset must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_offset_list_type_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "offset": [1, 2]}, tool_context)
        assert result.is_error
        assert "offset must be an integer" in result.error

    # --- Limit edge cases ---

    @pytest.mark.asyncio
    async def test_limit_none_type_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "limit": None}, tool_context)
        assert result.is_error
        assert "limit must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_limit_list_type_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("a\nb\n")
        result = await tool.execute({"file_path": "test.py", "limit": [5]}, tool_context)
        assert result.is_error
        assert "limit must be an integer" in result.error

    # --- Large file edge cases ---

    @pytest.mark.asyncio
    async def test_file_exactly_at_max_size(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "exact.bin"
        content = "x" * int(MAX_FILE_SIZE_KB * 1024)
        test_file.write_text(content)
        result = await tool.execute({"file_path": "exact.bin"}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_file_just_below_max_size(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "below.bin"
        content = "x" * int((MAX_FILE_SIZE_KB - 1) * 1024)
        test_file.write_text(content)
        result = await tool.execute({"file_path": "below.bin"}, tool_context)
        assert not result.is_error

    # --- Binary file detection edge cases ---

    @pytest.mark.asyncio
    async def test_binary_file_null_bytes(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "nulls.bin"
        test_file.write_bytes(b"\xff\xfe\xfd\xfc\xfb\xfa")
        result = await tool.execute({"file_path": "nulls.bin"}, tool_context)
        assert result.is_error
        assert "binary file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_binary_file_mixed_ascii_and_binary(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "mixed.bin"
        test_file.write_bytes(b"Hello\x80\x81\x82World")
        result = await tool.execute({"file_path": "mixed.bin"}, tool_context)
        assert result.is_error
        assert "binary file" in result.error.lower()

    # --- Notebook: display_data with string text/plain ---

    @pytest.mark.asyncio
    async def test_notebook_display_data_string_text(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "str_display.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import matplotlib"],
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/plain": "direct string output",
                            },
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "str_display.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: display_data]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_execute_result_string_text(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "str_result.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["42"],
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": "42 (string)",
                            },
                            "execution_count": 1,
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "str_result.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: execute_result]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_source_is_string(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "string_source.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "print('single string source')",
                    "outputs": [],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "string_source.ipynb"}, tool_context)
        assert not result.is_error
        assert "print('single string source')" in result.output

    @pytest.mark.asyncio
    async def test_notebook_stream_text_as_string(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "stream_str.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hi')"],
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": "direct stream text",
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "stream_str.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Output: stream]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_no_cells_key(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "no_cells.ipynb"
        nb_path.write_text(json.dumps({"metadata": {}, "nbformat": 4}))
        result = await tool.execute({"file_path": "no_cells.ipynb"}, tool_context)
        assert not result.is_error
        assert "empty notebook" in result.output.lower()

    @pytest.mark.asyncio
    async def test_notebook_empty_source_list(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "empty_source.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [],
                    "outputs": [],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "empty_source.ipynb"}, tool_context)
        assert not result.is_error
        assert "[Cell 0: code]" in result.output

    @pytest.mark.asyncio
    async def test_notebook_multiple_cells_with_outputs(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        nb_path = tool_context.cwd / "multi.ipynb"
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Title"],
                    "outputs": [],
                },
                {
                    "cell_type": "code",
                    "source": ["x = 1"],
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {"text/plain": ["1"]},
                            "execution_count": 1,
                        }
                    ],
                },
                {
                    "cell_type": "code",
                    "source": ["x/0"],
                    "outputs": [
                        {
                            "output_type": "error",
                            "ename": "ZeroDivisionError",
                            "evalue": "division by zero",
                        }
                    ],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb))
        result = await tool.execute({"file_path": "multi.ipynb"}, tool_context)
        assert not result.is_error
        assert "Cell 0: markdown" in result.output
        assert "Cell 1: code" in result.output
        assert "[Output: execute_result]" in result.output
        assert "Cell 2: code" in result.output
        assert "[Output: error]" in result.output

    # --- Absolute and relative path resolution ---

    @pytest.mark.asyncio
    async def test_absolute_path_within_cwd(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "abs_test.py"
        test_file.write_text("content\n")
        abs_path = str(test_file)
        result = await tool.execute({"file_path": abs_path}, tool_context)
        assert not result.is_error
        assert "content" in result.output
