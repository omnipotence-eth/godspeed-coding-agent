"""Tests for the traceback analyzer tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.traceback_analyzer import TracebackAnalyzerTool


@pytest.fixture
def ctx_with_dir(tmp_path: Path) -> ToolContext:
    py_file = tmp_path / "test.py"
    py_file.write_text("def foo():\n    return bar()\n\ndef bar():\n    return 1/0\n")
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.mark.asyncio
async def test_parse_python_traceback(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "Traceback (most recent call last):\n"
        '  File "test.py", line 2, in foo\n'
        "    return bar()\n"
        '  File "test.py", line 5, in bar\n'
        "    return 1/0\n"
        "ZeroDivisionError: division by zero\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_with_dir)
    assert not result.is_error
    assert "ZeroDivisionError" in result.output
    assert "test.py" in result.output


@pytest.mark.asyncio
async def test_empty_traceback(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    result = await tool.execute({"traceback": ""}, ctx_with_dir)
    assert result.is_error


@pytest.mark.asyncio
async def test_missing_parameter(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    result = await tool.execute({}, ctx_with_dir)
    assert result.is_error


class TestTracebackAnalyzerMetadata:
    def test_name(self) -> None:
        tool = TracebackAnalyzerTool()
        assert tool.name == "traceback_analyzer"

    def test_risk_level(self) -> None:
        tool = TracebackAnalyzerTool()
        assert tool.risk_level.value == "read_only"

    def test_schema(self) -> None:
        tool = TracebackAnalyzerTool()
        schema = tool.get_schema()
        assert "traceback" in schema["properties"]
