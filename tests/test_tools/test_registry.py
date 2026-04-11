"""Tests for tool registry."""

from __future__ import annotations

import pytest

from godspeed.tools.base import ToolCall, ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


class TestToolRegistry:
    """Test tool registration and dispatch."""

    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = MockTool(name="test_tool")
        registry.register(tool)
        assert registry.get("test_tool") is tool

    def test_get_unknown_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockTool(name="dup"))

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="a"))
        registry.register(MockTool(name="b"))
        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"a", "b"}

    def test_get_schemas(self) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", description="Read a file"))
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "file_read"
        assert schemas[0]["function"]["description"] == "Read a file"
        assert "properties" in schemas[0]["function"]["parameters"]

    @pytest.mark.asyncio
    async def test_dispatch_success(self, tool_context) -> None:
        registry = ToolRegistry()
        tool = MockTool(name="read", result=ToolResult.success("file contents"))
        registry.register(tool)
        call = ToolCall(tool_name="read", arguments={"path": "test.py"})
        result = await registry.dispatch(call, tool_context)
        assert result.output == "file contents"
        assert tool.last_arguments == {"path": "test.py"}

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, tool_context) -> None:
        registry = ToolRegistry()
        call = ToolCall(tool_name="unknown", arguments={})
        result = await registry.dispatch(call, tool_context)
        assert result.is_error is True
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_catches_exceptions(self, tool_context) -> None:
        registry = ToolRegistry()

        class FailingTool(MockTool):
            async def execute(self, arguments, context):
                raise RuntimeError("boom")

        registry.register(FailingTool(name="fail"))
        call = ToolCall(tool_name="fail", arguments={})
        result = await registry.dispatch(call, tool_context)
        assert result.is_error is True
        assert "boom" in result.error
