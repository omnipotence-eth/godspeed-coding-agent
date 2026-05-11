"""Tests for tool set definitions and get_allowed_tool_names."""

from __future__ import annotations

import pytest

from godspeed.tools.tool_sets import (
    TOOL_SET_FULL,
    TOOL_SET_LOCAL,
    TOOL_SET_SWEBENCH,
    TOOL_SET_WEB,
    get_allowed_tool_names,
)


class TestGetAllowedToolNames:
    def test_local_set_returns_frozenset(self) -> None:
        result = get_allowed_tool_names(TOOL_SET_LOCAL)
        assert isinstance(result, frozenset)
        assert len(result) > 0
        assert "file_read" in result
        assert "web_search" not in result

    def test_web_set_includes_local_and_web_tools(self) -> None:
        result = get_allowed_tool_names(TOOL_SET_WEB)
        assert isinstance(result, frozenset)
        assert "file_read" in result
        assert "web_search" in result
        assert "web_fetch" in result
        assert "github" in result

    def test_full_set_returns_none(self) -> None:
        result = get_allowed_tool_names(TOOL_SET_FULL)
        assert result is None

    def test_swebench_set_returns_frozenset(self) -> None:
        result = get_allowed_tool_names(TOOL_SET_SWEBENCH)
        assert isinstance(result, frozenset)
        assert len(result) > 0
        assert "file_read" in result
        assert "grep_search" in result
        assert "web_search" not in result

    def test_invalid_set_name_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool set"):
            get_allowed_tool_names("invalid_set")

    def test_invalid_empty_string_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool set"):
            get_allowed_tool_names("")
