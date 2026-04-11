"""Tests for hook configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from godspeed.hooks.config import HookDefinition


class TestHookDefinition:
    """Test HookDefinition model."""

    def test_valid_pre_tool_call(self) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command="echo {tool_name}",
            tools=["shell", "file_write"],
            timeout=10,
        )
        assert hook.event == "pre_tool_call"
        assert hook.tools == ["shell", "file_write"]
        assert hook.timeout == 10

    def test_valid_post_session(self) -> None:
        hook = HookDefinition(
            event="post_session",
            command="./scripts/cleanup.sh",
        )
        assert hook.event == "post_session"
        assert hook.tools is None
        assert hook.timeout == 30

    def test_all_event_types(self) -> None:
        for event in ["pre_tool_call", "post_tool_call", "pre_session", "post_session"]:
            hook = HookDefinition(event=event, command="echo test")
            assert hook.event == event

    def test_invalid_event_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HookDefinition(event="invalid_event", command="echo test")

    def test_default_timeout(self) -> None:
        hook = HookDefinition(event="pre_session", command="echo test")
        assert hook.timeout == 30

    def test_default_tools_is_none(self) -> None:
        hook = HookDefinition(event="pre_tool_call", command="echo test")
        assert hook.tools is None

    def test_timeout_bounds(self) -> None:
        with pytest.raises(ValidationError):
            HookDefinition(event="pre_session", command="echo", timeout=0)
        with pytest.raises(ValidationError):
            HookDefinition(event="pre_session", command="echo", timeout=301)

    def test_template_variables_in_command(self) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command="echo {tool_name} {session_id} {cwd}",
        )
        assert "{tool_name}" in hook.command
        assert "{session_id}" in hook.command
