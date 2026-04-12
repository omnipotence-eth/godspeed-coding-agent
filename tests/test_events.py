"""Tests for agent loop event types."""

from __future__ import annotations

import dataclasses
from typing import get_args

import pytest

from godspeed.agent.events import (
    AgentEvent,
    AssistantTextEvent,
    BudgetExceededEvent,
    ErrorEvent,
    ParallelBatchCompleteEvent,
    ParallelBatchStartEvent,
    PermissionDeniedEvent,
    PhaseChangeEvent,
    TextChunkEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)

ALL_EVENT_CLASSES = [
    ThinkingEvent,
    TextChunkEvent,
    AssistantTextEvent,
    ToolCallEvent,
    ToolResultEvent,
    PermissionDeniedEvent,
    ParallelBatchStartEvent,
    ParallelBatchCompleteEvent,
    BudgetExceededEvent,
    ErrorEvent,
    PhaseChangeEvent,
]


class TestEventInstantiation:
    """Each event dataclass can be instantiated with correct fields."""

    def test_thinking_event(self) -> None:
        evt = ThinkingEvent(text="reasoning about the problem")
        assert evt.text == "reasoning about the problem"

    def test_text_chunk_event(self) -> None:
        evt = TextChunkEvent(text="Hello")
        assert evt.text == "Hello"

    def test_assistant_text_event(self) -> None:
        evt = AssistantTextEvent(text="Full response here.")
        assert evt.text == "Full response here."

    def test_tool_call_event(self) -> None:
        evt = ToolCallEvent(
            tool_name="read_file",
            arguments={"path": "src/foo.py"},
            call_id="call_123",
        )
        assert evt.tool_name == "read_file"
        assert evt.arguments == {"path": "src/foo.py"}
        assert evt.call_id == "call_123"

    def test_tool_result_event(self) -> None:
        evt = ToolResultEvent(
            tool_name="read_file",
            output="file contents",
            is_error=False,
            call_id="call_123",
        )
        assert evt.tool_name == "read_file"
        assert evt.output == "file contents"
        assert evt.is_error is False
        assert evt.call_id == "call_123"

    def test_tool_result_event_error(self) -> None:
        evt = ToolResultEvent(
            tool_name="bash",
            output="command not found",
            is_error=True,
        )
        assert evt.is_error is True

    def test_permission_denied_event(self) -> None:
        evt = PermissionDeniedEvent(tool_name="bash", reason="not in allowlist")
        assert evt.tool_name == "bash"
        assert evt.reason == "not in allowlist"

    def test_parallel_batch_start_event(self) -> None:
        tools = [("read_file", {"path": "/a"}), ("read_file", {"path": "/b"})]
        evt = ParallelBatchStartEvent(tools=tools)
        assert len(evt.tools) == 2
        assert evt.tools[0] == ("read_file", {"path": "/a"})

    def test_parallel_batch_complete_event(self) -> None:
        results = [("read_file", "contents a", False), ("read_file", "contents b", False)]
        evt = ParallelBatchCompleteEvent(results=results)
        assert len(evt.results) == 2
        assert evt.results[1][2] is False

    def test_budget_exceeded_event(self) -> None:
        evt = BudgetExceededEvent(spent=5.50, limit=5.00)
        assert evt.spent == 5.50
        assert evt.limit == 5.00

    def test_error_event(self) -> None:
        evt = ErrorEvent(message="something broke")
        assert evt.message == "something broke"

    def test_phase_change_event(self) -> None:
        evt = PhaseChangeEvent(phase="plan", model="o3")
        assert evt.phase == "plan"
        assert evt.model == "o3"


class TestFrozen:
    """Events are frozen (immutable)."""

    @pytest.mark.parametrize("cls", ALL_EVENT_CLASSES, ids=lambda c: c.__name__)
    def test_frozen(self, cls: type) -> None:
        assert dataclasses.fields(cls)  # has fields
        evt = _make_instance(cls)
        first_field = dataclasses.fields(cls)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(evt, first_field, "mutated")


class TestSlots:
    """Events have slots (memory efficient)."""

    @pytest.mark.parametrize("cls", ALL_EVENT_CLASSES, ids=lambda c: c.__name__)
    def test_slots(self, cls: type) -> None:
        assert hasattr(cls, "__slots__"), f"{cls.__name__} missing __slots__"
        evt = _make_instance(cls)
        assert not hasattr(evt, "__dict__"), f"{cls.__name__} should not have __dict__"


class TestAgentEventUnion:
    """AgentEvent union type accepts all event types."""

    def test_all_classes_in_union(self) -> None:
        union_members = set(get_args(AgentEvent))
        for cls in ALL_EVENT_CLASSES:
            assert cls in union_members, f"{cls.__name__} missing from AgentEvent union"

    def test_union_has_no_extra_types(self) -> None:
        union_members = set(get_args(AgentEvent))
        expected = set(ALL_EVENT_CLASSES)
        assert union_members == expected, f"Extra types in union: {union_members - expected}"

    def test_isinstance_check_via_union(self) -> None:
        evt = ThinkingEvent(text="hi")
        assert isinstance(evt, get_args(AgentEvent))


class TestDefaults:
    """Default values work correctly."""

    def test_tool_call_event_default_call_id(self) -> None:
        evt = ToolCallEvent(tool_name="bash", arguments={"cmd": "ls"})
        assert evt.call_id == ""

    def test_tool_result_event_defaults(self) -> None:
        evt = ToolResultEvent(tool_name="bash", output="ok")
        assert evt.is_error is False
        assert evt.call_id == ""


class TestRepr:
    """Events have reasonable repr/str."""

    def test_thinking_event_repr(self) -> None:
        evt = ThinkingEvent(text="deep thought")
        r = repr(evt)
        assert "ThinkingEvent" in r
        assert "deep thought" in r

    def test_tool_call_event_repr(self) -> None:
        evt = ToolCallEvent(tool_name="bash", arguments={"cmd": "ls"}, call_id="c1")
        r = repr(evt)
        assert "ToolCallEvent" in r
        assert "bash" in r
        assert "c1" in r

    def test_budget_exceeded_repr(self) -> None:
        evt = BudgetExceededEvent(spent=3.14, limit=2.0)
        r = repr(evt)
        assert "3.14" in r
        assert "2.0" in r

    @pytest.mark.parametrize("cls", ALL_EVENT_CLASSES, ids=lambda c: c.__name__)
    def test_str_does_not_raise(self, cls: type) -> None:
        evt = _make_instance(cls)
        result = str(evt)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FACTORY: dict[type, tuple[tuple, dict]] = {
    ThinkingEvent: (("text",), {}),
    TextChunkEvent: (("chunk",), {}),
    AssistantTextEvent: (("full text",), {}),
    ToolCallEvent: ((), {"tool_name": "t", "arguments": {}}),
    ToolResultEvent: ((), {"tool_name": "t", "output": "o"}),
    PermissionDeniedEvent: ((), {"tool_name": "t", "reason": "r"}),
    ParallelBatchStartEvent: ((), {"tools": []}),
    ParallelBatchCompleteEvent: ((), {"results": []}),
    BudgetExceededEvent: ((), {"spent": 1.0, "limit": 2.0}),
    ErrorEvent: (("err",), {}),
    PhaseChangeEvent: ((), {"phase": "plan", "model": "m"}),
}


def _make_instance(cls: type) -> object:
    args, kwargs = _FACTORY[cls]
    return cls(*args, **kwargs)
