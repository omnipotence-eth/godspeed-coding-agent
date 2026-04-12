"""Typed event protocol for the agent loop.

Events replace callbacks for composable consumption. The callback-based
agent_loop() remains for backward compat — it wraps agent_loop_events().
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class ThinkingEvent:
    """Extended thinking content from the LLM."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class TextChunkEvent:
    """Streaming text chunk from the LLM."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class AssistantTextEvent:
    """Complete assistant text response (model decided to stop)."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A tool is about to be executed."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Result from a tool execution."""

    tool_name: str
    output: str
    is_error: bool = False
    call_id: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class PermissionDeniedEvent:
    """A tool call was denied by the permission engine."""

    tool_name: str
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelBatchStartEvent:
    """Multiple tools starting in parallel."""

    tools: list[tuple[str, dict[str, Any]]]


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelBatchCompleteEvent:
    """Parallel tool batch completed."""

    results: list[tuple[str, str, bool]]  # (tool_name, output, is_error)


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetExceededEvent:
    """Cost budget was exceeded."""

    spent: float
    limit: float


@dataclasses.dataclass(frozen=True, slots=True)
class ErrorEvent:
    """An error occurred in the agent loop."""

    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseChangeEvent:
    """Architect mode phase change."""

    phase: str  # "plan" or "execute"
    model: str


# Union type for all events
AgentEvent = (
    ThinkingEvent
    | TextChunkEvent
    | AssistantTextEvent
    | ToolCallEvent
    | ToolResultEvent
    | PermissionDeniedEvent
    | ParallelBatchStartEvent
    | ParallelBatchCompleteEvent
    | BudgetExceededEvent
    | ErrorEvent
    | PhaseChangeEvent
)
