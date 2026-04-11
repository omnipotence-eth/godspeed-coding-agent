"""Tool protocol, base types, and risk classification."""

from __future__ import annotations

import abc
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    """4-tier risk classification for tools.

    Determines default permission behavior:
    - READ_ONLY: auto-allowed, no prompt
    - LOW: ask once, then session-scoped allow
    - HIGH: ask every time (unless pattern-matched)
    - DESTRUCTIVE: blocked by default, requires explicit enable
    """

    READ_ONLY = "read_only"
    LOW = "low"
    HIGH = "high"
    DESTRUCTIVE = "destructive"


class ToolResult(BaseModel):
    """Result returned from a tool execution."""

    output: str = ""
    error: str | None = None
    is_error: bool = False

    @classmethod
    def success(cls, output: str) -> ToolResult:
        return cls(output=output)

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(output="", error=error, is_error=True)


class ToolCall(BaseModel):
    """A request to execute a tool with specific arguments."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""

    def format_for_permission(self) -> str:
        """Format as 'ToolName(arg_summary)' for permission rule matching.

        Prioritizes security-relevant keys over arbitrary first-string-value.
        For shell-like tools, always uses the 'command' key so that a benign
        'description' field cannot shadow the actual command being executed.
        """
        if isinstance(self.arguments, str):
            return f"{self.tool_name}({self.arguments})"
        if isinstance(self.arguments, dict):
            if not self.arguments:
                return f"{self.tool_name}()"
            # Prefer 'command' key for shell-like tools
            if "command" in self.arguments:
                return f"{self.tool_name}({self.arguments['command']})"
            # Then try 'file_path' for file tools
            if "file_path" in self.arguments:
                return f"{self.tool_name}({self.arguments['file_path']})"
            # Then 'action' for git-like tools
            if "action" in self.arguments:
                action = self.arguments["action"]
                return f"{self.tool_name}({action})"
            # Fall back to first string value
            for value in self.arguments.values():
                if isinstance(value, str):
                    return f"{self.tool_name}({value})"
        return f"{self.tool_name}(*)"


class ToolContext(BaseModel):
    """Execution context passed to every tool."""

    cwd: Path
    session_id: str
    # Use Any to avoid circular imports — set at runtime by the agent loop
    permissions: Any = None
    audit: Any = None

    model_config = {"arbitrary_types_allowed": True}


class Tool(abc.ABC):
    """Abstract base class for all Godspeed tools.

    Every tool declares its name, description, risk level, and parameter schema.
    The same protocol is used for built-in tools, MCP tools, and future
    computer-use tools.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM system prompt."""
        ...

    @property
    @abc.abstractmethod
    def risk_level(self) -> RiskLevel:
        """Risk classification determining default permission behavior."""
        ...

    @abc.abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the tool's parameters.

        This schema is sent to the LLM as part of the tool definition.
        Must follow the JSON Schema spec used by LLM tool-calling APIs.
        """
        ...

    @abc.abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with validated arguments.

        Args:
            arguments: Validated arguments matching get_schema().
            context: Execution context (cwd, session, permissions, audit).

        Returns:
            ToolResult with output or error.
        """
        ...
