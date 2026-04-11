"""Hook configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HookDefinition(BaseModel):
    """A single hook that runs a shell command at a lifecycle event.

    Events:
        pre_tool_call: Before a tool executes. Non-zero exit blocks the call.
        post_tool_call: After a tool executes.
        pre_session: When a session starts.
        post_session: When a session ends.

    Template variables in ``command``:
        {tool_name}: Name of the tool being called (tool events only).
        {session_id}: Current session ID.
        {cwd}: Working directory.
        {project_dir}: Project directory (same as cwd).
    """

    event: Literal["pre_tool_call", "post_tool_call", "pre_session", "post_session"]
    command: str
    tools: list[str] | None = Field(
        default=None,
        description="Tool names to match. None = all tools.",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Max seconds for hook execution.",
    )
