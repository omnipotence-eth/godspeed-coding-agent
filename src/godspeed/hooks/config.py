"""Hook configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from godspeed.hooks import HookEvent


class HookDefinition(BaseModel):
    """A single hook that runs a shell command at a lifecycle event.

    Events (27 total):
        Session: session_start, session_end, turn_end
        Permission: pre_permission_check, post_permission_check,
            permission_denied, permission_granted
        Tool: pre_tool_call, post_tool_call, tool_error, tool_retry
        File: pre_file_write, post_file_write, pre_file_read
        Context: pre_compaction, post_compaction, context_threshold_75,
            context_threshold_50, context_threshold_25
        Subagent: pre_subagent_spawn, post_subagent_complete, subagent_error
        Evolution: pre_evolution_run, post_evolution_run
        Safety: secret_detected, dangerous_command, stuck_loop_detected,
            budget_exceeded
        Audit: audit_write
        Graph: post_graph_build
        Workflow: workflow_phase_complete, workflow_complete, workflow_rejected

    Template variables in ``command``:
        {tool_name}: Name of the tool being called (tool events only).
        {session_id}: Current session ID.
        {cwd}: Working directory.
        {project_dir}: Project directory (same as cwd).
        {gs_event}: Event name (GS_EVENT).
        {gs_path}: File path (GS_PATH).
        {gs_cost_usd}: Running cost in USD (GS_COST_USD).
        {gs_timestamp}: ISO 8601 timestamp (GS_TIMESTAMP).
    """

    event: HookEvent
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
