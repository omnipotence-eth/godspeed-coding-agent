"""Hook system — run shell commands at agent lifecycle events."""

from __future__ import annotations

from enum import StrEnum


class HookEvent(StrEnum):
    """Lifecycle events that hooks can subscribe to.

    Events are grouped by category:
    - Session: session lifecycle
    - Permission: permission engine decisions
    - Tool: tool execution lifecycle
    - File: file operation lifecycle
    - Context: context management and compaction
    - Subagent: sub-agent spawn and completion
    - Evolution: harness evolution lifecycle
    - Safety: security and safety events
    - Audit: audit trail events
    - Workflow: orchestrated workflow phases
    """

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_END = "turn_end"

    # Permission lifecycle
    PRE_PERMISSION_CHECK = "pre_permission_check"
    POST_PERMISSION_CHECK = "post_permission_check"
    PERMISSION_DENIED = "permission_denied"
    PERMISSION_GRANTED = "permission_granted"

    # Tool lifecycle
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    TOOL_ERROR = "tool_error"
    TOOL_RETRY = "tool_retry"

    # File operation lifecycle
    PRE_FILE_WRITE = "pre_file_write"
    POST_FILE_WRITE = "post_file_write"
    PRE_FILE_READ = "pre_file_read"

    # Context management
    PRE_COMPACTION = "pre_compaction"
    POST_COMPACTION = "post_compaction"
    CONTEXT_THRESHOLD_75 = "context_threshold_75"
    CONTEXT_THRESHOLD_50 = "context_threshold_50"
    CONTEXT_THRESHOLD_25 = "context_threshold_25"

    # Subagent lifecycle
    PRE_SUBAGENT_SPAWN = "pre_subagent_spawn"
    POST_SUBAGENT_COMPLETE = "post_subagent_complete"
    SUBAGENT_ERROR = "subagent_error"

    # Evolution and training
    PRE_EVOLUTION_RUN = "pre_evolution_run"
    POST_EVOLUTION_RUN = "post_evolution_run"

    # Safety events
    SECRET_DETECTED = "secret_detected"  # noqa: S105
    DANGEROUS_COMMAND = "dangerous_command"
    STUCK_LOOP_DETECTED = "stuck_loop_detected"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Audit
    AUDIT_WRITE = "audit_write"

    # Graph (GCG)
    POST_GRAPH_BUILD = "post_graph_build"

    # Workflow
    WORKFLOW_PHASE_COMPLETE = "workflow_phase_complete"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_REJECTED = "workflow_rejected"
