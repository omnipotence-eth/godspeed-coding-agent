"""Core agent loop — the heart of Godspeed.

Hand-rolled loop following the pattern proven by mini-swe-agent (74%+ SWE-bench)
and Claude Code. The model decides when to stop. No framework overhead.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.agent.result import AgentCancelledError, AgentMetrics, ExitReason
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.llm.router import classify_task_type
from godspeed.tools.base import ToolCall, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["MAX_ITERATIONS", "MAX_RETRIES", "MAX_SPECULATIVE_CACHE_SIZE", "agent_loop"]

MAX_ITERATIONS = 50
MAX_RETRIES = 3
STUCK_LOOP_THRESHOLD = 3
AUTO_STASH_THRESHOLD = 3
MUST_FIX_CAP = 3
MAX_TOOL_OUTPUT_CHARS = 50000  # Prevent single tool call from blowing context window
MAX_SPECULATIVE_CACHE_SIZE = 10  # Max concurrent speculative tasks per iteration
VERIFIABLE_EXTENSIONS = (
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
)

# Callback type aliases for clarity
OnAssistantText = Callable[[str], None]
OnToolCall = Callable[[str, dict[str, Any]], None]
OnToolResult = Callable[[str, ToolResult], None]
OnPermissionDenied = Callable[[str, str], None]
OnChunk = Callable[[str], None]
OnParallelStart = Callable[[list[tuple[str, dict[str, Any]]]], None]
OnParallelComplete = Callable[[list[tuple[str, str, bool]]], None]
OnThinking = Callable[[str], None]


@dataclass
class _LoopState:
    """Mutable state and loop config shared across helper functions."""

    # --- Mutable state ---
    consecutive_writes: int = 0
    consecutive_successful_edits: int = 0
    recent_change_descriptions: list[str] = field(default_factory=list)
    auto_stashed: bool = False
    must_fix_injections: int = 0
    recent_error_hashes: list[str] = field(default_factory=list)
    speculative_cache: dict[str, asyncio.Task[ToolResult]] = field(default_factory=dict)

    # --- Loop config (set once at loop start) ---
    auto_fix_retries: int = 3
    auto_commit: bool = False
    auto_commit_threshold: int = 5
    stuck_threshold: int = 3
    stash_threshold: int = 3
    must_fix_cap: int = 3


async def agent_loop(
    user_input: str,
    conversation: Conversation,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    on_assistant_text: OnAssistantText | None = None,
    on_tool_call: OnToolCall | None = None,
    on_tool_result: OnToolResult | None = None,
    on_permission_denied: OnPermissionDenied | None = None,
    on_assistant_chunk: OnChunk | None = None,
    max_iterations: int | None = None,
    pause_event: asyncio.Event | None = None,
    cancel_event: asyncio.Event | None = None,
    hook_executor: Any | None = None,
    parallel_tool_calls: bool = True,
    skip_user_message: bool = False,
    auto_fix_retries: int = 3,
    auto_commit: bool = False,
    auto_commit_threshold: int = 5,
    max_retries: int | None = None,
    stuck_loop_threshold: int | None = None,
    auto_stash_threshold: int | None = None,
    must_fix_cap: int | None = None,
    on_parallel_start: OnParallelStart | None = None,
    on_parallel_complete: OnParallelComplete | None = None,
    on_thinking: OnThinking | None = None,
    metrics: AgentMetrics | None = None,
) -> str:
    """Run the agent loop until the model stops calling tools.

    Flow:
    1. Add user input to conversation
    2. Send conversation + tool schemas to LLM
    3. If response has tool_calls: check permissions, execute, record results
    4. If response is text-only: return it (model decided to stop)
    5. On malformed response: retry up to MAX_RETRIES

    Args:
        user_input: The user's message.
        conversation: Conversation history manager.
        llm_client: LLM client for API calls.
        tool_registry: Registry of available tools.
        tool_context: Execution context for tools.
        on_assistant_text: Callback(text) for complete assistant output.
        on_tool_call: Callback(tool_name, args) before tool execution.
        on_tool_result: Callback(tool_name, result) after tool execution.
        on_permission_denied: Callback(tool_name, reason) when permission denied.
        on_assistant_chunk: Callback(text) for streaming chunks. When provided,
            uses streaming LLM calls instead of batch calls.
        max_iterations: Override the default iteration limit (MAX_ITERATIONS).
        max_retries: Override MAX_RETRIES for malformed tool calls.
        stuck_loop_threshold: Override effective_stuck_threshold for stuck detection.
        auto_stash_threshold: Override effective_stash_threshold for auto-stash.
        must_fix_cap: Override effective_must_fix_cap for must-fix injections.
        pause_event: Optional asyncio.Event for pause/resume. When cleared,
            the loop waits at the top of each iteration until set again.
        cancel_event: Optional asyncio.Event for mid-turn cancellation.
            When set, the loop raises AgentCancelledError at the next safe
            checkpoint — between streaming chunks, before an LLM call,
            or before dispatching tools — so the user can interrupt a
            long-running turn immediately instead of waiting for the
            iteration boundary. The TUI binds Ctrl+C to set this event.
        parallel_tool_calls: Execute multiple tool calls concurrently when True
            (default). Falls back to sequential when False or for single calls.

    Returns:
        The final assistant text response.
    """
    iteration_limit = max_iterations if max_iterations is not None else MAX_ITERATIONS
    effective_max_retries = max_retries if max_retries is not None else MAX_RETRIES

    if not skip_user_message and user_input:
        conversation.add_user_message(user_input)
    tool_schemas = tool_registry.get_schemas()

    retries = 0
    final_text = ""
    state = _LoopState(
        auto_fix_retries=auto_fix_retries,
        auto_commit=auto_commit,
        auto_commit_threshold=auto_commit_threshold,
        stuck_threshold=(
            stuck_loop_threshold if stuck_loop_threshold is not None else STUCK_LOOP_THRESHOLD
        ),
        stash_threshold=(
            auto_stash_threshold if auto_stash_threshold is not None else AUTO_STASH_THRESHOLD
        ),
        must_fix_cap=must_fix_cap if must_fix_cap is not None else MUST_FIX_CAP,
    )

    for iteration in range(iteration_limit):
        # Cancel check: before pause check, so a cancel delivered during a
        # pause doesn't strand the loop. Raises AgentCancelledError; caller unwinds.
        _check_cancel(cancel_event)

        # Clear stale speculative tasks from previous iteration
        for task in state.speculative_cache.values():
            task.cancel()
        state.speculative_cache.clear()

        # Pause/resume: if pause_event exists and is cleared, wait for it
        if pause_event is not None and not pause_event.is_set():
            logger.info("Agent loop paused at iteration=%d", iteration)
            await pause_event.wait()
            logger.info("Agent loop resumed at iteration=%d", iteration)

        # Cancel check #2: may have been set while we were paused.
        _check_cancel(cancel_event)

        logger.debug("Agent loop iteration=%d tokens=%d", iteration, conversation.token_count)

        # Check if we need to compact
        if conversation.is_near_limit:
            await _compact_conversation(conversation, llm_client)

        # Task-aware routing: classify the upcoming call from conversation
        # state. Cheap heuristic (no extra LLM call); resolves to one of
        # plan/edit/read/shell. The router translates that to a model
        # via settings.routing (or the cheap_model/strong_model shortcuts).
        task_type = classify_task_type(conversation.messages)

        # Call LLM (streaming or batch)
        try:
            if on_assistant_chunk is not None:
                response = await _streaming_call(
                    llm_client,
                    conversation.messages,
                    tool_schemas if tool_schemas else None,
                    on_assistant_chunk,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    speculative_cache=state.speculative_cache,
                    cancel_event=cancel_event,
                    task_type=task_type,
                )
            else:
                response = await llm_client.chat(
                    messages=conversation.messages,
                    tools=tool_schemas if tool_schemas else None,
                    task_type=task_type,
                )
        except AgentCancelledError:
            # Finalize with INTERRUPTED and unwind — don't wrap in LLM_ERROR.
            logger.info("Agent loop cancelled mid-turn at iteration=%d", iteration)
            if metrics is not None:
                metrics.iterations_used = iteration
                metrics.finalize(ExitReason.INTERRUPTED)
            raise
        except Exception as exc:
            # Import here to avoid circular import at module level
            from godspeed.llm.client import BudgetExceededError

            if isinstance(exc, BudgetExceededError):
                msg = (
                    f"Budget exceeded (${exc.spent:.4f} / ${exc.limit:.2f} limit). "
                    "Use /budget to increase the limit."
                )
                logger.warning("Budget exceeded spent=%.4f limit=%.2f", exc.spent, exc.limit)
                if metrics is not None:
                    metrics.iterations_used = iteration
                    metrics.finalize(ExitReason.BUDGET_EXCEEDED)
                return msg
            logger.error("LLM call failed error=%s", exc, exc_info=True)
            if metrics is not None:
                metrics.iterations_used = iteration
                metrics.finalize(ExitReason.LLM_ERROR)
            return f"Error: LLM call failed — {exc}"

        # Display thinking blocks (extended thinking for Anthropic models)
        if response.thinking and on_thinking:
            on_thinking(response.thinking)

        # Handle text response (model decided to stop)
        if not response.has_tool_calls:
            final_text = response.content
            if final_text:
                conversation.add_assistant_message(content=final_text)
                # Skip Markdown re-render if we already streamed the text
                if on_assistant_text and on_assistant_chunk is None:
                    on_assistant_text(final_text)
            if metrics is not None:
                metrics.iterations_used = iteration + 1
                metrics.finalize(ExitReason.STOPPED)
            return final_text

        # Handle tool calls
        conversation.add_assistant_message(
            content=response.content,
            tool_calls=response.tool_calls,
        )

        if response.content and on_assistant_text:
            on_assistant_text(response.content)

        # --- Phase 1: Parse and pre-flight all tool calls ---
        parsed_calls: list[tuple[dict[str, Any], ToolCall | None]] = []
        for raw_tc in response.tool_calls:
            parsed_calls.append((raw_tc, _parse_tool_call(raw_tc)))

        # --- Phase 2: Permission checks + pre-tool hooks (sequential) ---
        # These are fast and order-sensitive, so always sequential.
        permitted: list[ToolCall] = []
        for raw_tc, tool_call in parsed_calls:
            _check_cancel(cancel_event)
            if tool_call is None:
                retries += 1
                if retries > effective_max_retries:
                    if metrics is not None:
                        metrics.iterations_used = iteration + 1
                        metrics.finalize(ExitReason.TOOL_ERROR)
                    return "Error: Too many malformed tool calls from the model."
                conversation.add_tool_result(
                    tool_call_id=raw_tc.get("id", ""),
                    content=(
                        "Error: Malformed tool call. Please try again with valid JSON arguments."
                    ),
                )
                continue

            retries = 0  # Reset on valid tool call

            # Permission check
            if tool_context.permissions is not None:
                if inspect.iscoroutinefunction(tool_context.permissions.evaluate):
                    decision = await tool_context.permissions.evaluate(tool_call)
                else:
                    decision = tool_context.permissions.evaluate(tool_call)
                if decision == "deny":
                    reason = f"Permission denied for {tool_call.format_for_permission()}"
                    logger.info("Permission denied tool=%s", tool_call.tool_name)
                    if on_permission_denied:
                        on_permission_denied(tool_call.tool_name, reason)
                    conversation.add_tool_result(
                        tool_call_id=tool_call.call_id,
                        content=(
                            f"DENIED: {reason}. "
                            "This tool call was blocked by the permission engine."
                        ),
                    )
                    continue

            if on_tool_call:
                on_tool_call(tool_call.tool_name, tool_call.arguments)

            # Pre-tool hook: can block execution
            if hook_executor is not None:
                hook_ok = await asyncio.get_running_loop().run_in_executor(
                    None, hook_executor.run_pre_tool, tool_call.tool_name
                )
                if not hook_ok:
                    logger.info("Pre-tool hook blocked tool=%s", tool_call.tool_name)
                    conversation.add_tool_result(
                        tool_call_id=tool_call.call_id,
                        content="BLOCKED: Pre-tool hook returned non-zero exit.",
                    )
                    continue

            permitted.append(tool_call)

        if not permitted:
            # All calls were malformed, denied, or blocked — continue to next LLM turn
            continue

        # --- Phase 3: Execute tools (parallel or sequential) ---
        if parallel_tool_calls and len(permitted) > 1:
            await _dispatch_parallel(
                permitted,
                tool_registry,
                tool_context,
                hook_executor,
                on_tool_result,
                on_parallel_start,
                on_parallel_complete,
                metrics,
                state,
                conversation,
                cancel_event,
                llm_client,
            )
        else:
            await _dispatch_sequential(
                permitted,
                tool_registry,
                tool_context,
                hook_executor,
                on_tool_result,
                metrics,
                state,
                conversation,
                cancel_event,
                llm_client,
            )

    if metrics is not None:
        metrics.iterations_used = iteration_limit
        metrics.finalize(ExitReason.MAX_ITERATIONS)
    return "Error: Reached maximum iterations. The task may be too complex for a single turn."


async def _dispatch_parallel(
    permitted: list[ToolCall],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    hook_executor: Any | None,
    on_tool_result: OnToolResult | None,
    on_parallel_start: OnParallelStart | None,
    on_parallel_complete: OnParallelComplete | None,
    metrics: AgentMetrics | None,
    state: _LoopState,
    conversation: Conversation,
    cancel_event: asyncio.Event | None,
    llm_client: LLMClient,
) -> None:
    """Execute tools in parallel (read-only) and sequential (write) batches.

    Handles auto-verify, auto-stash, auto-commit, and stuck-loop detection.
    """
    from godspeed.tools.base import RiskLevel

    # Partition into read-only (parallel-safe) and write (serial) groups
    read_only_calls: list[ToolCall] = []
    write_calls: list[ToolCall] = []
    for tc in permitted:
        tool = tool_registry.get(tc.tool_name)
        if tool is not None and tool.risk_level == RiskLevel.READ_ONLY:
            read_only_calls.append(tc)
        else:
            write_calls.append(tc)

    all_calls = read_only_calls + write_calls
    if on_parallel_start:
        on_parallel_start([(tc.tool_name, tc.arguments) for tc in all_calls])

    t0 = time.monotonic()
    parallel_results: list[ToolResult] = []
    if read_only_calls:
        coros = []
        for tc in read_only_calls:
            cached_task = state.speculative_cache.pop(tc.call_id, None)
            if cached_task is not None:
                logger.debug("Speculative hit tool=%s call_id=%s", tc.tool_name, tc.call_id)
                coros.append(cached_task)
            else:
                coros.append(asyncio.create_task(tool_registry.dispatch(tc, tool_context)))
        parallel_results = await asyncio.gather(*coros)

    serial_results: list[ToolResult] = []
    for tc in write_calls:
        _check_cancel(cancel_event)
        result = await tool_registry.dispatch(tc, tool_context)
        serial_results.append(result)

    results = parallel_results + serial_results
    permitted_ordered = all_calls
    batch_latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Parallel dispatch completed tools=%d latency_ms=%.1f",
        len(permitted_ordered),
        batch_latency_ms,
    )

    # Process results
    for tool_call, result in zip(permitted_ordered, results, strict=True):
        _check_cancel(cancel_event)
        if hook_executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, hook_executor.run_post_tool, tool_call.tool_name
            )
        if on_tool_result:
            on_tool_result(tool_call.tool_name, result)
        if metrics is not None:
            metrics.record_tool_call(tool_call.tool_name, result.is_error)
        if tool_context.audit is not None:
            await tool_context.audit.arecord(
                event_type="tool_call",
                detail={
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output_length": len(result.output),
                    "is_error": result.is_error,
                    "latency_ms": round(batch_latency_ms / len(permitted_ordered), 1),
                    "parallel": True,
                },
                outcome="error" if result.is_error else "success",
            )
        result_content = result.error if result.is_error else result.output
        conversation.add_tool_result(
            tool_call_id=tool_call.call_id,
            content=result_content or "",
        )

    # Post-processing: auto-verify, auto-stash, auto-commit, stuck-loop
    await _post_process_results(
        permitted_ordered,
        results,
        state,
        conversation,
        tool_registry,
        tool_context,
        llm_client,
        metrics,
    )

    if on_parallel_complete:
        on_parallel_complete(
            [
                (tc.tool_name, str(r.error) if r.is_error else str(r.output), r.is_error)
                for tc, r in zip(permitted_ordered, results, strict=True)
            ]
        )


async def _dispatch_sequential(
    permitted: list[ToolCall],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    hook_executor: Any | None,
    on_tool_result: OnToolResult | None,
    metrics: AgentMetrics | None,
    state: _LoopState,
    conversation: Conversation,
    cancel_event: asyncio.Event | None,
    llm_client: LLMClient,
) -> None:
    """Execute tools sequentially, one at a time."""
    for tool_call in permitted:
        _check_cancel(cancel_event)
        t0 = time.monotonic()
        cached_task = state.speculative_cache.pop(tool_call.call_id, None)
        if cached_task is not None:
            logger.debug(
                "Speculative hit (sequential) tool=%s call_id=%s",
                tool_call.tool_name,
                tool_call.call_id,
            )
            result = await cached_task
        else:
            result = await tool_registry.dispatch(tool_call, tool_context)
        latency_ms = (time.monotonic() - t0) * 1000

        if hook_executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, hook_executor.run_post_tool, tool_call.tool_name
            )
        if on_tool_result:
            on_tool_result(tool_call.tool_name, result)
        if metrics is not None:
            metrics.record_tool_call(tool_call.tool_name, result.is_error)
        if tool_context.audit is not None:
            await tool_context.audit.arecord(
                event_type="tool_call",
                detail={
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output_length": len(result.output),
                    "is_error": result.is_error,
                    "latency_ms": round(latency_ms, 1),
                },
                outcome="error" if result.is_error else "success",
            )
        result_content = result.error if result.is_error else result.output
        conversation.add_tool_result(
            tool_call_id=tool_call.call_id,
            content=result_content or "",
        )
        await _post_process_single_result(
            tool_call,
            result,
            state,
            conversation,
            tool_registry,
            tool_context,
            llm_client,
            metrics,
        )


async def _post_process_results(
    tool_calls: list[ToolCall],
    results: list[ToolResult],
    state: _LoopState,
    conversation: Conversation,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    llm_client: LLMClient,
    metrics: AgentMetrics | None,
) -> None:
    """Post-process a batch of tool results (auto-verify, auto-stash, etc.).

    Optimized: single-pass iteration over tool_calls/results to avoid repeated zip.
    """
    # Pre-compute zipped pairs once to avoid repeated iteration
    paired = list(zip(tool_calls, results, strict=True))

    # Single-pass: collect write operations and errors
    write_results: list[tuple[ToolCall, ToolResult]] = []
    has_non_write = False

    for tc, result in paired:
        # Check for write operations
        if not result.is_error and tc.tool_name in ("file_edit", "file_write"):
            write_results.append((tc, result))
        elif tc.tool_name not in ("file_edit", "file_write"):
            has_non_write = True

        # Stuck-loop detection (inline to avoid another pass)
        if result.is_error:
            error_hash = hashlib.sha256((result.error or "").encode()).hexdigest()
            state.recent_error_hashes.append(error_hash)
            if len(state.recent_error_hashes) > state.stuck_threshold:
                state.recent_error_hashes.pop(0)
        else:
            state.recent_error_hashes.clear()

        # Auto-verify for write operations
        if (
            not result.is_error
            and tc.tool_name in ("file_edit", "file_write")
            and tool_registry.has_tool("verify")
        ):
            file_path = tc.arguments.get("file_path", "")
            if file_path and file_path.endswith(VERIFIABLE_EXTENSIONS):
                verify_result = await _auto_verify_file(
                    file_path, tc.call_id, tool_registry, tool_context, state.auto_fix_retries
                )
                conversation.add_tool_result(
                    tool_call_id=f"{tc.call_id}_verify",
                    content=verify_result.output or "",
                )
                verify_text = verify_result.error or verify_result.output or ""
                state.must_fix_injections = _maybe_inject_must_fix(
                    conversation,
                    file_path,
                    verify_text,
                    state.must_fix_injections,
                    metrics,
                    state.must_fix_cap,
                )

    # Update write tracking from collected results
    batch_writes = len(write_results)
    if has_non_write:
        state.consecutive_writes = batch_writes
        state.consecutive_successful_edits = batch_writes
        state.recent_change_descriptions = [
            f"{tc.tool_name} {tc.arguments.get('file_path', '?')}" for tc, _ in write_results
        ]
    else:
        state.consecutive_writes += batch_writes

    # Stuck-loop detection: inject hint if needed
    if (
        len(state.recent_error_hashes) == state.stuck_threshold
        and len(set(state.recent_error_hashes)) == 1
    ):
        logger.warning("Stuck loop detected: %d identical errors", state.stuck_threshold)
        conversation.add_user_message(
            f"You have failed {state.stuck_threshold} times with the same error. "
            "Stop, explain what is wrong, and try a completely different approach."
        )
        state.recent_error_hashes.clear()

    # Auto-stash check
    if (
        state.consecutive_writes >= state.stash_threshold
        and not state.auto_stashed
        and tool_registry.has_tool("git")
    ):
        stash_call = ToolCall(
            tool_name="git",
            arguments={"action": "stash"},
            call_id=f"{tool_calls[-1].call_id}_autostash",
        )
        stash_result = await tool_registry.dispatch(stash_call, tool_context)
        if (
            not stash_result.is_error
            and "nothing to stash" not in (stash_result.output or "").lower()
        ):
            state.auto_stashed = True
            logger.info(
                "Auto-stash triggered after %d consecutive writes", state.consecutive_writes
            )
            conversation.add_tool_result(
                tool_call_id=stash_call.call_id,
                content=(
                    f"[Auto-stash] Saved working state after "
                    f"{state.consecutive_writes} consecutive file edits. "
                    "Use git stash_pop to restore if needed."
                ),
            )

    # Auto-commit tracking (uses write_results from single pass)
    for tc, _r in write_results:
        state.consecutive_successful_edits += 1
        desc = f"{tc.tool_name} {tc.arguments.get('file_path', '?')}"
        state.recent_change_descriptions.append(desc)

    if state.auto_commit and state.consecutive_successful_edits >= state.auto_commit_threshold:
        committed = await _try_auto_commit(
            list(state.recent_change_descriptions),
            tool_context,
            llm_client,
            conversation,
            tool_calls[-1].call_id,
        )
        if committed:
            state.consecutive_successful_edits = 0
            state.recent_change_descriptions.clear()


async def _post_process_single_result(
    tool_call: ToolCall,
    result: ToolResult,
    state: _LoopState,
    conversation: Conversation,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    llm_client: LLMClient,
    metrics: AgentMetrics | None,
) -> None:
    """Post-process a single tool result."""
    if (
        not result.is_error
        and tool_call.tool_name in ("file_edit", "file_write")
        and tool_registry.has_tool("verify")
    ):
        file_path = tool_call.arguments.get("file_path", "")
        if file_path and file_path.endswith(VERIFIABLE_EXTENSIONS):
            verify_result = await _auto_verify_file(
                file_path, tool_call.call_id, tool_registry, tool_context, state.auto_fix_retries
            )
            conversation.add_tool_result(
                tool_call_id=f"{tool_call.call_id}_verify",
                content=verify_result.output or "",
            )
            verify_text = verify_result.error or verify_result.output or ""
            state.must_fix_injections = _maybe_inject_must_fix(
                conversation,
                file_path,
                verify_text,
                state.must_fix_injections,
                metrics,
                state.must_fix_cap,
            )

    if not result.is_error and tool_call.tool_name in ("file_edit", "file_write"):
        state.consecutive_writes += 1
        if (
            state.consecutive_writes >= state.stash_threshold
            and not state.auto_stashed
            and tool_registry.has_tool("git")
        ):
            stash_call = ToolCall(
                tool_name="git",
                arguments={"action": "stash"},
                call_id=f"{tool_call.call_id}_autostash",
            )
            stash_result = await tool_registry.dispatch(stash_call, tool_context)
            if (
                not stash_result.is_error
                and "nothing to stash" not in (stash_result.output or "").lower()
            ):
                state.auto_stashed = True
                logger.info(
                    "Auto-stash triggered after %d consecutive writes", state.consecutive_writes
                )
                conversation.add_tool_result(
                    tool_call_id=stash_call.call_id,
                    content=(
                        f"[Auto-stash] Saved working state after "
                        f"{state.consecutive_writes} consecutive file edits. "
                        "Use git stash_pop to restore if needed."
                    ),
                )
        state.consecutive_successful_edits += 1
        desc = f"{tool_call.tool_name} {tool_call.arguments.get('file_path', '?')}"
        state.recent_change_descriptions.append(desc)
        if state.auto_commit and state.consecutive_successful_edits >= state.auto_commit_threshold:
            committed = await _try_auto_commit(
                list(state.recent_change_descriptions),
                tool_context,
                llm_client,
                conversation,
                tool_call.call_id,
            )
            if committed:
                state.consecutive_successful_edits = 0
                state.recent_change_descriptions.clear()
    else:
        state.consecutive_writes = 0
        state.consecutive_successful_edits = 0
        state.recent_change_descriptions = []

    # Stuck-loop detection
    if result.is_error:
        error_hash = hashlib.sha256((result.error or "").encode()).hexdigest()
        state.recent_error_hashes.append(error_hash)
        if len(state.recent_error_hashes) > state.stuck_threshold:
            state.recent_error_hashes.pop(0)
        if (
            len(state.recent_error_hashes) == state.stuck_threshold
            and len(set(state.recent_error_hashes)) == 1
        ):
            logger.warning("Stuck loop detected: %d identical errors", state.stuck_threshold)
            conversation.add_user_message(
                f"You have failed {state.stuck_threshold} times with the same error. "
                "Stop, explain what is wrong, and try a completely different approach."
            )
            state.recent_error_hashes.clear()
    else:
        state.recent_error_hashes.clear()


async def _auto_verify_file(
    file_path: str,
    parent_call_id: str,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    auto_fix_retries: int,
) -> ToolResult:
    """Run auto-verify on a file, using the retry loop when retries > 0.

    Falls back to plain verify dispatch when retries are disabled (0).
    """
    from pathlib import Path

    from godspeed.tools.verify import _EXTENSION_MAP, _verify_with_retry

    resolved = Path(file_path) if Path(file_path).is_absolute() else (tool_context.cwd / file_path)
    suffix = resolved.suffix.lower()
    lang = _EXTENSION_MAP.get(suffix)

    if auto_fix_retries > 0 and lang is not None:
        # Run in thread to avoid blocking the event loop
        return await asyncio.to_thread(
            _verify_with_retry,
            resolved=resolved,
            display_path=file_path,
            lang=lang,
            cwd=tool_context.cwd,
            max_retries=auto_fix_retries,
        )

    # Fallback: plain verify dispatch (one-shot)
    verify_call = ToolCall(
        tool_name="verify",
        arguments={"file_path": file_path},
        call_id=f"{parent_call_id}_verify",
    )
    return await tool_registry.dispatch(verify_call, tool_context)


def _maybe_inject_must_fix(
    conversation: Conversation,
    file_path: str,
    verify_output: str,
    injections: int,
    metrics: AgentMetrics | None,
    effective_must_fix_cap: int,
) -> int:
    """Force the model to address unresolved lint errors after auto-verify.

    verify_with_retry returns a success ToolResult even when lint errors
    persist (fingerprint: verify.REMAINING_ERRORS_FINGERPRINT). Without
    this gate the model sees a success marker and can proceed to unrelated
    edits while quality silently degrades. On detection, inject a
    user-role message naming the file and errors so the constraint is
    in-conversation.

    Caps at effective_must_fix_cap injections per session. After the cap we log a
    warning and fail open — better to let the agent try a different tack
    than to deadlock on a fundamentally unfixable error (broken ruff
    config, upstream dep bug, etc.).

    When `metrics` is provided, each successful injection is recorded so
    downstream RL can shape rewards against agent efficiency.
    """
    from godspeed.tools.verify import REMAINING_ERRORS_FINGERPRINT

    if REMAINING_ERRORS_FINGERPRINT not in (verify_output or ""):
        return injections
    if injections >= effective_must_fix_cap:
        logger.warning(
            "MUST-FIX cap reached for file=%s; allowing agent to proceed",
            file_path,
        )
        return injections
    conversation.add_user_message(
        f"VERIFY FAILED on {file_path}. Unresolved lint errors remain "
        f"after auto-fix attempts:\n\n{verify_output}\n\n"
        "You MUST fix these errors before any other edits or writes."
    )
    logger.info(
        "MUST-FIX injected file=%s count=%d/%d",
        file_path,
        injections + 1,
        effective_must_fix_cap,
    )
    if metrics is not None:
        metrics.record_must_fix_injection()
    return injections + 1


async def _try_auto_commit(
    change_descriptions: list[str],
    tool_context: ToolContext,
    llm_client: LLMClient,
    conversation: Conversation,
    parent_call_id: str,
) -> bool:
    """Attempt an auto-commit with LLM-generated message. Returns True on success."""
    from godspeed.agent.auto_commit import auto_commit, generate_commit_message

    try:
        message = await generate_commit_message(change_descriptions, llm_client)
        result = await auto_commit(tool_context.cwd, message)
        if not result.is_error:
            logger.info("Auto-commit succeeded message=%s", message)
            conversation.add_tool_result(
                tool_call_id=f"{parent_call_id}_autocommit",
                content=f"[Auto-commit] {result.output}",
            )
            return True
        logger.warning("Auto-commit failed: %s", result.error)
    except Exception as exc:
        logger.warning("Auto-commit error: %s", exc)
    return False


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall | None:
    """Parse a raw tool call from the LLM response.

    Returns None if the tool call is malformed (invalid JSON arguments, etc.).
    Common tool-name hallucinations (``read_file``, ``grep``, ``glob``, etc.)
    are rewritten to their canonical names via
    ``godspeed.tools.aliases.canonicalize_tool_name`` so weak models don't
    dead-end on a correct intent expressed with the wrong label.
    """
    from godspeed.tools.aliases import canonicalize_tool_name

    try:
        func = raw.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        arguments = json.loads(args_str) if isinstance(args_str, str) else args_str

        if not name:
            return None

        return ToolCall(
            tool_name=canonicalize_tool_name(name),
            arguments=arguments,
            call_id=raw.get("id", ""),
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Malformed tool call: %s", raw)
        return None


async def _compact_conversation(conversation: Conversation, llm_client: LLMClient) -> None:
    """Compact conversation by summarizing history via a separate LLM call.

    Uses model-aware compaction prompts — small models get aggressive summarization,
    frontier models get detailed preservation. Picks the cheapest available model
    from the fallback chain to minimize compaction cost.
    """
    from godspeed.context.compaction import get_compaction_prompt
    from godspeed.llm.cost import get_cheapest_model

    model_name = getattr(llm_client, "model", "")
    logger.info("Compacting conversation tokens=%d model=%s", conversation.token_count, model_name)

    # Use cheapest model for compaction
    candidates = [model_name, *getattr(llm_client, "fallback_models", [])]
    cheapest = get_cheapest_model(candidates)
    if cheapest and cheapest != model_name:
        logger.info("Compaction using cheaper model=%s (instead of %s)", cheapest, model_name)

    prompt = get_compaction_prompt(cheapest or model_name)
    context = conversation.get_compaction_context()
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    try:
        response = await llm_client.chat(
            messages=summary_messages,
            task_type="compaction",
        )
        conversation.compact(response.content)
    except Exception as exc:
        logger.error("Compaction failed error=%s", exc, exc_info=True)
        # Don't crash — try truncation as fallback
        with contextlib.suppress(Exception):
            conversation.compact(f"[Compaction failed: {exc}. Retaining most recent context.]")


def _check_cancel(cancel_event: asyncio.Event | None) -> None:
    """Raise AgentCancelledError if the event has been set.

    Called at checkpoint boundaries inside the agent loop: top of
    iteration, between streaming chunks, before tool dispatch. Cheap
    (single atomic is_set() read) — safe to sprinkle liberally.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise AgentCancelledError("cancel_event set by caller")


async def _streaming_call(
    llm_client: LLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    on_chunk: Callable[[str], None],
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
    speculative_cache: dict[str, asyncio.Task[ToolResult]] | None = None,
    cancel_event: asyncio.Event | None = None,
    task_type: str | None = None,
) -> ChatResponse:
    """Make a streaming LLM call, invoking on_chunk for each text delta.

    When tool_registry and tool_context are provided, speculatively dispatches
    READ_ONLY tool calls as soon as the final response arrives — before the
    main loop processes them. Results are stored in speculative_cache so the
    main loop can await them instead of re-dispatching.

    When cancel_event is provided, the chunk loop checks it between each
    yielded chunk and raises AgentCancelledError — closing the underlying
    litellm stream promptly (its aclose() fires on generator cleanup).

    Returns the final complete ChatResponse for conversation history.
    """
    final_response: ChatResponse | None = None

    stream = llm_client.stream_chat(messages=messages, tools=tools, task_type=task_type)
    try:
        async for chunk in stream:
            if chunk.finish_reason is None and chunk.content:
                # Intermediate chunk — stream text to caller first, THEN
                # check cancel. This way the user sees the text the model
                # already produced before we unwind — a cleaner UX than
                # cutting off mid-word.
                on_chunk(chunk.content)
            elif chunk.finish_reason is not None:
                # Final aggregated response
                final_response = chunk

            # Cancel checkpoint: between chunks. If the caller (TUI signal
            # handler, headless SIGINT) set cancel_event during the last
            # chunk's on_chunk callback — or any time before now — we raise
            # AgentCancelledError here. The generator cleanup path in `finally`
            # closes the underlying HTTP stream.
            _check_cancel(cancel_event)
    finally:
        # Ensure the underlying async generator is closed on cancel OR on
        # any exception. aclose() is idempotent and cheap.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()

    if final_response is None:
        # Stream ended without a finish_reason — shouldn't happen but be safe
        return ChatResponse(content="", tool_calls=[], finish_reason="stop", usage={})

    # Track streaming token usage (batch path does this inside LLMClient._call)
    if final_response.usage:
        llm_client.total_input_tokens += final_response.usage.get("prompt_tokens", 0)
        llm_client.total_output_tokens += final_response.usage.get("completion_tokens", 0)

    # Speculative execution: start READ_ONLY tools immediately
    if (
        final_response.has_tool_calls
        and tool_registry is not None
        and tool_context is not None
        and speculative_cache is not None
    ):
        _speculative_dispatch(
            final_response.tool_calls,
            tool_registry,
            tool_context,
            speculative_cache,
        )

    return final_response


def _speculative_dispatch(
    raw_tool_calls: list[dict[str, Any]],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    cache: dict[str, asyncio.Task[ToolResult]],
) -> None:
    """Start READ_ONLY tool calls speculatively as background tasks.

    Parses each tool call and checks risk level. If READ_ONLY, dispatches
    immediately and stores the asyncio.Task in cache keyed by call_id.
    The main loop checks the cache before dispatching to avoid double work.

    Enforces MAX_SPECULATIVE_CACHE_SIZE to prevent unbounded growth.
    """
    from godspeed.tools.base import RiskLevel

    for raw_tc in raw_tool_calls:
        # Enforce cache size limit to prevent memory growth
        if len(cache) >= MAX_SPECULATIVE_CACHE_SIZE:
            logger.debug(
                "Speculative cache full (max=%d), skipping remaining dispatches",
                MAX_SPECULATIVE_CACHE_SIZE,
            )
            break

        parsed = _parse_tool_call(raw_tc)
        if parsed is None:
            continue

        tool = tool_registry.get(parsed.tool_name)
        if tool is None or tool.risk_level != RiskLevel.READ_ONLY:
            continue

        call_id = parsed.call_id
        if call_id and call_id not in cache:
            logger.debug("Speculative dispatch tool=%s call_id=%s", parsed.tool_name, call_id)
            task = asyncio.create_task(tool_registry.dispatch(parsed, tool_context))
            cache[call_id] = task
