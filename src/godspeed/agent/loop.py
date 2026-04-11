"""Core agent loop — the heart of Godspeed.

Hand-rolled loop following the pattern proven by mini-swe-agent (74%+ SWE-bench)
and Claude Code. The model decides when to stop. No framework overhead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolCall, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 50
MAX_RETRIES = 3
STUCK_LOOP_THRESHOLD = 3
AUTO_STASH_THRESHOLD = 3

# Callback type aliases for clarity
OnAssistantText = Callable[[str], None]
OnToolCall = Callable[[str, dict[str, Any]], None]
OnToolResult = Callable[[str, ToolResult], None]
OnPermissionDenied = Callable[[str, str], None]
OnChunk = Callable[[str], None]


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
        pause_event: Optional asyncio.Event for pause/resume. When cleared,
            the loop waits at the top of each iteration until set again.

    Returns:
        The final assistant text response.
    """
    iteration_limit = max_iterations if max_iterations is not None else MAX_ITERATIONS
    conversation.add_user_message(user_input)
    tool_schemas = tool_registry.get_schemas()

    retries = 0
    final_text = ""
    recent_error_hashes: list[str] = []
    consecutive_writes = 0
    auto_stashed = False

    for iteration in range(iteration_limit):
        # Pause/resume: if pause_event exists and is cleared, wait for it
        if pause_event is not None and not pause_event.is_set():
            logger.info("Agent loop paused at iteration=%d", iteration)
            await pause_event.wait()
            logger.info("Agent loop resumed at iteration=%d", iteration)

        logger.debug("Agent loop iteration=%d tokens=%d", iteration, conversation.token_count)

        # Check if we need to compact
        if conversation.is_near_limit:
            await _compact_conversation(conversation, llm_client)

        # Call LLM (streaming or batch)
        try:
            if on_assistant_chunk is not None:
                response = await _streaming_call(
                    llm_client,
                    conversation.messages,
                    tool_schemas if tool_schemas else None,
                    on_assistant_chunk,
                )
            else:
                response = await llm_client.chat(
                    messages=conversation.messages,
                    tools=tool_schemas if tool_schemas else None,
                )
        except Exception as exc:
            logger.error("LLM call failed error=%s", exc, exc_info=True)
            return f"Error: LLM call failed — {exc}"

        # Handle text response (model decided to stop)
        if not response.has_tool_calls:
            final_text = response.content
            if final_text:
                conversation.add_assistant_message(content=final_text)
                if on_assistant_text:
                    on_assistant_text(final_text)
            return final_text

        # Handle tool calls
        conversation.add_assistant_message(
            content=response.content,
            tool_calls=response.tool_calls,
        )

        if response.content and on_assistant_text:
            on_assistant_text(response.content)

        for raw_tc in response.tool_calls:
            tool_call = _parse_tool_call(raw_tc)
            if tool_call is None:
                retries += 1
                if retries > MAX_RETRIES:
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

            # Execute tool with latency tracking
            t0 = time.monotonic()
            result = await tool_registry.dispatch(tool_call, tool_context)
            latency_ms = (time.monotonic() - t0) * 1000

            if on_tool_result:
                on_tool_result(tool_call.tool_name, result)

            # Record audit event with latency
            if tool_context.audit is not None:
                tool_context.audit.record(
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

            # Feed result back to conversation
            result_content = result.error if result.is_error else result.output
            conversation.add_tool_result(
                tool_call_id=tool_call.call_id,
                content=result_content or "",
            )

            # Auto-verify after successful file edits/writes
            if (
                not result.is_error
                and tool_call.tool_name in ("file_edit", "file_write")
                and tool_registry.has_tool("verify")
            ):
                file_path = tool_call.arguments.get("file_path", "")
                if file_path and file_path.endswith((".py", ".pyi")):
                    verify_call = ToolCall(
                        tool_name="verify",
                        arguments={"file_path": file_path},
                        call_id=f"{tool_call.call_id}_verify",
                    )
                    verify_result = await tool_registry.dispatch(verify_call, tool_context)
                    conversation.add_tool_result(
                        tool_call_id=verify_call.call_id,
                        content=verify_result.output or "",
                    )
                    logger.debug(
                        "Auto-verify file=%s passed=%s",
                        file_path,
                        "passed" in verify_result.output.lower(),
                    )

            # Auto-stash: track consecutive write operations
            if not result.is_error and tool_call.tool_name in ("file_edit", "file_write"):
                consecutive_writes += 1
                if (
                    consecutive_writes >= AUTO_STASH_THRESHOLD
                    and not auto_stashed
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
                        auto_stashed = True
                        logger.info(
                            "Auto-stash triggered after %d consecutive writes",
                            consecutive_writes,
                        )
                        conversation.add_tool_result(
                            tool_call_id=stash_call.call_id,
                            content=(
                                f"[Auto-stash] Saved working state after "
                                f"{consecutive_writes} consecutive file edits. "
                                f"Use git stash_pop to restore if needed."
                            ),
                        )
            else:
                consecutive_writes = 0

            # Stuck-loop detection: track repeated errors
            if result.is_error:
                error_hash = hashlib.sha256((result_content or "").encode()).hexdigest()
                recent_error_hashes.append(error_hash)
                if len(recent_error_hashes) > STUCK_LOOP_THRESHOLD:
                    recent_error_hashes.pop(0)

                if (
                    len(recent_error_hashes) == STUCK_LOOP_THRESHOLD
                    and len(set(recent_error_hashes)) == 1
                ):
                    logger.warning(
                        "Stuck loop detected: %d identical errors hash=%s",
                        STUCK_LOOP_THRESHOLD,
                        error_hash[:12],
                    )
                    conversation.add_user_message(
                        "You have failed 3 times with the same error. "
                        "Stop, explain what is wrong, and try a completely "
                        "different approach."
                    )
                    recent_error_hashes.clear()
            else:
                recent_error_hashes.clear()

    return "Error: Reached maximum iterations. The task may be too complex for a single turn."


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall | None:
    """Parse a raw tool call from the LLM response.

    Returns None if the tool call is malformed (invalid JSON arguments, etc.).
    """
    try:
        func = raw.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        arguments = json.loads(args_str) if isinstance(args_str, str) else args_str

        if not name:
            return None

        return ToolCall(
            tool_name=name,
            arguments=arguments,
            call_id=raw.get("id", ""),
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Malformed tool call: %s", raw)
        return None


async def _compact_conversation(conversation: Conversation, llm_client: LLMClient) -> None:
    """Compact conversation by summarizing history via a separate LLM call.

    Uses model-aware compaction prompts — small models get aggressive summarization,
    frontier models get detailed preservation.
    """
    from godspeed.context.compaction import get_compaction_prompt

    model_name = getattr(llm_client, "model", "")
    logger.info("Compacting conversation tokens=%d model=%s", conversation.token_count, model_name)

    prompt = get_compaction_prompt(model_name) if model_name else get_compaction_prompt("")
    context = conversation.get_compaction_context()
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    try:
        response = await llm_client.chat(messages=summary_messages)
        conversation.compact(response.content)
    except Exception as exc:
        logger.error("Compaction failed error=%s", exc, exc_info=True)
        # Don't crash — just warn and continue with full history


async def _streaming_call(
    llm_client: LLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    on_chunk: Callable[[str], None],
) -> ChatResponse:
    """Make a streaming LLM call, invoking on_chunk for each text delta.

    Returns the final complete ChatResponse for conversation history.
    """
    final_response: ChatResponse | None = None

    async for chunk in llm_client.stream_chat(messages=messages, tools=tools):
        if chunk.finish_reason is None and chunk.content:
            # Intermediate chunk — stream text to caller
            on_chunk(chunk.content)
        elif chunk.finish_reason is not None:
            # Final aggregated response
            final_response = chunk

    if final_response is None:
        # Stream ended without a finish_reason — shouldn't happen but be safe
        return ChatResponse(content="", tool_calls=[], finish_reason="stop", usage={})

    return final_response
