"""Core agent loop — the heart of Godspeed.

Hand-rolled loop following the pattern proven by mini-swe-agent (74%+ SWE-bench)
and Claude Code. The model decides when to stop. No framework overhead.
"""

from __future__ import annotations

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

    Returns:
        The final assistant text response.
    """
    conversation.add_user_message(user_input)
    tool_schemas = tool_registry.get_schemas()

    retries = 0
    final_text = ""

    for iteration in range(MAX_ITERATIONS):
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
    """Compact conversation by summarizing history via a separate LLM call."""
    logger.info("Compacting conversation tokens=%d", conversation.token_count)

    context = conversation.get_compaction_context()
    summary_messages = [
        {
            "role": "system",
            "content": (
                "Summarize the following conversation between a user and a coding agent. "
                "Preserve: architectural decisions, file paths modified, unresolved issues, "
                "current task state. Discard: redundant tool outputs, repeated attempts. "
                "Be concise but complete."
            ),
        },
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
