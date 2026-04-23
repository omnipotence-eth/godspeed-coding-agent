"""Sub-agent coordinator — spawn isolated agent loops for parallel sub-tasks.

The coordinator pattern keeps spawn-related state (depth tracking,
parent references, child registry) out of individual tools and off the
hot path of ToolContext. The ``SpawnAgentTool`` is a thin adapter that
forwards an LLM-issued spawn request to the coordinator.

Key design points:

- **Fresh conversation.** The child does NOT inherit the parent's
  history — only the task prompt.
- **Same cwd, permissions, audit, diff reviewer.** Constrained by the
  same 4-tier permission engine and recorded in the same hash-chained
  audit log.
- **Same LLMClient.** Cost / budget / routing / cache-hit telemetry
  are shared — parent's ``max_cost_usd`` covers both.
- **No recursive spawn.** The child registry omits ``spawn_agent`` so
  a child can't fork its own subagent (fork-bomb prevention; keeps
  latency predictable). This supersedes the older depth-counter which
  only slowed recursion down.
- **Bounded wall-clock.** Per-spawn ``timeout`` keeps a stuck child
  from blocking the parent indefinitely.
- **Full system prompt.** The child gets the real
  ``build_system_prompt`` output (workflows, anti-patterns, tool
  descriptions, repo-map) instead of a stub — dramatically better
  first-shot success rate on non-trivial subtasks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.agent.system_prompt import build_system_prompt
from godspeed.context.repo_summary import build_repo_summary
from godspeed.llm.client import LLMClient
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 25
MAX_ALLOWED_ITERATIONS = 50
DEFAULT_TIMEOUT_SECONDS = 300
MAX_ALLOWED_TIMEOUT_SECONDS = 1200

# Tools removed from the child registry by default. spawn_agent is
# removed to prevent recursion; other tools can be added here if they
# shouldn't be available to subagents (e.g. session-lifecycle tools).
_CHILD_TOOL_EXCLUDE: frozenset[str] = frozenset({"spawn_agent"})


class AgentCoordinator:
    """Coordinates sub-agent spawning with per-spawn timeouts + isolation."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        default_timeout: int = DEFAULT_TIMEOUT_SECONDS,
        default_max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._default_timeout = default_timeout
        self._default_max_iterations = default_max_iterations

    def _build_child_registry(self) -> ToolRegistry:
        """Return the parent registry with spawn_agent removed."""
        return self._tool_registry.without(*_CHILD_TOOL_EXCLUDE)

    async def spawn(
        self,
        task: str,
        *,
        timeout: int | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """Run an isolated sub-agent to completion and return its final text.

        Args:
            task: Sub-task description. Becomes the child's user message.
            timeout: Wall-clock limit in seconds. Defaults to
                ``DEFAULT_TIMEOUT_SECONDS``; clamped to
                ``[1, MAX_ALLOWED_TIMEOUT_SECONDS]``.
            max_iterations: Agent-loop iteration cap. Defaults to
                ``DEFAULT_MAX_ITERATIONS``; clamped to
                ``[1, MAX_ALLOWED_ITERATIONS]``.
        """
        effective_timeout = min(
            max(timeout if timeout is not None else self._default_timeout, 1),
            MAX_ALLOWED_TIMEOUT_SECONDS,
        )
        effective_iterations = min(
            max(
                max_iterations if max_iterations is not None else self._default_max_iterations,
                1,
            ),
            MAX_ALLOWED_ITERATIONS,
        )

        child_registry = self._build_child_registry()

        # Build the full child system prompt — tools list, workflows,
        # repo-map, project instructions don't flow from parent (fresh
        # conversation), so re-derive them from the same sources.
        repo_map_summary = build_repo_summary(self._tool_context.cwd)
        child_system_prompt = build_system_prompt(
            tools=child_registry.list_tools(),
            cwd=self._tool_context.cwd,
            repo_map_summary=repo_map_summary,
        )

        child_conversation = Conversation(
            system_prompt=child_system_prompt,
            model=self._llm_client.model,
            max_tokens=getattr(self._llm_client, "_max_tokens", 100_000),
        )

        logger.info(
            "Sub-agent spawn timeout=%ds max_iter=%d task_chars=%d",
            effective_timeout,
            effective_iterations,
            len(task),
        )

        try:
            result = await asyncio.wait_for(
                agent_loop(
                    user_input=task,
                    conversation=child_conversation,
                    llm_client=self._llm_client,
                    tool_registry=child_registry,
                    tool_context=self._tool_context,
                    max_iterations=effective_iterations,
                ),
                timeout=effective_timeout,
            )
        except TimeoutError:
            logger.warning("Sub-agent timed out after %ds", effective_timeout)
            return (
                f"Sub-agent timed out after {effective_timeout}s. "
                "Consider narrowing the task or raising the timeout."
            )
        except Exception as exc:
            logger.error("Sub-agent failed: %s", exc, exc_info=True)
            return f"Sub-agent error: {exc}"

        logger.info("Sub-agent completed result_len=%d", len(result))
        return result

    async def spawn_parallel(
        self,
        tasks: list[str],
        *,
        timeout: int | None = None,
        max_iterations: int | None = None,
    ) -> list[str]:
        """Spawn multiple sub-agents concurrently. Returns results in order."""
        logger.info("Sub-agent parallel spawn count=%d", len(tasks))
        coros = [self.spawn(task, timeout=timeout, max_iterations=max_iterations) for task in tasks]
        return list(await asyncio.gather(*coros, return_exceptions=False))


class SpawnAgentTool(Tool):
    """Tool for the main agent to delegate a scoped subtask to a child agent.

    The child gets a fresh conversation but shares the parent's cwd /
    permissions / audit / LLM client. Cost counts against the parent's
    budget. Cannot itself spawn further subagents (fork-bomb
    prevention). Good for parallel exploration, scoped code review,
    or test-authoring subtasks where the child should reason without
    the main conversation's baggage.
    """

    def __init__(self, coordinator: AgentCoordinator) -> None:
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        return "spawn_agent"

    @property
    def description(self) -> str:
        return (
            "Spawn a child coding agent with a fresh conversation to complete "
            "a focused subtask. Useful for parallel exploration, scoped code "
            "review, or test-authoring subtasks. The child shares your cwd, "
            "permissions, audit trail, and LLM client (cost counts against "
            "your budget). It does NOT inherit your conversation history and "
            "cannot itself spawn further subagents.\n\n"
            "Example: spawn_agent(task='Read src/parser.py and src/lexer.py "
            "and summarize the parse pipeline in 4 bullets')\n"
            "Example: spawn_agent(task='Run the full test suite and list any "
            "failures by file:line', max_iterations=15)"
        )

    @property
    def risk_level(self) -> RiskLevel:
        # HIGH — the user should see a prompt on every spawn. Any
        # destructive action the child takes is still gated by the
        # permission engine, but delegating to an autonomous child is
        # a meaningful decision the user may want to consent to.
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the subtask for the "
                        "child agent. Be specific about inputs, expected "
                        "output format, and stop condition."
                    ),
                },
                "max_iterations": {
                    "type": "integer",
                    "description": (
                        f"Maximum agent-loop iterations for the child "
                        f"(default {DEFAULT_MAX_ITERATIONS}, max "
                        f"{MAX_ALLOWED_ITERATIONS})."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Wall-clock timeout in seconds (default "
                        f"{DEFAULT_TIMEOUT_SECONDS}, max "
                        f"{MAX_ALLOWED_TIMEOUT_SECONDS})."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        task = arguments.get("task", "").strip()
        if not task:
            return ToolResult.failure(
                "spawn_agent requires a non-empty 'task' argument describing "
                "what the child agent should do."
            )

        timeout = arguments.get("timeout")
        max_iterations = arguments.get("max_iterations")

        result = await self._coordinator.spawn(
            task,
            timeout=int(timeout) if timeout is not None else None,
            max_iterations=int(max_iterations) if max_iterations is not None else None,
        )
        return ToolResult.success(result)
