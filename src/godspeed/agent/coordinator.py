"""Sub-agent coordinator — spawn isolated agent loops for parallel sub-tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.llm.client import LLMClient
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_SUB_AGENT_DEPTH = 3
SUB_AGENT_ITERATION_LIMIT = 25

SUB_AGENT_SYSTEM_PROMPT = """\
You are a sub-agent of Godspeed, a security-first coding agent. You have been \
spawned to handle a specific sub-task. Complete the task and return a concise \
summary of what you accomplished.

## Guidelines
- Focus on the specific task assigned to you
- Use tools efficiently — minimize unnecessary reads
- Return a clear summary when done
- Do not spawn further sub-agents unless absolutely necessary
"""


class AgentCoordinator:
    """Coordinates sub-agent spawning with depth limiting and isolation."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        max_depth: int = MAX_SUB_AGENT_DEPTH,
        iteration_limit: int = SUB_AGENT_ITERATION_LIMIT,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._max_depth = max_depth
        self._iteration_limit = iteration_limit
        self._current_depth = 0

    async def spawn(
        self,
        task: str,
        depth: int = 0,
    ) -> str:
        """Spawn an isolated sub-agent to handle a task.

        Args:
            task: The sub-task description.
            depth: Current nesting depth (0 = top-level spawn).

        Returns:
            The sub-agent's final text response.
        """
        if depth >= self._max_depth:
            return (
                f"Error: Maximum sub-agent depth ({self._max_depth}) reached. "
                f"Cannot spawn further sub-agents."
            )

        logger.info("Spawning sub-agent depth=%d task=%r", depth, task[:100])

        # Create isolated conversation for the sub-agent
        conversation = Conversation(
            system_prompt=SUB_AGENT_SYSTEM_PROMPT,
            model=self._llm_client.model,
            max_tokens=getattr(self._llm_client, "_max_tokens", 100_000),
        )

        try:
            result = await agent_loop(
                user_input=task,
                conversation=conversation,
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
                tool_context=self._tool_context,
                max_iterations=self._iteration_limit,
            )
            logger.info("Sub-agent completed depth=%d result_len=%d", depth, len(result))
            return result
        except Exception as exc:
            logger.error("Sub-agent failed depth=%d error=%s", depth, exc, exc_info=True)
            return f"Sub-agent error: {exc}"

    async def spawn_parallel(
        self,
        tasks: list[str],
        depth: int = 0,
    ) -> list[str]:
        """Spawn multiple sub-agents in parallel.

        Args:
            tasks: List of sub-task descriptions.
            depth: Current nesting depth.

        Returns:
            List of results (one per task, in order).
        """
        if depth >= self._max_depth:
            return [f"Error: Maximum sub-agent depth ({self._max_depth}) reached."] * len(tasks)

        logger.info("Spawning %d parallel sub-agents depth=%d", len(tasks), depth)
        coros = [self.spawn(task, depth=depth) for task in tasks]
        return list(await asyncio.gather(*coros, return_exceptions=False))


class SpawnAgentTool(Tool):
    """Tool for the LLM to spawn sub-agents for complex sub-tasks.

    The sub-agent gets its own isolated conversation but shares the same
    tool registry, LLM client, and tool context.
    """

    def __init__(self, coordinator: AgentCoordinator) -> None:
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        return "spawn_agent"

    @property
    def description(self) -> str:
        return (
            "Spawn a sub-agent to handle a specific sub-task independently. "
            "The sub-agent has its own conversation but shares your tools. "
            "Use for tasks that can be delegated (e.g., 'search for all "
            "usages of function X', 'refactor module Y'). "
            "Returns the sub-agent's final response."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The sub-task to delegate to the sub-agent",
                },
            },
            "required": ["task"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        task = arguments.get("task", "")
        if not task:
            return ToolResult.failure("task is required for spawn_agent")

        result = await self._coordinator.spawn(task, depth=0)
        return ToolResult.success(result)
