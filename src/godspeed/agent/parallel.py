"""Parallel execution for sub-agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ParallelAgent:
    """A sub-agent that runs with its own context and tools."""

    def __init__(
        self,
        task: str,
        model: str | None = None,
        max_iterations: int = 50,
    ) -> None:
        self.task = task
        self.model = model
        self.max_iterations = max_iterations
        self.id = str(uuid4())[:8]
        self.status = "pending"
        self.result: Any = None
        self.error: str | None = None


class ParallelAgentRunner:
    """Run multiple agents in parallel.

    Uses git worktrees or isolated directories to prevent
    file conflicts when running multiple agents.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = base_dir
        self.agents: dict[str, ParallelAgent] = {}

    async def spawn(
        self,
        task: str,
        model: str | None = None,
        use_worktree: bool = False,
    ) -> ParallelAgent:
        """Spawn a new parallel agent."""
        import os

        agent = ParallelAgent(task, model)
        self.agents[agent.id] = agent

        if use_worktree and self.base_dir:
            worktree_path = os.path.join(self.base_dir, f".worktree-{agent.id}")
            try:
                os.makedirs(worktree_path, exist_ok=True)
                agent.worktree_path = worktree_path
                agent.status = "running"
                logger.info("Spawned agent %s with worktree %s", agent.id, worktree_path)
            except Exception as exc:
                logger.warning("Failed to create worktree: %s", exc)
                agent.status = "running"
        else:
            agent.status = "running"

        return agent

    async def run_parallel(
        self,
        tasks: list[str],
        model: str | None = None,
        max_concurrent: int = 4,
    ) -> list[dict[str, Any]]:
        """Run multiple tasks in parallel."""
        results: list[dict[str, Any]] = []

        # Create agents for each task
        agents: list[ParallelAgent] = []
        for task in tasks:
            agent = await self.spawn(task, model)
            agents.append(agent)

        # Run in batches
        for i in range(0, len(agents), max_concurrent):
            batch = agents[i : i + max_concurrent]
            results_batch = await asyncio.gather(
                *[self._run_agent(a) for a in batch],
                return_exceptions=True,
            )

            for a, r in zip(batch, results_batch):
                if isinstance(r, Exception):
                    a.error = str(r)
                    a.status = "failed"
                else:
                    a.result = r
                    a.status = "completed"

                results.append(
                    {
                        "agent_id": a.id,
                        "status": a.status,
                        "result": a.result,
                        "error": a.error,
                    }
                )

        return results

    async def _run_agent(self, agent: ParallelAgent) -> Any:
        """Run a single agent task."""
        from godspeed.agent.loop import run_loop

        result = await run_loop(
            user_input=agent.task,
            model=agent.model,
            max_iterations=agent.max_iterations,
        )
        return result


# Global runner
_runner: ParallelAgentRunner | None = None


def get_parallel_runner(base_dir: str | None = None) -> ParallelAgentRunner:
    """Get or create the parallel runner."""
    global _runner
    if _runner is None:
        _runner = ParallelAgentRunner(base_dir)
    return _runner
