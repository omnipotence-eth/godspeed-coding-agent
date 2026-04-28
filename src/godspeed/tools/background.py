"""Background command execution — run long-running processes with status polling."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 10


@dataclass
class BackgroundProcess:
    """Tracked background subprocess."""

    id: int
    command: str
    process: asyncio.subprocess.Process
    started_at: float
    stdout_chunks: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)
    _collection_task: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    @property
    def elapsed_secs(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def stdout(self) -> str:
        return "".join(self.stdout_chunks)

    @property
    def stderr(self) -> str:
        return "".join(self.stderr_chunks)


class BackgroundRegistry:
    """Singleton registry tracking running background processes."""

    _instance: BackgroundRegistry | None = None

    def __init__(self) -> None:
        self._processes: dict[int, BackgroundProcess] = {}
        self._next_id = 1

    @classmethod
    def get(cls) -> BackgroundRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def add(self, proc: BackgroundProcess) -> None:
        self._processes[proc.id] = proc

    def get_process(self, pid: int) -> BackgroundProcess | None:
        return self._processes.get(pid)

    def list_all(self) -> list[BackgroundProcess]:
        return list(self._processes.values())

    @property
    def count(self) -> int:
        return len(self._processes)

    def next_id(self) -> int:
        pid = self._next_id
        self._next_id += 1
        return pid

    @property
    def active_count(self) -> int:
        return sum(1 for p in self._processes.values() if p.is_running)


async def _collect_output(proc: BackgroundProcess) -> None:
    """Collect stdout/stderr from a background process."""

    async def _read_stream(stream: asyncio.StreamReader | None, chunks: list[str]) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            chunks.append(line.decode("utf-8", errors="replace"))

    await asyncio.gather(
        _read_stream(proc.process.stdout, proc.stdout_chunks),
        _read_stream(proc.process.stderr, proc.stderr_chunks),
    )
    await proc.process.wait()


class BackgroundCheckTool(Tool):
    """Check status, get output, or kill background processes."""

    @property
    def name(self) -> str:
        return "background_check"

    @property
    def description(self) -> str:
        return (
            "Check on background processes. Actions: 'status' (show all), "
            "'output' (get stdout/stderr for a process), 'kill' (terminate a process)."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH  # kill action is destructive

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "output", "kill"],
                    "description": "The action to perform",
                },
                "id": {
                    "type": "integer",
                    "description": "Process ID (required for 'output' and 'kill')",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "")
        registry = BackgroundRegistry.get()

        if action == "status":
            return self._status(registry)
        elif action == "output":
            return self._output(registry, arguments)
        elif action == "kill":
            return await self._kill(registry, arguments)

        return ToolResult.failure(f"Invalid action: {action}")

    def _status(self, registry: BackgroundRegistry) -> ToolResult:
        processes = registry.list_all()
        if not processes:
            return ToolResult.success("No background processes.")

        lines = []
        for proc in processes:
            status = "running" if proc.is_running else f"exited ({proc.returncode})"
            elapsed = f"{proc.elapsed_secs:.1f}s"
            lines.append(f"  [{proc.id}] {status} ({elapsed}) — {proc.command[:80]}")

        return ToolResult.success("\n".join(lines))

    def _output(self, registry: BackgroundRegistry, arguments: dict[str, Any]) -> ToolResult:
        pid = arguments.get("id")
        if pid is None:
            return ToolResult.failure("id is required for 'output' action")

        proc = registry.get_process(pid)
        if proc is None:
            return ToolResult.failure(f"No process with id {pid}")

        parts = []
        if proc.stdout:
            parts.append(f"STDOUT:\n{proc.stdout}")
        if proc.stderr:
            parts.append(f"STDERR:\n{proc.stderr}")
        if not parts:
            parts.append("(no output yet)")

        status = "running" if proc.is_running else f"exited ({proc.returncode})"
        header = f"Process {pid} [{status}]"
        return ToolResult.success(f"{header}\n" + "\n".join(parts))

    async def _kill(self, registry: BackgroundRegistry, arguments: dict[str, Any]) -> ToolResult:
        pid = arguments.get("id")
        if pid is None:
            return ToolResult.failure("id is required for 'kill' action")

        proc = registry.get_process(pid)
        if proc is None:
            return ToolResult.failure(f"No process with id {pid}")

        if not proc.is_running:
            return ToolResult.success(f"Process {pid} already exited ({proc.returncode})")

        proc.process.terminate()
        try:
            await asyncio.wait_for(proc.process.wait(), timeout=5)
        except TimeoutError:
            proc.process.kill()

        return ToolResult.success(f"Process {pid} terminated")
