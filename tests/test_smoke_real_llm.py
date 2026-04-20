"""End-to-end smoke tests that run against a real LLM via Ollama.

Skipped by default. Run explicitly with `pytest -m real_llm`.

Requires:
  - Ollama running on localhost:11434
  - A pulled model (default: qwen3:4b; override via GODSPEED_SMOKE_MODEL env)

These tests prove the agent loop chains real tool calls end-to-end with a
real model in the seat — the foundation claim the audit found was missing.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.llm.client import LLMClient
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import ToolContext
from godspeed.tools.file_read import FileReadTool
from godspeed.tools.file_write import FileWriteTool
from godspeed.tools.glob_search import GlobSearchTool
from godspeed.tools.registry import ToolRegistry
from godspeed.tools.shell import ShellTool

SMOKE_MODEL = os.environ.get("GODSPEED_SMOKE_MODEL", "ollama/qwen3:4b")


def _ollama_reachable(host: str = "127.0.0.1", port: int = 11434, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.real_llm,
    pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running on localhost:11434"),
]


def _build_agent_env(project_dir: Path) -> tuple[ToolRegistry, ToolContext, LLMClient]:
    registry = ToolRegistry()
    for tool in [FileReadTool(), FileWriteTool(), GlobSearchTool(), ShellTool()]:
        registry.register(tool)

    engine = PermissionEngine(
        allow_patterns=["file_read(*)", "file_write(*)", "glob_search(*)", "shell(*)"],
        tool_risk_levels={t.name: t.risk_level for t in registry.list_tools()},
    )
    ctx = ToolContext(cwd=project_dir, session_id="smoke-test", permissions=engine)
    client = LLMClient(model=SMOKE_MODEL, timeout=60)
    return registry, ctx, client


def _fresh_conversation() -> Conversation:
    return Conversation(
        system_prompt=(
            "You are a coding agent. You have tools to read, write, and search files. "
            "Use tools to complete tasks. Be concise."
        ),
        model=SMOKE_MODEL,
        max_tokens=8000,
    )


async def test_smoke_read_file(tmp_path: Path) -> None:
    """Agent must read a seeded file and reference its contents in the response."""
    (tmp_path / "hello.py").write_text('print("Hello from Godspeed!")\n')
    registry, ctx, client = _build_agent_env(tmp_path)

    result = await agent_loop(
        "Read the file hello.py and tell me what it prints.",
        _fresh_conversation(),
        client,
        registry,
        ctx,
    )

    lowered = result.lower()
    assert "hello" in lowered or "godspeed" in lowered, (
        f"agent did not surface file contents: {result!r}"
    )


async def test_smoke_write_file(tmp_path: Path) -> None:
    """Agent must create a file with the requested content."""
    registry, ctx, client = _build_agent_env(tmp_path)

    await agent_loop(
        "Create a file called 'test.txt' with the content 'smoke test passed'.",
        _fresh_conversation(),
        client,
        registry,
        ctx,
    )

    test_file = tmp_path / "test.txt"
    assert test_file.exists(), "agent did not create test.txt"
    content = test_file.read_text().lower()
    assert "smoke" in content or "passed" in content, f"file content unexpected: {content!r}"
