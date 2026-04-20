"""Multi-turn integration tests that run against a real LLM via Ollama.

Skipped by default. Run explicitly with `pytest -m real_llm`.

Purpose: prove the agent LOOP — not model IQ — correctly chains
``file_read → file_edit → verify`` with a real model in the seat. Failures
here indicate agent-layer bugs (prompt, tool dispatch, conversation wiring),
not a weak model choice.

Requires:
  - Ollama running on localhost:11434
  - A coder-competent model pulled. Default: ``qwen3-coder:latest`` (the
    4B non-coder model fails at this task, which is itself useful
    signal). Override via ``GODSPEED_SMOKE_MODEL`` env.
"""

from __future__ import annotations

import ast
import os
import socket
from pathlib import Path

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.llm.client import LLMClient
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import ToolContext
from godspeed.tools.file_edit import FileEditTool
from godspeed.tools.file_read import FileReadTool
from godspeed.tools.file_write import FileWriteTool
from godspeed.tools.glob_search import GlobSearchTool
from godspeed.tools.registry import ToolRegistry
from godspeed.tools.shell import ShellTool

SMOKE_MODEL = os.environ.get("GODSPEED_SMOKE_MODEL", "ollama/qwen3-coder:latest")


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
    for tool in [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GlobSearchTool(),
        ShellTool(),
    ]:
        registry.register(tool)

    allow = [
        "file_read(*)",
        "file_write(*)",
        "file_edit(*)",
        "glob_search(*)",
        "shell(*)",
    ]
    engine = PermissionEngine(
        allow_patterns=allow,
        tool_risk_levels={t.name: t.risk_level for t in registry.list_tools()},
    )
    ctx = ToolContext(cwd=project_dir, session_id="integration-real", permissions=engine)
    client = LLMClient(model=SMOKE_MODEL, timeout=90)
    return registry, ctx, client


def _conversation() -> Conversation:
    return Conversation(
        system_prompt=(
            "You are a coding agent. Use file_read to inspect code, file_edit "
            "to make targeted changes, and shell to verify. Be concise — make "
            "the smallest edit that fixes the problem."
        ),
        model=SMOKE_MODEL,
        max_tokens=8000,
    )


# A broken Python module with a clear syntax error the agent must locate + fix.
# Missing comma in the list literal — `ast.parse` rejects it until fixed.
BROKEN_SOURCE = '''"""Helpers for computing totals."""


def add(a, b):
    return a + b


def total(values):
    return sum(values)


NUMBERS = [1, 2 3, 4]
'''

FIXED_SOURCE_MARKER = "[1, 2, 3, 4]"


async def test_multi_turn_read_edit_verify(tmp_path: Path) -> None:
    """Agent must locate a syntax error, edit the file, and leave it parseable.

    Proves the agent loop chains at least two tool calls (read → edit) with
    a real model. We do not assert the exact tool sequence — just that the
    end state is correct, which is what matters for downstream fine-tuning.
    """
    broken = tmp_path / "totals.py"
    broken.write_text(BROKEN_SOURCE, encoding="utf-8")

    with pytest.raises(SyntaxError):
        ast.parse(broken.read_text(encoding="utf-8"))

    registry, ctx, client = _build_agent_env(tmp_path)

    await agent_loop(
        (
            "The file totals.py has a syntax error on the NUMBERS line. "
            "Read the file, identify the missing comma, and fix it with the "
            "smallest possible edit. Do not change anything else."
        ),
        _conversation(),
        client,
        registry,
        ctx,
    )

    fixed = broken.read_text(encoding="utf-8")
    ast.parse(fixed)
    assert FIXED_SOURCE_MARKER in fixed, (
        f"agent produced a parseable file but not the expected fix; got:\n{fixed}"
    )
