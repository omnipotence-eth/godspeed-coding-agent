"""Build a fully-registered ``ToolRegistry`` with all 21 canonical Godspeed tools.

Mirrors ``data/build_godspeed_tools.py`` in the ml-lab experiment but returns
live tool instances plus the registry. Uses MagicMock for external services
that tools need (coordinator for spawn_agent, index for code_search).

The 21 tools (matches ``data/godspeed_tools.json``):
    file_read, file_write, file_edit, diff_apply,
    glob_search, grep_search, code_search, repo_map,
    shell, test_runner, verify, background_check,
    git, github,
    web_search, web_fetch, image_read, pdf_read, notebook_edit,
    tasks, spawn_agent
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Canonical ordering — 21 tools grouped by category
TOOL_CATEGORIES: dict[str, list[str]] = {
    "file_ops": ["file_read", "file_write", "file_edit", "diff_apply"],
    "search": ["glob_search", "grep_search", "code_search", "repo_map"],
    "execution": ["shell", "test_runner", "verify", "background_check"],
    "vcs": ["git", "github"],
    "web_media": ["web_search", "web_fetch", "image_read", "pdf_read", "notebook_edit"],
    "coordination": ["tasks", "spawn_agent"],
}

ALL_TOOLS: tuple[str, ...] = tuple(t for tools in TOOL_CATEGORIES.values() for t in tools)

# Tools that require real filesystem/git/shell sandbox to execute deterministically.
SANDBOX_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "file_read",
        "file_write",
        "file_edit",
        "diff_apply",
        "glob_search",
        "grep_search",
        "repo_map",
        "shell",
        "test_runner",
        "verify",
        "background_check",
        "git",
        "notebook_edit",
        "tasks",
    }
)

# Tools whose output we stub from a fixture library (no real service calls).
FIXTURE_BACKED_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "web_fetch",
        "github",
        "pdf_read",
        "image_read",
        "code_search",
        "spawn_agent",
    }
)


def build_registry() -> ToolRegistry:
    """Return a ``ToolRegistry`` with all 21 canonical tools registered."""
    from godspeed.agent.coordinator import SpawnAgentTool
    from godspeed.tools.background import BackgroundCheckTool
    from godspeed.tools.code_search import CodeSearchTool
    from godspeed.tools.diff_apply import DiffApplyTool
    from godspeed.tools.file_edit import FileEditTool
    from godspeed.tools.file_read import FileReadTool
    from godspeed.tools.file_write import FileWriteTool
    from godspeed.tools.git import GitTool
    from godspeed.tools.github import GithubTool
    from godspeed.tools.glob_search import GlobSearchTool
    from godspeed.tools.grep_search import GrepSearchTool
    from godspeed.tools.image_read import ImageReadTool
    from godspeed.tools.notebook import NotebookEditTool
    from godspeed.tools.pdf_read import PdfReadTool
    from godspeed.tools.repo_map import RepoMapTool
    from godspeed.tools.shell import ShellTool
    from godspeed.tools.tasks import TaskStore, TaskTool
    from godspeed.tools.test_runner import TestRunnerTool
    from godspeed.tools.verify import VerifyTool
    from godspeed.tools.web_fetch import WebFetchTool
    from godspeed.tools.web_search import WebSearchTool

    registry = ToolRegistry()

    instances = [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        DiffApplyTool(),
        GlobSearchTool(),
        GrepSearchTool(),
        CodeSearchTool(MagicMock()),  # real index not needed; executor stubs dispatch
        RepoMapTool(),
        ShellTool(),
        TestRunnerTool(),
        VerifyTool(),
        BackgroundCheckTool(),
        GitTool(),
        GithubTool(),
        WebSearchTool(),
        WebFetchTool(),
        ImageReadTool(),
        PdfReadTool(),
        NotebookEditTool(),
        TaskTool(TaskStore()),
        SpawnAgentTool(MagicMock()),  # coordinator stub; executor stubs dispatch
    ]

    for tool in instances:
        registry.register(tool)

    registered = {t.name for t in registry.list_tools()}
    expected = set(ALL_TOOLS)
    missing = expected - registered
    extra = registered - expected
    if missing or extra:
        msg = f"registry mismatch. missing={sorted(missing)} extra={sorted(extra)}"
        raise RuntimeError(msg)

    logger.debug("registered %d tools: %s", len(registry.list_tools()), sorted(registered))
    return registry


def get_tool_schemas() -> list[dict[str, Any]]:
    """Shortcut: build registry and return its 21-tool OpenAI-format schema list."""
    return build_registry().get_schemas()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reg = build_registry()
    for t in reg.list_tools():
        cat = next(
            (c for c, names in TOOL_CATEGORIES.items() if t.name in names),
            "unknown",
        )
        logger.info("%-18s [%s]  risk=%s", t.name, cat, t.risk_level)
    logger.info("total: %d tools", len(reg.list_tools()))
