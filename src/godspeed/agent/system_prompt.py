"""System prompt assembly for the agent."""

from __future__ import annotations

import logging
from pathlib import Path

from godspeed.tools.base import Tool

logger = logging.getLogger(__name__)

CORE_PROMPT = """\
You are Godspeed, a security-first coding agent. You help users with software \
engineering tasks by reading, writing, and editing code in their project.

## Capabilities
- Read, write, and edit files
- Run shell commands
- Search for files and content (glob, grep)
- Git operations (status, diff, commit, undo)

## Guidelines
- Read files before modifying them — never guess at content
- Use search tools to understand the codebase before making changes
- Make minimal, focused changes — don't over-engineer
- Test your changes when possible (run tests, lint)
- Never hardcode secrets — use environment variables
- Explain what you're doing concisely

## Tool Usage
- Use file_read to examine files before editing
- Use file_edit (search/replace) for precise modifications
- Use file_write only for new files
- Use grep_search to find code patterns
- Use glob_search to find files by name
- Use shell to run commands (tests, linters, builds)
- Use git for version control operations

## Safety
- Never execute destructive commands without explicit user confirmation
- Never modify files outside the project directory
- Never expose secrets, API keys, or credentials
- If unsure, ask the user before proceeding
"""


PLAN_MODE_PROMPT = """\

## Plan Mode Active
You are in PLAN MODE. You may only explore and plan — do NOT write files, \
run commands, or make changes. Use read-only tools (file_read, grep_search, \
glob_search, repo_map) to understand the codebase. Propose a plan but do not \
execute it.
"""


def build_system_prompt(
    tools: list[Tool],
    project_instructions: str | None = None,
    cwd: Path | None = None,
    plan_mode: bool = False,
) -> str:
    """Assemble the full system prompt.

    Combines:
    1. Core agent prompt (role, guidelines, safety)
    2. Project instructions from GODSPEED.md (if present)
    3. Available tool descriptions
    4. Working directory context
    """
    parts = [CORE_PROMPT]

    if plan_mode:
        parts.append(PLAN_MODE_PROMPT)

    if cwd:
        parts.append(f"\n## Working Directory\n{cwd}\n")

    if project_instructions:
        parts.append(f"\n## Project Instructions\n{project_instructions}\n")

    if tools:
        tool_descriptions = "\n## Available Tools\n"
        for tool in tools:
            tool_descriptions += (
                f"\n### {tool.name}\n{tool.description}\nRisk level: {tool.risk_level}\n"
            )
        parts.append(tool_descriptions)

    return "\n".join(parts)


# Note: load_project_instructions lives in context/project_instructions.py
# which supports walk-up-tree loading for nested projects.
