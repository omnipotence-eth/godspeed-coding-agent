"""Auto-commit workflow — generates commit messages and commits via GitPython."""

from __future__ import annotations

import logging
from pathlib import Path

import git

from godspeed.llm.client import LLMClient
from godspeed.tools.base import ToolResult

logger = logging.getLogger(__name__)

COMMIT_SYSTEM_PROMPT = (
    "You are a git commit message generator. Write a conventional commit message "
    "for these changes. Format: type(scope): description. One line, under 72 chars. "
    "Return ONLY the commit message."
)

FALLBACK_MESSAGE = "chore: auto-commit from godspeed"


async def generate_commit_message(changes: list[str], llm_client: LLMClient) -> str:
    """Generate a conventional commit message from a list of change descriptions.

    Makes a cheap LLM call to produce a one-line commit message. Falls back to a
    generic message on any failure.

    Args:
        changes: List of change descriptions (e.g. "edited src/main.py").
        llm_client: LLM client for the generation call.

    Returns:
        A conventional commit message string.
    """
    try:
        messages = [
            {"role": "system", "content": COMMIT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(changes)},
        ]
        response = await llm_client.chat(messages=messages)
        message = response.content.strip()
        if message:
            return message
        logger.warning("LLM returned empty commit message, using fallback")
        return FALLBACK_MESSAGE
    except Exception as exc:
        logger.warning("Failed to generate commit message error=%s, using fallback", exc)
        return FALLBACK_MESSAGE


async def auto_commit(repo_path: Path, message: str) -> ToolResult:
    """Stage tracked changes and commit with attribution.

    Uses GitPython to stage modified tracked files and create a commit with
    Godspeed co-author attribution.

    Args:
        repo_path: Path to (or inside) the git repository.
        message: Commit message to use.

    Returns:
        ToolResult with success or failure info.
    """
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        repo.git.add("-u")
        commit = repo.index.commit(f"{message}\n\nCo-Authored-By: Godspeed <noreply@godspeed.dev>")
        sha = commit.hexsha[:8]
        logger.info("Auto-committed sha=%s message=%s", sha, message)
        return ToolResult.success(f"Auto-committed: {sha} {message}")
    except Exception as exc:
        logger.warning("Auto-commit failed error=%s", exc)
        return ToolResult.failure(str(exc))
