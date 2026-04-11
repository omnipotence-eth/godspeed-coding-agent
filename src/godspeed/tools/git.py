"""Git tool — git operations via GitPython."""

from __future__ import annotations

import logging
from typing import Any

from git import InvalidGitRepositoryError, Repo
from git.exc import GitCommandError

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({"status", "diff", "commit", "log", "undo"})


class GitTool(Tool):
    """Git operations: status, diff, commit, log, undo.

    Uses GitPython to interact with the repository. The undo action
    reverts the last commit (soft reset, keeping changes staged).
    """

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Run git operations: status, diff, commit, log, undo. "
            "Commit stages all changes and commits with the given message. "
            "Undo reverts the last commit (soft reset, changes remain staged)."
        )

    @property
    def risk_level(self) -> RiskLevel:
        # Commit/undo are write operations but LOW since they're reversible
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(VALID_ACTIONS),
                    "description": "Git action to perform",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (required for 'commit' action)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "")
        if action not in VALID_ACTIONS:
            return ToolResult.failure(
                f"Invalid action: '{action}'. Must be one of: {', '.join(sorted(VALID_ACTIONS))}"
            )

        try:
            repo = Repo(str(context.cwd), search_parent_directories=True)
        except InvalidGitRepositoryError:
            return ToolResult.failure("Not a git repository (or any parent)")

        logger.info("git.execute action=%s cwd=%s", action, context.cwd)

        if action == "status":
            return await self._status(repo)
        if action == "diff":
            return await self._diff(repo)
        if action == "commit":
            return await self._commit(repo, arguments.get("message"))
        if action == "log":
            return await self._log(repo)
        if action == "undo":
            return await self._undo(repo)

        return ToolResult.failure(f"Unhandled action: {action}")

    async def _status(self, repo: Repo) -> ToolResult:
        """Show working tree status."""
        try:
            output = repo.git.status()
        except GitCommandError as exc:
            return ToolResult.failure(f"git status failed: {exc}")
        return ToolResult.success(output)

    async def _diff(self, repo: Repo) -> ToolResult:
        """Show staged and unstaged changes."""
        try:
            staged = repo.git.diff("--cached")
            unstaged = repo.git.diff()
        except GitCommandError as exc:
            return ToolResult.failure(f"git diff failed: {exc}")

        parts: list[str] = []
        if staged:
            parts.append(f"=== Staged Changes ===\n{staged}")
        if unstaged:
            parts.append(f"=== Unstaged Changes ===\n{unstaged}")
        if not parts:
            return ToolResult.success("No changes")

        return ToolResult.success("\n\n".join(parts))

    async def _commit(self, repo: Repo, message: str | None) -> ToolResult:
        """Stage all changes and commit."""
        if not message:
            return ToolResult.failure("message is required for commit action")

        try:
            # Stage only tracked files that have been modified or deleted
            repo.git.add("-u")

            # Check if there's anything to commit
            has_head = True
            try:
                repo.head.commit  # noqa: B018
            except ValueError:
                has_head = False  # No HEAD yet — initial commit is fine

            if has_head and not repo.is_dirty(untracked_files=False):
                return ToolResult.failure("Nothing to commit (working tree clean)")

            commit = repo.index.commit(message)
            short_sha = commit.hexsha[:8]
            logger.info("git.commit sha=%s message=%r", short_sha, message)
            return ToolResult.success(
                f"Committed {short_sha}: {message}\n"
                "Only tracked files staged. Use shell to `git add <file>` for new files."
            )
        except GitCommandError as exc:
            return ToolResult.failure(f"git commit failed: {exc}")

    async def _log(self, repo: Repo) -> ToolResult:
        """Show recent commit log (last 10)."""
        try:
            output = repo.git.log("--oneline", "-10", "--no-color")
        except GitCommandError as exc:
            return ToolResult.failure(f"git log failed: {exc}")
        if not output:
            return ToolResult.success("No commits yet")
        return ToolResult.success(output)

    async def _undo(self, repo: Repo) -> ToolResult:
        """Undo the last commit (soft reset, keeps changes staged)."""
        try:
            repo.head.commit  # noqa: B018 — verify HEAD exists
        except ValueError:
            return ToolResult.failure("No commits to undo")

        try:
            repo.git.reset("--soft", "HEAD~1")
            logger.info("git.undo soft reset HEAD~1")
            return ToolResult.success("Undid last commit (changes remain staged)")
        except GitCommandError as exc:
            return ToolResult.failure(f"git undo failed: {exc}")
