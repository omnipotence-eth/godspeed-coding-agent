"""GitHub tool — PR/issue workflow via the gh CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset(
    {
        "create_pr",
        "list_prs",
        "get_pr",
        "list_issues",
        "get_issue",
        "create_issue",
        "comment_issue",
        "comment_pr",
    }
)

_GH_TIMEOUT = 30


async def _run_gh(args: list[str], cwd: str) -> tuple[int, str, str]:
    """Run a gh CLI command via asyncio subprocess and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "gh",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=_GH_TIMEOUT)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        raise

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    return proc.returncode or 0, stdout, stderr


class GithubTool(Tool):
    """GitHub PR and issue operations via the gh CLI.

    Supports creating/listing/viewing PRs and issues, and adding comments.
    Requires the ``gh`` CLI to be installed and authenticated.
    """

    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return (
            "GitHub operations via the gh CLI: create/list/view pull requests and issues, "
            "add comments. Requires gh to be installed and authenticated."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(VALID_ACTIONS),
                    "description": "GitHub action to perform",
                },
                "title": {
                    "type": "string",
                    "description": "Title for create_pr or create_issue",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Body text for create_pr, create_issue, comment_issue, or comment_pr"
                    ),
                },
                "base": {
                    "type": "string",
                    "description": "Base branch for create_pr (default: main)",
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter state for list_prs / list_issues (default: open)",
                },
                "labels": {
                    "type": "string",
                    "description": "Comma-separated labels for list_issues or create_issue",
                },
                "number": {
                    "type": "integer",
                    "description": (
                        "PR or issue number for get_pr, get_issue, comment_pr, comment_issue"
                    ),
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

        if not shutil.which("gh"):
            return ToolResult.failure(
                "gh CLI is not installed. Install from https://cli.github.com/"
            )

        cwd = str(context.cwd)
        logger.info("github.execute action=%s cwd=%s", action, cwd)

        try:
            if action == "create_pr":
                return await self._create_pr(arguments, cwd)
            if action == "list_prs":
                return await self._list_prs(arguments, cwd)
            if action == "get_pr":
                return await self._get_pr(arguments, cwd)
            if action == "list_issues":
                return await self._list_issues(arguments, cwd)
            if action == "get_issue":
                return await self._get_issue(arguments, cwd)
            if action == "create_issue":
                return await self._create_issue(arguments, cwd)
            if action == "comment_issue":
                return await self._comment_issue(arguments, cwd)
            if action == "comment_pr":
                return await self._comment_pr(arguments, cwd)
        except TimeoutError:
            logger.warning("github.timeout action=%s", action)
            return ToolResult.failure(f"gh command timed out after {_GH_TIMEOUT}s")

        return ToolResult.failure(f"Unhandled action: {action}")

    # ------------------------------------------------------------------
    # Pull Request actions
    # ------------------------------------------------------------------

    async def _create_pr(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        title = arguments.get("title")
        if not title:
            return ToolResult.failure("title is required for create_pr")

        body = arguments.get("body", "")
        base = arguments.get("base", "main")

        args = [
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            base,
        ]
        returncode, stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh pr create failed: {stderr.strip()}")

        return ToolResult.success(stdout.strip())

    async def _list_prs(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        state = arguments.get("state", "open")

        args = [
            "pr",
            "list",
            "--state",
            state,
            "--json",
            "number,title,state,author",
        ]
        returncode, stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh pr list failed: {stderr.strip()}")

        return self._format_json(stdout)

    async def _get_pr(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        number = arguments.get("number")
        if number is None:
            return ToolResult.failure("number is required for get_pr")

        args = [
            "pr",
            "view",
            str(number),
            "--json",
            "number,title,state,body,additions,deletions,files",
        ]
        returncode, stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh pr view failed: {stderr.strip()}")

        return self._format_json(stdout)

    # ------------------------------------------------------------------
    # Issue actions
    # ------------------------------------------------------------------

    async def _list_issues(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        state = arguments.get("state", "open")
        labels = arguments.get("labels")

        args = [
            "issue",
            "list",
            "--state",
            state,
            "--json",
            "number,title,state,labels",
        ]
        if labels:
            args.extend(["--label", labels])

        returncode, stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh issue list failed: {stderr.strip()}")

        return self._format_json(stdout)

    async def _get_issue(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        number = arguments.get("number")
        if number is None:
            return ToolResult.failure("number is required for get_issue")

        args = [
            "issue",
            "view",
            str(number),
            "--json",
            "number,title,state,body,labels,comments",
        ]
        returncode, stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh issue view failed: {stderr.strip()}")

        return self._format_json(stdout)

    async def _create_issue(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        title = arguments.get("title")
        if not title:
            return ToolResult.failure("title is required for create_issue")

        body = arguments.get("body", "")
        labels = arguments.get("labels")

        args = [
            "issue",
            "create",
            "--title",
            title,
            "--body",
            body,
        ]
        if labels:
            args.extend(["--label", labels])

        returncode, stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh issue create failed: {stderr.strip()}")

        return ToolResult.success(stdout.strip())

    # ------------------------------------------------------------------
    # Comment actions
    # ------------------------------------------------------------------

    async def _comment_issue(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        number = arguments.get("number")
        body = arguments.get("body")
        if number is None:
            return ToolResult.failure("number is required for comment_issue")
        if not body:
            return ToolResult.failure("body is required for comment_issue")

        args = ["issue", "comment", str(number), "--body", body]
        returncode, _stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh issue comment failed: {stderr.strip()}")

        return ToolResult.success(f"Commented on issue #{number}")

    async def _comment_pr(self, arguments: dict[str, Any], cwd: str) -> ToolResult:
        number = arguments.get("number")
        body = arguments.get("body")
        if number is None:
            return ToolResult.failure("number is required for comment_pr")
        if not body:
            return ToolResult.failure("body is required for comment_pr")

        args = ["pr", "comment", str(number), "--body", body]
        returncode, _stdout, stderr = await _run_gh(args, cwd)
        if returncode != 0:
            return ToolResult.failure(f"gh pr comment failed: {stderr.strip()}")

        return ToolResult.success(f"Commented on PR #{number}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_json(raw: str) -> ToolResult:
        """Parse gh JSON output and re-format for readability."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ToolResult.success(raw.strip())

        return ToolResult.success(json.dumps(data, indent=2))
