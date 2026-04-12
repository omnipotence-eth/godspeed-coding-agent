"""Tests for GitHub tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.github import GithubTool


@pytest.fixture
def tool() -> GithubTool:
    return GithubTool()


def _mock_run_gh(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> AsyncMock:
    """Build an AsyncMock that returns (returncode, stdout, stderr) tuple."""
    return AsyncMock(return_value=(returncode, stdout, stderr))


class TestGithubToolMetadata:
    """Basic tool metadata checks."""

    def test_name(self, tool: GithubTool) -> None:
        assert tool.name == "github"

    def test_risk_level(self, tool: GithubTool) -> None:
        assert tool.risk_level == "high"

    def test_schema_has_action(self, tool: GithubTool) -> None:
        schema = tool.get_schema()
        assert "action" in schema["properties"]
        assert schema["required"] == ["action"]


class TestGhNotInstalled:
    """gh CLI missing — graceful error."""

    @pytest.mark.asyncio
    async def test_gh_not_installed(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value=None):
            result = await tool.execute({"action": "list_prs"}, tool_context)

        assert result.is_error
        assert "gh CLI is not installed" in (result.error or "")


class TestInvalidAction:
    """Unknown action rejected."""

    @pytest.mark.asyncio
    async def test_invalid_action(self, tool: GithubTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"action": "delete_repo"}, tool_context)
        assert result.is_error
        assert "Invalid action" in (result.error or "")


class TestCreatePr:
    """create_pr action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        pr_url = "https://github.com/o/r/pull/42"
        mock = _mock_run_gh(stdout=pr_url)

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute(
                {"action": "create_pr", "title": "feat", "body": "desc", "base": "main"},
                tool_context,
            )

        assert not result.is_error
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_missing_title(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"):
            result = await tool.execute({"action": "create_pr"}, tool_context)

        assert result.is_error
        assert "title is required" in (result.error or "")


class TestListPrs:
    """list_prs action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        data = [{"number": 1, "title": "pr1", "state": "open", "author": {"login": "u"}}]
        mock = _mock_run_gh(stdout=json.dumps(data))

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute({"action": "list_prs"}, tool_context)

        assert not result.is_error
        assert "pr1" in result.output


class TestGetPr:
    """get_pr action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        data = {"number": 5, "title": "fix bug", "state": "open", "body": "details"}
        mock = _mock_run_gh(stdout=json.dumps(data))

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute({"action": "get_pr", "number": 5}, tool_context)

        assert not result.is_error
        assert "fix bug" in result.output

    @pytest.mark.asyncio
    async def test_missing_number(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"):
            result = await tool.execute({"action": "get_pr"}, tool_context)

        assert result.is_error
        assert "number is required" in (result.error or "")


class TestListIssues:
    """list_issues action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        data = [{"number": 10, "title": "bug", "state": "open", "labels": []}]
        mock = _mock_run_gh(stdout=json.dumps(data))

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute(
                {"action": "list_issues", "state": "open", "labels": "bug"},
                tool_context,
            )

        assert not result.is_error
        assert "bug" in result.output


class TestGetIssue:
    """get_issue action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        data = {"number": 3, "title": "feature req", "state": "open", "body": "want this"}
        mock = _mock_run_gh(stdout=json.dumps(data))

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute({"action": "get_issue", "number": 3}, tool_context)

        assert not result.is_error
        assert "feature req" in result.output

    @pytest.mark.asyncio
    async def test_missing_number(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"):
            result = await tool.execute({"action": "get_issue"}, tool_context)

        assert result.is_error
        assert "number is required" in (result.error or "")


class TestCreateIssue:
    """create_issue action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        issue_url = "https://github.com/o/r/issues/7"
        mock = _mock_run_gh(stdout=issue_url)

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute(
                {"action": "create_issue", "title": "new", "body": "desc", "labels": "enhancement"},
                tool_context,
            )

        assert not result.is_error
        assert "7" in result.output

    @pytest.mark.asyncio
    async def test_missing_title(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"):
            result = await tool.execute({"action": "create_issue"}, tool_context)

        assert result.is_error
        assert "title is required" in (result.error or "")


class TestCommentIssue:
    """comment_issue action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        mock = _mock_run_gh(stdout="")

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute(
                {"action": "comment_issue", "number": 10, "body": "LGTM"},
                tool_context,
            )

        assert not result.is_error
        assert "Commented on issue #10" in result.output

    @pytest.mark.asyncio
    async def test_missing_body(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"):
            result = await tool.execute({"action": "comment_issue", "number": 10}, tool_context)

        assert result.is_error
        assert "body is required" in (result.error or "")


class TestCommentPr:
    """comment_pr action."""

    @pytest.mark.asyncio
    async def test_success(self, tool: GithubTool, tool_context: ToolContext) -> None:
        mock = _mock_run_gh(stdout="")

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute(
                {"action": "comment_pr", "number": 5, "body": "Nice work"},
                tool_context,
            )

        assert not result.is_error
        assert "Commented on PR #5" in result.output

    @pytest.mark.asyncio
    async def test_missing_body(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"):
            result = await tool.execute({"action": "comment_pr", "number": 5}, tool_context)

        assert result.is_error
        assert "body is required" in (result.error or "")


class TestSubprocessErrors:
    """Timeout and failure cases."""

    @pytest.mark.asyncio
    async def test_timeout(self, tool: GithubTool, tool_context: ToolContext) -> None:
        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", AsyncMock(side_effect=TimeoutError)),
        ):
            result = await tool.execute({"action": "list_prs"}, tool_context)

        assert result.is_error
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, tool: GithubTool, tool_context: ToolContext) -> None:
        mock = _mock_run_gh(stderr="not found", returncode=1)

        with (
            patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh"),
            patch("godspeed.tools.github._run_gh", mock),
        ):
            result = await tool.execute({"action": "list_prs"}, tool_context)

        assert result.is_error
        assert "not found" in (result.error or "")
