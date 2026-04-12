"""Tests for the GitHub PR/issue workflow tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.github import GithubTool


@pytest.fixture
def tool() -> GithubTool:
    return GithubTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


def _make_process(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> AsyncMock:
    """Build a mock asyncio subprocess that returns the given outputs."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


# ------------------------------------------------------------------
# Tool metadata
# ------------------------------------------------------------------


def test_name_and_risk(tool: GithubTool) -> None:
    assert tool.name == "github"
    assert tool.risk_level == RiskLevel.HIGH


def test_schema_has_required_action(tool: GithubTool) -> None:
    schema = tool.get_schema()
    assert schema["required"] == ["action"]
    assert "action" in schema["properties"]
    actions = schema["properties"]["action"]["enum"]
    assert "create_pr" in actions
    assert "list_issues" in actions


# ------------------------------------------------------------------
# gh not installed
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value=None)
async def test_gh_not_installed(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "list_prs"}, ctx)
    assert result.is_error
    assert "gh CLI is not installed" in (result.error or "")


# ------------------------------------------------------------------
# Invalid action
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
async def test_invalid_action(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "nope"}, ctx)
    assert result.is_error
    assert "Invalid action" in (result.error or "")


# ------------------------------------------------------------------
# create_pr
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_create_pr_success(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="https://github.com/owner/repo/pull/42")
    result = await tool.execute(
        {"action": "create_pr", "title": "feat: add widget", "body": "Adds widget support"},
        ctx,
    )
    assert not result.is_error
    assert "pull/42" in result.output

    # Verify the gh args
    call_args = mock_exec.call_args
    positional = call_args[0]
    assert positional[0] == "gh"
    assert "pr" in positional
    assert "create" in positional
    assert "--title" in positional
    assert "feat: add widget" in positional
    assert "--base" in positional
    assert "main" in positional


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
async def test_create_pr_missing_title(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "create_pr"}, ctx)
    assert result.is_error
    assert "title is required" in (result.error or "")


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_create_pr_custom_base(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="https://github.com/o/r/pull/1")
    result = await tool.execute(
        {"action": "create_pr", "title": "fix: bug", "base": "develop"},
        ctx,
    )
    assert not result.is_error
    positional = mock_exec.call_args[0]
    assert "develop" in positional


# ------------------------------------------------------------------
# list_prs
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_list_prs(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    payload = [
        {"number": 1, "title": "PR one", "state": "OPEN", "author": {"login": "dev"}},
        {"number": 2, "title": "PR two", "state": "OPEN", "author": {"login": "dev"}},
    ]
    mock_exec.return_value = _make_process(stdout=json.dumps(payload))
    result = await tool.execute({"action": "list_prs"}, ctx)
    assert not result.is_error
    data = json.loads(result.output)
    assert len(data) == 2
    assert data[0]["number"] == 1


# ------------------------------------------------------------------
# get_pr
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_get_pr(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    payload = {
        "number": 5,
        "title": "feat: stuff",
        "state": "OPEN",
        "body": "desc",
        "additions": 10,
        "deletions": 3,
        "files": [{"path": "a.py"}],
    }
    mock_exec.return_value = _make_process(stdout=json.dumps(payload))
    result = await tool.execute({"action": "get_pr", "number": 5}, ctx)
    assert not result.is_error
    data = json.loads(result.output)
    assert data["number"] == 5
    assert data["additions"] == 10


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
async def test_get_pr_missing_number(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "get_pr"}, ctx)
    assert result.is_error
    assert "number is required" in (result.error or "")


# ------------------------------------------------------------------
# list_issues
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_list_issues(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    payload = [
        {"number": 10, "title": "Bug report", "state": "OPEN", "labels": []},
    ]
    mock_exec.return_value = _make_process(stdout=json.dumps(payload))
    result = await tool.execute({"action": "list_issues", "state": "open"}, ctx)
    assert not result.is_error
    data = json.loads(result.output)
    assert data[0]["number"] == 10


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_list_issues_with_labels(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="[]")
    result = await tool.execute(
        {"action": "list_issues", "labels": "bug,priority"},
        ctx,
    )
    assert not result.is_error
    positional = mock_exec.call_args[0]
    assert "--label" in positional
    assert "bug,priority" in positional


# ------------------------------------------------------------------
# get_issue
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_get_issue(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    payload = {
        "number": 7,
        "title": "Feature request",
        "state": "OPEN",
        "body": "Please add X",
        "labels": [{"name": "enhancement"}],
        "comments": [],
    }
    mock_exec.return_value = _make_process(stdout=json.dumps(payload))
    result = await tool.execute({"action": "get_issue", "number": 7}, ctx)
    assert not result.is_error
    data = json.loads(result.output)
    assert data["number"] == 7
    assert data["labels"][0]["name"] == "enhancement"


# ------------------------------------------------------------------
# create_issue
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_create_issue(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="https://github.com/owner/repo/issues/15")
    result = await tool.execute(
        {"action": "create_issue", "title": "New bug", "body": "It breaks"},
        ctx,
    )
    assert not result.is_error
    assert "issues/15" in result.output


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
async def test_create_issue_missing_title(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "create_issue"}, ctx)
    assert result.is_error
    assert "title is required" in (result.error or "")


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_create_issue_with_labels(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="https://github.com/o/r/issues/20")
    result = await tool.execute(
        {"action": "create_issue", "title": "Bug", "labels": "bug,urgent"},
        ctx,
    )
    assert not result.is_error
    positional = mock_exec.call_args[0]
    assert "--label" in positional
    assert "bug,urgent" in positional


# ------------------------------------------------------------------
# comment_issue
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_comment_issue(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="")
    result = await tool.execute(
        {"action": "comment_issue", "number": 3, "body": "LGTM"},
        ctx,
    )
    assert not result.is_error
    assert "Commented on issue #3" in result.output


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
async def test_comment_issue_missing_body(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "comment_issue", "number": 3}, ctx)
    assert result.is_error
    assert "body is required" in (result.error or "")


# ------------------------------------------------------------------
# comment_pr
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_comment_pr(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(stdout="")
    result = await tool.execute(
        {"action": "comment_pr", "number": 9, "body": "Looks good"},
        ctx,
    )
    assert not result.is_error
    assert "Commented on PR #9" in result.output


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
async def test_comment_pr_missing_number(
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    result = await tool.execute({"action": "comment_pr", "body": "hi"}, ctx)
    assert result.is_error
    assert "number is required" in (result.error or "")


# ------------------------------------------------------------------
# gh CLI failure (nonzero exit)
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_gh_cli_failure(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    mock_exec.return_value = _make_process(returncode=1, stderr="not authenticated")
    result = await tool.execute({"action": "list_prs"}, ctx)
    assert result.is_error
    assert "not authenticated" in (result.error or "")


# ------------------------------------------------------------------
# Timeout
# ------------------------------------------------------------------


@pytest.mark.asyncio
@patch("godspeed.tools.github.shutil.which", return_value="/usr/bin/gh")
@patch("godspeed.tools.github.asyncio.create_subprocess_exec")
async def test_timeout(
    mock_exec: AsyncMock,
    _mock_which: MagicMock,
    tool: GithubTool,
    ctx: ToolContext,
) -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(side_effect=TimeoutError)
    proc.kill = MagicMock()
    mock_exec.return_value = proc
    result = await tool.execute({"action": "list_prs"}, ctx)
    assert result.is_error
    assert "timed out" in (result.error or "")
