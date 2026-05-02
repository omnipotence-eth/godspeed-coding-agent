"""Tests for godspeed.tools.github."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.tools.github import GithubTool, _run_gh


class TestGithubToolMetadata:
    def test_name(self):
        tool = GithubTool()
        assert tool.name == "github"

    def test_risk_level(self):
        tool = GithubTool()
        assert tool.risk_level.value == "high"

    def test_description_contains_keywords(self):
        tool = GithubTool()
        desc = tool.description.lower()
        assert "github" in desc or "gh" in desc

    def test_get_schema(self):
        tool = GithubTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert "action" in schema["required"]


class TestRunGh:
    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        """Test successful gh command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate.return_value = (b'{"pr": 1}', b"")
            proc.returncode = 0
            mock_exec.return_value = proc

            returncode, stdout, _stderr = await _run_gh(["pr", "list"], str(tmp_path))
            assert returncode == 0
            assert "pr" in stdout

    @pytest.mark.asyncio
    async def test_failure(self, tmp_path):
        """Test gh command failure."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate.return_value = (b"", b"error message")
            proc.returncode = 1
            mock_exec.return_value = proc

            returncode, _stdout, stderr = await _run_gh(["pr", "list"], str(tmp_path))
            assert returncode == 1
            assert "error" in stderr

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        """Test gh command timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate.side_effect = TimeoutError()
            proc.kill = MagicMock()
            mock_exec.return_value = proc

            with pytest.raises(asyncio.TimeoutError):
                await _run_gh(["pr", "list"], str(tmp_path))


class TestGithubToolExecute:
    @pytest.mark.asyncio
    async def test_missing_action(self):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = "/project"
        result = await tool.execute({}, context)
        assert result.is_error is True
        assert "action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = "/project"
        result = await tool.execute({"action": "invalid_action"}, context)
        assert result.is_error is True
        assert "invalid" in result.error.lower() or "unhandled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_pr_missing_title(self):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = "/project"
        result = await tool.execute({"action": "create_pr"}, context)
        assert result.is_error is True
        assert "title" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_pr_missing_number(self):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = "/project"
        result = await tool.execute({"action": "get_pr"}, context)
        assert result.is_error is True
        assert "number" in result.error.lower()

    @pytest.mark.asyncio
    async def test_successful_list_prs(self, tmp_path):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = str(tmp_path)

        with patch("godspeed.tools.github._run_gh") as mock_gh:
            mock_gh.return_value = (0, '[{"number": 1, "title": "Test PR"}]', "")

            result = await tool.execute({"action": "list_prs"}, context)
            assert result.is_error is False
            assert "PR" in result.output or "1" in result.output

    @pytest.mark.asyncio
    async def test_gh_failure(self, tmp_path):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = str(tmp_path)

        with patch("godspeed.tools.github._run_gh") as mock_gh:
            mock_gh.return_value = (1, "", "gh not found")

            result = await tool.execute({"action": "list_prs"}, context)
            assert result.is_error is True
            assert "failed" in result.error.lower() or "gh" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        tool = GithubTool()
        context = MagicMock()
        context.cwd = str(tmp_path)

        with patch("godspeed.tools.github._run_gh") as mock_gh:
            mock_gh.side_effect = TimeoutError()

            result = await tool.execute({"action": "list_prs"}, context)
            assert result.is_error is True
            assert "timed out" in result.error.lower()
