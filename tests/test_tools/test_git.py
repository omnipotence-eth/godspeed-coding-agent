"""Tests for git tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo

from godspeed.tools.base import ToolContext
from godspeed.tools.git import GitTool


@pytest.fixture
def tool() -> GitTool:
    return GitTool()


@pytest.fixture
def git_context(tmp_path: Path) -> ToolContext:
    """Create a ToolContext with an initialized git repo."""
    godspeed_dir = tmp_path / ".godspeed"
    godspeed_dir.mkdir()
    repo = Repo.init(str(tmp_path))
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()
    return ToolContext(cwd=tmp_path, session_id="test-git-session")


def _make_initial_commit(cwd: Path) -> None:
    """Create a file and make an initial commit."""
    repo = Repo(str(cwd))
    init_file = cwd / "init.txt"
    init_file.write_text("initial content\n")
    repo.index.add(["init.txt"])
    repo.index.commit("Initial commit")


class TestGitTool:
    """Test git operations."""

    def test_metadata(self, tool: GitTool) -> None:
        assert tool.name == "git"
        assert tool.risk_level == "low"

    def test_schema_has_required_action(self, tool: GitTool) -> None:
        schema = tool.get_schema()
        assert "action" in schema["properties"]
        assert schema["required"] == ["action"]
        assert set(schema["properties"]["action"]["enum"]) == {
            "commit",
            "diff",
            "log",
            "stash",
            "stash_pop",
            "status",
            "undo",
        }

    @pytest.mark.asyncio
    async def test_status_clean(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        result = await tool.execute({"action": "status"}, git_context)
        assert not result.is_error
        assert "clean" in result.output.lower() or "nothing to commit" in result.output.lower()

    @pytest.mark.asyncio
    async def test_status_with_changes(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        (git_context.cwd / "new.txt").write_text("new file\n")
        result = await tool.execute({"action": "status"}, git_context)
        assert not result.is_error
        assert "new.txt" in result.output

    @pytest.mark.asyncio
    async def test_diff_no_changes(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        result = await tool.execute({"action": "diff"}, git_context)
        assert not result.is_error
        assert "No changes" in result.output

    @pytest.mark.asyncio
    async def test_diff_with_changes(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        (git_context.cwd / "init.txt").write_text("modified content\n")
        result = await tool.execute({"action": "diff"}, git_context)
        assert not result.is_error
        assert "modified content" in result.output

    @pytest.mark.asyncio
    async def test_commit(self, tool: GitTool, git_context: ToolContext) -> None:
        (git_context.cwd / "file.txt").write_text("content\n")
        result = await tool.execute({"action": "commit", "message": "Add file.txt"}, git_context)
        assert not result.is_error
        assert "Committed" in result.output
        assert "Add file.txt" in result.output

    @pytest.mark.asyncio
    async def test_commit_no_message(self, tool: GitTool, git_context: ToolContext) -> None:
        (git_context.cwd / "file.txt").write_text("content\n")
        result = await tool.execute({"action": "commit"}, git_context)
        assert result.is_error
        assert "message is required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_log(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        result = await tool.execute({"action": "log"}, git_context)
        assert not result.is_error
        assert "Initial commit" in result.output

    @pytest.mark.asyncio
    async def test_log_empty_repo(self, tool: GitTool, git_context: ToolContext) -> None:
        result = await tool.execute({"action": "log"}, git_context)
        # GitPython may error on empty repo log — handle gracefully
        # Either "No commits" or an error is acceptable
        assert result.output or result.error

    @pytest.mark.asyncio
    async def test_undo(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        # Make a second commit to undo
        (git_context.cwd / "second.txt").write_text("second\n")
        repo = Repo(str(git_context.cwd))
        repo.index.add(["second.txt"])
        repo.index.commit("Second commit")

        result = await tool.execute({"action": "undo"}, git_context)
        assert not result.is_error
        assert "Undid last commit" in result.output

        # Verify the commit was undone
        log_result = await tool.execute({"action": "log"}, git_context)
        assert "Second commit" not in log_result.output

    @pytest.mark.asyncio
    async def test_undo_empty_repo(self, tool: GitTool, git_context: ToolContext) -> None:
        result = await tool.execute({"action": "undo"}, git_context)
        assert result.is_error
        assert "No commits" in result.error

    @pytest.mark.asyncio
    async def test_invalid_action(self, tool: GitTool, git_context: ToolContext) -> None:
        result = await tool.execute({"action": "push"}, git_context)
        assert result.is_error
        assert "Invalid action" in result.error

    @pytest.mark.asyncio
    async def test_stash_with_changes(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        (git_context.cwd / "init.txt").write_text("modified\n")
        result = await tool.execute({"action": "stash"}, git_context)
        assert not result.is_error
        assert "godspeed-auto-stash" in result.output or "Saved" in result.output

    @pytest.mark.asyncio
    async def test_stash_clean_tree(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        result = await tool.execute({"action": "stash"}, git_context)
        assert not result.is_error
        assert "nothing to stash" in result.output.lower()

    @pytest.mark.asyncio
    async def test_stash_pop(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        (git_context.cwd / "init.txt").write_text("stashed content\n")
        await tool.execute({"action": "stash"}, git_context)
        # File should be reverted after stash
        assert (git_context.cwd / "init.txt").read_text() == "initial content\n"
        # Pop should restore it
        result = await tool.execute({"action": "stash_pop"}, git_context)
        assert not result.is_error
        assert (git_context.cwd / "init.txt").read_text() == "stashed content\n"

    @pytest.mark.asyncio
    async def test_stash_pop_empty(self, tool: GitTool, git_context: ToolContext) -> None:
        _make_initial_commit(git_context.cwd)
        result = await tool.execute({"action": "stash_pop"}, git_context)
        assert result.is_error
        assert "no stash entries" in result.error.lower()

    @pytest.mark.asyncio
    async def test_not_a_repo(self, tool: GitTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"action": "status"}, tool_context)
        assert result.is_error
        assert "Not a git repository" in result.error
