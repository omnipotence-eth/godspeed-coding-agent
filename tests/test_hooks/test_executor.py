"""Tests for hook executor."""

from __future__ import annotations

import sys
from pathlib import Path

from godspeed.hooks.config import HookDefinition
from godspeed.hooks.executor import HookExecutor


def _make_executor(
    hooks: list[HookDefinition],
    tmp_path: Path,
) -> HookExecutor:
    return HookExecutor(hooks=hooks, cwd=tmp_path, session_id="test-session-123")


class TestPreToolHooks:
    """Test pre_tool_call hook execution."""

    def test_allows_when_no_hooks(self, tmp_path: Path) -> None:
        executor = _make_executor([], tmp_path)
        assert executor.run_pre_tool("shell") is True

    def test_allows_on_success(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=f"{sys.executable} -c \"print('ok')\"",
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is True

    def test_blocks_on_failure(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "raise SystemExit(1)"',
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is False

    def test_tool_filter_matches(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "raise SystemExit(1)"',
            tools=["shell"],
        )
        executor = _make_executor([hook], tmp_path)
        # Should block shell
        assert executor.run_pre_tool("shell") is False
        # Should allow file_read (not in tool list)
        assert executor.run_pre_tool("file_read") is True

    def test_tool_filter_none_matches_all(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "raise SystemExit(1)"',
            tools=None,
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is False
        assert executor.run_pre_tool("file_read") is False

    def test_template_variable_expansion(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        hook = HookDefinition(
            event="pre_tool_call",
            command=(
                f'{sys.executable} -c "'
                "import sys; "
                f"open(r'{marker}', 'w').write(sys.argv[0])"
                '" {tool_name}'
            ),
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_pre_tool("shell")
        # Hook ran successfully (marker file created)
        assert marker.exists()


class TestPostToolHooks:
    """Test post_tool_call hook execution."""

    def test_runs_post_hook(self, tmp_path: Path) -> None:
        marker = tmp_path / "post_marker.txt"
        hook = HookDefinition(
            event="post_tool_call",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('done')\"",
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_post_tool("shell")
        assert marker.exists()
        assert marker.read_text() == "done"

    def test_post_hook_tool_filter(self, tmp_path: Path) -> None:
        marker = tmp_path / "post_marker.txt"
        hook = HookDefinition(
            event="post_tool_call",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('done')\"",
            tools=["shell"],
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_post_tool("file_read")
        assert not marker.exists()


class TestSessionHooks:
    """Test session lifecycle hooks."""

    def test_pre_session(self, tmp_path: Path) -> None:
        marker = tmp_path / "pre_session.txt"
        hook = HookDefinition(
            event="pre_session",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('started')\"",
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_pre_session()
        assert marker.exists()

    def test_post_session(self, tmp_path: Path) -> None:
        marker = tmp_path / "post_session.txt"
        hook = HookDefinition(
            event="post_session",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('ended')\"",
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_post_session()
        assert marker.exists()

    def test_only_session_hooks_run(self, tmp_path: Path) -> None:
        marker = tmp_path / "wrong.txt"
        hook = HookDefinition(
            event="pre_tool_call",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('wrong')\"",
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_pre_session()
        executor.run_post_session()
        assert not marker.exists()


class TestHookTimeout:
    """Test hook timeout handling."""

    def test_timeout_returns_nonzero(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "import time; time.sleep(10)"',
            timeout=1,
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is False


class TestHookErrors:
    """Test hook error handling."""

    def test_bad_template_variable(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command="echo {nonexistent_var}",
        )
        executor = _make_executor([hook], tmp_path)
        # Bad template returns exit code 1, blocking the tool
        assert executor.run_pre_tool("shell") is False

    def test_multiple_hooks_all_must_pass(self, tmp_path: Path) -> None:
        hook1 = HookDefinition(
            event="pre_tool_call",
            command=f"{sys.executable} -c \"print('ok')\"",
        )
        hook2 = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "raise SystemExit(1)"',
        )
        executor = _make_executor([hook1, hook2], tmp_path)
        assert executor.run_pre_tool("shell") is False
