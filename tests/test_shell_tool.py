"""Tests for ShellTool timeout + process-tree-kill behavior.

Focus: the bug discovered during SWE-Bench dev-23 attempt #3 where
subprocess.run's timeout cleanup left grandchildren holding pipes,
blocking the Python runner for 60-100+ minutes. Fix uses Popen +
_kill_process_tree helper; these tests guard that the new code path
actually times out cleanly.
"""

from __future__ import annotations

import platform
import shutil
import sys
import time
from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.shell import ShellTool, _kill_process_tree


def _has_bash() -> bool:
    """True when a bash interpreter is on PATH (needed for Windows sleep tests)."""
    return shutil.which("bash") is not None


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-shell")


# ---------------------------------------------------------------------------
# Normal execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_echo_succeeds(tool: ShellTool, ctx: ToolContext) -> None:
    result = await tool.execute({"command": "echo hello"}, ctx)
    assert result.is_error is False
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_nonzero_exit_returns_failure(tool: ShellTool, ctx: ToolContext) -> None:
    # exit 2 works on both bash and cmd
    result = await tool.execute({"command": "exit 2"}, ctx)
    assert result.is_error is True
    assert "Exit code 2" in (result.error or "")


# ---------------------------------------------------------------------------
# Timeout + force-kill — the bug-fix regression tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    platform.system() == "Windows" and not _has_bash(),
    reason="sleep requires bash; cmd doesn't have it",
)
@pytest.mark.asyncio
async def test_timeout_kills_hanging_command_promptly(tool: ShellTool, ctx: ToolContext) -> None:
    """A long sleep with a short timeout must return within ~timeout seconds,
    not block indefinitely."""
    t0 = time.monotonic()
    result = await tool.execute({"command": "sleep 30", "timeout": 2}, ctx)
    elapsed = time.monotonic() - t0

    assert result.is_error is True
    assert "timed out" in (result.error or "").lower()
    # Should complete within timeout + reasonable cleanup margin (5s drain).
    # The old bug had this hanging for 60+ minutes; we're checking <15s.
    assert elapsed < 15, (
        f"timeout took {elapsed:.1f}s; the Popen+kill_tree path should "
        "terminate within timeout + ~5s cleanup."
    )


@pytest.mark.skipif(
    platform.system() == "Windows" and not _has_bash(),
    reason="test requires bash to spawn a grandchild sleep",
)
@pytest.mark.asyncio
async def test_timeout_error_message_mentions_force_kill(tool: ShellTool, ctx: ToolContext) -> None:
    result = await tool.execute({"command": "sleep 30", "timeout": 2}, ctx)
    assert result.is_error is True
    # Error message must flag that a force-kill happened, so callers know
    # this wasn't a graceful exit.
    assert (
        "force-killed" in (result.error or "").lower()
        or "force-kill" in (result.error or "").lower()
    )


# ---------------------------------------------------------------------------
# _kill_process_tree helper
# ---------------------------------------------------------------------------


def test_kill_process_tree_handles_nonexistent_pid() -> None:
    """Passing a PID that doesn't exist must not raise."""
    # Pick a very high PID unlikely to exist; psutil.NoSuchProcess swallowed.
    _kill_process_tree(99999999)
    # If we got here without an exception, the suppression worked.


def test_kill_process_tree_kills_real_subprocess() -> None:
    """Actually kill a real subprocess via the helper."""
    import subprocess

    if platform.system() == "Windows":
        # Spawn something safe-ish that'll live a few seconds
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
    else:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    try:
        # Give it a moment to start
        time.sleep(0.3)
        assert proc.poll() is None, "subprocess should still be running"
        _kill_process_tree(proc.pid)
        # After kill, poll should return a non-None returncode within 2s.
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert proc.poll() is not None, "subprocess was not killed by _kill_process_tree"
    finally:
        # Belt-and-suspenders cleanup
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
