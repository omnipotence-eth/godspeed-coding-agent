"""Tests for background command execution (Unit 7)."""

from __future__ import annotations

import asyncio
import sys

import pytest

from godspeed.tools.background import BackgroundCheckTool, BackgroundRegistry
from godspeed.tools.base import ToolContext
from godspeed.tools.shell import ShellTool


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the singleton registry before each test."""
    BackgroundRegistry.reset()
    yield
    BackgroundRegistry.reset()


# ---------------------------------------------------------------------------
# BackgroundRegistry
# ---------------------------------------------------------------------------


def test_registry_singleton():
    """BackgroundRegistry.get() returns the same instance."""
    r1 = BackgroundRegistry.get()
    r2 = BackgroundRegistry.get()
    assert r1 is r2


def test_registry_empty():
    """Empty registry has no processes."""
    reg = BackgroundRegistry.get()
    assert reg.count == 0
    assert reg.list_all() == []


def test_registry_next_id():
    """IDs increment sequentially."""
    reg = BackgroundRegistry.get()
    assert reg.next_id() == 1
    assert reg.next_id() == 2
    assert reg.next_id() == 3


# ---------------------------------------------------------------------------
# ShellTool: background=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows ProactorEventLoop subprocess cleanup is flaky; CI covers this on Linux.",
)
async def test_shell_background_starts_process(tmp_path):
    """shell with background=True spawns a process and returns immediately."""
    tool = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    cmd = "sleep 0.1 && echo done"
    result = await tool.execute({"command": cmd, "background": True}, ctx)

    assert not result.is_error
    assert "background process" in result.output.lower()
    assert "1" in result.output  # Process ID 1

    # Verify it's in the registry
    reg = BackgroundRegistry.get()
    assert reg.count == 1
    proc = reg.get_process(1)
    assert proc is not None
    assert proc.command == cmd

    # Wait for it to finish
    await asyncio.sleep(0.5)
    assert not proc.is_running


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows ProactorEventLoop subprocess cleanup is flaky; CI covers this on Linux.",
)
async def test_shell_background_max_concurrent(tmp_path):
    """Rejects background process when max concurrent reached."""
    from godspeed.tools.background import MAX_CONCURRENT

    tool = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    # Fill up with sleeping processes
    for _ in range(MAX_CONCURRENT):
        result = await tool.execute({"command": "sleep 60", "background": True}, ctx)
        assert not result.is_error

    # Next one should fail
    result = await tool.execute(
        {
            "command": "echo over",
            "background": True,
        },
        ctx,
    )
    assert result.is_error
    assert "Too many" in result.error

    # Clean up
    reg = BackgroundRegistry.get()
    for proc in reg.list_all():
        proc.process.terminate()
        await asyncio.wait_for(proc.process.wait(), timeout=5)


# ---------------------------------------------------------------------------
# BackgroundCheckTool: status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_no_processes(tmp_path):
    """Status with no processes returns informative message."""
    tool = BackgroundCheckTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await tool.execute({"action": "status"}, ctx)
    assert "No background processes" in result.output


@pytest.mark.asyncio
async def test_status_shows_running(tmp_path):
    """Status shows running processes."""
    shell = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    await shell.execute({"command": "sleep 5", "background": True}, ctx)

    check = BackgroundCheckTool()
    result = await check.execute({"action": "status"}, ctx)

    assert "running" in result.output
    assert "[1]" in result.output

    # Clean up
    reg = BackgroundRegistry.get()
    proc = reg.get_process(1)
    proc.process.terminate()
    await asyncio.wait_for(proc.process.wait(), timeout=5)


# ---------------------------------------------------------------------------
# BackgroundCheckTool: output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_after_completion(tmp_path):
    """Output shows captured stdout after process completes."""
    shell = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    cmd = "echo hello_background"
    await shell.execute({"command": cmd, "background": True}, ctx)

    # Wait for process to finish and output to be collected
    await asyncio.sleep(1.0)

    check = BackgroundCheckTool()
    result = await check.execute({"action": "output", "id": 1}, ctx)

    assert not result.is_error
    assert "hello_background" in result.output


@pytest.mark.asyncio
async def test_output_missing_id(tmp_path):
    """Output without id returns error."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await check.execute({"action": "output"}, ctx)
    assert result.is_error
    assert "id is required" in result.error


@pytest.mark.asyncio
async def test_output_invalid_id(tmp_path):
    """Output with non-existent id returns error."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await check.execute({"action": "output", "id": 999}, ctx)
    assert result.is_error
    assert "999" in result.error


# ---------------------------------------------------------------------------
# BackgroundCheckTool: kill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_running_process(tmp_path):
    """Kill terminates a running process."""
    shell = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    await shell.execute({"command": "sleep 30", "background": True}, ctx)

    reg = BackgroundRegistry.get()
    proc = reg.get_process(1)
    assert proc.is_running

    check = BackgroundCheckTool()
    result = await check.execute({"action": "kill", "id": 1}, ctx)

    assert not result.is_error
    assert "terminated" in result.output.lower()
    # Wait a moment for termination
    await asyncio.sleep(0.5)
    assert not proc.is_running


@pytest.mark.asyncio
async def test_kill_already_exited(tmp_path):
    """Kill on already-exited process returns success."""
    shell = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    await shell.execute({"command": "echo quick", "background": True}, ctx)

    await asyncio.sleep(1.0)

    check = BackgroundCheckTool()
    result = await check.execute({"action": "kill", "id": 1}, ctx)

    assert not result.is_error
    assert "already exited" in result.output.lower()


@pytest.mark.asyncio
async def test_invalid_action(tmp_path):
    """Invalid action returns error."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await check.execute({"action": "restart"}, ctx)
    assert result.is_error
    assert "Invalid action" in result.error
