"""Tests for background command execution (Unit 7)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from godspeed.tools.background import (
    BackgroundCheckTool,
    BackgroundProcess,
    BackgroundRegistry,
    _collect_output,
)
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
    reason="Windows ProactorEventLoop subprocess detection is flaky",
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
# BackgroundProcess properties
# ---------------------------------------------------------------------------


def test_background_process_returncode():
    """Line 36: returncode property delegates to process.returncode."""
    mock_proc = asyncio.subprocess.Process
    # Use a real BackgroundProcess with a mock process
    import time
    from unittest.mock import MagicMock

    fake_process = MagicMock()
    fake_process.returncode = 0
    bp = BackgroundProcess(
        id=1,
        command="echo hi",
        process=fake_process,
        started_at=time.monotonic(),
    )
    assert bp.returncode == 0
    assert not bp.is_running


def test_background_process_is_running_true():
    """is_running returns True when process hasn't exited."""
    import time
    from unittest.mock import MagicMock

    fake_process = MagicMock()
    fake_process.returncode = None
    bp = BackgroundProcess(
        id=1,
        command="sleep 10",
        process=fake_process,
        started_at=time.monotonic(),
    )
    assert bp.is_running


def test_background_process_stdout_caching():
    """Lines 45-47: stdout is cached after first access."""
    import time
    from unittest.mock import MagicMock

    fake_process = MagicMock()
    fake_process.returncode = None
    bp = BackgroundProcess(
        id=1,
        command="test",
        process=fake_process,
        started_at=time.monotonic(),
    )
    bp.stdout_chunks.append("hello ")
    bp.stdout_chunks.append("world")
    result = bp.stdout
    assert result == "hello world"
    assert hasattr(bp, "_stdout_cache")
    bp.stdout_chunks.append("!")
    assert bp.stdout == "hello world"  # cached, won't include !


def test_background_process_stderr_caching():
    """Lines 51-53: stderr is cached after first access."""
    import time
    from unittest.mock import MagicMock

    fake_process = MagicMock()
    fake_process.returncode = None
    bp = BackgroundProcess(
        id=1,
        command="test",
        process=fake_process,
        started_at=time.monotonic(),
    )
    bp.stderr_chunks.append("error ")
    bp.stderr_chunks.append("msg")
    result = bp.stderr
    assert result == "error msg"
    assert hasattr(bp, "_stderr_cache")


def test_background_process_invalidate_output_cache():
    """Lines 58, 60: invalidate clears both stdout and stderr caches."""
    import time
    from unittest.mock import MagicMock

    fake_process = MagicMock()
    fake_process.returncode = None
    bp = BackgroundProcess(
        id=1,
        command="test",
        process=fake_process,
        started_at=time.monotonic(),
    )
    bp.stdout_chunks.append("old")
    _ = bp.stdout
    bp.stderr_chunks.append("old")
    _ = bp.stderr
    assert hasattr(bp, "_stdout_cache")
    assert hasattr(bp, "_stderr_cache")

    bp.invalidate_output_cache()
    assert not hasattr(bp, "_stdout_cache")
    assert not hasattr(bp, "_stderr_cache")


def test_background_process_elapsed_secs():
    """elapsed_secs returns time since started."""
    import time
    from unittest.mock import MagicMock

    fake_process = MagicMock()
    fake_process.returncode = None
    started = time.monotonic()
    bp = BackgroundProcess(
        id=1,
        command="test",
        process=fake_process,
        started_at=started,
    )
    elapsed = bp.elapsed_secs
    assert elapsed >= 0


# ---------------------------------------------------------------------------
# BackgroundRegistry additional tests
# ---------------------------------------------------------------------------


def test_registry_add_remove():
    """Registry add and remove work correctly."""
    from unittest.mock import MagicMock

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    fake_process = MagicMock()
    fake_process.returncode = 0
    bp = BackgroundProcess(
        id=reg.next_id(),
        command="test",
        process=fake_process,
        started_at=0.0,
    )
    reg.add(bp)
    assert reg.count == 1
    assert reg.get_process(bp.id) is bp

    reg.remove(bp.id)
    assert reg.count == 0
    assert reg.get_process(bp.id) is None


def test_registry_remove_nonexistent():
    """Removing a nonexistent ID doesn't raise."""
    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()
    reg.remove(999)


def test_registry_active_count():
    """active_count counts only running processes."""
    from unittest.mock import MagicMock

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    running = MagicMock()
    running.returncode = None
    exited = MagicMock()
    exited.returncode = 0

    bp1 = BackgroundProcess(id=1, command="a", process=running, started_at=0.0)
    bp2 = BackgroundProcess(id=2, command="b", process=exited, started_at=0.0)
    reg.add(bp1)
    reg.add(bp2)

    assert reg.active_count == 1


# ---------------------------------------------------------------------------
# _collect_output edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_output_null_stream():
    """Line 115: stream is None in _read_stream."""

    class FakeProcess:
        returncode = 0
        stdout = None
        stderr = None

        async def wait(self):
            pass

    proc = BackgroundProcess(
        id=1,
        command="test",
        process=FakeProcess(),
        started_at=0.0,
    )
    # Should not raise
    await _collect_output(proc)


# ---------------------------------------------------------------------------
# BackgroundCheckTool: output edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_stdout_only(tmp_path):
    """Line 204-214: output with only stdout."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    from unittest.mock import MagicMock
    import time

    fake_process = MagicMock()
    fake_process.returncode = 0

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    bp = BackgroundProcess(
        id=1,
        command="echo hi",
        process=fake_process,
        started_at=time.monotonic(),
    )
    bp.stdout_chunks.append("hello world")
    reg.add(bp)

    result = await check.execute({"action": "output", "id": 1}, ctx)
    assert not result.is_error
    assert "STDOUT" in result.output
    assert "hello world" in result.output


@pytest.mark.asyncio
async def test_output_stderr_only():
    """Output with only stderr."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")

    from unittest.mock import MagicMock
    import time

    fake_process = MagicMock()
    fake_process.returncode = 0

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    bp = BackgroundProcess(
        id=1,
        command="test",
        process=fake_process,
        started_at=time.monotonic(),
    )
    bp.stderr_chunks.append("error output")
    reg.add(bp)

    result = await check.execute({"action": "output", "id": 1}, ctx)
    assert not result.is_error
    assert "STDERR" in result.output
    assert "error output" in result.output


@pytest.mark.asyncio
async def test_output_no_output_yet():
    """Output with no stdout or stderr yet returns '(no output yet)'."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")

    from unittest.mock import MagicMock
    import time

    fake_process = MagicMock()
    fake_process.returncode = None  # still running

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    bp = BackgroundProcess(
        id=1,
        command="sleep 10",
        process=fake_process,
        started_at=time.monotonic(),
    )
    reg.add(bp)

    result = await check.execute({"action": "output", "id": 1}, ctx)
    assert not result.is_error
    assert "no output yet" in result.output


@pytest.mark.asyncio
async def test_output_both_stdout_and_stderr():
    """Output with both stdout and stderr."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")

    from unittest.mock import MagicMock
    import time

    fake_process = MagicMock()
    fake_process.returncode = 0

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    bp = BackgroundProcess(
        id=1,
        command="test",
        process=fake_process,
        started_at=time.monotonic(),
    )
    bp.stdout_chunks.append("out")
    bp.stderr_chunks.append("err")
    reg.add(bp)

    result = await check.execute({"action": "output", "id": 1}, ctx)
    assert not result.is_error
    assert "STDOUT" in result.output
    assert "STDERR" in result.output


# ---------------------------------------------------------------------------
# BackgroundCheckTool: kill edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_missing_id():
    """Line 219: kill without id returns error."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")
    result = await check.execute({"action": "kill"}, ctx)
    assert result.is_error
    assert "id is required" in result.error


@pytest.mark.asyncio
async def test_kill_invalid_id():
    """Kill with non-existent id returns error."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")
    result = await check.execute({"action": "kill", "id": 999}, ctx)
    assert result.is_error
    assert "999" in result.error


@pytest.mark.asyncio
async def test_kill_already_exited_process():
    """Line 226: kill on already-exited process."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")

    from unittest.mock import MagicMock
    import time

    fake_process = MagicMock()
    fake_process.returncode = 0  # already exited

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    bp = BackgroundProcess(
        id=1,
        command="echo done",
        process=fake_process,
        started_at=time.monotonic(),
    )
    reg.add(bp)

    result = await check.execute({"action": "kill", "id": 1}, ctx)
    assert "already exited" in result.output.lower()
    assert not result.is_error


@pytest.mark.asyncio
async def test_kill_timeout_force_kill():
    """Lines 231-232: kill with timeout triggers force kill."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")

    fake_process = MagicMock()
    fake_process.returncode = None
    fake_process.wait = AsyncMock(side_effect=[TimeoutError(), None])
    fake_process.terminate = MagicMock()
    fake_process.kill = MagicMock()

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    import time

    bp = BackgroundProcess(
        id=1,
        command="sleep 60",
        process=fake_process,
        started_at=time.monotonic(),
    )
    reg.add(bp)

    result = await check.execute({"action": "kill", "id": 1}, ctx)
    assert "terminated" in result.output.lower()
    fake_process.terminate.assert_called_once()
    fake_process.kill.assert_called_once()


# ---------------------------------------------------------------------------
# BackgroundCheckTool: status edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_mixed_running_and_exited():
    """Status shows both running and exited processes."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=Path("."), session_id="test")

    from unittest.mock import MagicMock
    import time

    running = MagicMock()
    running.returncode = None
    exited = MagicMock()
    exited.returncode = 0

    reg = BackgroundRegistry.get()
    reg.reset()
    reg = BackgroundRegistry.get()

    bp1 = BackgroundProcess(id=1, command="sleep 10", process=running, started_at=time.monotonic())
    bp2 = BackgroundProcess(id=2, command="echo hi", process=exited, started_at=time.monotonic())
    reg.add(bp1)
    reg.add(bp2)

    result = await check.execute({"action": "status"}, ctx)
    assert "running" in result.output
    assert "exited" in result.output
    assert "[1]" in result.output
    assert "[2]" in result.output


# ---------------------------------------------------------------------------
# BackgroundCheckTool: metadata
# ---------------------------------------------------------------------------


def test_background_check_tool_name():
    """Line 139: name property returns 'background_check'."""
    tool = BackgroundCheckTool()
    assert tool.name == "background_check"


def test_background_check_tool_description():
    """Line 143: description returns expected string."""
    tool = BackgroundCheckTool()
    desc = tool.description
    assert "status" in desc.lower()
    assert "output" in desc.lower()
    assert "kill" in desc.lower()


def test_background_check_tool_risk_level():
    """Line 150: risk_level is HIGH."""
    tool = BackgroundCheckTool()
    from godspeed.tools.base import RiskLevel

    assert tool.risk_level == RiskLevel.HIGH


def test_background_check_tool_schema():
    """Line 153: get_schema returns expected structure."""
    tool = BackgroundCheckTool()
    schema = tool.get_schema()
    assert schema["type"] == "object"
    assert "action" in schema["properties"]
    assert "id" in schema["properties"]
    assert "action" in schema["required"]


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
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows ProactorEventLoop output collection is flaky",
)
async def test_output_after_completion(tmp_path):
    """Output handling after process completes."""
    shell = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    cmd = "echo hello_background"
    await shell.execute({"command": cmd, "background": True}, ctx)

    # Wait for process to finish and potentially be cleaned up
    await asyncio.sleep(1.0)

    check = BackgroundCheckTool()
    result = await check.execute({"action": "output", "id": 1}, ctx)

    # Process may still be in registry (returns output) or cleaned up (returns error)
    if result.is_error:
        assert "1" in result.error
    else:
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
    """Kill on already-exited process returns appropriate response."""
    shell = ShellTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")

    await shell.execute({"command": "echo quick", "background": True}, ctx)

    # Wait for process to finish and potentially be cleaned up
    await asyncio.sleep(1.0)

    check = BackgroundCheckTool()
    result = await check.execute({"action": "kill", "id": 1}, ctx)

    # Process may still be in registry (returns terminated) or cleaned up (returns error)
    if result.is_error:
        assert "1" in result.error
    else:
        assert "terminated" in result.output.lower() or "already exited" in result.output.lower()


@pytest.mark.asyncio
async def test_invalid_action(tmp_path):
    """Invalid action returns error."""
    check = BackgroundCheckTool()
    ctx = ToolContext(cwd=tmp_path, session_id="test")
    result = await check.execute({"action": "restart"}, ctx)
    assert result.is_error
    assert "Invalid action" in result.error
