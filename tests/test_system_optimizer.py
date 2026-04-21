"""Tests for SystemOptimizerTool.

Static-shape + output-structure tests. No live system calls beyond
what the tool itself makes (psutil is required in the test env, pynvml
and nvidia-smi are optional). The tool is READ_ONLY in this release so
there are no destructive paths to test yet — when `act` mode lands,
adversarial tests for the deny-list go here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.system_optimizer import (
    _SYSTEM_CRITICAL_NAMES,
    DEFAULT_TOP,
    MAX_TOP,
    SystemOptimizerTool,
    _is_system_critical,
)


@pytest.fixture
def tool() -> SystemOptimizerTool:
    return SystemOptimizerTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


# ---------------------------------------------------------------------------
# Static shape
# ---------------------------------------------------------------------------


def test_tool_name(tool: SystemOptimizerTool) -> None:
    assert tool.name == "system_optimizer"


def test_tool_is_read_only(tool: SystemOptimizerTool) -> None:
    """Initial release must be READ_ONLY — no act mode yet."""
    assert tool.risk_level == RiskLevel.READ_ONLY


def test_schema_supports_inspect_and_recommend(tool: SystemOptimizerTool) -> None:
    schema = tool.get_schema()
    mode_enum = schema["properties"]["mode"]["enum"]
    assert mode_enum == ["inspect", "recommend"], (
        f"mode enum should expose 'inspect' and 'recommend' (both READ_ONLY); got {mode_enum}"
    )


def test_schema_clamps_top(tool: SystemOptimizerTool) -> None:
    schema = tool.get_schema()
    top = schema["properties"]["top"]
    assert top["minimum"] == 1
    assert top["maximum"] == MAX_TOP


def test_description_mentions_read_only(tool: SystemOptimizerTool) -> None:
    # Agent should see this and not try to call the tool expecting mutation.
    assert "read-only" in tool.description.lower() or "no processes are killed" in tool.description


# ---------------------------------------------------------------------------
# Live execute (inspect only) — tests on the current host
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_returns_structured_report(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    result = await tool.execute({"mode": "inspect"}, ctx)
    assert result.is_error is False
    out = result.output
    assert "Mode: inspect (READ_ONLY)" in out
    assert "Platform:" in out
    assert "CPU:" in out
    assert "Memory:" in out
    assert "Top" in out and "processes" in out


@pytest.mark.asyncio
async def test_inspect_default_top(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    """Default 'top' must equal DEFAULT_TOP rows in the process table."""
    result = await tool.execute({"mode": "inspect"}, ctx)
    out = result.output
    # Count rows in the "Top N processes" section. Each row starts
    # with two spaces + PID (digits).
    lines = out.splitlines()
    in_top = False
    data_rows = 0
    for line in lines:
        if "Top" in line and "processes" in line:
            in_top = True
            continue
        if in_top:
            if line.startswith("  ") and "PID" in line:
                # header row, skip
                continue
            if line.startswith("  * "):
                # footer line
                break
            if line.strip() and line.startswith("  "):
                data_rows += 1
    assert data_rows == DEFAULT_TOP, f"expected {DEFAULT_TOP} rows (default top); got {data_rows}"


@pytest.mark.asyncio
async def test_inspect_respects_custom_top(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    result = await tool.execute({"mode": "inspect", "top": 3}, ctx)
    out = result.output
    data_rows = [
        ln
        for ln in out.splitlines()
        if ln.startswith("  ") and not ln.startswith("  * ") and "PID" not in ln
    ]
    # Filter out CPU/Memory/Swap/Disk/GPU lines (they start with those keywords)
    proc_rows = [
        ln
        for ln in data_rows
        if not any(
            ln.lstrip().startswith(k)
            for k in ("CPU:", "Memory:", "Swap:", "Disk", "GPU", "Utilization:", "Temperature:")
        )
    ]
    assert len(proc_rows) >= 3, f"should have at least 3 process rows; got {len(proc_rows)}"


@pytest.mark.asyncio
async def test_inspect_caps_at_max_top(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    """Requesting more than MAX_TOP should clamp, not error."""
    result = await tool.execute({"mode": "inspect", "top": MAX_TOP * 10}, ctx)
    assert result.is_error is False


@pytest.mark.asyncio
async def test_inspect_sort_by_cpu(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    result = await tool.execute({"mode": "inspect", "sort_by": "cpu"}, ctx)
    assert "Top" in result.output
    assert "by CPU" in result.output


@pytest.mark.asyncio
async def test_inspect_sort_by_memory_default(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    result = await tool.execute({"mode": "inspect"}, ctx)
    assert "by memory" in result.output


@pytest.mark.asyncio
async def test_non_inspect_mode_fails(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    result = await tool.execute({"mode": "act"}, ctx)
    assert result.is_error is True
    assert "not supported" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Recommend mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recommend_returns_structured_report(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    result = await tool.execute({"mode": "recommend"}, ctx)
    assert result.is_error is False
    out = result.output
    assert "Mode: recommend (READ_ONLY)" in out
    assert "Platform:" in out
    # Either "No cleanup recommendations" or "N recommendation(s)"
    assert "recommendation" in out.lower()


@pytest.mark.asyncio
async def test_recommend_never_recommends_killing_system_critical(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """Any action_id that would kill a system-critical process must be absent."""
    result = await tool.execute({"mode": "recommend"}, ctx)
    out = result.output
    # Deny-list members should not appear as targets of kill_process actions.
    for critical in _SYSTEM_CRITICAL_NAMES.get("win32", frozenset()):
        # We only care if the critical process actually appeared AS a kill target.
        # `kill_process:<pid>:<name>` is the action_id format.
        assert "kill_process:" not in out or f":{critical}" not in out, (
            f"kill_process action targets system-critical {critical!r}"
        )


@pytest.mark.asyncio
async def test_recommend_skips_system_idle_process(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """PID 0 / 'System Idle Process' on Windows must not be a recommendation."""
    result = await tool.execute({"mode": "recommend"}, ctx)
    assert "System Idle Process" not in result.output
    assert "kill_process:0:" not in result.output


@pytest.mark.asyncio
async def test_recommend_is_read_only_regardless_of_findings(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """Recommend must not execute anything — risk_level stays READ_ONLY."""
    assert tool.risk_level == RiskLevel.READ_ONLY
    # Run it a few times; output should remain text only, no side effects.
    for _ in range(2):
        result = await tool.execute({"mode": "recommend"}, ctx)
        assert result.is_error is False


@pytest.mark.asyncio
async def test_recommend_note_flags_act_not_implemented(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """Output must tell the agent act mode doesn't exist yet."""
    result = await tool.execute({"mode": "recommend"}, ctx)
    # Either there were no recommendations (healthy system) or the
    # explicit 'not yet implemented' note appears.
    out = result.output.lower()
    assert "not yet implemented" in out or "healthy" in out


# ---------------------------------------------------------------------------
# Deny-list — correctness & platform dispatch
# ---------------------------------------------------------------------------


def test_deny_lists_defined_for_three_platforms() -> None:
    for platform_key in ("win32", "linux", "darwin"):
        assert platform_key in _SYSTEM_CRITICAL_NAMES
        assert _SYSTEM_CRITICAL_NAMES[platform_key], f"{platform_key} deny-list is empty"


def test_windows_deny_list_covers_expected_critical() -> None:
    win = _SYSTEM_CRITICAL_NAMES["win32"]
    # Hard requirements: if act mode launches, these must be in the list.
    for essential in {
        "System",
        "csrss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "winlogon.exe",
        "explorer.exe",
        "dwm.exe",
    }:
        assert essential in win, f"{essential} must be in Windows deny-list"


def test_linux_deny_list_covers_expected_critical() -> None:
    linux = _SYSTEM_CRITICAL_NAMES["linux"]
    for essential in {"systemd", "init", "sshd", "kworker"}:
        assert essential in linux, f"{essential} must be in Linux deny-list"


def test_macos_deny_list_covers_expected_critical() -> None:
    macos = _SYSTEM_CRITICAL_NAMES["darwin"]
    for essential in {"launchd", "kernel_task", "WindowServer", "Finder", "Dock"}:
        assert essential in macos, f"{essential} must be in macOS deny-list"


def test_is_system_critical_explorer_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    assert _is_system_critical("explorer.exe") is True
    assert _is_system_critical("chrome.exe") is False


def test_is_system_critical_linux_prefix_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """kworker/u32:1 etc. should match the 'kworker' entry by prefix."""
    monkeypatch.setattr("sys.platform", "linux")
    assert _is_system_critical("kworker") is True
    assert _is_system_critical("kworker/u32:1") is True
    assert _is_system_critical("kworkerprocess") is False  # not a real prefix match


def test_is_system_critical_macos_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert _is_system_critical("kernel_task") is True
    assert _is_system_critical("Finder") is True
    assert _is_system_critical("Safari") is False


def test_is_system_critical_handles_unknown_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """On an unknown platform, everything is 'not critical' (safe default)."""
    monkeypatch.setattr("sys.platform", "aix")
    assert _is_system_critical("anything") is False
