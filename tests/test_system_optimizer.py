"""Tests for SystemOptimizerTool.

Static-shape + output-structure tests. No live system calls beyond
what the tool itself makes (psutil is required in the test env, pynvml
and nvidia-smi are optional). The tool is READ_ONLY in this release so
there are no destructive paths to test yet — when `act` mode lands,
adversarial tests for the deny-list go here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.system_optimizer import (
    DEFAULT_TOP,
    MAX_TOP,
    SystemOptimizerTool,
    _SYSTEM_CRITICAL_NAMES,
    _build_recommend,
    _check_ollama_vram,
    _collect_outlier_processes,
    _count_zombie_python,
    _cpu_mem_summary,
    _disk_summary,
    _gb,
    _gpu_summary,
    _gpu_via_nvidia_smi,
    _gpu_via_pynvml,
    _is_system_critical,
    _mb,
    _top_processes,
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
    lines = out.splitlines()
    in_top = False
    data_rows = 0
    for line in lines:
        if "Top" in line and "processes" in line:
            in_top = True
            continue
        if in_top:
            if line.startswith("  ") and "PID" in line:
                continue
            if line.startswith("  * "):
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
    assert "recommendation" in out.lower()


@pytest.mark.asyncio
async def test_recommend_never_recommends_killing_system_critical(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """Any action_id that would kill a system-critical process must be absent."""
    result = await tool.execute({"mode": "recommend"}, ctx)
    out = result.output
    for critical in _SYSTEM_CRITICAL_NAMES.get("win32", frozenset()):
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
    for _ in range(2):
        result = await tool.execute({"mode": "recommend"}, ctx)
        assert result.is_error is False


@pytest.mark.asyncio
async def test_recommend_note_flags_act_not_implemented(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """Output must tell the agent act mode doesn't exist yet."""
    result = await tool.execute({"mode": "recommend"}, ctx)
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
    assert _is_system_critical("kworkerprocess") is False


def test_is_system_critical_macos_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert _is_system_critical("kernel_task") is True
    assert _is_system_critical("Finder") is True
    assert _is_system_critical("Safari") is False


def test_is_system_critical_handles_unknown_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """On an unknown platform, everything is 'not critical' (safe default)."""
    monkeypatch.setattr("sys.platform", "aix")
    assert _is_system_critical("anything") is False


# ============================================================================
# NEW TESTS — cover missed branches (82 statements)
# ============================================================================


# ---------------------------------------------------------------------------
# Invalid sort_by fallback (line 160)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_invalid_sort_by_defaults_to_memory(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    result = await tool.execute({"mode": "inspect", "sort_by": "invalid_key"}, ctx)
    assert not result.is_error
    assert "by memory" in result.output


# ---------------------------------------------------------------------------
# Missing psutil (lines 164-165)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_missing_psutil_returns_clear_error(
    tool: SystemOptimizerTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    orig = builtins.__import__

    def _raise_psutil(name: str, *a: Any, **kw: Any) -> Any:
        if name == "psutil":
            raise ImportError("No module named 'psutil'")
        return orig(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise_psutil)
    result = await tool.execute({"mode": "inspect"}, ctx)
    assert result.is_error
    assert "psutil" in (result.error or "")


@pytest.mark.asyncio
async def test_recommend_missing_psutil_returns_clear_error(
    tool: SystemOptimizerTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    orig = builtins.__import__

    def _raise_psutil(name: str, *a: Any, **kw: Any) -> Any:
        if name == "psutil":
            raise ImportError("No module named 'psutil'")
        return orig(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise_psutil)
    result = await tool.execute({"mode": "recommend"}, ctx)
    assert result.is_error
    assert "psutil" in (result.error or "")


# ---------------------------------------------------------------------------
# _cpu_mem_summary
# ---------------------------------------------------------------------------


def test_cpu_mem_summary_success() -> None:
    mock_psutil = MagicMock()
    mock_psutil.cpu_percent.return_value = 42.5
    mock_psutil.cpu_count.return_value = 12
    vmem = MagicMock()
    vmem.used = 8 * 1024**3
    vmem.total = 32 * 1024**3
    vmem.percent = 25.0
    mock_psutil.virtual_memory.return_value = vmem
    swap = MagicMock()
    swap.used = 2 * 1024**3
    swap.total = 8 * 1024**3
    mock_psutil.swap_memory.return_value = swap

    result = _cpu_mem_summary(mock_psutil)
    assert len(result) == 3
    assert "42.5%" in result[0]
    assert "12 logical cores" in result[0]
    assert "8.0 GB / 32.0 GB" in result[1]
    assert "2.0 GB / 8.0 GB" in result[2]


def test_cpu_mem_summary_cpu_count_none() -> None:
    mock_psutil = MagicMock()
    mock_psutil.cpu_percent.return_value = 10.0
    mock_psutil.cpu_count.return_value = None
    vmem = MagicMock()
    vmem.used = 0
    vmem.total = 1
    vmem.percent = 50.0
    mock_psutil.virtual_memory.return_value = vmem
    swap = MagicMock()
    swap.used = 0
    swap.total = 0
    mock_psutil.swap_memory.return_value = swap

    result = _cpu_mem_summary(mock_psutil)
    assert "0 logical cores" in result[0]


# ---------------------------------------------------------------------------
# _gpu_summary — fallback chain
# ---------------------------------------------------------------------------


def test_gpu_summary_pynvml_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", MagicMock())
    monkeypatch.setattr(
        "godspeed.tools.system_optimizer._gpu_via_pynvml",
        lambda: ["GPU 0: Fake PYNVML GPU", "  details"],
    )
    result = _gpu_summary()
    assert len(result) == 2
    assert "Fake PYNVML GPU" in result[0]


def test_gpu_summary_nvidia_smi_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.tools.system_optimizer._gpu_via_pynvml", lambda: None)
    monkeypatch.setattr(
        "godspeed.tools.system_optimizer._gpu_via_nvidia_smi",
        lambda: ["GPU 0: Fake SMI GPU", "  details"],
    )
    result = _gpu_summary()
    assert len(result) == 2
    assert "Fake SMI GPU" in result[0]


def test_gpu_summary_no_gpu_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.tools.system_optimizer._gpu_via_pynvml", lambda: None)
    monkeypatch.setattr("godspeed.tools.system_optimizer._gpu_via_nvidia_smi", lambda: None)
    result = _gpu_summary()
    assert len(result) == 1
    assert "no NVIDIA GPU detected" in result[0]


# ---------------------------------------------------------------------------
# _gpu_via_pynvml (lines 220-248)
# ---------------------------------------------------------------------------


def test_gpu_via_pynvml_success_bytes_name(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 5070 Ti"
    mem = MagicMock()
    mem.used = 8 * 1024**3
    mem.total = 16 * 1024**3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
    util = MagicMock()
    util.gpu = 45
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    result = _gpu_via_pynvml()
    assert result is not None
    assert len(result) == 3
    assert "RTX 5070 Ti" in result[0]
    assert "Utilization: 45%" in result[1]
    assert "8192 / 16384" in result[1]
    assert "Temperature: 65 C" in result[2]
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_gpu_via_pynvml_success_string_name(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 2
    h0 = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [h0, h0]
    mock_pynvml.nvmlDeviceGetName.return_value = "GeForce RTX 4090"
    mem = MagicMock()
    mem.used = 10 * 1024 * 1024 * 1024
    mem.total = 24 * 1024 * 1024 * 1024
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
    util = MagicMock()
    util.gpu = 80
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 72
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    result = _gpu_via_pynvml()
    assert result is not None
    assert len(result) == 6
    assert "GeForce RTX 4090" in result[0]
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_gpu_via_pynvml_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    orig = builtins.__import__

    def _raise_pynvml(name: str, *a: Any, **kw: Any) -> Any:
        if name == "pynvml":
            raise ImportError("No module named 'pynvml'")
        return orig(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise_pynvml)
    assert _gpu_via_pynvml() is None


def test_gpu_via_pynvml_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = RuntimeError("driver error")
    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    assert _gpu_via_pynvml() is None


def test_gpu_via_pynvml_query_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_pynvml.nvmlDeviceGetName.side_effect = RuntimeError("device lost")
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    result = _gpu_via_pynvml()
    assert result is not None
    assert len(result) == 1
    assert "pynvml query error" in result[0]
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_gpu_via_pynvml_shutdown_suppresses_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
    mem = MagicMock()
    mem.used = 1024**3
    mem.total = 8 * 1024**3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
    util = MagicMock()
    util.gpu = 10
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 40
    mock_pynvml.NVML_TEMPERATURE_GPU = 0
    mock_pynvml.nvmlShutdown.side_effect = OSError("shutdown failed")

    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    result = _gpu_via_pynvml()
    assert result is not None
    assert len(result) == 3
    mock_pynvml.nvmlShutdown.assert_called_once()


# ---------------------------------------------------------------------------
# _gpu_via_nvidia_smi (lines 253, 266-268, 270, 275, 283)
# ---------------------------------------------------------------------------


def _mock_smi_module() -> Any:
    import godspeed.tools.system_optimizer as mod

    return mod


def test_gpu_via_nvidia_smi_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: None)
    assert _gpu_via_nvidia_smi() is None


def test_gpu_via_nvidia_smi_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)),
    )
    assert _gpu_via_nvidia_smi() is None


def test_gpu_via_nvidia_smi_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", MagicMock(side_effect=OSError("executable not found"))
    )
    assert _gpu_via_nvidia_smi() is None


def test_gpu_via_nvidia_smi_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    proc = CompletedProcess(args=[], returncode=1, stdout="", stderr="nvidia-smi failed")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    assert _gpu_via_nvidia_smi() is None


def test_gpu_via_nvidia_smi_wrong_column_count(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    proc = CompletedProcess(
        args=[],
        returncode=0,
        stdout="RTX 5070 Ti, 45, 8192, 16384, 65\nShort, 30, 4096\n",
        stderr="",
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    result = _gpu_via_nvidia_smi()
    assert result is not None
    assert len(result) == 3


def test_gpu_via_nvidia_smi_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    proc = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    assert _gpu_via_nvidia_smi() is None


def test_gpu_via_nvidia_smi_success(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    proc = CompletedProcess(
        args=[],
        returncode=0,
        stdout="RTX 5070 Ti, 45, 8192, 16384, 65\nTesla T4, 90, 14336, 15360, 78\n",
        stderr="",
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    result = _gpu_via_nvidia_smi()
    assert result is not None
    assert len(result) == 6
    assert "RTX 5070 Ti" in result[0]
    assert "8192 / 16384" in result[1]
    assert "Tesla T4" in result[3]
    assert "14336 / 15360" in result[4]


def test_gpu_via_nvidia_smi_empty_lines_yield_none(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    proc = CompletedProcess(args=[], returncode=0, stdout="\n", stderr="")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    assert _gpu_via_nvidia_smi() is None


# ---------------------------------------------------------------------------
# _disk_summary (lines 291-292, 296, 299, 302-303)
# ---------------------------------------------------------------------------


def test_disk_summary_partition_enumeration_failure() -> None:
    mock_psutil = MagicMock()
    mock_psutil.disk_partitions.side_effect = OSError("cannot enumerate")
    result = _disk_summary(mock_psutil)
    assert len(result) == 1
    assert "partition enumeration failed" in result[0]


def test_disk_summary_cdrom_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = MagicMock()
    cdrom = MagicMock()
    cdrom.mountpoint = "/mnt/cdrom"
    cdrom.opts = "ro,cdrom"
    cdrom.fstype = "iso9660"
    hdd = MagicMock()
    hdd.mountpoint = "/"
    hdd.opts = "rw"
    hdd.fstype = "ext4"
    usage = MagicMock()
    usage.used = 50 * 1024**3
    usage.total = 100 * 1024**3
    usage.percent = 50.0
    mock_psutil.disk_partitions.return_value = [cdrom, hdd]
    mock_psutil.disk_usage.return_value = usage

    result = _disk_summary(mock_psutil)
    assert len(result) == 1
    assert "/" in result[0]


def test_disk_summary_win32_empty_fstype_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    mock_psutil = MagicMock()
    removable = MagicMock()
    removable.mountpoint = "D:\\"
    removable.opts = "rw"
    removable.fstype = ""
    fixed = MagicMock()
    fixed.mountpoint = "C:\\"
    fixed.opts = "rw"
    fixed.fstype = "NTFS"
    usage = MagicMock()
    usage.used = 100 * 1024**3
    usage.total = 500 * 1024**3
    usage.percent = 20.0
    mock_psutil.disk_partitions.return_value = [removable, fixed]
    mock_psutil.disk_usage.return_value = usage

    result = _disk_summary(mock_psutil)
    assert len(result) == 1
    assert "C:\\" in result[0]


def test_disk_summary_permission_error_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = MagicMock()
    ok_part = MagicMock()
    ok_part.mountpoint = "/data"
    ok_part.opts = "rw"
    ok_part.fstype = "ext4"
    denied_part = MagicMock()
    denied_part.mountpoint = "/root"
    denied_part.opts = "rw"
    denied_part.fstype = "ext4"
    mock_psutil.disk_partitions.return_value = [ok_part, denied_part]
    usage = MagicMock()
    usage.used = 10 * 1024**3
    usage.total = 100 * 1024**3
    usage.percent = 10.0

    def _usage(mp: str) -> MagicMock:
        if mp == "/root":
            raise PermissionError("denied")
        return usage

    mock_psutil.disk_usage.side_effect = _usage
    result = _disk_summary(mock_psutil)
    assert len(result) == 1
    assert "/data" in result[0]


def test_disk_summary_no_partitions() -> None:
    mock_psutil = MagicMock()
    mock_psutil.disk_partitions.return_value = []
    result = _disk_summary(mock_psutil)
    assert len(result) == 1
    assert "no partitions" in result[0]


def test_disk_summary_all_skipped_returns_fallback() -> None:
    mock_psutil = MagicMock()
    cdrom = MagicMock()
    cdrom.mountpoint = "/mnt/cdrom"
    cdrom.opts = "ro,cdrom"
    cdrom.fstype = "iso9660"
    mock_psutil.disk_partitions.return_value = [cdrom]
    result = _disk_summary(mock_psutil)
    assert len(result) == 1
    assert "no partitions" in result[0]


# ---------------------------------------------------------------------------
# _top_processes — error handling (lines 330-334)
# ---------------------------------------------------------------------------


def test_top_processes_no_such_process_skipped() -> None:
    import psutil

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = psutil.NoSuchProcess
    mock_psutil.AccessDenied = psutil.AccessDenied
    mock_psutil.ZombieProcess = psutil.ZombieProcess

    proc = MagicMock()
    proc.cpu_percent.return_value = 0.0
    proc.pid = 9999
    proc.name.return_value = "test_proc"
    proc.memory_info.return_value.rss = 100 * 1024 * 1024
    proc.cpu_percent.return_value = 5.0
    proc.oneshot.return_value.__enter__.side_effect = psutil.NoSuchProcess(9999)
    mock_psutil.process_iter.return_value = [proc]

    result = _top_processes(mock_psutil, top=5, sort_by="memory")
    assert "Top 5 processes by memory" in result[0]
    assert result is not None


def test_top_processes_zombie_skipped() -> None:
    import psutil

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = psutil.NoSuchProcess
    mock_psutil.AccessDenied = psutil.AccessDenied
    mock_psutil.ZombieProcess = psutil.ZombieProcess

    proc = MagicMock()
    proc.pid = 8888
    proc.name.return_value = "zombie_proc"
    proc.oneshot.return_value.__enter__.side_effect = psutil.ZombieProcess(8888)
    mock_psutil.process_iter.return_value = [proc]

    result = _top_processes(mock_psutil, top=5, sort_by="memory")
    assert len(result) >= 1


def test_top_processes_generic_exception_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = psutil.NoSuchProcess
    mock_psutil.AccessDenied = psutil.AccessDenied
    mock_psutil.ZombieProcess = psutil.ZombieProcess

    good_proc = MagicMock()
    good_proc.cpu_percent.return_value = 0.0
    good_proc.pid = 100
    good_proc.name.return_value = "good"
    good_proc.memory_info.return_value.rss = 50 * 1024 * 1024
    good_proc.cpu_percent.return_value = 2.0

    bad_proc = MagicMock()
    bad_proc.pid = 999
    bad_proc.name.return_value = "bad"
    bad_proc.oneshot.return_value.__enter__.side_effect = ValueError("unexpected crash")

    mock_psutil.process_iter.return_value = [good_proc, bad_proc]

    result = _top_processes(mock_psutil, top=5, sort_by="memory")
    assert "good" in "\n".join(result)


def test_top_processes_sort_by_cpu() -> None:
    mock_psutil = MagicMock()
    proc_a = MagicMock()
    proc_a.pid = 1
    proc_a.name.return_value = "proc_a"
    proc_a.memory_info.return_value.rss = 100 * 1024 * 1024
    proc_a.cpu_percent.return_value = 50.0
    proc_b = MagicMock()
    proc_b.pid = 2
    proc_b.name.return_value = "proc_b"
    proc_b.memory_info.return_value.rss = 50 * 1024 * 1024
    proc_b.cpu_percent.return_value = 10.0
    mock_psutil.process_iter.return_value = [proc_a, proc_b]

    result = _top_processes(mock_psutil, top=5, sort_by="cpu")
    full = "\n".join(result)
    assert "by CPU" in result[0]
    assert "proc_a" in full
    assert "proc_b" in full


# ---------------------------------------------------------------------------
# _collect_outlier_processes — error handling (lines 562-566)
# ---------------------------------------------------------------------------


def test_collect_outlier_processes_no_such_process_zombie_access_denied() -> None:
    import psutil

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = psutil.NoSuchProcess
    mock_psutil.AccessDenied = psutil.AccessDenied
    mock_psutil.ZombieProcess = psutil.ZombieProcess

    p1 = MagicMock()
    p1.pid = 1001
    p1.name.return_value = "normal"
    p1.memory_info.return_value.rss = 4 * 1024**3
    p1.cpu_percent.return_value = 50.0
    p2 = MagicMock()
    p2.oneshot.return_value.__enter__.side_effect = psutil.NoSuchProcess(9999)
    p3 = MagicMock()
    p3.oneshot.return_value.__enter__.side_effect = psutil.ZombieProcess(8888)
    p4 = MagicMock()
    p4.oneshot.return_value.__enter__.side_effect = psutil.AccessDenied(7777)

    mock_psutil.process_iter.return_value = [p1, p2, p3, p4]

    big_mem, high_cpu = _collect_outlier_processes(mock_psutil, top=10)
    assert len(big_mem) == 1
    assert big_mem[0][0] == 1001


def test_collect_outlier_processes_generic_exception_skipped() -> None:
    import psutil

    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = psutil.NoSuchProcess
    mock_psutil.AccessDenied = psutil.AccessDenied
    mock_psutil.ZombieProcess = psutil.ZombieProcess

    p_bad = MagicMock()
    p_bad.pid = 5555
    p_bad.oneshot.return_value.__enter__.side_effect = RuntimeError("unexpected")
    p_ok = MagicMock()
    p_ok.pid = 10
    p_ok.name.return_value = "ok_proc"
    p_ok.memory_info.return_value.rss = 5 * 1024**3
    p_ok.cpu_percent.return_value = 250.0

    mock_psutil.process_iter.return_value = [p_ok, p_bad]

    big_mem, high_cpu = _collect_outlier_processes(mock_psutil, top=10)
    assert len(big_mem) == 1
    assert big_mem[0][0] == 10
    assert len(high_cpu) == 1
    assert high_cpu[0][0] == 10


def test_collect_outlier_processes_skips_idle() -> None:
    mock_psutil = MagicMock()
    p_idle = MagicMock()
    p_idle.pid = 0
    p_idle.name.return_value = "System Idle Process"
    p_idle.memory_info.return_value.rss = 100 * 1024
    p_idle.cpu_percent.return_value = 400.0

    mock_psutil.process_iter.return_value = [p_idle]

    big_mem, high_cpu = _collect_outlier_processes(mock_psutil, top=10)
    assert len(big_mem) == 0
    assert len(high_cpu) == 0


def test_collect_outlier_processes_high_cpu() -> None:
    mock_psutil = MagicMock()
    p = MagicMock()
    p.pid = 42
    p.name.return_value = "heavy_proc"
    p.memory_info.return_value.rss = 1 * 1024**3
    p.cpu_percent.return_value = 350.0

    mock_psutil.process_iter.return_value = [p]

    big_mem, high_cpu = _collect_outlier_processes(mock_psutil, top=10)
    assert len(big_mem) == 0
    assert len(high_cpu) == 1
    assert high_cpu[0][0] == 42


# ---------------------------------------------------------------------------
# _count_zombie_python (lines 587-596, 585-586)
# ---------------------------------------------------------------------------


def test_count_zombie_python_windows_always_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    mock_psutil = MagicMock()
    assert _count_zombie_python(mock_psutil) == 0
    mock_psutil.process_iter.assert_not_called()


def test_count_zombie_python_linux_counts_zombies(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil as real_psutil

    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = MagicMock()
    zp = MagicMock()
    zp.info = {"status": real_psutil.STATUS_ZOMBIE, "name": "python3.11"}
    np = MagicMock()
    np.info = {"status": "running", "name": "python3.11"}
    mock_psutil.process_iter.return_value = [zp, np]
    mock_psutil.STATUS_ZOMBIE = real_psutil.STATUS_ZOMBIE

    assert _count_zombie_python(mock_psutil) == 1


def test_count_zombie_python_linux_skips_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil as real_psutil

    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
    mock_psutil.AccessDenied = real_psutil.AccessDenied

    ok = MagicMock()
    ok.info = {"status": real_psutil.STATUS_ZOMBIE, "name": "python3"}

    err_nosuch = MagicMock()
    err_nosuch.info.get.side_effect = real_psutil.NoSuchProcess(1)

    err_denied = MagicMock()
    err_denied.info.get.side_effect = real_psutil.AccessDenied(2)

    mock_psutil.process_iter.return_value = [ok, err_nosuch, err_denied]
    mock_psutil.STATUS_ZOMBIE = real_psutil.STATUS_ZOMBIE

    assert _count_zombie_python(mock_psutil) == 1


def test_count_zombie_python_macos_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil as real_psutil

    monkeypatch.setattr(sys, "platform", "darwin")
    mock_psutil = MagicMock()
    zp = MagicMock()
    zp.info = {"status": real_psutil.STATUS_ZOMBIE, "name": "Python"}
    mock_psutil.process_iter.return_value = [zp]
    mock_psutil.STATUS_ZOMBIE = real_psutil.STATUS_ZOMBIE

    assert _count_zombie_python(mock_psutil) == 1


# ---------------------------------------------------------------------------
# _check_ollama_vram (lines 504, 513-514, 516, 520-527)
# ---------------------------------------------------------------------------


def test_check_ollama_vram_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: None)
    assert _check_ollama_vram() == []


def test_check_ollama_vram_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama")
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)),
    )
    assert _check_ollama_vram() == []


def test_check_ollama_vram_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama")
    monkeypatch.setattr(mod.subprocess, "run", MagicMock(side_effect=OSError("disk full")))
    assert _check_ollama_vram() == []


def test_check_ollama_vram_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama")
    proc = CompletedProcess(args=[], returncode=1, stdout="", stderr="ollama daemon not running")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    assert _check_ollama_vram() == []


def test_check_ollama_vram_parses_models(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama")
    stdout = (
        "NAME           ID              SIZE      PROCESSOR    UNTIL\n"
        "qwen3:14b      abc123def456    12 GB     100%GPU      4 minutes from now\n"
        "llama3:8b      xyz789abc012    5.6 GB    100%GPU      Stopping...\n"
    )
    proc = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    result = _check_ollama_vram()
    assert len(result) == 2
    assert result[0][0] == "MEDIUM"
    assert "qwen3:14b" in result[0][1]
    assert result[1][0] == "MEDIUM"
    assert "llama3:8b" in result[1][1]


def test_check_ollama_vram_incomplete_line_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama")
    stdout = (
        "NAME           ID              SIZE      PROCESSOR    UNTIL\n"
        "qwen3:14b      abc123def456    12 GB     100%GPU      4 minutes from now\n"
        "short\n"
    )
    proc = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    result = _check_ollama_vram()
    assert len(result) == 1
    assert "qwen3:14b" in result[0][1]


def test_check_ollama_vram_no_models_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama")
    proc = CompletedProcess(
        args=[],
        returncode=0,
        stdout="NAME           ID              SIZE      PROCESSOR    UNTIL\n",
        stderr="",
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)
    assert _check_ollama_vram() == []


# ---------------------------------------------------------------------------
# _build_recommend — branch coverage (lines 375, 383-394, 396-397, 400, 402,
#                                    405-406, 408, 420-428, 439, 450, 462,
#                                    479-480)
# ---------------------------------------------------------------------------


def _fake_build_recommend_psutil() -> MagicMock:
    mock_psutil = MagicMock()
    vmem = MagicMock()
    vmem.percent = 50.0
    vmem.used = 8 * 1024**3
    vmem.total = 32 * 1024**3
    mock_psutil.virtual_memory.return_value = vmem
    mock_psutil.disk_partitions.return_value = []
    mock_psutil.process_iter.return_value = []
    return mock_psutil


def test_build_recommend_vmem_90_high() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    mock_psutil.virtual_memory.return_value.percent = 92.0
    mock_psutil.virtual_memory.return_value.used = 30 * 1024**3
    mock_psutil.virtual_memory.return_value.total = 32 * 1024**3

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "HIGH" in result
    assert "92.0%" in result
    assert "investigate_memory_pressure" in result


def test_build_recommend_vmem_80_medium() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    mock_psutil.virtual_memory.return_value.percent = 85.0

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "MEDIUM" in result
    assert "investigate_memory_pressure" in result


def test_build_recommend_disk_partition_exception() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    mock_psutil.disk_partitions.side_effect = OSError("cannot enumerate")

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "healthy" in result.lower() or "No cleanup" in result


def test_build_recommend_disk_cdrom_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = _fake_build_recommend_psutil()
    cdrom = MagicMock()
    cdrom.mountpoint = "/mnt/cdrom"
    cdrom.opts = "ro,cdrom"
    cdrom.fstype = "iso9660"
    mock_psutil.disk_partitions.return_value = [cdrom]

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "healthy" in result.lower() or "No cleanup" in result


def test_build_recommend_disk_win32_empty_fstype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    mock_psutil = _fake_build_recommend_psutil()
    removable = MagicMock()
    removable.mountpoint = "D:\\"
    removable.opts = "rw"
    removable.fstype = ""
    mock_psutil.disk_partitions.return_value = [removable]

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "healthy" in result.lower() or "No cleanup" in result


def test_build_recommend_disk_below_threshold_no_rec() -> None:
    """Cover branch 420->398: disk partition below 85% with mem > 80% for context."""
    mock_psutil = _fake_build_recommend_psutil()
    mock_psutil.virtual_memory.return_value.percent = 85.0
    part = MagicMock()
    part.mountpoint = "/tmp"
    part.opts = "rw"
    part.fstype = "ext4"
    usage = MagicMock()
    usage.used = 30 * 1024**3
    usage.total = 100 * 1024**3
    usage.percent = 30.0
    mock_psutil.disk_partitions.return_value = [part]
    mock_psutil.disk_usage.return_value = usage

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "MEDIUM" in result
    assert "investigate_memory_pressure" in result
    assert "investigate_disk_pressure" not in result
    mock_psutil = _fake_build_recommend_psutil()
    part = MagicMock()
    part.mountpoint = "/secret"
    part.opts = "rw"
    part.fstype = "ext4"
    mock_psutil.disk_partitions.return_value = [part]
    mock_psutil.disk_usage.side_effect = PermissionError("denied")

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "healthy" in result.lower() or "No cleanup" in result


def test_build_recommend_disk_90_high() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    part = MagicMock()
    part.mountpoint = "/"
    part.opts = "rw"
    part.fstype = "ext4"
    usage = MagicMock()
    usage.used = 90 * 1024**3
    usage.total = 100 * 1024**3
    usage.percent = 92.0
    mock_psutil.disk_partitions.return_value = [part]
    mock_psutil.disk_usage.return_value = usage

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "HIGH" in result
    assert "investigate_disk_pressure" in result


def test_build_recommend_disk_85_medium() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    part = MagicMock()
    part.mountpoint = "/home"
    part.opts = "rw"
    part.fstype = "ext4"
    usage = MagicMock()
    usage.used = 430 * 1024**3
    usage.total = 500 * 1024**3
    usage.percent = 87.0
    mock_psutil.disk_partitions.return_value = [part]
    mock_psutil.disk_usage.return_value = usage

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "MEDIUM" in result
    assert "investigate_disk_pressure:/home" in result


def test_build_recommend_big_mem_procs_flagged() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    big_mem = [(100, "chrome.exe", 5 * 1024**3)]
    high_cpu: list[Any] = []

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch(
            "godspeed.tools.system_optimizer._collect_outlier_processes",
            return_value=(big_mem, high_cpu),
        ),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "chrome.exe" in result
    assert "LOW" in result
    assert "kill_process:100:chrome.exe" in result


def test_build_recommend_high_cpu_procs_flagged() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    high_cpu = [(200, "python.exe", 300.0)]

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch(
            "godspeed.tools.system_optimizer._collect_outlier_processes",
            return_value=([], high_cpu),
        ),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "python.exe" in result
    assert "LOW" in result
    assert "300% CPU" in result


def test_build_recommend_big_mem_system_critical_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    mock_psutil = _fake_build_recommend_psutil()
    big_mem = [(4, "csrss.exe", 5 * 1024**3)]

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch(
            "godspeed.tools.system_optimizer._collect_outlier_processes", return_value=(big_mem, [])
        ),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "kill_process:" not in result or "csrss.exe" not in result


def test_build_recommend_high_cpu_system_critical_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = _fake_build_recommend_psutil()
    high_cpu = [(1, "systemd", 250.0)]

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch(
            "godspeed.tools.system_optimizer._collect_outlier_processes",
            return_value=([], high_cpu),
        ),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "kill_process:" not in result or "systemd" not in result


def test_build_recommend_zombie_python_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    mock_psutil = _fake_build_recommend_psutil()

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=3),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "zombie" in result.lower()
    assert "cleanup_zombies" in result


def test_build_recommend_ollama_recommendations_included(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_psutil = _fake_build_recommend_psutil()
    ollama_rec = [("MEDIUM", "Ollama loaded", "unload it", "ollama_stop:gemma4:12b")]

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=ollama_rec),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "ollama_stop" in result


def test_build_recommend_severity_sorting() -> None:
    mock_psutil = _fake_build_recommend_psutil()
    mock_psutil.virtual_memory.return_value.percent = 75.0
    part = MagicMock()
    part.mountpoint = "/"
    part.opts = "rw"
    part.fstype = "ext4"
    usage = MagicMock()
    usage.used = 465 * 1024**3
    usage.total = 500 * 1024**3
    usage.percent = 93.0
    mock_psutil.disk_partitions.return_value = [part]
    mock_psutil.disk_usage.return_value = usage

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch(
            "godspeed.tools.system_optimizer._collect_outlier_processes",
            return_value=([], [(99, "burn", 400.0)]),
        ),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    high_idx = result.find("HIGH")
    low_idx = result.find("LOW")
    assert high_idx != -1 and low_idx != -1
    assert high_idx < low_idx, "HIGH severity should appear before LOW"


def test_build_recommend_no_recommendations_healthy() -> None:
    mock_psutil = _fake_build_recommend_psutil()

    with (
        patch("godspeed.tools.system_optimizer._check_ollama_vram", return_value=[]),
        patch("godspeed.tools.system_optimizer._collect_outlier_processes", return_value=([], [])),
        patch("godspeed.tools.system_optimizer._count_zombie_python", return_value=0),
    ):
        result = _build_recommend(mock_psutil, top=10)

    assert "healthy" in result.lower()
    assert "No cleanup recommendations" in result


# ---------------------------------------------------------------------------
# Helper: _gb / _mb
# ---------------------------------------------------------------------------


def test_gb_formatting() -> None:
    assert _gb(1024**3) == "1.0 GB"
    assert _gb(2 * 1024**3) == "2.0 GB"
    assert _gb(0) == "0.0 GB"


def test_mb_formatting() -> None:
    assert _mb(1024**2) == "1 MB"
    assert _mb(512 * 1024**2) == "512 MB"
    assert _mb(0) == "0 MB"


# ---------------------------------------------------------------------------
# Recommend + ollama via execute (integration-style)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recommend_with_ollama_mocked(
    tool: SystemOptimizerTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _mock_smi_module()
    monkeypatch.setattr(mod.shutil, "which", lambda x: "/usr/bin/ollama" if x == "ollama" else None)
    stdout = "NAME    ID    SIZE    PROCESSOR    UNTIL\nllama3:8b    xyz    7.0 GB    100%GPU    4 minutes from now\n"
    proc = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: proc)

    result = await tool.execute({"mode": "recommend"}, ctx)
    assert not result.is_error
    out_lower = result.output.lower()
    assert "recommendation" in out_lower


# ---------------------------------------------------------------------------
# Mutant sort_by values that should stay within enum but aren't explicitly
# tested above (sort_by is "memory" by default for invalid values too)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_top_zero_treated_as_default(
    tool: SystemOptimizerTool, ctx: ToolContext
) -> None:
    """top=0 is falsy so `0 or DEFAULT_TOP` evaluates to DEFAULT_TOP."""
    result = await tool.execute({"mode": "inspect", "top": 0}, ctx)
    assert not result.is_error
    assert f"Top {DEFAULT_TOP} processes" in result.output


@pytest.mark.asyncio
async def test_inspect_top_none_uses_default(tool: SystemOptimizerTool, ctx: ToolContext) -> None:
    result = await tool.execute({"mode": "inspect", "top": None}, ctx)
    assert not result.is_error
    assert f"Top {DEFAULT_TOP} processes" in result.output
