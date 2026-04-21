"""System optimizer tool — inspect host resources safely.

Cross-platform tool that lets the agent (or the user via direct call)
see what's consuming CPU, memory, GPU VRAM, and disk on the host. The
productized version of what a human would do with ``Task Manager``,
``htop``, and ``nvidia-smi`` — but structured for an agent.

This commit ships the READ_ONLY ``inspect`` mode only. The ``recommend``
(READ_ONLY) and ``act`` (DESTRUCTIVE) modes land in follow-up commits
once the inspect output format is battle-tested.

Safety guarantees on this initial commit:
  - RiskLevel.READ_ONLY — no process kills, no config changes
  - No shell execution of user-supplied commands
  - GPU introspection via pynvml (in-process, no subprocess) when
    available; falls back to nvidia-smi subprocess if not
  - Hard cap on returned rows (configurable, default 15) — agent can't
    consume megabytes of context with a large process table

Usage from the agent:

    {"tool": "system_optimizer", "arguments": {"mode": "inspect", "top": 10}}

Returns a structured report:

    Mode: inspect (READ_ONLY)
    Platform: win32
    CPU: 28.5% avg over 0.5s (16 logical cores)
    Memory: 42.3 GB / 95.6 GB used (44.2%)
    Swap: 0 B / 0 B used
    GPU 0: NVIDIA RTX 5070 Ti
      Utilization: 73% GPU, 6517 / 16303 MiB VRAM (40.0%)
      Temperature: 63 C
    Disk C:/: 247 GB / 931 GB used (26.5%)

    Top 10 processes by memory:
      PID     MEM        CPU%   NAME
      12345   1234 MB    45.2%  python.exe (experiments/swebench_lite/run.py ...)
      ...

Deny-list design (for future ``act`` mode) is documented in
``_SYSTEM_CRITICAL_NAMES`` below but is NOT enforced yet — there is
nothing to deny in ``inspect``.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import shutil
import subprocess
import sys
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Default limit on process rows returned. Keeps the agent's context small.
DEFAULT_TOP = 10
MAX_TOP = 25

# Per-OS process names that must never be killed by the future ``act`` mode.
# Enforced later; documented here so reviewers see the safety surface up front.
_SYSTEM_CRITICAL_NAMES: dict[str, frozenset[str]] = {
    "win32": frozenset(
        {
            "System",
            "Registry",
            "smss.exe",
            "csrss.exe",
            "wininit.exe",
            "services.exe",
            "lsass.exe",
            "winlogon.exe",
            "explorer.exe",
            "dwm.exe",
            "fontdrvhost.exe",
            "ctfmon.exe",
            "svchost.exe",
            "vmmem",
            "vmmemWSL",
            "Docker Desktop.exe",
            "com.docker.backend.exe",
        }
    ),
    "linux": frozenset(
        {
            "systemd",
            "init",
            "kthreadd",
            "kworker",
            "kswapd0",
            "ksoftirqd",
            "migration",
            "rcu_sched",
            "sshd",
            "Xorg",
            "gnome-shell",
        }
    ),
    "darwin": frozenset(
        {
            "launchd",
            "kernel_task",
            "WindowServer",
            "loginwindow",
            "Finder",
            "Dock",
            "SystemUIServer",
            "ControlCenter",
        }
    ),
}


class SystemOptimizerTool(Tool):
    """Inspect host resources — CPU, memory, GPU, disk, top processes.

    READ_ONLY in this commit. Later commits add ``recommend`` (also
    READ_ONLY, suggests actions based on inspect output) and ``act``
    (DESTRUCTIVE, executes a suggested action with confirmation).
    """

    @property
    def name(self) -> str:
        return "system_optimizer"

    @property
    def description(self) -> str:
        return (
            "Inspect host system resources: CPU/memory/GPU utilization, disk "
            "usage, and top processes by CPU or memory. Read-only; no processes "
            "are killed or modified. Use this when the user reports slowness, "
            "when a benchmark run is behaving oddly, or before recommending "
            "resource cleanup (Ollama model unload, etc.)."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["inspect", "recommend"],
                    "description": (
                        "Operation mode. 'inspect' returns raw resource state; "
                        "'recommend' returns a ranked list of safe cleanup actions "
                        "(e.g. 'unload Ollama model X — idle 20 min, 15 GB VRAM'). "
                        "Both modes are READ_ONLY."
                    ),
                },
                "top": {
                    "type": "integer",
                    "description": (
                        f"How many top processes to list (default {DEFAULT_TOP}, max {MAX_TOP})."
                    ),
                    "minimum": 1,
                    "maximum": MAX_TOP,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["memory", "cpu"],
                    "description": "Sort top processes by 'memory' (default) or 'cpu'.",
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        mode = arguments.get("mode", "inspect")
        if mode not in ("inspect", "recommend"):
            return ToolResult.failure(
                f"mode={mode!r} not supported in this release; "
                "choose 'inspect' or 'recommend' (both READ_ONLY)."
            )

        top = min(MAX_TOP, max(1, int(arguments.get("top") or DEFAULT_TOP)))
        sort_by = arguments.get("sort_by", "memory")
        if sort_by not in ("memory", "cpu"):
            sort_by = "memory"

        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return ToolResult.failure(
                "psutil is not installed. Add the [system] optional dependency: "
                "pip install psutil (>=5.9)."
            )

        if mode == "recommend":
            return ToolResult.success(_build_recommend(psutil, top=top))

        lines: list[str] = []
        lines.append("Mode: inspect (READ_ONLY)")
        lines.append(f"Platform: {sys.platform} ({platform.system()} {platform.release()})")
        lines.append("")
        lines.extend(_cpu_mem_summary(psutil))
        lines.append("")
        lines.extend(_gpu_summary())
        lines.append("")
        lines.extend(_disk_summary(psutil))
        lines.append("")
        lines.extend(_top_processes(psutil, top=top, sort_by=sort_by))
        return ToolResult.success("\n".join(lines))


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _cpu_mem_summary(psutil: Any) -> list[str]:
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count(logical=True) or 0
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return [
        f"CPU: {cpu_percent:.1f}% avg over 0.5s ({cpu_count} logical cores)",
        (f"Memory: {_gb(vmem.used)} / {_gb(vmem.total)} used ({vmem.percent:.1f}%)"),
        f"Swap: {_gb(swap.used)} / {_gb(swap.total)} used",
    ]


def _gpu_summary() -> list[str]:
    """Try pynvml first (in-process); fall back to nvidia-smi subprocess."""
    lines = _gpu_via_pynvml()
    if lines is not None:
        return lines
    lines = _gpu_via_nvidia_smi()
    if lines is not None:
        return lines
    return ["GPU: no NVIDIA GPU detected (or pynvml + nvidia-smi both unavailable)"]


def _gpu_via_pynvml() -> list[str] | None:
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    out: list[str] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            out.append(f"GPU {i}: {name}")
            out.append(
                f"  Utilization: {util.gpu}% GPU, "
                f"{mem.used // (1024 * 1024)} / {mem.total // (1024 * 1024)} MiB VRAM "
                f"({100 * mem.used / max(1, mem.total):.1f}%)"
            )
            out.append(f"  Temperature: {temp} C")
    except Exception as exc:
        logger.debug("pynvml query failed: %s", exc)
        out.append(f"GPU: pynvml query error ({exc})")
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return out


def _gpu_via_nvidia_smi() -> list[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("nvidia-smi call failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    lines: list[str] = []
    for i, row in enumerate(result.stdout.strip().splitlines()):
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 5:
            continue
        name, util, mem_used, mem_total, temp = parts
        lines.append(f"GPU {i}: {name}")
        lines.append(
            f"  Utilization: {util}% GPU, {mem_used} / {mem_total} MiB VRAM "
            f"({100 * int(mem_used) / max(1, int(mem_total)):.1f}%)"
        )
        lines.append(f"  Temperature: {temp} C")
    return lines or None


def _disk_summary(psutil: Any) -> list[str]:
    """Report usage of mounted partitions we can safely query."""
    out: list[str] = []
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:
        return ["Disk: partition enumeration failed"]
    # On Windows, filter out removable / optical drives.
    for part in partitions:
        if "cdrom" in (part.opts or "").lower():
            continue
        if sys.platform == "win32" and part.fstype == "":
            # Often removable media with no media inserted.
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        out.append(
            f"Disk {part.mountpoint}: {_gb(usage.used)} / {_gb(usage.total)} used "
            f"({usage.percent:.1f}%)"
        )
    return out or ["Disk: no partitions reported"]


def _top_processes(psutil: Any, *, top: int, sort_by: str) -> list[str]:
    """Return the top-N processes by memory (default) or CPU."""
    procs: list[tuple[int, str, float, int]] = []
    # First pass: prime cpu_percent (psutil needs two samples).
    for p in psutil.process_iter(attrs=["pid"]):
        with contextlib.suppress(Exception):
            p.cpu_percent(interval=None)
    # Small wait so the second cpu_percent reads a real delta.
    import time as _time

    _time.sleep(0.2)

    for p in psutil.process_iter(attrs=["pid", "name"]):
        try:
            with p.oneshot():
                pid = p.pid
                name = p.name() or "?"
                cpu = p.cpu_percent(interval=None)
                mem = p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:  # noqa: S112 - per-process read is best-effort
            continue
        procs.append((pid, name, cpu, mem))

    if sort_by == "cpu":
        procs.sort(key=lambda r: (r[2], r[3]), reverse=True)
        header = f"Top {top} processes by CPU:"
    else:
        procs.sort(key=lambda r: (r[3], r[2]), reverse=True)
        header = f"Top {top} processes by memory:"

    lines = [header]
    lines.append(f"  {'PID':>7}  {'MEM':>9}  {'CPU%':>6}  NAME")
    for pid, name, cpu, mem in procs[:top]:
        star = " *" if _is_system_critical(name) else ""
        lines.append(f"  {pid:>7}  {_mb(mem):>9}  {cpu:>6.1f}  {name}{star}")
    lines.append("  * = system-critical (will be protected by future 'act' mode)")
    return lines


def _is_system_critical(name: str) -> bool:
    deny = _SYSTEM_CRITICAL_NAMES.get(sys.platform, frozenset())
    # On linux, kworker/* etc. — prefix match for that one pattern.
    if sys.platform == "linux":
        return any(name == critical or name.startswith(critical + "/") for critical in deny)
    return name in deny


def _build_recommend(psutil: Any, *, top: int) -> str:
    """Return a ranked list of safe cleanup recommendations.

    READ_ONLY — only *suggests* actions; doesn't execute them. Each
    recommendation carries a severity (HIGH / MEDIUM / LOW), a short
    rationale, and an ``action_id`` that the future ``act`` mode will
    use to dispatch. Recommendations that would touch a system-critical
    process are omitted.
    """
    recs: list[tuple[str, str, str, str]] = []  # (severity, title, rationale, action_id)

    # Collect raw state (duplicates some of the inspect-mode queries
    # intentionally; keeps the two modes independent).
    vmem = psutil.virtual_memory()
    if vmem.percent >= 90:
        recs.append(
            (
                "HIGH",
                f"System memory at {vmem.percent:.1f}% ({_gb(vmem.used)}/{_gb(vmem.total)})",
                "Close memory-hog user applications; consider swap expansion.",
                "investigate_memory_pressure",
            )
        )
    elif vmem.percent >= 80:
        recs.append(
            (
                "MEDIUM",
                f"System memory at {vmem.percent:.1f}% ({_gb(vmem.used)}/{_gb(vmem.total)})",
                "Review top memory consumers before starting heavy workloads.",
                "investigate_memory_pressure",
            )
        )

    # Disk: any partition > 85%
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:
        partitions = []
    for part in partitions:
        if "cdrom" in (part.opts or "").lower():
            continue
        if sys.platform == "win32" and part.fstype == "":
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        if usage.percent >= 90:
            recs.append(
                (
                    "HIGH",
                    f"Disk {part.mountpoint} at {usage.percent:.1f}% "
                    f"({_gb(usage.used)} / {_gb(usage.total)})",
                    (
                        "Risk of write failures. Clear cache/temp dirs, old logs, "
                        "orphaned build artifacts."
                    ),
                    f"investigate_disk_pressure:{part.mountpoint}",
                )
            )
        elif usage.percent >= 85:
            recs.append(
                (
                    "MEDIUM",
                    f"Disk {part.mountpoint} at {usage.percent:.1f}%",
                    "Review cache dirs; plan cleanup before hitting 90%.",
                    f"investigate_disk_pressure:{part.mountpoint}",
                )
            )

    # GPU: VRAM high and Ollama listed as a consumer
    ollama_vram_recs = _check_ollama_vram()
    recs.extend(ollama_vram_recs)

    # Top processes: flag non-critical processes consuming > 3 GB memory or > 200% CPU
    big_mem_procs, high_cpu_procs = _collect_outlier_processes(psutil, top=top)
    for pid, name, mem in big_mem_procs:
        if _is_system_critical(name):
            continue
        recs.append(
            (
                "LOW",
                f"{name} (pid {pid}) holds {_gb(mem)} memory",
                ("Consider closing if idle — this is the largest non-system process on the host."),
                f"kill_process:{pid}:{name}",
            )
        )
    for pid, name, cpu in high_cpu_procs:
        if _is_system_critical(name):
            continue
        recs.append(
            (
                "LOW",
                f"{name} (pid {pid}) using {cpu:.0f}% CPU",
                "Sustained high CPU. Expected if this is the active app (e.g. game).",
                f"kill_process:{pid}:{name}",
            )
        )

    # Zombie Python processes (platform: linux/macos)
    zombie_procs = _count_zombie_python(psutil)
    if zombie_procs:
        recs.append(
            (
                "LOW",
                f"{zombie_procs} zombie Python process(es)",
                "Orphaned children of a killed parent; reap by restarting the parent.",
                "cleanup_zombies",
            )
        )

    # Order by severity (HIGH > MEDIUM > LOW) then alphabetically by title
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: (rank.get(r[0], 99), r[1]))

    out = ["Mode: recommend (READ_ONLY)"]
    out.append(f"Platform: {sys.platform} ({platform.system()} {platform.release()})")
    out.append("")
    if not recs:
        out.append("No cleanup recommendations — system state is healthy.")
        return "\n".join(out)

    out.append(f"{len(recs)} recommendation(s), most severe first:")
    out.append("")
    for i, (sev, title, rationale, action_id) in enumerate(recs, 1):
        out.append(f"  {i}. [{sev}] {title}")
        out.append(f"       {rationale}")
        out.append(f"       action_id: {action_id!r}")
        out.append("")
    out.append(
        "Note: 'act' mode (to execute these) is NOT yet implemented. "
        "Until it ships, treat these as suggestions only; execute manually "
        "after reviewing the specifics."
    )
    return "\n".join(out)


def _check_ollama_vram() -> list[tuple[str, str, str, str]]:
    """If Ollama has any model loaded, recommend unloading.

    Reads `ollama ps` output. Returns empty list if ollama CLI is absent
    or no models are loaded.
    """
    if shutil.which("ollama") is None:
        return []
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    # Parse `ollama ps`: first line is header, then each line is a model.
    recs: list[tuple[str, str, str, str]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        size = " ".join(parts[2:4]) if len(parts) >= 4 else "?"
        # Heuristic: any loaded model is a candidate for unload when user
        # reports slowness. Phase 7 part 3's 'act' mode can dispatch the
        # actual 'ollama stop' call.
        recs.append(
            (
                "MEDIUM",
                f"Ollama has model {name!r} loaded ({size})",
                (
                    "Holds VRAM even when idle. If not actively serving Godspeed, "
                    "`ollama stop` to reclaim."
                ),
                f"ollama_stop:{name}",
            )
        )
    return recs


def _collect_outlier_processes(
    psutil: Any, *, top: int
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, float]]]:
    """Return (big_memory, high_cpu) outlier lists, each bounded by top."""
    big_mem: list[tuple[int, str, int]] = []
    high_cpu: list[tuple[int, str, float]] = []
    # Prime cpu_percent
    for p in psutil.process_iter(attrs=["pid"]):
        with contextlib.suppress(Exception):
            p.cpu_percent(interval=None)
    import time as _time

    _time.sleep(0.2)

    for p in psutil.process_iter(attrs=["pid", "name"]):
        try:
            with p.oneshot():
                pid = p.pid
                name = p.name() or "?"
                mem = p.memory_info().rss
                cpu = p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:  # noqa: S112
            continue
        # Skip synthetic OS accounting processes that psutil exposes but
        # that aren't real workloads. "System Idle Process" (PID 0 on
        # Windows) is Windows' idle-time counter; it will always look
        # like 100-400% CPU depending on cores.
        if pid == 0 or name.lower() in ("system idle process", "idle"):
            continue
        if mem > 3 * 1024**3:  # 3 GB
            big_mem.append((pid, name, mem))
        if cpu > 200.0:
            high_cpu.append((pid, name, cpu))

    big_mem.sort(key=lambda r: r[2], reverse=True)
    high_cpu.sort(key=lambda r: r[2], reverse=True)
    return big_mem[:top], high_cpu[:top]


def _count_zombie_python(psutil: Any) -> int:
    """Count zombie-state Python processes. Meaningful on linux/macos."""
    if sys.platform == "win32":
        return 0
    count = 0
    for p in psutil.process_iter(attrs=["pid", "name", "status"]):
        try:
            if p.info.get("status") == psutil.STATUS_ZOMBIE and (
                p.info.get("name") or ""
            ).lower().startswith("python"):
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count


def _gb(n: int) -> str:
    return f"{n / (1024**3):.1f} GB"


def _mb(n: int) -> str:
    return f"{n / (1024**2):.0f} MB"
