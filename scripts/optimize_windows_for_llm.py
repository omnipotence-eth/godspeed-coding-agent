"""Windows system optimizer for local LLM development.

Configures pagefile, enables long paths, sets performance power plan,
and disables unnecessary services that compete for RAM/VRAM.

Requires administrator privileges for some operations.

Usage:
    python scripts/optimize_windows_for_llm.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Optimal pagefile size for 96GB RAM + 12GB VRAM running 35B models
TARGET_PAGEFILE_MIN_GB = 48
TARGET_PAGEFILE_MAX_GB = 64


def _is_admin() -> bool:
    """Check if the script is running with administrator privileges."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _get_current_pagefile_gb() -> tuple[int, int] | None:
    """Return (initial_gb, max_gb) from registry, or None if auto-managed."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management",
        ) as key:
            paging_files, _ = winreg.QueryValueEx(key, "PagingFiles")
            if isinstance(paging_files, list):
                paging_files = paging_files[0]
            if not isinstance(paging_files, str):
                return None
            # Format: "C:\pagefile.sys 49152 65536" or "C:\pagefile.sys"
            parts = paging_files.split()
            if len(parts) >= 3:
                return int(parts[1]) // 1024, int(parts[2]) // 1024
            # Single value or auto-managed
            return None
    except Exception as exc:
        logger.debug("Could not read pagefile registry: %s", exc)
        return None


def _set_pagefile(min_gb: int, max_gb: int, *, dry_run: bool = False) -> bool:
    """Set Windows pagefile to fixed size. Returns True on success."""
    if sys.platform != "win32":
        logger.warning("Pagefile optimization is Windows-only")
        return False

    if not _is_admin():
        logger.error("Administrator privileges required to change pagefile size")
        return False

    current = _get_current_pagefile_gb()
    if current == (min_gb, max_gb):
        logger.info("Pagefile already set to %d-%d GB", min_gb, max_gb)
        return True

    logger.info(
        "Setting pagefile: %d-%d GB (current: %s)",
        min_gb,
        max_gb,
        f"{current[0]}-{current[1]} GB" if current else "auto-managed",
    )

    if dry_run:
        logger.info("[DRY-RUN] Would set pagefile to %d-%d GB", min_gb, max_gb)
        return True

    try:
        import winreg

        min_mb = min_gb * 1024
        max_mb = max_gb * 1024
        value = f"C:\\pagefile.sys {min_mb} {max_mb}"

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "PagingFiles", 0, winreg.REG_MULTI_SZ, [value])

        logger.info("Pagefile set to %d-%d GB. Reboot required to take effect.", min_gb, max_gb)
        return True
    except Exception as exc:
        logger.error("Failed to set pagefile: %s", exc)
        return False


def _enable_long_paths(*, dry_run: bool = False) -> bool:
    """Enable Windows long path support (>260 chars). Returns True on success."""
    if sys.platform != "win32":
        logger.warning("Long paths optimization is Windows-only")
        return False

    if not _is_admin():
        logger.error("Administrator privileges required to enable long paths")
        return False

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0,
            winreg.KEY_QUERY_VALUE,
        ) as key:
            current, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            if current == 1:
                logger.info("Long paths already enabled")
                return True
    except Exception as exc:
        logger.debug("Could not query long paths: %s", exc)

    logger.info("Enabling Windows long path support")

    if dry_run:
        logger.info("[DRY-RUN] Would enable long paths via registry")
        return True

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)

        logger.info("Long paths enabled. Reboot required to take effect.")
        return True
    except Exception as exc:
        logger.error("Failed to enable long paths: %s", exc)
        return False


def _set_high_performance_power_plan(*, dry_run: bool = False) -> bool:
    """Set Windows power plan to High Performance. Returns True on success."""
    if sys.platform != "win32":
        logger.warning("Power plan optimization is Windows-only")
        return False

    import shutil
    import subprocess

    if shutil.which("powercfg") is None:
        logger.warning("powercfg not found")
        return False

    # High Performance GUID
    high_perf_guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

    try:
        result = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            check=False,
        )
        if high_perf_guid in result.stdout:
            logger.info("High Performance power plan already active")
            return True
    except Exception as exc:
        logger.debug("Could not check active power plan: %s", exc)

    logger.info("Setting High Performance power plan")

    if dry_run:
        logger.info("[DRY-RUN] Would set High Performance power plan")
        return True

    try:
        subprocess.run(
            ["powercfg", "/setactive", high_perf_guid],
            capture_output=True,
            check=False,
        )
        logger.info("High Performance power plan activated")
        return True
    except Exception as exc:
        logger.error("Failed to set power plan: %s", exc)
        return False


def _disable_windows_search(*, dry_run: bool = False) -> bool:
    """Disable Windows Search indexing to free RAM. Returns True on success."""
    if sys.platform != "win32":
        return False

    import subprocess

    logger.info("Disabling Windows Search indexing")

    if dry_run:
        logger.info("[DRY-RUN] Would disable Windows Search service")
        return True

    try:
        subprocess.run(
            ["sc", "config", "wsearch", "start=", "disabled"],
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["sc", "stop", "wsearch"],
            capture_output=True,
            check=False,
        )
        logger.info("Windows Search disabled")
        return True
    except Exception as exc:
        logger.error("Failed to disable Windows Search: %s", exc)
        return False


def _disable_sysmain(*, dry_run: bool = False) -> bool:
    """Disable SysMain (Superfetch) to reduce background disk I/O. Returns True."""
    if sys.platform != "win32":
        return False

    import subprocess

    logger.info("Disabling SysMain (Superfetch)")

    if dry_run:
        logger.info("[DRY-RUN] Would disable SysMain service")
        return True

    try:
        subprocess.run(
            ["sc", "config", "SysMain", "start=", "disabled"],
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["sc", "stop", "SysMain"],
            capture_output=True,
            check=False,
        )
        logger.info("SysMain disabled")
        return True
    except Exception as exc:
        logger.error("Failed to disable SysMain: %s", exc)
        return False


def _create_directories(*, dry_run: bool = False) -> bool:
    """Create standard directories for llama.cpp models and cache."""
    dirs = [
        Path.home() / ".llamacpp" / "models",
        Path.home() / ".llamacpp" / "cache",
        Path.home() / ".godspeed",
    ]

    for d in dirs:
        if dry_run:
            logger.info("[DRY-RUN] Would create directory: %s", d)
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Directory ready: %s", d)
        except Exception as exc:
            logger.error("Failed to create %s: %s", d, exc)
            return False
    return True


def main() -> int:
    """Run all optimizations. Returns 0 on success, 1 on partial failure."""
    parser = argparse.ArgumentParser(
        description="Optimize Windows for local LLM development",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making modifications",
    )
    parser.add_argument(
        "--pagefile-min-gb",
        type=int,
        default=TARGET_PAGEFILE_MIN_GB,
        help=f"Minimum pagefile size in GB (default: {TARGET_PAGEFILE_MIN_GB})",
    )
    parser.add_argument(
        "--pagefile-max-gb",
        type=int,
        default=TARGET_PAGEFILE_MAX_GB,
        help=f"Maximum pagefile size in GB (default: {TARGET_PAGEFILE_MAX_GB})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if sys.platform != "win32":
        logger.error("This script is Windows-only")
        return 1

    if not _is_admin():
        logger.warning("Not running as administrator — some changes will be skipped")
        logger.warning("Re-run as admin for full optimization: right-click → Run as administrator")

    results: list[tuple[str, bool]] = []

    results.append(
        (
            "Pagefile",
            _set_pagefile(
                args.pagefile_min_gb,
                args.pagefile_max_gb,
                dry_run=args.dry_run,
            ),
        )
    )
    results.append(("Long paths", _enable_long_paths(dry_run=args.dry_run)))
    results.append(
        ("High Performance power plan", _set_high_performance_power_plan(dry_run=args.dry_run))
    )
    results.append(("Windows Search", _disable_windows_search(dry_run=args.dry_run)))
    results.append(("SysMain", _disable_sysmain(dry_run=args.dry_run)))
    results.append(("Directories", _create_directories(dry_run=args.dry_run)))

    logger.info("")
    logger.info("Optimization summary:")
    all_passed = True
    for name, ok in results:
        status = "OK" if ok else "FAIL"
        logger.info("  %-30s %s", name, status)
        if not ok:
            all_passed = False

    if args.dry_run:
        logger.info("")
        logger.info("This was a dry run. Remove --dry-run to apply changes.")
        return 0

    reboot_needed = any(name in ("Pagefile", "Long paths") and ok for name, ok in results)
    if reboot_needed:
        logger.info("")
        logger.info("REBOOT REQUIRED for some changes to take effect.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
