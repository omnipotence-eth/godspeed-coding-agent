"""Pre-flight checks for benchmark runs.

Verifies the execution environment before starting a long benchmark run:
- NVIDIA NIM API key connectivity
- Docker availability (required for SWE-bench verification)
- WSL availability (Windows hosts)
- Disk space sufficiency
- Python environment (swebench package, sb-cli)
- Network connectivity to NIM endpoint

Run with:
    python -m godspeed.benchmarks.preflight --all
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

NIM_ENDPOINT = "https://api.nvidia.com/v1"
MIN_DISK_GB = 20
MIN_DOCKER_VERSION = (20, 0)
POOR_CONNECTIVITY_MS = 5000


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    fatal: bool = False


@dataclass
class PreFlightReport:
    results: list[CheckResult] = field(default_factory=list)
    all_passed: bool = True

    def add(self, name: str, passed: bool, detail: str = "", fatal: bool = False) -> None:
        self.results.append(CheckResult(name=name, passed=passed, detail=detail, fatal=fatal))
        if not passed:
            self.all_passed = False


def check_nim_connectivity(report: PreFlightReport, keys_env: str = "NVIDIA_NIM_API_KEYS") -> None:
    """Verify NIM API keys can authenticate against the NVIDIA endpoint."""
    raw = os.environ.get(keys_env, os.environ.get("NVIDIA_NIM_API_KEY", ""))
    if not raw:
        report.add(
            "NIM keys", False, f"Neither {keys_env} nor NVIDIA_NIM_API_KEY is set", fatal=True
        )
        return

    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        report.add("NIM keys", False, "No non-empty keys found", fatal=True)
        return

    report.add("NIM key count", True, f"{len(keys)} key(s) configured")

    import urllib.request

    healthy = 0
    for i, key in enumerate(keys, 1):
        try:
            req = urllib.request.Request(  # noqa: S310
                f"{NIM_ENDPOINT}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            urllib.request.urlopen(req, timeout=10)  # noqa: S310
            healthy += 1
        except Exception as e:  # noqa: BLE001
            error = str(e)[:120]
            report.add(f"NIM key #{i}", False, f"Key ending ...{key[-8:]} failed: {error}")

    if healthy == 0:
        report.add("NIM connectivity", False, "All keys failed to authenticate", fatal=True)
    elif healthy < len(keys):
        report.add("NIM connectivity", True, f"{healthy}/{len(keys)} keys healthy (some degraded)")
    else:
        report.add("NIM connectivity", True, f"All {len(keys)} keys authenticated")


def check_python_env(report: PreFlightReport) -> None:
    """Verify required Python packages are importable."""
    packages = {
        "godspeed": "godspeed package",
        "litellm": "LiteLLM",
        "datasets": "HuggingFace datasets (SWE-bench)",
    }
    for pkg, label in packages.items():
        try:
            __import__(pkg)
            report.add(f"Python: {label}", True, "installed")
        except ImportError:
            report.add(
                f"Python: {label}",
                False,
                f"{pkg} not installed — run: pip install {pkg}",
                fatal=pkg == "godspeed",
            )

    # swebench CLI (optional but recommended)
    if shutil.which("swebench") or shutil.which("sb-cli"):
        report.add("Python: sb-cli", True, "available")
    else:
        report.add(
            "Python: sb-cli",
            True,
            "not found — needed to submit results. Install: pip install sb-cli",
        )


def check_docker(report: PreFlightReport) -> None:
    """Verify Docker daemon is running and functional."""
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        report.add("Docker", False, "docker command not found on PATH", fatal=False)
        report.add(
            "Docker note",
            True,
            "SWE-bench verification requires Docker; run without --agent-in-loop to skip",
        )
        return

    try:
        result = subprocess.run(
            [docker_bin, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            report.add("Docker", True, f"v{version} running")
        else:
            stderr_tail = result.stderr.strip()[-200:] if result.stderr else "unknown error"
            report.add("Docker", False, f"daemon not accessible: {stderr_tail}")
    except subprocess.TimeoutExpired:
        report.add("Docker", False, "command timed out — daemon unresponsive")
    except FileNotFoundError:
        report.add("Docker", False, "docker executable vanished mid-check")


def check_wsl(report: PreFlightReport) -> None:
    """Verify WSL availability on Windows hosts."""
    if sys.platform != "win32":
        return  # Not Windows — nothing to check

    wsl_bin = shutil.which("wsl.exe") or shutil.which("wsl")
    if wsl_bin is None:
        report.add(
            "WSL", False, "WSL not found; SWE-bench Docker verification unavailable on Windows"
        )
        return

    try:
        result = subprocess.run(
            [str(wsl_bin), "-d", "Ubuntu", "-e", "bash", "-c", "echo ok"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            report.add("WSL", True, "Ubuntu available")
        else:
            report.add("WSL", False, f"Ubuntu WSL not responding: {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        report.add("WSL", False, "WSL command timed out")
    except FileNotFoundError:
        report.add("WSL", False, "WSL binary not found")


def check_disk_space(report: PreFlightReport, min_gb: int = MIN_DISK_GB) -> None:
    """Verify sufficient free disk space."""
    try:
        usage = shutil.disk_usage(Path.cwd())
        free_gb = usage.free / (1024**3)
        if free_gb >= min_gb:
            report.add("Disk space", True, f"{free_gb:.1f} GB free (≥ {min_gb} GB required)")
        else:
            report.add(
                "Disk space",
                False,
                f"Only {free_gb:.1f} GB free ({min_gb} GB required). "
                f"SWE-bench clones repositories — free up space before running.",
                fatal=True,
            )
    except OSError as e:
        report.add("Disk space", False, f"Unable to check: {e}")


def check_nim_rpm(report: PreFlightReport, keys_env: str = "NVIDIA_NIM_API_KEYS") -> None:
    """Estimate effective RPM capacity with configured keys."""
    raw = os.environ.get(keys_env, os.environ.get("NVIDIA_NIM_API_KEY", ""))
    if not raw:
        return
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    rpm = len(keys) * 30
    report.add("NIM RPM capacity", True, f"{len(keys)} keys x 30 RPM = {rpm} RPM effective")


def run_all_checks() -> PreFlightReport:
    """Run all pre-flight checks. Returns a report with pass/fail per check."""
    report = PreFlightReport()
    t0 = time.monotonic()

    check_nim_connectivity(report)
    check_nim_rpm(report)
    check_python_env(report)
    check_docker(report)
    check_wsl(report)
    check_disk_space(report)

    ms = (time.monotonic() - t0) * 1000
    report.add("Pre-flight total time", True, f"{ms:.0f}ms")

    return report


def print_report(report: PreFlightReport) -> int:
    """Pretty-print the pre-flight report. Returns 0 if all pass, 1 otherwise."""
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        if r.fatal:
            status = "FATAL"
        detail_str = f" — {r.detail}" if r.detail else ""
        print(f"  [{status:>5}] {r.name}{detail_str}")  # noqa: T201

    fatal_count = sum(1 for r in report.results if r.fatal and not r.passed)
    fail_count = sum(1 for r in report.results if not r.fatal and not r.passed)

    if report.all_passed:
        return 0

    summary_parts = []
    if fatal_count:
        summary_parts.append(f"{fatal_count} FATAL")
    if fail_count:
        summary_parts.append(f"{fail_count} failed")
    return 1


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Godspeed benchmark pre-flight checks")
    parser.add_argument("--all", action="store_true", default=True, help="Run all checks")
    parser.add_argument("--check-nim", action="store_true", help="Check NIM connectivity only")
    parser.add_argument("--check-docker", action="store_true", help="Check Docker only")
    parser.add_argument("--check-disk", action="store_true", help="Check disk space only")
    parser.add_argument("--quiet", action="store_true", help="Exit code only, no output")
    args = parser.parse_args()

    report = run_all_checks()

    if args.quiet:
        return 0 if report.all_passed else 1

    return print_report(report)


if __name__ == "__main__":
    sys.exit(main())
