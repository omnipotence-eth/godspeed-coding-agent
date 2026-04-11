"""Test runner tool — detect and run project test suites.

Auto-detects the project's test framework and runs targeted tests after edits.
Supports pytest, jest/vitest, go test, and cargo test.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Timeout for test execution (generous — tests can be slow)
TEST_TIMEOUT = 60

# Max output to capture from test runner
MAX_OUTPUT_CHARS = 5000


class TestRunnerTool(Tool):
    """Run project tests to validate changes.

    Auto-detects the test framework based on project files:
    - pytest (pyproject.toml/setup.py/conftest.py)
    - jest/vitest (package.json with test script)
    - go test (go.mod)
    - cargo test (Cargo.toml)

    Can run all tests or target specific files/directories.
    """

    @property
    def name(self) -> str:
        return "test_runner"

    @property
    def description(self) -> str:
        return (
            "Run project tests to validate changes. Auto-detects the test framework. "
            "Optionally target a specific file or directory. Returns pass/fail with output."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Optional: specific test file or directory to run. "
                        "If empty, runs the full test suite."
                    ),
                },
                "framework": {
                    "type": "string",
                    "description": (
                        "Optional: force a specific framework (pytest, jest, vitest, "
                        "go, cargo). Auto-detected if omitted."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        target = arguments.get("target", "")
        framework = arguments.get("framework", "")

        if not framework:
            framework = detect_framework(context.cwd)
            if not framework:
                return ToolResult.success(
                    "No test framework detected. Looked for: "
                    "pytest (conftest.py/pyproject.toml), jest/vitest (package.json), "
                    "go test (go.mod), cargo test (Cargo.toml)."
                )

        runner = _RUNNERS.get(framework)
        if runner is None:
            return ToolResult.failure(
                f"Unknown test framework: {framework}. Supported: {', '.join(_RUNNERS.keys())}"
            )

        return runner(context.cwd, target)


def detect_framework(cwd: Path) -> str:
    """Detect the project's test framework from project files."""
    # Python: pytest
    if (cwd / "conftest.py").exists() or (cwd / "tests" / "conftest.py").exists():
        return "pytest"
    if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
        # Check if pytest is in pyproject.toml
        pyproject = cwd / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "pytest" in content:
                    return "pytest"
            except OSError:
                pass
        return "pytest"  # Default for Python projects

    # JavaScript/TypeScript: jest or vitest
    package_json = cwd / "package.json"
    if package_json.exists():
        try:
            import json

            data = json.loads(package_json.read_text(encoding="utf-8"))
            scripts = data.get("scripts", {})
            if "test" in scripts:
                test_cmd = scripts["test"]
                if "vitest" in test_cmd:
                    return "vitest"
                return "jest"
        except (OSError, json.JSONDecodeError):
            pass

    # Go
    if (cwd / "go.mod").exists():
        return "go"

    # Rust
    if (cwd / "Cargo.toml").exists():
        return "cargo"

    return ""


def _run_tests(
    cmd: list[str],
    cwd: Path,
    framework: str,
) -> ToolResult:
    """Run a test command and return structured results."""
    bin_name = cmd[0]
    resolved_bin = shutil.which(bin_name)
    if resolved_bin is None:
        return ToolResult.success(f"{bin_name} not found — cannot run tests.")

    cmd[0] = resolved_bin

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(f"Tests timed out after {TEST_TIMEOUT}s ({framework})")
    except OSError as exc:
        return ToolResult.failure(f"Failed to run {framework}: {exc}")

    output = result.stdout.strip()
    errors = result.stderr.strip()
    combined = output
    if errors:
        combined = f"{output}\n\nSTDERR:\n{errors}" if output else errors

    # Truncate long output
    if len(combined) > MAX_OUTPUT_CHARS:
        truncated = len(combined) - MAX_OUTPUT_CHARS
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n... ({truncated} chars truncated)"

    if result.returncode == 0:
        return ToolResult.success(f"Tests PASSED ({framework}):\n{combined}")

    return ToolResult.success(f"Tests FAILED ({framework}, exit={result.returncode}):\n{combined}")


def _run_pytest(cwd: Path, target: str) -> ToolResult:
    """Run pytest."""
    cmd = ["pytest", "-x", "--tb=short", "-q"]
    if target:
        cmd.append(target)
    return _run_tests(cmd, cwd, "pytest")


def _run_jest(cwd: Path, target: str) -> ToolResult:
    """Run jest via npx."""
    cmd = ["npx", "jest", "--no-coverage", "--bail"]
    if target:
        cmd.append(target)
    return _run_tests(cmd, cwd, "jest")


def _run_vitest(cwd: Path, target: str) -> ToolResult:
    """Run vitest via npx."""
    cmd = ["npx", "vitest", "run", "--reporter=verbose"]
    if target:
        cmd.append(target)
    return _run_tests(cmd, cwd, "vitest")


def _run_go_test(cwd: Path, target: str) -> ToolResult:
    """Run go test."""
    cmd = ["go", "test", "-v"]
    if target:
        cmd.append(target)
    else:
        cmd.append("./...")
    return _run_tests(cmd, cwd, "go")


def _run_cargo_test(cwd: Path, target: str) -> ToolResult:
    """Run cargo test."""
    cmd = ["cargo", "test"]
    if target:
        cmd.extend(["--", target])
    return _run_tests(cmd, cwd, "cargo")


_RUNNERS: dict[str, Any] = {
    "pytest": _run_pytest,
    "jest": _run_jest,
    "vitest": _run_vitest,
    "go": _run_go_test,
    "cargo": _run_cargo_test,
}
