"""Agent-in-loop tool: run the SWE-Bench Docker harness from inside the agent session.

This tool is scaffolding for the agent-in-loop Docker work described in
`AGENT_IN_LOOP_DESIGN.md`. It is *not yet wired into `run.py`* — that is a
follow-up commit requiring user sign-off (see design doc).

Usage (once wired):

    from experiments.swebench_lite.docker_test_tool import SWEBenchVerifyTool

    registry.register(
        SWEBenchVerifyTool(
            instance_id="sqlfluff__sqlfluff-2419",
            model_name="nvidia_nim/moonshotai/kimi-k2.5",
            workdir=Path("experiments/swebench_lite"),
            split="dev",
        )
    )

The agent sees the tool in its toolset and can call
`swebench_verify_patch` with no arguments. The tool captures the current
working-tree diff via `git diff`, feeds it to `verify_patch.verify_patch`,
and returns the harness verdict + a tail of the test output.

Each call spins up a fresh swebench container - ~60-90 seconds per
invocation. Agents should be prompted to call it sparingly (e.g.
only after believing the fix is complete).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

# verify_patch lives in the same directory; make it importable whether
# this tool is loaded as a script or via the experiments package.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from verify_patch import verify_patch  # noqa: E402

logger = logging.getLogger(__name__)

MAX_OUTPUT_TAIL_CHARS = 2000


class SWEBenchVerifyTool(Tool):
    """Run the SWE-Bench test harness on the agent's current edits.

    Bound to a single instance at construction. The agent only sees
    the tool's name/description/schema — the instance id and model name
    are closed over at registration time so the agent cannot spoof them.
    """

    def __init__(
        self,
        instance_id: str,
        model_name: str,
        workdir: Path,
        split: str = "dev",
        timeout_s: int = 900,
    ) -> None:
        self.instance_id = instance_id
        self.model_name = model_name
        self.workdir = workdir
        self.split = split
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return "swebench_verify_patch"

    @property
    def description(self) -> str:
        return (
            "Run the SWE-Bench test harness on your current edits. "
            "This runs the failing test (and related tests) inside a Docker "
            "container configured for this instance. Returns whether the "
            "instance is now resolved plus a tail of the test output. "
            "Each call takes ~60-90 seconds — use sparingly, typically only "
            "after you believe your fix is complete."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        diff = _capture_diff(context.cwd)
        if not diff.strip():
            return ToolResult.success(
                "resolved=False\n\n"
                "(working tree has no changes — nothing to verify. "
                "Make your edits first, then call this tool.)"
            )

        logger.info(
            "swebench_verify_patch: instance=%s diff_lines=%d",
            self.instance_id,
            diff.count("\n"),
        )
        try:
            resolved, test_output = verify_patch(
                instance_id=self.instance_id,
                model_name=self.model_name,
                model_patch=diff,
                workdir=self.workdir,
                timeout_s=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return ToolResult.failure(
                f"harness timed out after {e.timeout}s — the container may be "
                f"slow or the test may hang. Consider narrowing your edit."
            )
        except Exception as exc:
            logger.exception("verify_patch raised")
            return ToolResult.failure(f"harness invocation failed: {exc}")

        tail = test_output[-MAX_OUTPUT_TAIL_CHARS:]
        return ToolResult.success(f"resolved={resolved}\n\n{tail}")


def _capture_diff(cwd: Path) -> str:
    """Capture the current working-tree diff.

    Matches ``run.py``'s ``_capture_patch`` retry logic: on Windows the
    default git config may produce empty diffs for CRLF-normalized files,
    so we retry with ``core.autocrlf=false --text`` if the first attempt
    is empty but ``status --porcelain`` shows staged / unstaged changes.
    """
    result = subprocess.run(
        ["git", "diff"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diff = result.stdout

    if diff.strip():
        return diff

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if not status.stdout.strip():
        return ""

    # Dirty tree but empty diff — retry with CRLF normalization disabled
    retry = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "diff", "--text"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return retry.stdout
