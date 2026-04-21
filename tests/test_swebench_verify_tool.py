"""Tests for experiments/swebench_lite/docker_test_tool.py SWEBenchVerifyTool.

These live in tests/ (not experiments/) because (a) they exercise the
Tool protocol from the core package and (b) we want them to run in the
default CI matrix without needing WSL/Docker — verify_patch is mocked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import RiskLevel, ToolContext

# The tool module lives in experiments/swebench_lite/, which isn't a
# Python package. Add it to sys.path so we can import docker_test_tool
# directly (mirroring the scaffolded tool's own sys.path.insert trick).
_EXPERIMENTS_DIR = (Path(__file__).parent.parent / "experiments" / "swebench_lite").resolve()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from docker_test_tool import (  # noqa: E402
    HARD_VERIFY_CAP,
    MAX_OUTPUT_TAIL_CHARS,
    MAX_VERIFY_CALLS,
    SWEBenchVerifyTool,
)


@pytest.fixture
def tool(tmp_path: Path) -> SWEBenchVerifyTool:
    return SWEBenchVerifyTool(
        instance_id="sqlfluff__sqlfluff-2419",
        model_name="nvidia_nim/moonshotai/kimi-k2.5",
        workdir=tmp_path,
        split="dev",
        timeout_s=600,
    )


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    # Create a real git repo so _capture_diff's subprocess calls succeed.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "file.py").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return ToolContext(cwd=repo, session_id="test-session")


def _make_edit(ctx: ToolContext, contents: str) -> None:
    (ctx.cwd / "file.py").write_text(contents, encoding="utf-8")


# ---------------------------------------------------------------------------
# Static shape
# ---------------------------------------------------------------------------


def test_tool_name_is_stable(tool: SWEBenchVerifyTool) -> None:
    assert tool.name == "swebench_verify_patch"


def test_tool_risk_level_is_low(tool: SWEBenchVerifyTool) -> None:
    assert tool.risk_level == RiskLevel.LOW


def test_tool_schema_takes_no_arguments(tool: SWEBenchVerifyTool) -> None:
    schema = tool.get_schema()
    assert schema == {"type": "object", "properties": {}, "required": []}


def test_description_mentions_budget(tool: SWEBenchVerifyTool) -> None:
    assert str(MAX_VERIFY_CALLS) in tool.description


# ---------------------------------------------------------------------------
# Execute behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_diff_returns_unresolved_without_harness(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    # No edits yet -> _capture_diff returns "" -> tool returns unresolved shortcut
    with patch("docker_test_tool.verify_patch") as mock_verify:
        result = await tool.execute({}, tool_context)
    assert mock_verify.call_count == 0
    assert result.is_error is False
    assert "resolved=False" in result.output
    assert "no changes" in result.output.lower()


@pytest.mark.asyncio
async def test_first_call_invokes_verify_and_formats_output(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    _make_edit(tool_context, "fixed\n")
    test_tail = "FAILED test_thing.py::test_case - assertion failed"
    with patch("docker_test_tool.verify_patch", return_value=(False, test_tail)) as mock_verify:
        result = await tool.execute({}, tool_context)
    assert mock_verify.call_count == 1
    assert result.is_error is False
    assert result.output.startswith("resolved=False\n\n")
    assert test_tail in result.output


@pytest.mark.asyncio
async def test_resolved_true_surfaces_in_output(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    _make_edit(tool_context, "correct fix\n")
    with patch("docker_test_tool.verify_patch", return_value=(True, "PASSED")):
        result = await tool.execute({}, tool_context)
    assert result.output.startswith("resolved=True\n\n")


@pytest.mark.asyncio
async def test_no_edit_short_circuit_returns_cached(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    _make_edit(tool_context, "same\n")
    with patch(
        "docker_test_tool.verify_patch", return_value=(False, "first run output")
    ) as mock_verify:
        first = await tool.execute({}, tool_context)
        # Same working tree, no new edits — must NOT re-run the harness
        second = await tool.execute({}, tool_context)
    assert mock_verify.call_count == 1
    assert first.output.startswith("resolved=False")
    assert second.output.startswith("resolved=False")
    assert "cached verdict" in second.output
    assert "first run output" in second.output


@pytest.mark.asyncio
async def test_new_edit_triggers_fresh_harness_call(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    _make_edit(tool_context, "v1\n")
    with patch(
        "docker_test_tool.verify_patch", side_effect=[(False, "out v1"), (True, "out v2")]
    ) as mock_verify:
        await tool.execute({}, tool_context)
        _make_edit(tool_context, "v2\n")
        result = await tool.execute({}, tool_context)
    assert mock_verify.call_count == 2
    assert result.output.startswith("resolved=True")


@pytest.mark.asyncio
async def test_hard_cap_blocks_further_calls(tool_context: ToolContext, tmp_path: Path) -> None:
    # Use the real constants so if we change them, this test catches it.
    tool = SWEBenchVerifyTool(
        instance_id="x",
        model_name="m",
        workdir=tmp_path,
        split="dev",
        max_calls=2,
        hard_cap=3,
    )

    # Vary the working tree each call so we actually invoke verify_patch
    # rather than hitting the no-edit short-circuit.
    def _unique_edit(i: int) -> None:
        _make_edit(tool_context, f"v{i}\n")

    with patch(
        "docker_test_tool.verify_patch",
        side_effect=[(False, "a"), (False, "b"), (False, "c")],
    ) as mock_verify:
        _unique_edit(1)
        await tool.execute({}, tool_context)
        _unique_edit(2)
        await tool.execute({}, tool_context)
        _unique_edit(3)
        await tool.execute({}, tool_context)
        _unique_edit(4)
        blocked = await tool.execute({}, tool_context)

    assert mock_verify.call_count == 3
    assert blocked.is_error is True
    assert "budget exhausted" in (blocked.error or "")


@pytest.mark.asyncio
async def test_timeout_surfaces_as_tool_failure(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    _make_edit(tool_context, "edit\n")
    with patch(
        "docker_test_tool.verify_patch",
        side_effect=subprocess.TimeoutExpired(cmd="wsl", timeout=900),
    ):
        result = await tool.execute({}, tool_context)
    assert result.is_error is True
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_arbitrary_harness_exception_surfaces_as_failure(
    tool: SWEBenchVerifyTool, tool_context: ToolContext
) -> None:
    _make_edit(tool_context, "edit\n")
    with patch("docker_test_tool.verify_patch", side_effect=RuntimeError("boom")):
        result = await tool.execute({}, tool_context)
    assert result.is_error is True
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_output_tail_is_bounded(tool: SWEBenchVerifyTool, tool_context: ToolContext) -> None:
    _make_edit(tool_context, "edit\n")
    huge = "x" * (MAX_OUTPUT_TAIL_CHARS * 3)
    with patch("docker_test_tool.verify_patch", return_value=(False, huge)):
        result = await tool.execute({}, tool_context)
    # header "resolved=False\n\n" + tail
    tail_section = result.output.split("\n\n", 1)[1]
    assert len(tail_section) <= MAX_OUTPUT_TAIL_CHARS


def test_default_budget_constants() -> None:
    """Guardrail: if we shrink the budget, callers that relied on 5 break."""
    assert MAX_VERIFY_CALLS == 5
    assert HARD_VERIFY_CAP == 8
