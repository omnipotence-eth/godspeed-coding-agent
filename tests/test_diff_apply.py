"""Tests for the unified diff apply tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.diff_apply import DiffApplyTool


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


def _make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def tool() -> DiffApplyTool:
    return DiffApplyTool()


# ── Single file, single hunk: add lines ─────────────────────────────


@pytest.mark.asyncio()
async def test_add_lines(tmp_path: Path, tool: DiffApplyTool) -> None:
    _make_file(tmp_path, "hello.txt", "line1\nline2\nline3\n")
    diff = (
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1,3 +1,5 @@\n"
        " line1\n"
        " line2\n"
        "+inserted_a\n"
        "+inserted_b\n"
        " line3\n"
    )
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    assert "1 hunks" in result.output
    content = (tmp_path / "hello.txt").read_text(encoding="utf-8")
    assert content == "line1\nline2\ninserted_a\ninserted_b\nline3\n"


# ── Single file, single hunk: remove lines ──────────────────────────


@pytest.mark.asyncio()
async def test_remove_lines(tmp_path: Path, tool: DiffApplyTool) -> None:
    _make_file(tmp_path, "hello.txt", "line1\nline2\nline3\nline4\n")
    diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,4 +1,2 @@\n line1\n-line2\n-line3\n line4\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    content = (tmp_path / "hello.txt").read_text(encoding="utf-8")
    assert content == "line1\nline4\n"


# ── Single file, single hunk: modify lines ──────────────────────────


@pytest.mark.asyncio()
async def test_modify_lines(tmp_path: Path, tool: DiffApplyTool) -> None:
    _make_file(tmp_path, "hello.txt", "alpha\nbeta\ngamma\n")
    diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    content = (tmp_path / "hello.txt").read_text(encoding="utf-8")
    assert content == "alpha\nBETA\ngamma\n"


# ── Multi-hunk single file ──────────────────────────────────────────


@pytest.mark.asyncio()
async def test_multi_hunk(tmp_path: Path, tool: DiffApplyTool) -> None:
    _make_file(
        tmp_path,
        "code.py",
        "import os\nimport sys\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n",
    )
    diff = (
        "--- a/code.py\n"
        "+++ b/code.py\n"
        "@@ -1,2 +1,3 @@\n"
        " import os\n"
        " import sys\n"
        "+import json\n"
        "@@ -7,2 +8,2 @@\n"
        " def bar():\n"
        "-    pass\n"
        "+    return 42\n"
    )
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    assert "2 hunks" in result.output
    content = (tmp_path / "code.py").read_text(encoding="utf-8")
    assert "import json" in content
    assert "return 42" in content
    assert content.count("pass") == 1  # only foo's pass remains


# ── Multi-file diff ─────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_multi_file(tmp_path: Path, tool: DiffApplyTool) -> None:
    _make_file(tmp_path, "a.txt", "aaa\nbbb\n")
    _make_file(tmp_path, "b.txt", "xxx\nyyy\n")
    diff = (
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-aaa\n"
        "+AAA\n"
        " bbb\n"
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " xxx\n"
        "-yyy\n"
        "+YYY\n"
    )
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    assert "2 files" in result.output
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "AAA\nbbb\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "xxx\nYYY\n"


# ── Fuzzy context matching ──────────────────────────────────────────


@pytest.mark.asyncio()
async def test_fuzzy_matching(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Context lines are off by 2 positions — fuzzy should handle it."""
    _make_file(
        tmp_path,
        "shift.txt",
        "extra1\nextra2\nalpha\nbeta\ngamma\n",
    )
    # Hunk says old_start=1 but actual match starts at line 3
    diff = "--- a/shift.txt\n+++ b/shift.txt\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    assert "fuzzy" in result.output
    content = (tmp_path / "shift.txt").read_text(encoding="utf-8")
    assert "BETA" in content
    assert "beta" not in content


# ── Reject on context mismatch ──────────────────────────────────────


@pytest.mark.asyncio()
async def test_context_mismatch_rejected(tmp_path: Path, tool: DiffApplyTool) -> None:
    _make_file(tmp_path, "nope.txt", "aaa\nbbb\nccc\n")
    diff = "--- a/nope.txt\n+++ b/nope.txt\n@@ -1,3 +1,3 @@\n xxx\n-yyy\n+zzz\n www\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert result.is_error
    assert "failed to match" in result.error.lower()


# ── Empty diff → error ──────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_empty_diff(tmp_path: Path, tool: DiffApplyTool) -> None:
    result = await tool.execute({"diff": ""}, _ctx(tmp_path))
    assert result.is_error


@pytest.mark.asyncio()
async def test_whitespace_only_diff(tmp_path: Path, tool: DiffApplyTool) -> None:
    result = await tool.execute({"diff": "   \n\n  "}, _ctx(tmp_path))
    assert result.is_error


# ── Path traversal blocked ──────────────────────────────────────────


@pytest.mark.asyncio()
async def test_path_traversal_blocked(tmp_path: Path, tool: DiffApplyTool) -> None:
    diff = "--- a/../../../etc/passwd\n+++ b/../../../etc/passwd\n@@ -1,1 +1,2 @@\n root\n+hacked\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert result.is_error
    assert "access denied" in result.error.lower() or "outside" in result.error.lower()


# ── New file creation (--- /dev/null) ────────────────────────────────


@pytest.mark.asyncio()
async def test_new_file_creation(tmp_path: Path, tool: DiffApplyTool) -> None:
    diff = "--- /dev/null\n+++ b/newfile.txt\n@@ -0,0 +1,3 @@\n+hello\n+world\n+!\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    created = tmp_path / "newfile.txt"
    assert created.exists()
    content = created.read_text(encoding="utf-8")
    assert "hello" in content
    assert "world" in content
    assert "!" in content


@pytest.mark.asyncio()
async def test_new_file_in_subdirectory(tmp_path: Path, tool: DiffApplyTool) -> None:
    diff = (
        "--- /dev/null\n+++ b/sub/dir/new.py\n@@ -0,0 +1,2 @@\n+print('hello')\n+print('world')\n"
    )
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    created = tmp_path / "sub" / "dir" / "new.py"
    assert created.exists()


# ── Tool metadata ────────────────────────────────────────────────────


def test_tool_name(tool: DiffApplyTool) -> None:
    assert tool.name == "diff_apply"


def test_tool_risk_level(tool: DiffApplyTool) -> None:
    from godspeed.tools.base import RiskLevel

    assert tool.risk_level == RiskLevel.LOW


def test_tool_schema_has_diff(tool: DiffApplyTool) -> None:
    schema = tool.get_schema()
    assert "diff" in schema["properties"]
    assert "diff" in schema["required"]
