"""Tests for the unified diff apply tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.diff_apply import (
    DiffApplyTool,
    FileDiff,
    Hunk,
    _apply_hunk_to_lines,
    _extract_path,
    apply_file_diff,
    parse_unified_diff,
)


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


# ── Malformed diff: missing +++ header ──────────────────────────────


def test_parse_missing_plus_header() -> None:
    """Line 54-55: +++ header missing after ---."""
    diff = "--- a/file.txt\n some context\n"
    with pytest.raises(ValueError, match=r"Expected.*\+\+\+"):
        parse_unified_diff(diff)


# ── Malformed diff: bad hunk header ──────────────────────────────────


def test_parse_malformed_hunk_header() -> None:
    """Line 107-108: hunk header that doesn't match the regex."""
    diff = "--- a/file.txt\n+++ b/file.txt\n@@ not a valid hunk @@\n text\n"
    with pytest.raises(ValueError, match="Malformed"):
        parse_unified_diff(diff)


# ── _extract_path edge cases ────────────────────────────────────────


def test_extract_path_dev_null() -> None:
    """Line 94: /dev/null path extraction."""
    assert _extract_path("--- /dev/null") == "/dev/null"


def test_extract_path_no_prefix() -> None:
    """Path without a/ or b/ prefix."""
    assert _extract_path("--- path/to/file") == "path/to/file"


# ── Hunk parsing: bare empty line handling ──────────────────────────


def test_bare_empty_line_as_context() -> None:
    """Lines 138-139: bare empty line followed by continued hunk lines."""
    lines = [
        "",
        "@@ -1,3 +1,3 @@",
        " line1",
        "",
        " line2",
        "--- a/other.txt",
        "",
    ]
    _dummy_hunk, _dummy_next_i = (
        Hunk(old_start=1, old_count=3, new_start=1, new_count=3),
        5,
    )  # dummy for type checker
    # Actually use _parse_hunk internals
    from godspeed.tools.diff_apply import _parse_hunk

    hunk, next_i = _parse_hunk(lines, 1)
    assert len(hunk.lines) == 3
    # second item should be " " (bare empty line replaced with space)
    assert hunk.lines[1] == " "


def test_hunk_break_on_non_hunk_line() -> None:
    """Line 143: break when encountering non-diff line after bare empty line."""
    diff = "--- a/file.txt\n+++ b/file.txt\n@@ -1,3 +1,3 @@\n line1\n\nNOT_A_DIFF_LINE\n"
    result = parse_unified_diff(diff)
    assert len(result) == 1
    assert len(result[0].hunks) == 1
    assert len(result[0].hunks[0].lines) == 1  # only " line1"; bare empty + non-hunk breaks


# ── New file: dry_run ────────────────────────────────────────────────


@patch("godspeed.tools.diff_apply.resolve_tool_path")
def test_new_file_dry_run(mock_resolve: MagicMock, tmp_path: Path) -> None:
    """Line 229->228: dry_run new file doesn't write to disk."""
    target = tmp_path / "new.txt"
    mock_resolve.return_value = target

    fd = FileDiff(
        old_path="/dev/null",
        new_path="new.txt",
        hunks=[
            Hunk(old_start=0, old_count=0, new_start=1, new_count=2, lines=["+hello", "+world"])
        ],
        is_new_file=True,
    )
    hunks, fuzzy = apply_file_diff(fd, tmp_path, dry_run=True)
    assert hunks == 1
    assert fuzzy == 0
    assert not target.exists()


# ── File not found during apply ──────────────────────────────────────


@patch("godspeed.tools.diff_apply.resolve_tool_path")
def test_apply_file_not_found(mock_resolve: MagicMock, tmp_path: Path) -> None:
    """Line 246-247: file not found during patch application."""
    target = tmp_path / "nonexistent.txt"
    mock_resolve.return_value = target

    fd = FileDiff(
        old_path="nonexistent.txt",
        new_path="nonexistent.txt",
        hunks=[Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-old", "+new"])],
    )
    with pytest.raises(ValueError, match="File not found"):
        apply_file_diff(fd, tmp_path)


# ── Trailing newline handling ────────────────────────────────────────


def test_trailing_newline_preserved(tmp_path: Path) -> None:
    """Lines 254-268: trailing newline is preserved."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("line1\nline2\n")
    fd = FileDiff(
        old_path="test.txt",
        new_path="test.txt",
        hunks=[
            Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-line1", "+LINE1"])
        ],
    )
    hunks, fuzzy = apply_file_diff(fd, tmp_path)
    assert hunks == 1
    result = file_path.read_text(encoding="utf-8")
    assert result == "LINE1\nline2\n"


def test_no_trailing_newline_preserved(tmp_path: Path) -> None:
    """File without trailing newline stays without."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("line1\nline2")
    fd = FileDiff(
        old_path="test.txt",
        new_path="test.txt",
        hunks=[
            Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-line1", "+LINE1"])
        ],
    )
    hunks, fuzzy = apply_file_diff(fd, tmp_path)
    assert hunks == 1
    result = file_path.read_text(encoding="utf-8")
    assert result == "LINE1\nline2"


def test_trailing_newline_empty_last_line(tmp_path: Path) -> None:
    """Content ending with \\n results in empty final list element that gets stripped."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("alpha\nbeta\n")
    fd = FileDiff(
        old_path="test.txt",
        new_path="test.txt",
        hunks=[Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=["-beta", "+BETA"])],
    )
    apply_file_diff(fd, tmp_path)
    result = file_path.read_text(encoding="utf-8")
    assert result == "alpha\nBETA\n"


# ── Whitespace in diff ────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_diff_with_crlf(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Diff with \\r\\n line endings should be normalized."""
    _make_file(tmp_path, "hello.txt", "line1\nline2\nline3\n")
    diff = (
        "--- a/hello.txt\r\n"
        "+++ b/hello.txt\r\n"
        "@@ -1,3 +1,4 @@\r\n"
        " line1\r\n"
        " line2\r\n"
        "+added\r\n"
        " line3\r\n"
    )
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    content = (tmp_path / "hello.txt").read_text(encoding="utf-8")
    assert "added" in content


# ── Diff parse ValueError in execute ──────────────────────────────────


@pytest.mark.asyncio()
async def test_execute_parse_error(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Line 334-335: ValueError raised from parse_unified_diff."""
    result = await tool.execute({"diff": "--- a/x\nNOT_VALID\n"}, _ctx(tmp_path))
    assert result.is_error
    assert "parse" in result.error.lower() or "Failed" in result.error.lower()


# ── No file diffs found ────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_execute_no_file_diffs(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Line 338: no file diffs found in valid-looking diff."""
    result = await tool.execute({"diff": "some text\nnot a diff\n"}, _ctx(tmp_path))
    assert result.is_error
    assert "No file diffs" in result.error


# ── Dry run summary ────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_dry_run_summary(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Line 382: dry_run returns a different summary message."""
    _make_file(tmp_path, "hello.txt", "line1\nline2\n")
    diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,2 +1,3 @@\n line1\n line2\n+line3\n"
    result = await tool.execute({"diff": diff, "dry_run": True}, _ctx(tmp_path))
    assert not result.is_error
    assert "Dry run" in result.output
    assert "would apply" in result.output
    # Verify file unchanged
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "line1\nline2\n"


# ── Diff reviewer gate ────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_diff_reviewer_accept(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Line 346-361: diff_reviewer accepts the diff."""
    _make_file(tmp_path, "hello.txt", "line1\nline2\n")
    ctx = _ctx(tmp_path)
    ctx.diff_reviewer = AsyncMock()
    ctx.diff_reviewer.review.return_value = "accept"

    diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,2 +1,2 @@\n line1\n-line2\n+LINE2\n"
    result = await tool.execute({"diff": diff}, ctx)
    assert not result.is_error
    assert "1 hunks" in result.output
    ctx.diff_reviewer.review.assert_called_once()


@pytest.mark.asyncio()
async def test_diff_reviewer_reject(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Line 361: diff_reviewer rejects the diff."""
    _make_file(tmp_path, "hello.txt", "line1\nline2\n")
    ctx = _ctx(tmp_path)
    ctx.diff_reviewer = AsyncMock()
    ctx.diff_reviewer.review.return_value = "reject"

    diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,2 +1,2 @@\n line1\n-line2\n+LINE2\n"
    result = await tool.execute({"diff": diff}, ctx)
    assert result.is_error
    assert "rejected" in result.error.lower()
    # Verify file unchanged
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "line1\nline2\n"


@pytest.mark.asyncio()
async def test_diff_reviewer_multi_file_path_summary(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Path summary includes ... when more than 5 files."""
    _make_file(tmp_path, "a.txt", "aaa\n")
    _make_file(tmp_path, "b.txt", "bbb\n")
    _make_file(tmp_path, "c.txt", "ccc\n")
    _make_file(tmp_path, "d.txt", "ddd\n")
    _make_file(tmp_path, "e.txt", "eee\n")
    _make_file(tmp_path, "f.txt", "fff\n")

    diff_parts = []
    for name in ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt"]:
        content = {"a": "aaa", "b": "bbb", "c": "ccc", "d": "ddd", "e": "eee", "f": "fff"}
        base = name[:-4]
        diff_parts.append(
            f"--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,2 @@\n {content[base]}\n+new_line\n"
        )
    diff = "".join(diff_parts)

    ctx = _ctx(tmp_path)
    ctx.diff_reviewer = AsyncMock()
    ctx.diff_reviewer.review.return_value = "accept"

    result = await tool.execute({"diff": diff}, ctx)
    assert not result.is_error
    call_kwargs = ctx.diff_reviewer.review.call_args.kwargs
    assert "... (1 more)" in call_kwargs["path"] or "more" in call_kwargs["path"]


# ── _apply_hunk_to_lines edge cases ──────────────────────────────────


def test_apply_hunk_exact_match() -> None:
    """Apply hunk with exact match at target position."""
    file_lines = ["alpha", "beta", "gamma"]
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=["-beta", "+BETA"])
    new_lines, used_fuzzy = _apply_hunk_to_lines(file_lines, hunk)
    assert new_lines == ["alpha", "BETA", "gamma"]
    assert not used_fuzzy


def test_apply_hunk_offset_below_zero() -> None:
    """Fuzzy offsets that go below zero are skipped."""
    file_lines = ["alpha", "beta"]
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=["-beta", "+BETA"])
    # offset -3 would try pos=-1, which is < 0 — skipped
    new_lines, used_fuzzy = _apply_hunk_to_lines(file_lines, hunk, fuzzy_range=5)
    assert new_lines == ["alpha", "BETA"]
    assert not used_fuzzy


def test_apply_hunk_offset_beyond_end() -> None:
    """Fuzzy offsets beyond file length are skipped."""
    file_lines = ["alpha", "beta"]
    hunk = Hunk(old_start=2, old_count=2, new_start=2, new_count=2, lines=["-beta", "+BETA"])
    # old_block = ["beta"], target_line=1, offset +3 → pos=4, len=1, 4+1>2 → skipped
    new_lines, used_fuzzy = _apply_hunk_to_lines(file_lines, hunk, fuzzy_range=5)
    assert new_lines == ["alpha", "BETA"]
    assert not used_fuzzy


def test_apply_hunk_no_match() -> None:
    """Hunk that cannot match at any offset raises ValueError."""
    file_lines = ["alpha", "beta", "gamma"]
    hunk = Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-nope", "+yes"])
    with pytest.raises(ValueError, match="failed to match"):
        _apply_hunk_to_lines(file_lines, hunk, fuzzy_range=2)


# ── New file creation: actual write ──────────────────────────────────


@patch("godspeed.tools.diff_apply.resolve_tool_path")
def test_new_file_writes_content(mock_resolve: MagicMock, tmp_path: Path) -> None:
    """Line 231->237: new file non-dry_run writes content."""
    target = tmp_path / "sub" / "new.py"
    mock_resolve.return_value = target

    fd = FileDiff(
        old_path="/dev/null",
        new_path="sub/new.py",
        hunks=[
            Hunk(
                old_start=0,
                old_count=0,
                new_start=1,
                new_count=3,
                lines=["+print('hello')", "+print('world')", " final"],
            )
        ],
        is_new_file=True,
    )
    hunks, fuzzy = apply_file_diff(fd, tmp_path)
    assert hunks == 1
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "hello" in content
    assert "world" in content


def test_new_file_created_in_same_dir(tmp_path: Path) -> None:
    """New file creation in root with no subdirs."""
    target = tmp_path / "new.py"
    fd = FileDiff(
        old_path="/dev/null",
        new_path="new.py",
        hunks=[Hunk(old_start=0, old_count=0, new_start=1, new_count=1, lines=["+x = 1"])],
        is_new_file=True,
    )
    hunks, fuzzy = apply_file_diff(fd, tmp_path)
    assert hunks == 1
    assert target.read_text(encoding="utf-8") == "x = 1\n"


# ── Hunk loop natural exit ──────────────────────────────────────────


def test_hunk_loop_natural_exit() -> None:
    """Line 123->145: while loop exits naturally when hunk is last in input."""
    diff = "--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n line1\n line2\n"
    result = parse_unified_diff(diff)
    assert len(result) == 1
    assert len(result[0].hunks) == 1
    assert len(result[0].hunks[0].lines) == 2


# ── Fuzzy summary in output ──────────────────────────────────────────


@pytest.mark.asyncio()
async def test_fuzzy_summary_in_result(tmp_path: Path, tool: DiffApplyTool) -> None:
    """Fuzzy count appears in the result output."""
    _make_file(tmp_path, "test.txt", "extra\nextra\nalpha\nbeta\n")
    diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n"
    result = await tool.execute({"diff": diff}, _ctx(tmp_path))
    assert not result.is_error
    assert "fuzzy" in result.output


# ── Tool metadata ────────────────────────────────────────────────────


def test_tool_name(tool: DiffApplyTool) -> None:
    assert tool.name == "diff_apply"


def test_tool_description(tool: DiffApplyTool) -> None:
    desc = tool.description
    assert "unified diff" in desc.lower()
    assert "fuzzy" in desc.lower()


def test_tool_risk_level(tool: DiffApplyTool) -> None:
    from godspeed.tools.base import RiskLevel

    assert tool.risk_level == RiskLevel.LOW


def test_tool_schema_has_diff(tool: DiffApplyTool) -> None:
    schema = tool.get_schema()
    assert "diff" in schema["properties"]
    assert "diff" in schema["required"]
