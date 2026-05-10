"""Tests for file write tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import DiffReviewer, ToolContext
from godspeed.tools.file_write import FileWriteTool


def _ctx(cwd: Path) -> ToolContext:
    return ToolContext(cwd=cwd, session_id="test")


class _FakeDiffReviewer(DiffReviewer):
    """Stub implementing DiffReviewer protocol for tests."""

    def __init__(self, decision: str, accept_call_assertions: callable | None = None) -> None:
        self._decision = decision
        self._accept_call_assertions = accept_call_assertions
        self.call_count = 0
        self.last_kwargs: dict | None = None

    async def review(self, *, tool_name: str, path: str, before: str, after: str) -> str:
        self.call_count += 1
        self.last_kwargs = {
            "tool_name": tool_name,
            "path": path,
            "before": before,
            "after": after,
        }
        return self._decision


class TestFileWriteTool:
    """Test FileWriteTool."""

    def test_metadata(self) -> None:
        tool = FileWriteTool()
        assert tool.name == "file_write"
        assert tool.risk_level == "low"
        assert "file_path" in tool.get_schema()["required"]
        assert "creates the file and parent directories" in tool.description.lower()
        assert tool.produces_diff is True

    def test_write_new_file(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "hello.txt", "content": "Hello, world!"}, _ctx(tmp_path))
        )
        assert result.success
        assert (tmp_path / "hello.txt").read_text() == "Hello, world!"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute(
                {"file_path": "a/b/c/deep.txt", "content": "nested"},
                _ctx(tmp_path),
            )
        )
        assert result.success
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text() == "nested"

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        (tmp_path / "existing.txt").write_text("old content")
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute(
                {"file_path": "existing.txt", "content": "new content"},
                _ctx(tmp_path),
            )
        )
        assert result.success
        assert (tmp_path / "existing.txt").read_text() == "new content"

    def test_empty_path_fails(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(tool.execute({"file_path": "", "content": "x"}, _ctx(tmp_path)))
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    def test_empty_content_writes_empty(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "empty.txt", "content": ""}, _ctx(tmp_path))
        )
        assert result.success
        assert (tmp_path / "empty.txt").read_text() == ""

    def test_reports_bytes_written(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "size.txt", "content": "12345"}, _ctx(tmp_path))
        )
        assert "5 bytes" in result.output

    def test_file_size_limit(self, tmp_path: Path) -> None:
        """Test that files exceeding MAX_FILE_SIZE are rejected."""
        tool = FileWriteTool()
        # Create content that exceeds 10 MB limit
        large_content = "x" * (11 * 1024 * 1024)  # 11 MB
        result = asyncio.run(
            tool.execute({"file_path": "large.txt", "content": large_content}, _ctx(tmp_path))
        )
        assert result.is_error
        assert "exceeds maximum size" in result.error.lower()

    def test_non_string_content_fails(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = asyncio.run(
            tool.execute({"file_path": "test.txt", "content": 12345}, _ctx(tmp_path))
        )
        assert result.is_error
        assert "must be a string" in result.error.lower()

    # --- Diff reviewer tests ---

    @pytest.mark.asyncio
    async def test_diff_reviewer_accept(self, tmp_path: Path) -> None:
        reviewer = _FakeDiffReviewer("accept")
        ctx = ToolContext(cwd=tmp_path, session_id="test", diff_reviewer=reviewer)
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "approved.txt", "content": "approved"}, ctx)
        assert result.success
        assert (tmp_path / "approved.txt").read_text() == "approved"
        assert reviewer.call_count == 1

    @pytest.mark.asyncio
    async def test_diff_reviewer_reject(self, tmp_path: Path) -> None:
        reviewer = _FakeDiffReviewer("reject")
        ctx = ToolContext(cwd=tmp_path, session_id="test", diff_reviewer=reviewer)
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "rejected.txt", "content": "rejected"}, ctx)
        assert result.is_error
        assert "rejected by reviewer" in result.error.lower()
        assert not (tmp_path / "rejected.txt").exists()
        assert reviewer.call_count == 1

    @pytest.mark.asyncio
    async def test_diff_reviewer_overwrite_accept(self, tmp_path: Path) -> None:
        (tmp_path / "existing.txt").write_text("before")
        reviewer = _FakeDiffReviewer("accept")
        ctx = ToolContext(cwd=tmp_path, session_id="test", diff_reviewer=reviewer)
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "existing.txt", "content": "after"}, ctx)
        assert result.success
        assert (tmp_path / "existing.txt").read_text() == "after"
        assert reviewer.last_kwargs["before"] == "before"
        assert reviewer.last_kwargs["after"] == "after"

    @pytest.mark.asyncio
    async def test_diff_reviewer_with_binary_existing(self, tmp_path: Path) -> None:
        (tmp_path / "binfile").write_bytes(b"\x00\x01\x02\xff\xfe")
        reviewer = _FakeDiffReviewer("accept")
        ctx = ToolContext(cwd=tmp_path, session_id="test", diff_reviewer=reviewer)
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "binfile", "content": "new"}, ctx)
        assert result.success
        assert reviewer.last_kwargs["before"] == "<binary>"
        assert reviewer.last_kwargs["after"] == "new"

    @pytest.mark.asyncio
    async def test_diff_reviewer_non_accept_decisions(self, tmp_path: Path) -> None:
        for decision in ("reject", "edit", "skip", "later"):
            sub = tmp_path / decision
            sub.mkdir()
            reviewer = _FakeDiffReviewer(decision)
            ctx = ToolContext(cwd=sub, session_id="test", diff_reviewer=reviewer)
            tool = FileWriteTool()
            result = await tool.execute({"file_path": "test.txt", "content": "data"}, ctx)
            assert result.is_error
            assert "rejected by reviewer" in result.error.lower()

    # --- Atomic write exception handling ---

    @pytest.mark.asyncio
    async def test_atomic_write_temp_cleanup_on_error(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        ctx = _ctx(tmp_path)

        with patch("godspeed.tools.file_write.os.replace", side_effect=OSError("disk full")):
            result = await tool.execute({"file_path": "fail.txt", "content": "data"}, ctx)

        assert result.is_error
        assert "Failed to write" in result.error

    @pytest.mark.asyncio
    async def test_oserror_during_write(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        ctx = _ctx(tmp_path)

        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            result = await tool.execute({"file_path": "sub/nope.txt", "content": "data"}, ctx)

        assert result.is_error
        assert "Failed to write" in result.error

    @pytest.mark.asyncio
    async def test_path_outside_cwd(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        ctx = _ctx(tmp_path)
        result = await tool.execute({"file_path": "../outside.txt", "content": "data"}, ctx)
        assert result.is_error
        assert "Access denied" in result.error

    # --- Additional diff review edge cases ---

    @pytest.mark.asyncio
    async def test_diff_reviewer_new_file_empty_before(self, tmp_path: Path) -> None:
        reviewer = _FakeDiffReviewer("accept")
        ctx = ToolContext(cwd=tmp_path, session_id="test", diff_reviewer=reviewer)
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "new_file.txt", "content": "brand new"}, ctx)
        assert result.success
        assert reviewer.last_kwargs["before"] == ""
        assert reviewer.last_kwargs["after"] == "brand new"

    @pytest.mark.asyncio
    async def test_diff_reviewer_skip_decision(self, tmp_path: Path) -> None:
        reviewer = _FakeDiffReviewer("skip")
        ctx = ToolContext(cwd=tmp_path, session_id="test", diff_reviewer=reviewer)
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "skipped.txt", "content": "data"}, ctx)
        assert result.is_error
        assert "rejected by reviewer" in result.error.lower()

    # --- Atomic write cleanup when unlink also fails ---

    @pytest.mark.asyncio
    async def test_atomic_write_cleanup_unlink_also_fails(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        ctx = _ctx(tmp_path)
        with patch("godspeed.tools.file_write.os.replace", side_effect=OSError("disk full")):
            with patch(
                "godspeed.tools.file_write.os.unlink", side_effect=OSError("unlink also failed")
            ):
                result = await tool.execute({"file_path": "fail.txt", "content": "data"}, ctx)
        assert result.is_error
        assert "Failed to write" in result.error

    # --- Parent dir creation edge cases ---

    @pytest.mark.asyncio
    async def test_create_deeply_nested_dirs(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = await tool.execute(
            {"file_path": "x/y/z/w/deep.txt", "content": "deep"},
            _ctx(tmp_path),
        )
        assert result.success
        assert (tmp_path / "x" / "y" / "z" / "w" / "deep.txt").read_text() == "deep"

    @pytest.mark.asyncio
    async def test_dirs_already_exist(self, tmp_path: Path) -> None:
        (tmp_path / "existing_dir").mkdir()
        tool = FileWriteTool()
        result = await tool.execute(
            {"file_path": "existing_dir/file.txt", "content": "inside"},
            _ctx(tmp_path),
        )
        assert result.success
        assert (tmp_path / "existing_dir" / "file.txt").read_text() == "inside"

    # --- Encoded content handling ---

    @pytest.mark.asyncio
    async def test_unicode_content(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        content = "Hello \u4e16\u754c \U0001f600 \u00e9\u00e8\u00fc"
        result = await tool.execute(
            {"file_path": "unicode.txt", "content": content}, _ctx(tmp_path)
        )
        assert result.success
        assert (tmp_path / "unicode.txt").read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_special_path_chars(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = await tool.execute(
            {"file_path": "my-app_v2.0/config.json", "content": "{}"}, _ctx(tmp_path)
        )
        assert result.success
        assert (tmp_path / "my-app_v2.0" / "config.json").read_text() == "{}"

    @pytest.mark.asyncio
    async def test_overwrite_reports_correct_byte_count(self, tmp_path: Path) -> None:
        (tmp_path / "existing.txt").write_text("old")
        tool = FileWriteTool()
        content = "new_content"
        result = await tool.execute(
            {"file_path": "existing.txt", "content": content}, _ctx(tmp_path)
        )
        assert result.success
        assert f"{len(content)} bytes" in result.output

    # --- Path validation edge cases ---

    @pytest.mark.asyncio
    async def test_non_string_file_path_type(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = await tool.execute({"file_path": 456, "content": "x"}, _ctx(tmp_path))
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_with_spaces(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = await tool.execute(
            {"file_path": "my project/src/utils.py", "content": "# utils"}, _ctx(tmp_path)
        )
        assert result.success
        assert (tmp_path / "my project" / "src" / "utils.py").read_text() == "# utils"

    @pytest.mark.asyncio
    async def test_write_with_dotfile(self, tmp_path: Path) -> None:
        tool = FileWriteTool()
        result = await tool.execute({"file_path": ".env", "content": "DEBUG=true"}, _ctx(tmp_path))
        assert result.success
        assert (tmp_path / ".env").read_text() == "DEBUG=true"
