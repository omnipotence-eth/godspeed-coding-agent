"""Tests for tool name canonicalization aliases."""

from __future__ import annotations

import logging
from unittest.mock import patch

from godspeed.tools.aliases import canonicalize_tool_name


class TestCanonicalizeToolName:
    def test_empty_string_returns_empty(self) -> None:
        assert canonicalize_tool_name("") == ""

    def test_unknown_name_passes_through(self) -> None:
        assert canonicalize_tool_name("nonexistent_tool") == "nonexistent_tool"

    def test_grep_alias_rewritten(self) -> None:
        with patch.object(logging.getLogger("godspeed.tools.aliases"), "info") as mock_log:
            result = canonicalize_tool_name("grep")
            assert result == "grep_search"
            mock_log.assert_called_once()

    def test_read_file_alias_rewritten(self) -> None:
        with patch.object(logging.getLogger("godspeed.tools.aliases"), "info") as mock_log:
            result = canonicalize_tool_name("read_file")
            assert result == "file_read"
            mock_log.assert_called_once()

    def test_write_file_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("write_file")
        assert result == "file_write"

    def test_edit_file_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("edit_file")
        assert result == "file_edit"

    def test_file_edit_hyphen_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("file-edit")
        assert result == "file_edit"

    def test_file_read_hyphen_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("file-read")
        assert result == "file_read"

    def test_file_write_hyphen_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("file-write")
        assert result == "file_write"

    def test_glob_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("glob")
        assert result == "glob_search"

    def test_search_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("search")
        assert result == "grep_search"

    def test_search_code_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("search_code")
        assert result == "code_search"

    def test_runtests_camelcase_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("runTests")
        assert result == "test_runner"

    def test_run_tests_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("run_tests")
        assert result == "test_runner"

    def test_runtests_lowercase_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("runtests")
        assert result == "test_runner"

    def test_git_status_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("git_status")
        assert result == "git"

    def test_background_alias_rewritten(self) -> None:
        result = canonicalize_tool_name("background")
        assert result == "background_check"

    def test_logger_called_for_all_rewrites(self) -> None:
        aliases = [
            ("grep", "grep_search"),
            ("read_file", "file_read"),
            ("write_file", "file_write"),
            ("edit_file", "file_edit"),
            ("file-edit", "file_edit"),
            ("file-read", "file_read"),
            ("file-write", "file_write"),
            ("glob", "glob_search"),
            ("search", "grep_search"),
            ("search_code", "code_search"),
            ("runTests", "test_runner"),
            ("run_tests", "test_runner"),
            ("runtests", "test_runner"),
            ("git_status", "git"),
            ("background", "background_check"),
        ]
        with patch.object(logging.getLogger("godspeed.tools.aliases"), "info") as mock_log:
            for alias, _ in aliases:
                canonicalize_tool_name(alias)
            assert mock_log.call_count == len(aliases)
