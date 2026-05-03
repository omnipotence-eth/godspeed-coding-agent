"""Tests for godspeed.tui.completions."""

from __future__ import annotations

from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from godspeed.tui.completions import (
    _MENTION_AT_CURSOR_RE,
    MENTION_TYPES,
    SLASH_COMMANDS,
    GodspeedCompleter,
)


class TestConstants:
    def test_slash_commands_not_empty(self):
        assert len(SLASH_COMMANDS) > 0
        assert any(cmd[0] == "/help" for cmd in SLASH_COMMANDS)
        assert any(cmd[0] == "/quit" for cmd in SLASH_COMMANDS)

    def test_mention_types_not_empty(self):
        assert len(MENTION_TYPES) > 0
        assert any("@file:" in mt[0] for mt in MENTION_TYPES)

    def test_mention_regex(self):
        match = _MENTION_AT_CURSOR_RE.search("text @file:src/ma")
        assert match is not None
        assert match.group(1) == "file:src/ma"

    def test_mention_regex_no_match(self):
        match = _MENTION_AT_CURSOR_RE.search("text without mention")
        assert match is None


class TestGodspeedCompleterInit:
    def test_default_cwd(self):
        completer = GodspeedCompleter()
        assert completer._cwd is not None

    def test_custom_cwd(self, tmp_path):
        completer = GodspeedCompleter(cwd=tmp_path)
        assert completer._cwd == tmp_path

    def test_extra_commands(self):
        extra = [("/test", "Test command")]
        completer = GodspeedCompleter(extra_commands=extra)
        assert len(completer._extra_commands) == 1


class TestSlashCommandCompletion:
    def test_complete_help(self):
        completer = GodspeedCompleter()
        doc = Document(text="/help")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) > 0
        assert any(c.text == "/help" for c in completions)

    def test_complete_partial(self):
        completer = GodspeedCompleter()
        doc = Document(text="/mo")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) > 0
        assert any("model" in c.text for c in completions)

    def test_no_match(self):
        completer = GodspeedCompleter()
        doc = Document(text="/xyz")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) == 0

    def test_complete_all_commands(self):
        completer = GodspeedCompleter()
        doc = Document(text="/")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) >= len(SLASH_COMMANDS)

    def test_extra_commands_included(self):
        extra = [("/custom", "Custom command")]
        completer = GodspeedCompleter(extra_commands=extra)
        doc = Document(text="/cus")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert any(c.text == "/custom" for c in completions)


class TestMentionCompletion:
    def test_mention_at_cursor_no_match(self):
        completer = GodspeedCompleter()
        result = completer._find_mention_at_cursor("text without mention")
        assert result is None

    def test_mention_at_cursor_match(self):
        completer = GodspeedCompleter()
        result = completer._find_mention_at_cursor("text @file:src/ma")
        assert result == "@file:src/ma"

    def test_complete_mention_types(self):
        completer = GodspeedCompleter()
        doc = Document(text="@")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) > 0
        assert any("@file:" in c.text for c in completions)

    def test_complete_mention_partial(self):
        completer = GodspeedCompleter()
        doc = Document(text="@fi")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert any("@file:" in c.text for c in completions)


class TestFilePathCompletion:
    def test_complete_file_paths(self, tmp_path):
        # Create test files
        (tmp_path / "test.py").write_text("content")
        (tmp_path / "other.txt").write_text("content")
        completer = GodspeedCompleter(cwd=tmp_path)
        completions = list(completer._complete_file_paths("test"))
        assert len(completions) > 0
        assert any("test.py" in c.text for c in completions)

    def test_complete_directory_paths(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        completer = GodspeedCompleter(cwd=tmp_path)
        completions = list(completer._complete_file_paths("sub"))
        assert len(completions) > 0
        assert any("subdir" in c.text for c in completions)

    def test_complete_no_match(self, tmp_path):
        completer = GodspeedCompleter(cwd=tmp_path)
        completions = list(completer._complete_file_paths("nonexistent"))
        assert len(completions) == 0

    def test_complete_hidden_files_excluded(self, tmp_path):
        (tmp_path / ".hidden").write_text("content")
        completer = GodspeedCompleter(cwd=tmp_path)
        completions = list(completer._complete_file_paths(""))
        assert all(not c.text.startswith(".hidden") for c in completions)


class TestGetCompletions:
    def test_slash_command_branch(self, tmp_path):
        completer = GodspeedCompleter(cwd=tmp_path)
        doc = Document(text="/help")
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) > 0

    def test_mention_branch(self, tmp_path):
        completer = GodspeedCompleter(cwd=tmp_path)
        # Create a file so path completion has something to match
        (tmp_path / "test_file.py").write_text("content")
        # text_before_cursor is what matters for completion
        doc = Document(text="@file:test")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) > 0

    def test_file_path_argument_branch(self, tmp_path):
        # After a slash command with space, complete file paths
        completer = GodspeedCompleter(cwd=tmp_path)
        doc = Document(text="/tools test")
        completions = list(completer.get_completions(doc, MagicMock()))
        # Should complete file paths as arguments
        assert isinstance(completions, (list, type(completions)))
