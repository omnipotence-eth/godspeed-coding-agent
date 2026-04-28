"""Tab completion for the Godspeed TUI."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

logger = logging.getLogger(__name__)

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/model", "Show or switch the active model"),
    ("/clear", "Clear conversation history"),
    ("/undo", "Undo last git commit"),
    ("/audit", "Show audit trail stats and verify chain"),
    ("/permissions", "Show current permission rules"),
    ("/autocommit", "Toggle auto-commit or set threshold"),
    ("/architect", "Toggle architect mode (plan then execute)"),
    ("/think", "Toggle extended thinking or set token budget"),
    ("/budget", "Show/set cost budget in USD"),
    ("/evolve", "Self-evolution: status|run|history|rollback|review"),
    ("/quit", "Exit Godspeed"),
    ("/exit", "Exit Godspeed"),
]

MENTION_TYPES: list[tuple[str, str]] = [
    ("@file:", "Include file content"),
    ("@folder:", "Include directory listing"),
    ("@web:", "Fetch web page content (HTTPS only)"),
]


class GodspeedCompleter(Completer):
    """Completer for slash commands and file paths.

    Completes:
    - Slash commands when input starts with /
    - File paths as arguments to certain commands
    """

    def __init__(
        self,
        cwd: Path | None = None,
        extra_commands: list[tuple[str, str]] | None = None,
    ) -> None:
        self._cwd = cwd or Path(".")
        self._extra_commands = extra_commands or []

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Yield completions for the current input."""
        text = document.text_before_cursor
        stripped = text.lstrip()

        # Slash command completion
        if stripped.startswith("/"):
            yield from self._complete_slash_commands(stripped)
            return

        # @-mention completion
        mention_match = self._find_mention_at_cursor(text)
        if mention_match is not None:
            yield from self._complete_mentions(mention_match)
            return

        # File path completion when there's a space (argument position)
        parts = stripped.split(maxsplit=1)
        if len(parts) == 2 and parts[0].startswith("/"):
            yield from self._complete_file_paths(parts[1])

    def _complete_slash_commands(self, text: str) -> Iterable[Completion]:
        """Complete slash commands (built-in + dynamic skill commands)."""
        all_commands = SLASH_COMMANDS + self._extra_commands
        for cmd, description in all_commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=description,
                )

    def _find_mention_at_cursor(self, text: str) -> str | None:
        """Find an @-mention being typed at the cursor position.

        Returns the partial mention text (e.g. "@file:src/m") or None.
        """
        # Walk backwards from cursor to find @ that starts a mention
        import re

        match = re.search(r"@(\S*)$", text)
        if match:
            return match.group(0)
        return None

    def _complete_mentions(self, partial: str) -> Iterable[Completion]:
        """Complete @-mentions: type prefixes and file/folder paths."""
        # If just "@" or partial type, suggest mention types
        if ":" not in partial:
            for mention, description in MENTION_TYPES:
                if mention.startswith(partial):
                    yield Completion(
                        mention,
                        start_position=-len(partial),
                        display_meta=description,
                    )
            return

        # If we have a type prefix (e.g. @file:src/), complete paths
        prefix, path_part = partial.split(":", 1)
        mention_type = prefix[1:]  # strip @

        if mention_type in ("file", "folder"):
            for completion in self._complete_file_paths(path_part):
                # Re-wrap as full mention
                yield Completion(
                    f"{prefix}:{completion.text}",
                    start_position=-len(partial),
                    display_meta=completion.display_meta,
                )

    def _complete_file_paths(self, partial: str) -> Iterable[Completion]:
        """Complete file paths relative to cwd."""
        try:
            partial_path = Path(partial)
            if partial.endswith("/") or partial.endswith("\\"):
                search_dir = self._cwd / partial_path
                prefix = ""
            else:
                search_dir = self._cwd / partial_path.parent
                prefix = partial_path.name

            if not search_dir.is_dir():
                return

            for entry in sorted(search_dir.iterdir()):
                name = entry.name
                if name.startswith("."):
                    continue
                if prefix and not name.lower().startswith(prefix.lower()):
                    continue

                relative = entry.relative_to(self._cwd)
                display = str(relative) + ("/" if entry.is_dir() else "")
                yield Completion(
                    display,
                    start_position=-len(partial),
                    display_meta="dir" if entry.is_dir() else "file",
                )
        except (OSError, ValueError):
            return
