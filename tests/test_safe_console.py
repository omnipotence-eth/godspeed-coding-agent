"""Tests for godspeed.tui.safe_console helpers."""

from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from godspeed.tui.safe_console import escape_markup, print_markup_safe, print_plain_safe

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def test_escape_markup_is_idempotent_on_plain_text() -> None:
    assert escape_markup("hello") == "hello"
    assert escape_markup("") == ""


def test_print_plain_safe_no_markup_interpretation() -> None:
    buf = StringIO()
    c = Console(file=buf, force_terminal=True, width=80)
    print_plain_safe(c, "plain [bold]not bold[/bold]")
    out = _ANSI_RE.sub("", buf.getvalue())
    assert "[bold]" in out


def test_print_markup_safe_falls_back_on_bad_markup() -> None:
    buf = StringIO()
    c = Console(file=buf, force_terminal=True, width=80)
    print_markup_safe(c, "[/not] broken")
    out = _ANSI_RE.sub("", buf.getvalue())
    assert "broken" in out or "[" in out
