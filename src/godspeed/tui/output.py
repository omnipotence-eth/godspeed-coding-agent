"""Rich output formatting for the Godspeed TUI — Midnight Gold theme.

Design philosophy: function first with beautiful form. Whitespace as structure,
restraint in color, information density that respects the developer's attention.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from godspeed.tui.theme import (
    BOLD_ERROR,
    BOLD_PRIMARY,
    BOLD_WARNING,
    BRAND_TAGLINE,
    DECORATOR,
    DIM,
    ERROR,
    GUTTER,
    GUTTER_STYLE,
    MARKER_ERROR,
    MARKER_INFO,
    MARKER_SUCCESS,
    MARKER_TOOL,
    MARKER_WARNING,
    MUTED,
    PROMPT_ICON,
    RULE_CHAR,
    SECONDARY,
    SEPARATOR_DOT,
    SUCCESS,
    SYNTAX_THEME,
    TABLE_KEY,
    TABLE_VALUE,
    WARNING,
    brand,
    styled,
)

logger = logging.getLogger(__name__)

console = Console()

# Max lines for inline tool result display
_RESULT_MAX_LINES = 10
_RESULT_MAX_CHARS = 2000

# Rule width for horizontal separators
_RULE_WIDTH = 35


def _rule() -> str:
    """Return a thin horizontal rule string."""
    return styled(RULE_CHAR * _RULE_WIDTH, MUTED)


def _gutter_lines(text: str) -> None:
    """Print text with a left gutter border on each line."""
    gutter = styled(GUTTER, GUTTER_STYLE)
    for line in text.splitlines():
        console.print(f"    {gutter} {line}")


# =============================================================================
# Status-typed message formatters
# =============================================================================


def format_info(message: str) -> None:
    """Display an info message with ● indicator."""
    console.print(f"  {styled(MARKER_INFO, SECONDARY)} {styled(message, DIM)}")


def format_success(message: str) -> None:
    """Display a success message with ✓ indicator."""
    console.print(f"  {styled(MARKER_SUCCESS, SUCCESS)} {message}")


def format_warning(message: str) -> None:
    """Display a warning message with ⚠ indicator."""
    console.print(f"  {styled(MARKER_WARNING, WARNING)} {message}")


def format_error(message: str) -> None:
    """Display an error message with ✗ indicator."""
    console.print(f"  {styled(MARKER_ERROR, ERROR)} {styled(f'Error: {message}', BOLD_ERROR)}")


# =============================================================================
# Tool call / result display
# =============================================================================


def format_tool_call(name: str, args: dict[str, Any]) -> None:
    """Display a tool call — compact inline for simple calls, expanded for complex ones."""
    marker = styled(MARKER_TOOL, MUTED)
    tool = styled(name, BOLD_PRIMARY)

    # Simple single-arg tools: compact inline
    if name in ("file_read", "glob_search", "repo_map") and args.get("file_path"):
        console.print(f"  {marker} {tool}  {args['file_path']}")
        return

    if name == "grep_search" and args.get("pattern"):
        path = args.get("path", "")
        suffix = f"  {path}" if path else ""
        console.print(f"  {marker} {tool}  {styled(args['pattern'], DIM)}{suffix}")
        return

    if name == "git" and args.get("action"):
        action = args["action"]
        extra = args.get("message", args.get("branch", ""))
        suffix = f"  {extra}" if extra else ""
        console.print(f"  {marker} {tool}  {action}{suffix}")
        return

    # Shell: show command with $ prefix and gutter
    if name == "shell" and args.get("command"):
        console.print(f"  {marker} {tool}")
        _gutter_lines(f"$ {args['command']}")
        return

    # File edit: show compact diff with gutter
    if name == "file_edit" and args.get("file_path"):
        console.print(f"  {marker} {tool}  {args['file_path']}")
        if args.get("old_string") and args.get("new_string"):
            import difflib

            old_lines = args["old_string"].splitlines()
            new_lines = args["new_string"].splitlines()
            diff_output = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))
            # Skip --- / +++ headers
            diff_lines = diff_output[2:] if len(diff_output) > 2 else diff_output
            if diff_lines:
                diff_text = "\n".join(diff_lines[:15])
                _gutter_lines(diff_text)
        return

    # File write: show path and line count
    if name == "file_write" and args.get("file_path"):
        content = args.get("content", "")
        line_count = len(content.splitlines())
        count_label = styled(f"({line_count} lines)", DIM)
        console.print(f"  {marker} {tool}  {args['file_path']}  {count_label}")
        return

    # Default: JSON args with gutter
    try:
        args_text = json.dumps(args, indent=2, default=str)
    except (TypeError, ValueError):
        args_text = str(args)

    console.print(f"  {marker} {tool}")
    _gutter_lines(args_text)


def format_tool_result(name: str, result: str, is_error: bool = False) -> None:
    """Display a tool result — compact for success, expanded for errors."""
    if is_error:
        marker = styled(MARKER_ERROR, ERROR)
        tool = styled(f"{name}", BOLD_ERROR)

        # Show full error output
        display = result
        if len(result) > _RESULT_MAX_CHARS:
            remaining = len(result) - _RESULT_MAX_CHARS
            display = result[:_RESULT_MAX_CHARS] + f"\n... ({remaining} more chars)"

        lines = display.splitlines()
        if len(lines) <= 3:
            # Short error inline
            console.print(f"  {marker} {tool}  {lines[0] if lines else ''}")
            for line in lines[1:]:
                console.print(f"    {line}")
        else:
            console.print(f"  {marker} {tool}")
            # Indent error output
            for line in lines[:20]:
                console.print(f"    {styled(line, DIM)}")
            if len(lines) > 20:
                console.print(f"    {styled(f'... ({len(lines) - 20} more lines)', DIM)}")
    else:
        marker = styled(MARKER_SUCCESS, SUCCESS)
        tool = styled(name, MUTED)

        # Summarize success output
        lines = result.splitlines()
        line_count = len(lines)

        if not result.strip():
            console.print(f"  {marker} {tool}")
            return

        # Short results: show inline
        if line_count <= _RESULT_MAX_LINES and len(result) <= 500:
            console.print(f"  {marker} {tool}")
            for line in lines:
                console.print(f"    {styled(line, DIM)}")
        else:
            # Long results: show summary
            console.print(f"  {marker} {tool}  {styled(f'({line_count} lines)', DIM)}")


def format_assistant_text(text: str) -> None:
    """Render assistant text as Rich Markdown."""
    if not text.strip():
        return
    md = Markdown(text)
    console.print(md)


def format_permission_prompt(
    tool_name: str,
    reason: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Display a permission request with contextual detail.

    Shows tool-specific previews:
    - file_edit: unified diff of old_string -> new_string
    - file_write: first 15 lines of content
    - shell: syntax-highlighted command
    - file_read / grep_search / repo_map: file path
    """
    console.print()

    # Warning marker header
    warn_icon = styled(MARKER_WARNING, WARNING)
    warn_text = styled("Permission required", BOLD_WARNING)
    console.print(f"  {warn_icon}  {warn_text}")
    console.print()

    args = arguments or {}

    # Tool name and primary arg
    console.print(f"    {styled(tool_name, BOLD_PRIMARY)}", end="")

    if tool_name == "file_edit" and args.get("old_string") and args.get("new_string"):
        import difflib

        file_path = args.get("file_path", "unknown")
        old = args["old_string"]
        new = args["new_string"]

        # Line change stats
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        added = sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("+ "))
        removed = sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("- "))
        stats = f"+{added} -{removed} lines"

        console.print(f"  {file_path}  {styled(stats, DIM)}")

        # Unified diff
        diff_output = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="before",
                tofile="after",
                lineterm="",
                n=2,
            )
        )
        diff_content = diff_output[2:] if len(diff_output) > 2 else diff_output
        diff_text = "\n".join(diff_content[:30])
        console.print(Syntax(diff_text, "diff", theme=SYNTAX_THEME, word_wrap=True))
        if len(diff_content) > 30:
            console.print(f"    {styled(f'... ({len(diff_content) - 30} more lines)', DIM)}")

    elif tool_name == "file_write" and args.get("content"):
        from pathlib import Path

        file_path = args.get("file_path", "unknown")
        all_lines = args["content"].splitlines()
        line_count = len(all_lines)

        target = Path(file_path)
        if not target.is_absolute():
            action = "write"
        elif target.exists():
            action = "overwrite"
        else:
            action = "create"
        action_style = WARNING if action == "overwrite" else MUTED
        console.print(f"  {file_path}  {styled(f'({action}, {line_count} lines)', action_style)}")

        preview = "\n".join(all_lines[:15])
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else "text"
        lexer_map = {"py": "python", "js": "javascript", "ts": "typescript", "yaml": "yaml"}
        lexer = lexer_map.get(ext, ext)
        console.print(Syntax(preview, lexer, theme=SYNTAX_THEME, word_wrap=True))
        if len(all_lines) > 15:
            console.print(f"    {styled(f'... ({len(all_lines) - 15} more lines)', DIM)}")

    elif tool_name == "shell" and args.get("command"):
        console.print()
        cmd = f"    $ {args['command']}"
        console.print(Syntax(cmd, "bash", theme=SYNTAX_THEME, word_wrap=True))

    elif args.get("file_path"):
        console.print(f"  {args['file_path']}")

    elif args.get("pattern"):
        console.print(f"  {styled(args['pattern'], DIM)}")

    else:
        console.print()

    # Prompt line
    console.print()
    console.print(f"    {styled(reason, DIM)}")
    console.print(
        f"    {styled('Allow?', WARNING)}"
        f" {styled(f'(y)es {SEPARATOR_DOT} (n)o {SEPARATOR_DOT} (a)lways this session', DIM)}"
    )
    return ""


def format_permission_denied(tool_name: str, reason: str) -> None:
    """Display a permission denied notice."""
    marker = styled(MARKER_ERROR, ERROR)
    console.print(f"  {marker} {styled('Blocked:', BOLD_ERROR)} {tool_name} -- {reason}")


def format_stats(
    input_tokens: int,
    output_tokens: int,
    model: str,
    session_id: str,
    cost: float | None = None,
) -> None:
    """Display session statistics (used by /stats command)."""
    table = Table(show_header=False, border_style=MUTED, expand=False, padding=(0, 2))
    table.add_column("Key", style=TABLE_KEY)
    table.add_column("Value", style=TABLE_VALUE)
    table.add_row("Model", model)
    table.add_row("Session", session_id[:12] + "...")
    table.add_row("Input tokens", f"{input_tokens:,}")
    table.add_row("Output tokens", f"{output_tokens:,}")
    table.add_row("Total tokens", f"{input_tokens + output_tokens:,}")
    if cost is not None:
        table.add_row("Estimated cost", f"${cost:.4f}")

    panel = Panel(
        table,
        title=styled("Session Stats", BOLD_PRIMARY),
        border_style=SECONDARY,
        expand=False,
    )
    console.print(panel)


def format_welcome(
    model: str,
    project_dir: str,
    tools: list[str] | None = None,
    deny_rules: list[str] | None = None,
    audit_enabled: bool = True,
) -> None:
    """Display welcome banner — clean, minimal, function-first."""
    from godspeed import __version__

    console.print()

    # Decorated branded header
    dec = styled(f"{DECORATOR}{DECORATOR}{DECORATOR}", MUTED)
    header = f"  {dec} {PROMPT_ICON} {brand(__version__)} {dec}"
    console.print(header)
    console.print(f"  {styled(BRAND_TAGLINE, DIM)}")
    console.print()

    # Thin rule separator
    console.print(f"  {_rule()}")

    # Key info — aligned, clean
    audit_status = styled("enabled", SUCCESS) if audit_enabled else styled("disabled", ERROR)
    console.print(f"  {styled('Model', MUTED)}    {model}")
    console.print(f"  {styled('Project', MUTED)}  {project_dir}")
    console.print(f"  {styled('Audit', MUTED)}    {audit_status}")

    # Hint line
    console.print(
        f"\n  {styled(f'Type /help for commands {SEPARATOR_DOT} /plan for read-only mode', DIM)}\n"
    )


def format_session_summary(
    duration_secs: float,
    input_tokens: int,
    output_tokens: int,
    cost: float | None = None,
    tool_calls: int = 0,
    tool_errors: int = 0,
    tool_denied: int = 0,
) -> None:
    """Display session summary on quit — clean, compact."""
    console.print()
    console.print(f"  {_rule()}")
    console.print(f"  {styled('Session complete', DIM)}")
    console.print()

    # Duration
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)
    dur = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    console.print(f"    {styled('Duration', MUTED)}  {dur}")

    # Tokens
    total = input_tokens + output_tokens
    console.print(
        f"    {styled('Tokens', MUTED)}    {total:,}"
        f"  {styled(f'(in: {input_tokens:,} {SEPARATOR_DOT} out: {output_tokens:,})', DIM)}"
    )

    # Cost
    if cost is not None and cost > 0:
        console.print(f"    {styled('Cost', MUTED)}      ${cost:.4f}")
    elif cost is not None:
        console.print(f"    {styled('Cost', MUTED)}      {styled('free', SUCCESS)}")

    # Tool summary
    if tool_calls > 0:
        success = tool_calls - tool_errors - tool_denied
        parts = [f"{success} {MARKER_SUCCESS}"]
        if tool_errors > 0:
            parts.append(f"{tool_errors} {MARKER_ERROR}")
        if tool_denied > 0:
            parts.append(f"{tool_denied} denied")
        summary = f" {SEPARATOR_DOT} ".join(parts)
        console.print(
            f"    {styled('Tools', MUTED)}     {tool_calls} calls  {styled(f'({summary})', DIM)}"
        )

    # Branded sign-off with decorative slashes
    dec = styled(f"{DECORATOR}{DECORATOR}{DECORATOR}", MUTED)
    console.print(f"\n  {dec} {PROMPT_ICON} {styled('Godspeed', BOLD_PRIMARY)} {dec}\n")
