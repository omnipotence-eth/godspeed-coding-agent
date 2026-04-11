"""Rich output formatting for the Godspeed TUI — Midnight Gold theme."""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from godspeed.tui.theme import (
    BOLD_ERROR,
    BOLD_PRIMARY,
    BOLD_SUCCESS,
    BOLD_WARNING,
    BORDER_BRAND,
    BORDER_ERROR,
    BORDER_INFO,
    BORDER_SUCCESS,
    BORDER_TOOL,
    BORDER_WARNING,
    BRAND_TAGLINE,
    DIM,
    ERROR,
    MUTED,
    PERM_DENY,
    PROMPT_ICON,
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


def format_tool_call(name: str, args: dict[str, Any]) -> None:
    """Display a tool call as a Rich Panel with the tool name as header."""
    try:
        args_text = json.dumps(args, indent=2, default=str)
    except (TypeError, ValueError):
        args_text = str(args)

    syntax = Syntax(args_text, "json", theme=SYNTAX_THEME, word_wrap=True)
    panel = Panel(
        syntax,
        title=styled(name, BOLD_PRIMARY),
        border_style=BORDER_TOOL,
        expand=False,
    )
    console.print(panel)


def format_tool_result(name: str, result: str, is_error: bool = False) -> None:
    """Display a tool result with color based on success/error."""
    if is_error:
        border = BORDER_ERROR
        title = styled(f"{name} (error)", BOLD_ERROR)
        text_style = ERROR
    else:
        border = BORDER_SUCCESS
        title = styled(name, BOLD_SUCCESS)
        text_style = DIM

    # Truncate very long outputs for display
    max_display = 2000
    display_text = result
    if len(result) > max_display:
        display_text = result[:max_display] + f"\n... ({len(result) - max_display} chars truncated)"

    panel = Panel(
        Text(display_text, style=text_style),
        title=title,
        border_style=border,
        expand=False,
    )
    console.print(panel)


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
    - file_edit: mini-diff of old_string -> new_string
    - file_write: first 10 lines of content
    - shell: syntax-highlighted command
    - file_read / grep_search / repo_map: file path
    """
    console.print()

    detail_parts: list[Any] = []
    args = arguments or {}

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

        detail_parts.append(
            Text.from_markup(f"[{MUTED}]File:[/{MUTED}] {file_path}  [{DIM}]({stats})[/{DIM}]\n")
        )

        # Unified diff with context
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
        # Skip the --- / +++ headers, keep @@ hunks and content
        diff_content = [line for line in diff_output[2:]] if len(diff_output) > 2 else diff_output
        diff_text = "\n".join(diff_content[:30])
        detail_parts.append(Syntax(diff_text, "diff", theme=SYNTAX_THEME, word_wrap=True))
        if len(diff_content) > 30:
            detail_parts.append(
                Text.from_markup(f"[{DIM}]... ({len(diff_content) - 30} more lines)[/{DIM}]")
            )

    elif tool_name == "file_write" and args.get("content"):
        from pathlib import Path

        file_path = args.get("file_path", "unknown")
        all_lines = args["content"].splitlines()
        line_count = len(all_lines)

        # Detect overwrite vs create
        target = Path(file_path)
        if not target.is_absolute():
            # Best-effort check — may not resolve perfectly without context.cwd
            action = "write"
        elif target.exists():
            action = "overwrite"
        else:
            action = "create"
        action_label = f"[{WARNING}]{action}[/{WARNING}]" if action == "overwrite" else action

        detail_parts.append(
            Text.from_markup(
                f"[{MUTED}]File:[/{MUTED}] {file_path}  "
                f"[{DIM}]({action_label}, {line_count} lines)[/{DIM}]\n"
            )
        )
        preview = "\n".join(all_lines[:15])
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else "text"
        lexer_map = {"py": "python", "js": "javascript", "ts": "typescript", "yaml": "yaml"}
        lexer = lexer_map.get(ext, ext)
        detail_parts.append(Syntax(preview, lexer, theme=SYNTAX_THEME, word_wrap=True))
        if len(all_lines) > 15:
            detail_parts.append(
                Text.from_markup(f"[{DIM}]... ({len(all_lines) - 15} more lines)[/{DIM}]")
            )

    elif tool_name == "shell" and args.get("command"):
        detail_parts.append(Syntax(args["command"], "bash", theme=SYNTAX_THEME, word_wrap=True))

    elif args.get("file_path"):
        detail_parts.append(Text.from_markup(f"[{MUTED}]Path:[/{MUTED}] {args['file_path']}"))

    elif args.get("pattern"):
        detail_parts.append(Text.from_markup(f"[{MUTED}]Pattern:[/{MUTED}] {args['pattern']}"))

    # Build the panel content
    from rich.console import Group

    content_parts: list[Any] = [
        Text.from_markup(f"[{BOLD_PRIMARY}]{tool_name}[/{BOLD_PRIMARY}]\n"),
    ]
    if detail_parts:
        content_parts.append(Text(""))
        content_parts.extend(detail_parts)
        content_parts.append(Text(""))

    content_parts.append(
        Text.from_markup(
            f"[{DIM}]{reason}[/{DIM}]\n\n"
            f"[{WARNING}]Allow this tool call?[/{WARNING}] "
            f"[{DIM}](y)es / (n)o / (a)lways for this session[/{DIM}]"
        )
    )

    panel = Panel(
        Group(*content_parts),
        title=styled("Permission Required", BOLD_WARNING),
        border_style=BORDER_WARNING,
        expand=False,
    )
    console.print(panel)
    return ""


def format_permission_denied(tool_name: str, reason: str) -> None:
    """Display a permission denied notice."""
    console.print(
        Text(f"  Blocked: {tool_name} -- {reason}", style=BOLD_ERROR),
    )


def format_stats(
    input_tokens: int,
    output_tokens: int,
    model: str,
    session_id: str,
    cost: float | None = None,
) -> None:
    """Display session statistics."""
    table = Table(show_header=False, border_style=MUTED, expand=False)
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
        border_style=BORDER_INFO,
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
    """Display welcome banner — Midnight Gold theme."""
    from godspeed import __version__

    console.print()
    audit_status = styled("enabled", SUCCESS) if audit_enabled else styled("disabled", ERROR)
    console.print(
        Panel(
            Text.from_markup(
                f"{PROMPT_ICON} {brand(__version__)} -- {styled(BRAND_TAGLINE, DIM)}\n\n"
                f"[{MUTED}]Model:[/{MUTED}]   {model}\n"
                f"[{MUTED}]Project:[/{MUTED}] {project_dir}\n"
                f"[{MUTED}]Audit:[/{MUTED}]   {audit_status}"
            ),
            border_style=BORDER_BRAND,
            expand=False,
        )
    )

    if tools:
        tool_list = ", ".join(tools)
        console.print(f"\n  [{BOLD_PRIMARY}]Tools:[/{BOLD_PRIMARY}] [{DIM}]{tool_list}[/{DIM}]")

    console.print(
        f"\n  [{BOLD_WARNING}]Safety:[/{BOLD_WARNING}]"
        " All tool calls require permission."
        " Destructive commands are blocked by default."
    )

    if deny_rules:
        sample = deny_rules[:5]
        deny_display = ", ".join(sample)
        if len(deny_rules) > 5:
            deny_display += f", ... (+{len(deny_rules) - 5} more)"
        console.print(f"  [{PERM_DENY}]Deny:[/{PERM_DENY}]   [{DIM}]{deny_display}[/{DIM}]")

    console.print(
        f"\n  [{DIM}]Type /help for commands, Ctrl+C to interrupt, /quit to exit.[/{DIM}]\n"
    )


def format_error(message: str) -> None:
    """Display an error message."""
    console.print(Text(f"Error: {message}", style=BOLD_ERROR))
