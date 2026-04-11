"""Rich output formatting for the Godspeed TUI."""

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

logger = logging.getLogger(__name__)

console = Console()


def format_tool_call(name: str, args: dict[str, Any]) -> None:
    """Display a tool call as a Rich Panel with the tool name as header."""
    try:
        args_text = json.dumps(args, indent=2, default=str)
    except (TypeError, ValueError):
        args_text = str(args)

    syntax = Syntax(args_text, "json", theme="monokai", word_wrap=True)
    panel = Panel(
        syntax,
        title=f"[bold cyan]{name}[/bold cyan]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


def format_tool_result(name: str, result: str, is_error: bool = False) -> None:
    """Display a tool result with color based on success/error."""
    if is_error:
        style = "red"
        title = f"[bold red]{name} (error)[/bold red]"
    else:
        style = "green"
        title = f"[bold green]{name}[/bold green]"

    # Truncate very long outputs for display
    max_display = 2000
    display_text = result
    if len(result) > max_display:
        display_text = result[:max_display] + f"\n... ({len(result) - max_display} chars truncated)"

    panel = Panel(
        Text(display_text, style="dim" if not is_error else "red"),
        title=title,
        border_style=style,
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
    - file_edit: mini-diff of old_string → new_string
    - file_write: first 10 lines of content
    - shell: syntax-highlighted command
    - file_read / grep_search / repo_map: file path
    """
    console.print()

    detail_parts: list[Any] = []
    args = arguments or {}

    if tool_name == "file_edit" and args.get("old_string") and args.get("new_string"):
        file_path = args.get("file_path", "unknown")
        detail_parts.append(Text.from_markup(f"[dim]File:[/dim] {file_path}\n"))
        old = args["old_string"]
        new = args["new_string"]
        diff_lines = []
        for line in old.splitlines():
            diff_lines.append(f"- {line}")
        for line in new.splitlines():
            diff_lines.append(f"+ {line}")
        diff_text = "\n".join(diff_lines[:20])
        detail_parts.append(Syntax(diff_text, "diff", theme="monokai", word_wrap=True))
        if len(diff_lines) > 20:
            detail_parts.append(
                Text.from_markup(f"[dim]... ({len(diff_lines) - 20} more lines)[/dim]")
            )

    elif tool_name == "file_write" and args.get("content"):
        file_path = args.get("file_path", "unknown")
        detail_parts.append(Text.from_markup(f"[dim]File:[/dim] {file_path}\n"))
        all_lines = args["content"].splitlines()
        preview = "\n".join(all_lines[:10])
        # Guess lexer from file extension
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else "text"
        lexer_map = {"py": "python", "js": "javascript", "ts": "typescript", "yaml": "yaml"}
        lexer = lexer_map.get(ext, ext)
        detail_parts.append(Syntax(preview, lexer, theme="monokai", word_wrap=True))
        if len(all_lines) > 10:
            detail_parts.append(
                Text.from_markup(f"[dim]... ({len(all_lines) - 10} more lines)[/dim]")
            )

    elif tool_name == "shell" and args.get("command"):
        detail_parts.append(Syntax(args["command"], "bash", theme="monokai", word_wrap=True))

    elif args.get("file_path"):
        detail_parts.append(Text.from_markup(f"[dim]Path:[/dim] {args['file_path']}"))

    elif args.get("pattern"):
        detail_parts.append(Text.from_markup(f"[dim]Pattern:[/dim] {args['pattern']}"))

    # Build the panel content
    from rich.console import Group

    content_parts: list[Any] = [
        Text.from_markup(f"[bold]{tool_name}[/bold]\n"),
    ]
    if detail_parts:
        content_parts.append(Text(""))  # spacer
        content_parts.extend(detail_parts)
        content_parts.append(Text(""))  # spacer

    content_parts.append(
        Text.from_markup(
            f"[dim]{reason}[/dim]\n\n"
            "[yellow]Allow this tool call?[/yellow] "
            "[dim](y)es / (n)o / (a)lways for this session[/dim]"
        )
    )

    panel = Panel(
        Group(*content_parts),
        title="[bold yellow]Permission Required[/bold yellow]",
        border_style="yellow",
        expand=False,
    )
    console.print(panel)
    return ""


def format_permission_denied(tool_name: str, reason: str) -> None:
    """Display a permission denied notice."""
    console.print(
        Text(f"  Blocked: {tool_name} -- {reason}", style="bold red"),
    )


def format_stats(
    input_tokens: int,
    output_tokens: int,
    model: str,
    session_id: str,
) -> None:
    """Display session statistics."""
    table = Table(show_header=False, border_style="dim", expand=False)
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Model", model)
    table.add_row("Session", session_id[:12] + "...")
    table.add_row("Input tokens", f"{input_tokens:,}")
    table.add_row("Output tokens", f"{output_tokens:,}")
    table.add_row("Total tokens", f"{input_tokens + output_tokens:,}")

    panel = Panel(table, title="[bold]Session Stats[/bold]", border_style="blue", expand=False)
    console.print(panel)


def format_welcome(
    model: str,
    project_dir: str,
    tools: list[str] | None = None,
    deny_rules: list[str] | None = None,
    audit_enabled: bool = True,
) -> None:
    """Display welcome banner with safety disclosure (Claude Code style)."""
    from godspeed import __version__

    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Godspeed v{__version__}[/bold] -- Security-first coding agent\n\n"
                f"[dim]Model:[/dim]   {model}\n"
                f"[dim]Project:[/dim] {project_dir}\n"
                f"[dim]Audit:[/dim]   {'enabled' if audit_enabled else '[red]disabled[/red]'}"
            ),
            border_style="bright_blue",
            expand=False,
        )
    )

    # Safety disclosure — what the agent can do
    if tools:
        tool_list = ", ".join(tools)
        console.print(f"\n  [bold]Tools:[/bold] [dim]{tool_list}[/dim]")

    console.print(
        "\n  [bold yellow]Safety:[/bold yellow]"
        " All tool calls require permission."
        " Destructive commands are blocked by default."
    )

    if deny_rules:
        sample = deny_rules[:5]
        deny_display = ", ".join(sample)
        if len(deny_rules) > 5:
            deny_display += f", ... (+{len(deny_rules) - 5} more)"
        console.print(f"  [red]Deny:[/red]   [dim]{deny_display}[/dim]")

    console.print("\n  [dim]Type /help for commands, Ctrl+C to interrupt, /quit to exit.[/dim]\n")


def format_error(message: str) -> None:
    """Display an error message."""
    console.print(Text(f"Error: {message}", style="bold red"))
