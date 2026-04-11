"""Slash commands for the Godspeed TUI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from godspeed.tui.output import console, format_error, format_stats

logger = logging.getLogger(__name__)


class CommandResult:
    """Result from executing a slash command."""

    def __init__(
        self,
        handled: bool = True,
        should_quit: bool = False,
        message: str = "",
    ) -> None:
        self.handled = handled
        self.should_quit = should_quit
        self.message = message


CommandHandler = Callable[..., CommandResult]


class Commands:
    """Registry of slash commands with dispatch.

    Usage:
        commands = Commands(...)
        result = commands.dispatch("/help")
    """

    def __init__(
        self,
        conversation: Any,
        llm_client: Any,
        permission_engine: Any,
        audit_trail: Any | None,
        session_id: str,
        cwd: Path,
    ) -> None:
        self._conversation = conversation
        self._llm_client = llm_client
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id
        self._cwd = cwd
        self._handlers: dict[str, CommandHandler] = {}
        self.max_iterations: int | None = None  # None = use default
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in slash commands."""
        self._handlers["/help"] = self._cmd_help
        self._handlers["/model"] = self._cmd_model
        self._handlers["/clear"] = self._cmd_clear
        self._handlers["/undo"] = self._cmd_undo
        self._handlers["/audit"] = self._cmd_audit
        self._handlers["/permissions"] = self._cmd_permissions
        self._handlers["/extend"] = self._cmd_extend
        self._handlers["/context"] = self._cmd_context
        self._handlers["/quit"] = self._cmd_quit
        self._handlers["/exit"] = self._cmd_quit

    def register(self, name: str, handler: CommandHandler) -> None:
        """Register a custom slash command."""
        if not name.startswith("/"):
            name = "/" + name
        self._handlers[name] = handler

    def dispatch(self, raw_input: str) -> CommandResult | None:
        """Dispatch a slash command. Returns None if input is not a command."""
        stripped = raw_input.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = self._handlers.get(cmd)
        if handler is None:
            format_error(f"Unknown command: {cmd}. Type /help for available commands.")
            return CommandResult(handled=True)

        return handler(args)

    # -- Built-in command handlers ------------------------------------------------

    def _cmd_help(self, _args: str = "") -> CommandResult:
        """Show available commands."""
        from rich.table import Table

        table = Table(title="Commands", border_style="blue", expand=False)
        table.add_column("Command", style="bold cyan")
        table.add_column("Description")

        table.add_row("/help", "Show this help message")
        table.add_row("/model [name]", "Show or switch the active model")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/undo", "Undo last git commit (git reset --soft HEAD~1)")
        table.add_row("/audit", "Show audit trail stats and verify chain integrity")
        table.add_row("/permissions", "Show current permission rules")
        table.add_row("/extend [N]", "Set max iterations per turn (default: 50)")
        table.add_row("/context", "Show context window usage")
        table.add_row("/quit, /exit", "Exit Godspeed")

        console.print(table)
        return CommandResult(handled=True)

    def _cmd_model(self, args: str = "") -> CommandResult:
        """Show or switch the active model."""
        if args.strip():
            old_model = self._llm_client.model
            self._llm_client.model = args.strip()
            console.print(
                f"  Model switched: [dim]{old_model}[/dim] -> [bold]{self._llm_client.model}[/bold]"
            )
        else:
            console.print(f"  Active model: [bold]{self._llm_client.model}[/bold]")
            if self._llm_client.fallback_models:
                fallbacks = ", ".join(self._llm_client.fallback_models)
                console.print(f"  Fallbacks: [dim]{fallbacks}[/dim]")
        return CommandResult(handled=True)

    def _cmd_clear(self, _args: str = "") -> CommandResult:
        """Clear conversation history."""
        self._conversation.clear()
        console.print("  [dim]Conversation cleared.[/dim]")
        return CommandResult(handled=True)

    def _cmd_undo(self, _args: str = "") -> CommandResult:
        """Undo last git commit with git reset --soft HEAD~1."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=10,
            )
            if result.returncode != 0:
                format_error("Not a git repository or no commits to undo.")
                return CommandResult(handled=True)

            last_commit = result.stdout.strip()
            console.print(f"  Undoing: [dim]{last_commit}[/dim]")

            undo_result = subprocess.run(
                ["git", "reset", "--soft", "HEAD~1"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=10,
            )
            if undo_result.returncode == 0:
                console.print("  [green]Last commit undone (changes preserved in staging).[/green]")
            else:
                format_error(f"git reset failed: {undo_result.stderr.strip()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            format_error(f"Failed to run git: {exc}")

        return CommandResult(handled=True)

    def _cmd_audit(self, _args: str = "") -> CommandResult:
        """Show audit trail stats and verify chain."""
        if self._audit_trail is None:
            console.print("  [dim]Audit trail is disabled.[/dim]")
            return CommandResult(handled=True)

        from rich.table import Table

        table = Table(show_header=False, border_style="dim", expand=False)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("Session", self._session_id[:12] + "...")
        table.add_row("Records", str(self._audit_trail.record_count))
        table.add_row("Log file", str(self._audit_trail.log_path))

        console.print(table)

        # Verify chain integrity
        is_valid, message = self._audit_trail.verify_chain()
        if is_valid:
            console.print(f"  [green]Chain integrity: VALID -- {message}[/green]")
        else:
            console.print(f"  [red]Chain integrity: BROKEN -- {message}[/red]")

        return CommandResult(handled=True)

    def _cmd_permissions(self, _args: str = "") -> CommandResult:
        """Show current permission rules."""
        from rich.table import Table

        if self._permission_engine is None:
            console.print("  [dim]Permission engine not loaded.[/dim]")
            return CommandResult(handled=True)

        table = Table(title="Permission Rules", border_style="yellow", expand=False)
        table.add_column("Action", style="bold")
        table.add_column("Pattern")

        for rule in self._permission_engine.deny_rules:
            table.add_row("[red]DENY[/red]", rule.pattern)
        for rule in self._permission_engine.allow_rules:
            table.add_row("[green]ALLOW[/green]", rule.pattern)
        for rule in self._permission_engine.ask_rules:
            table.add_row("[yellow]ASK[/yellow]", rule.pattern)

        # Session grants
        for grant in self._permission_engine.session_grants:
            table.add_row("[blue]SESSION[/blue]", grant)

        console.print(table)
        return CommandResult(handled=True)

    def _cmd_extend(self, args: str = "") -> CommandResult:
        """Set or show the max iterations per agent turn."""
        from godspeed.agent.loop import MAX_ITERATIONS

        if not args.strip():
            current = self.max_iterations if self.max_iterations is not None else MAX_ITERATIONS
            console.print(f"  Max iterations: [bold]{current}[/bold] (default: {MAX_ITERATIONS})")
            return CommandResult(handled=True)

        try:
            value = int(args.strip())
        except ValueError:
            format_error(f"Invalid number: {args.strip()}")
            return CommandResult(handled=True)

        if value < 1:
            format_error("Max iterations must be at least 1.")
            return CommandResult(handled=True)

        self.max_iterations = value
        console.print(f"  Max iterations set to [bold]{value}[/bold]")
        return CommandResult(handled=True)

    def _cmd_context(self, _args: str = "") -> CommandResult:
        """Show context window usage."""
        tokens = self._conversation.token_count
        max_tokens = self._conversation.max_tokens
        pct = (tokens / max_tokens * 100) if max_tokens > 0 else 0

        if pct < 50:
            color = "green"
        elif pct < 80:
            color = "yellow"
        else:
            color = "red"

        console.print(f"  [{color}]tokens: {tokens:,} / {max_tokens:,} ({pct:.0f}%)[/{color}]")
        msg_count = len(self._conversation.messages)
        console.print(f"  [dim]messages: {msg_count}[/dim]")
        return CommandResult(handled=True)

    def _cmd_quit(self, _args: str = "") -> CommandResult:
        """Exit Godspeed."""
        format_stats(
            input_tokens=self._llm_client.total_input_tokens,
            output_tokens=self._llm_client.total_output_tokens,
            model=self._llm_client.model,
            session_id=self._session_id,
        )
        console.print("  [dim]Goodbye.[/dim]")
        return CommandResult(handled=True, should_quit=True)
