"""Slash commands for the Godspeed TUI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any

from godspeed.tui.output import console, format_error, format_stats
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    BOLD_SUCCESS,
    BOLD_WARNING,
    CTX_CRITICAL,
    CTX_OK,
    CTX_WARN,
    DIM,
    ERROR,
    MUTED,
    PERM_ALLOW,
    PERM_ASK,
    PERM_DENY,
    PERM_SESSION,
    SUCCESS,
    TABLE_BORDER,
    TABLE_KEY,
    TABLE_VALUE,
    WARNING,
)

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
        pause_event: Any | None = None,
    ) -> None:
        self._conversation = conversation
        self._llm_client = llm_client
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id
        self._cwd = cwd
        self._pause_event = pause_event
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
        self._handlers["/plan"] = self._cmd_plan
        self._handlers["/checkpoint"] = self._cmd_checkpoint
        self._handlers["/restore"] = self._cmd_restore
        self._handlers["/pause"] = self._cmd_pause
        self._handlers["/resume"] = self._cmd_resume
        self._handlers["/guidance"] = self._cmd_guidance
        self._handlers["/tasks"] = self._cmd_tasks
        self._handlers["/quit"] = self._cmd_quit
        self._handlers["/exit"] = self._cmd_quit

    # Task store reference — set externally after Commands init
    _task_store: Any | None = None

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

        table = Table(title="Commands", border_style=TABLE_BORDER, expand=False)
        table.add_column("Command", style=BOLD_PRIMARY)
        table.add_column("Description")

        table.add_row("/help", "Show this help message")
        table.add_row("/model [name]", "Show or switch the active model")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/undo", "Undo last git commit (git reset --soft HEAD~1)")
        table.add_row("/audit", "Show audit trail stats and verify chain integrity")
        table.add_row("/permissions", "Show current permission rules")
        table.add_row("/extend [N]", "Set max iterations per turn (default: 50)")
        table.add_row("/context", "Show context window usage")
        table.add_row("/plan", "Toggle plan mode (read-only -- explore and plan only)")
        table.add_row("/checkpoint [name]", "Save checkpoint, or list if no name")
        table.add_row("/restore <name>", "Restore a saved checkpoint")
        table.add_row("/pause", "Pause the agent loop at next iteration")
        table.add_row("/resume", "Resume a paused agent loop")
        table.add_row("/guidance <msg>", "Inject guidance and resume paused agent")
        table.add_row("/quit, /exit", "Exit Godspeed")

        console.print(table)
        return CommandResult(handled=True)

    def _cmd_model(self, args: str = "") -> CommandResult:
        """Show or switch the active model."""
        if args.strip():
            old_model = self._llm_client.model
            self._llm_client.model = args.strip()
            new_model = self._llm_client.model
            console.print(
                f"  Model switched: [{MUTED}]{old_model}[/{MUTED}]"
                f" -> [{BOLD_PRIMARY}]{new_model}[/{BOLD_PRIMARY}]"
            )
        else:
            model = self._llm_client.model
            console.print(f"  Active model: [{BOLD_PRIMARY}]{model}[/{BOLD_PRIMARY}]")
            if self._llm_client.fallback_models:
                fallbacks = ", ".join(self._llm_client.fallback_models)
                console.print(f"  Fallbacks: [{DIM}]{fallbacks}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_clear(self, _args: str = "") -> CommandResult:
        """Clear conversation history."""
        self._conversation.clear()
        console.print(f"  [{DIM}]Conversation cleared.[/{DIM}]")
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
            console.print(f"  Undoing: [{MUTED}]{last_commit}[/{MUTED}]")

            undo_result = subprocess.run(
                ["git", "reset", "--soft", "HEAD~1"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=10,
            )
            if undo_result.returncode == 0:
                console.print(
                    f"  [{SUCCESS}]Last commit undone (changes preserved in staging).[/{SUCCESS}]"
                )
            else:
                format_error(f"git reset failed: {undo_result.stderr.strip()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            format_error(f"Failed to run git: {exc}")

        return CommandResult(handled=True)

    def _cmd_audit(self, _args: str = "") -> CommandResult:
        """Show audit trail stats and verify chain."""
        if self._audit_trail is None:
            console.print(f"  [{DIM}]Audit trail is disabled.[/{DIM}]")
            return CommandResult(handled=True)

        from rich.table import Table

        table = Table(show_header=False, border_style=MUTED, expand=False)
        table.add_column("Key", style=TABLE_KEY)
        table.add_column("Value", style=TABLE_VALUE)
        table.add_row("Session", self._session_id[:12] + "...")
        table.add_row("Records", str(self._audit_trail.record_count))
        table.add_row("Log file", str(self._audit_trail.log_path))

        console.print(table)

        # Verify chain integrity
        is_valid, message = self._audit_trail.verify_chain()
        if is_valid:
            console.print(f"  [{SUCCESS}]Chain integrity: VALID -- {message}[/{SUCCESS}]")
        else:
            console.print(f"  [{ERROR}]Chain integrity: BROKEN -- {message}[/{ERROR}]")

        return CommandResult(handled=True)

    def _cmd_permissions(self, _args: str = "") -> CommandResult:
        """Show current permission rules."""
        from rich.table import Table

        if self._permission_engine is None:
            console.print(f"  [{DIM}]Permission engine not loaded.[/{DIM}]")
            return CommandResult(handled=True)

        table = Table(title="Permission Rules", border_style=WARNING, expand=False)
        table.add_column("Action", style="bold")
        table.add_column("Pattern")

        for rule in self._permission_engine.deny_rules:
            table.add_row(f"[{PERM_DENY}]DENY[/{PERM_DENY}]", rule.pattern)
        for rule in self._permission_engine.allow_rules:
            table.add_row(f"[{PERM_ALLOW}]ALLOW[/{PERM_ALLOW}]", rule.pattern)
        for rule in self._permission_engine.ask_rules:
            table.add_row(f"[{PERM_ASK}]ASK[/{PERM_ASK}]", rule.pattern)

        # Session grants
        for grant in self._permission_engine.session_grants:
            table.add_row(f"[{PERM_SESSION}]SESSION[/{PERM_SESSION}]", grant)

        console.print(table)
        return CommandResult(handled=True)

    def _cmd_plan(self, _args: str = "") -> CommandResult:
        """Toggle plan mode — read-only, explore and plan only."""
        if self._permission_engine is None:
            format_error("Permission engine not loaded — cannot toggle plan mode.")
            return CommandResult(handled=True)

        self._permission_engine.plan_mode = not self._permission_engine.plan_mode
        if self._permission_engine.plan_mode:
            console.print(
                f"  [{BOLD_WARNING}]Plan mode ON[/{BOLD_WARNING}] — "
                "read-only tools only. Use /plan again to exit."
            )
        else:
            console.print(
                f"  [{BOLD_SUCCESS}]Plan mode OFF[/{BOLD_SUCCESS}] — full tool access restored."
            )
        return CommandResult(handled=True)

    def _cmd_extend(self, args: str = "") -> CommandResult:
        """Set or show the max iterations per agent turn."""
        from godspeed.agent.loop import MAX_ITERATIONS

        if not args.strip():
            current = self.max_iterations if self.max_iterations is not None else MAX_ITERATIONS
            console.print(
                f"  Max iterations: [{BOLD_PRIMARY}]{current}[/{BOLD_PRIMARY}]"
                f" (default: {MAX_ITERATIONS})"
            )
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
        console.print(f"  Max iterations set to [{BOLD_PRIMARY}]{value}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_context(self, _args: str = "") -> CommandResult:
        """Show context window usage."""
        tokens = self._conversation.token_count
        max_tokens = self._conversation.max_tokens
        pct = (tokens / max_tokens * 100) if max_tokens > 0 else 0

        if pct < 50:
            color = CTX_OK
        elif pct < 80:
            color = CTX_WARN
        else:
            color = CTX_CRITICAL

        console.print(f"  [{color}]tokens: {tokens:,} / {max_tokens:,} ({pct:.0f}%)[/{color}]")
        msg_count = len(self._conversation.messages)
        console.print(f"  [{DIM}]messages: {msg_count}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_checkpoint(self, args: str = "") -> CommandResult:
        """Save a checkpoint or list available checkpoints."""
        from godspeed.context.checkpoint import list_checkpoints, save_checkpoint

        name = args.strip()

        if not name or name == "list":
            # List checkpoints
            checkpoints = list_checkpoints(self._cwd)
            if not checkpoints:
                console.print(f"  [{DIM}]No checkpoints saved yet.[/{DIM}]")
                return CommandResult(handled=True)

            from datetime import datetime

            from rich.table import Table

            table = Table(title="Checkpoints", border_style=TABLE_BORDER, expand=False)
            table.add_column("Name", style=BOLD_PRIMARY)
            table.add_column("Time", style=MUTED)
            table.add_column("Model")
            table.add_column("Tokens", justify="right")
            table.add_column("Messages", justify="right")

            for cp in checkpoints:
                ts = datetime.fromtimestamp(cp["timestamp"], tz=UTC)
                table.add_row(
                    cp["name"],
                    ts.strftime("%Y-%m-%d %H:%M"),
                    cp["model"],
                    f"{cp['token_count']:,}",
                    str(cp["message_count"]),
                )

            console.print(table)
            return CommandResult(handled=True)

        # Save checkpoint
        system_msg = self._conversation.messages[0]
        system_prompt = system_msg.get("content", "")
        # Messages excluding system prompt
        messages = self._conversation.messages[1:]

        path = save_checkpoint(
            name=name,
            system_prompt=system_prompt,
            messages=messages,
            model=self._llm_client.model,
            token_count=self._conversation.token_count,
            project_dir=self._cwd,
        )
        console.print(
            f"  [{SUCCESS}]Checkpoint saved:[/{SUCCESS}] [{BOLD_PRIMARY}]{name}[/{BOLD_PRIMARY}]"
        )
        console.print(f"  [{DIM}]{path}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_restore(self, args: str = "") -> CommandResult:
        """Restore a saved checkpoint."""
        from godspeed.context.checkpoint import load_checkpoint

        name = args.strip()
        if not name:
            format_error("Usage: /restore <name>")
            return CommandResult(handled=True)

        data = load_checkpoint(name, self._cwd)
        if data is None:
            format_error(f"Checkpoint not found: {name}")
            return CommandResult(handled=True)

        # Restore conversation state
        self._conversation.clear()
        for msg in data.get("messages", []):
            role = msg.get("role", "")
            if role == "user":
                self._conversation.add_user_message(msg.get("content", ""))
            elif role == "assistant":
                self._conversation.add_assistant_message(
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls"),
                )
            elif role == "tool":
                self._conversation.add_tool_result(
                    tool_call_id=msg.get("tool_call_id", ""),
                    content=msg.get("content", ""),
                )

        token_count = self._conversation.token_count
        msg_count = len(self._conversation.messages) - 1  # exclude system prompt
        console.print(
            f"  [{SUCCESS}]Restored checkpoint:[/{SUCCESS}]"
            f" [{BOLD_PRIMARY}]{name}[/{BOLD_PRIMARY}]"
            f" ({msg_count} messages, {token_count:,} tokens)"
        )
        return CommandResult(handled=True)

    def _cmd_pause(self, _args: str = "") -> CommandResult:
        """Pause the agent loop at the next iteration."""
        if self._pause_event is None:
            format_error("Pause/resume not available in this session.")
            return CommandResult(handled=True)

        self._pause_event.clear()
        console.print(
            f"  [{BOLD_WARNING}]Agent paused.[/{BOLD_WARNING}] Use /resume or /guidance <msg>."
        )
        return CommandResult(handled=True)

    def _cmd_resume(self, _args: str = "") -> CommandResult:
        """Resume a paused agent loop."""
        if self._pause_event is None:
            format_error("Pause/resume not available in this session.")
            return CommandResult(handled=True)

        if self._pause_event.is_set():
            console.print(f"  [{DIM}]Agent is not paused.[/{DIM}]")
            return CommandResult(handled=True)

        self._pause_event.set()
        console.print(f"  [{BOLD_SUCCESS}]Agent resumed.[/{BOLD_SUCCESS}]")
        return CommandResult(handled=True)

    def _cmd_guidance(self, args: str = "") -> CommandResult:
        """Inject guidance as a user message and resume the paused agent."""
        if not args.strip():
            format_error("Usage: /guidance <your guidance message>")
            return CommandResult(handled=True)

        # Inject guidance into conversation
        self._conversation.add_user_message(f"[User guidance]: {args.strip()}")
        console.print(f"  [{DIM}]Guidance injected: {args.strip()}[/{DIM}]")

        # Resume if paused
        if self._pause_event is not None and not self._pause_event.is_set():
            self._pause_event.set()
            console.print(f"  [{BOLD_SUCCESS}]Agent resumed with guidance.[/{BOLD_SUCCESS}]")

        return CommandResult(handled=True)

    def _cmd_tasks(self, _args: str = "") -> CommandResult:
        """Show current task list."""
        if self._task_store is None:
            console.print(f"  [{MUTED}]Task tracking not enabled.[/{MUTED}]")
            return CommandResult()

        tasks = self._task_store.list_all()
        if not tasks:
            console.print(f"  [{MUTED}]No tasks.[/{MUTED}]")
            return CommandResult()

        from rich.table import Table

        table = Table(title="Tasks", border_style=TABLE_BORDER, expand=False)
        table.add_column("ID", style=BOLD_PRIMARY, width=4)
        table.add_column("Title")
        table.add_column("Status")

        for t in tasks:
            if t.status == "completed":
                status_style = f"[{SUCCESS}]{t.status}[/{SUCCESS}]"
            elif t.status == "in_progress":
                status_style = f"[{WARNING}]{t.status}[/{WARNING}]"
            else:
                status_style = f"[{MUTED}]{t.status}[/{MUTED}]"
            table.add_row(str(t.id), t.title, status_style)

        console.print(table)
        return CommandResult()

    def _cmd_quit(self, _args: str = "") -> CommandResult:
        """Exit Godspeed."""
        format_stats(
            input_tokens=self._llm_client.total_input_tokens,
            output_tokens=self._llm_client.total_output_tokens,
            model=self._llm_client.model,
            session_id=self._session_id,
        )
        console.print(f"  [{DIM}]Goodbye.[/{DIM}]")
        return CommandResult(handled=True, should_quit=True)
