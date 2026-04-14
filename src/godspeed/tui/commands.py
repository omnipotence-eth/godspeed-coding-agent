"""Slash commands for the Godspeed TUI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any

from godspeed.tui.output import (
    console,
    format_error,
    format_info,
    format_stats,
    format_success,
    format_warning,
)
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    CTX_CRITICAL,
    CTX_OK,
    CTX_WARN,
    DIM,
    MUTED,
    PERM_ALLOW,
    PERM_ASK,
    PERM_DENY,
    PERM_SESSION,
    RULE_CHAR,
    SUCCESS,
    TABLE_BORDER,
    TABLE_KEY,
    TABLE_VALUE,
    WARNING,
    styled,
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
        self.auto_commit: bool = False
        self.auto_commit_threshold: int = 5
        self.architect_mode: bool = False
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
        self._handlers["/reindex"] = self._cmd_reindex
        self._handlers["/stats"] = self._cmd_stats
        self._handlers["/autocommit"] = self._cmd_autocommit
        self._handlers["/architect"] = self._cmd_architect
        self._handlers["/think"] = self._cmd_think
        self._handlers["/budget"] = self._cmd_budget
        self._handlers["/evolve"] = self._cmd_evolve
        self._handlers["/export"] = self._cmd_export
        self._handlers["/quit"] = self._cmd_quit
        self._handlers["/exit"] = self._cmd_quit

    # External references — set after Commands init
    _task_store: Any | None = None
    _codebase_index: Any | None = None

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
        """Show available commands — grouped by category."""
        rule = styled(RULE_CHAR * 40, MUTED)
        console.print()
        console.print(f"  {styled('Commands', BOLD_PRIMARY)}")
        console.print(f"  {rule}")

        groups: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Session",
                [
                    ("/model [name]", "Show or switch the active model"),
                    ("/clear", "Clear conversation history"),
                    ("/stats", "Show token usage and estimated cost"),
                    ("/export [name]", "Export conversation as markdown"),
                    ("/quit, /exit", "Exit Godspeed"),
                ],
            ),
            (
                "Agent Control",
                [
                    ("/plan", "Toggle plan mode (read-only)"),
                    ("/extend [N]", "Set max iterations per turn"),
                    ("/autocommit [on|off|N]", "Toggle auto-commit or set threshold"),
                    ("/architect", "Toggle architect mode (plan then execute)"),
                    ("/think [budget]", "Toggle extended thinking or set token budget"),
                    ("/budget [amount]", "Show/set cost budget in USD"),
                    ("/evolve [cmd]", "Self-evolution: status|run|history|rollback|review"),
                    ("/pause", "Pause the agent loop"),
                    ("/resume", "Resume a paused agent"),
                    ("/guidance <msg>", "Inject guidance and resume"),
                ],
            ),
            (
                "Context",
                [
                    ("/context", "Show context window usage"),
                    ("/checkpoint [name]", "Save/list checkpoints"),
                    ("/restore <name>", "Restore a checkpoint"),
                    ("/tasks", "Show task list"),
                    ("/reindex", "Rebuild codebase search index"),
                ],
            ),
            (
                "Security",
                [
                    ("/audit", "Show audit trail and verify chain"),
                    ("/permissions", "Show permission rules"),
                    ("/undo", "Undo last git commit"),
                ],
            ),
        ]

        for group_name, cmds in groups:
            console.print()
            console.print(f"  {styled(group_name, MUTED)}")
            for cmd_name, desc in cmds:
                console.print(f"    {styled(cmd_name, BOLD_PRIMARY):28s} {styled(desc, DIM)}")

        console.print()
        return CommandResult(handled=True)

    def _cmd_model(self, args: str = "") -> CommandResult:
        """Show or switch the active model."""
        if args.strip():
            old_model = self._llm_client.model
            self._llm_client.model = args.strip()
            new_model = self._llm_client.model
            format_success(
                f"Model switched: [{MUTED}]{old_model}[/{MUTED}]"
                f" -> [{BOLD_PRIMARY}]{new_model}[/{BOLD_PRIMARY}]"
            )
        else:
            model = self._llm_client.model
            format_info(f"Active model: [{BOLD_PRIMARY}]{model}[/{BOLD_PRIMARY}]")
            if self._llm_client.fallback_models:
                fallbacks = ", ".join(self._llm_client.fallback_models)
                console.print(f"    [{DIM}]Fallbacks: {fallbacks}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_clear(self, _args: str = "") -> CommandResult:
        """Clear conversation history."""
        self._conversation.clear()
        format_info("Conversation cleared.")
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
            format_info(f"Undoing: {last_commit}")

            undo_result = subprocess.run(
                ["git", "reset", "--soft", "HEAD~1"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=10,
            )
            if undo_result.returncode == 0:
                format_success("Last commit undone (changes preserved in staging).")
            else:
                format_error(f"git reset failed: {undo_result.stderr.strip()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            format_error(f"Failed to run git: {exc}")

        return CommandResult(handled=True)

    def _cmd_audit(self, _args: str = "") -> CommandResult:
        """Show audit trail stats and verify chain."""
        if self._audit_trail is None:
            format_info("Audit trail is disabled.")
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
            format_success(f"Chain integrity: VALID -- {message}")
        else:
            format_error(f"Chain integrity: BROKEN -- {message}")

        return CommandResult(handled=True)

    def _cmd_permissions(self, _args: str = "") -> CommandResult:
        """Show current permission rules."""
        from rich.table import Table

        if self._permission_engine is None:
            format_info("Permission engine not loaded.")
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
            format_warning("Plan mode ON — read-only tools only. Use /plan again to exit.")
        else:
            format_success("Plan mode OFF — full tool access restored.")
        return CommandResult(handled=True)

    def _cmd_extend(self, args: str = "") -> CommandResult:
        """Set or show the max iterations per agent turn."""
        from godspeed.agent.loop import MAX_ITERATIONS

        if not args.strip():
            current = self.max_iterations if self.max_iterations is not None else MAX_ITERATIONS
            format_info(
                f"Max iterations: [{BOLD_PRIMARY}]{current}[/{BOLD_PRIMARY}]"
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
        format_success(f"Max iterations set to [{BOLD_PRIMARY}]{value}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_autocommit(self, args: str = "") -> CommandResult:
        """Toggle auto-commit or set the file-change threshold."""
        arg = args.strip().lower()

        if arg == "on":
            self.auto_commit = True
            logger.info("autocommit toggled state=on threshold=%d", self.auto_commit_threshold)
            format_success(
                f"Auto-commit [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}]"
                f" (threshold: {self.auto_commit_threshold} files)"
            )
        elif arg == "off":
            self.auto_commit = False
            logger.info("autocommit toggled state=off")
            format_info(f"Auto-commit [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
        elif arg:
            # Numeric threshold
            try:
                value = int(arg)
            except ValueError:
                format_error(f"Invalid argument: {args.strip()}. Use on, off, or a number.")
                return CommandResult(handled=True)

            if value < 1:
                format_error("Threshold must be at least 1.")
                return CommandResult(handled=True)

            self.auto_commit_threshold = value
            self.auto_commit = True
            logger.info(
                "autocommit threshold_set threshold=%d state=on", self.auto_commit_threshold
            )
            format_success(
                f"Auto-commit [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}]"
                f" — threshold set to [{BOLD_PRIMARY}]{value}[/{BOLD_PRIMARY}] files"
            )
        else:
            # No args — toggle
            self.auto_commit = not self.auto_commit
            state = "ON" if self.auto_commit else "OFF"
            logger.info(
                "autocommit toggled state=%s threshold=%d",
                state.lower(),
                self.auto_commit_threshold,
            )
            format_info(
                f"Auto-commit [{BOLD_PRIMARY}]{state}[/{BOLD_PRIMARY}]"
                f" (threshold: {self.auto_commit_threshold} files)"
            )

        return CommandResult(handled=True)

    def _cmd_architect(self, args: str = "") -> CommandResult:
        """Toggle architect mode — two-phase plan-then-execute."""
        self.architect_mode = not self.architect_mode
        if self.architect_mode:
            format_success(
                f"Architect mode [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}] "
                "— requests will be planned before execution"
            )
        else:
            format_info(f"Architect mode [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_think(self, args: str = "") -> CommandResult:
        """Toggle extended thinking or set the thinking token budget."""
        arg = args.strip()

        if not arg:
            # Toggle: off → default 10k, on → off
            current = self._llm_client.thinking_budget
            if current > 0:
                self._llm_client.thinking_budget = 0
                format_info(f"Extended thinking [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
            else:
                self._llm_client.thinking_budget = 10_000
                format_success(
                    f"Extended thinking [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}] (budget: 10,000 tokens)"
                )
            return CommandResult(handled=True)

        if arg.lower() == "off":
            self._llm_client.thinking_budget = 0
            format_info(f"Extended thinking [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        try:
            budget = int(arg.replace(",", "").replace("_", ""))
        except ValueError:
            format_error(f"Invalid budget: {arg}. Use a number or 'off'.")
            return CommandResult(handled=True)

        if budget < 1000:
            format_error("Thinking budget must be at least 1,000 tokens.")
            return CommandResult(handled=True)

        self._llm_client.thinking_budget = budget
        format_success(
            f"Extended thinking [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}] (budget: {budget:,} tokens)"
        )
        return CommandResult(handled=True)

    def _cmd_budget(self, args: str = "") -> CommandResult:
        """Show or set the cost budget in USD."""
        from godspeed.llm.cost import format_cost

        arg = args.strip()

        if not arg:
            # Show current cost and budget
            spent = self._llm_client.total_cost_usd
            limit = self._llm_client.max_cost_usd
            model = self._llm_client.model
            input_tokens = self._llm_client.total_input_tokens
            output_tokens = self._llm_client.total_output_tokens

            spent_str = format_cost(spent)
            if limit > 0:
                pct = (spent / limit * 100) if limit > 0 else 0
                format_info(
                    f"Cost: [{BOLD_PRIMARY}]{spent_str}[/{BOLD_PRIMARY}]"
                    f" / ${limit:.2f} ({pct:.0f}%)"
                )
            else:
                format_info(
                    f"Cost: [{BOLD_PRIMARY}]{spent_str}[/{BOLD_PRIMARY}]"
                    f" [{DIM}](no budget limit)[/{DIM}]"
                )
            console.print(
                f"    [{DIM}]{input_tokens:,} input + {output_tokens:,} output tokens"
                f" ({model})[/{DIM}]"
            )
            return CommandResult(handled=True)

        if arg.lower() in ("off", "unlimited", "0"):
            self._llm_client.max_cost_usd = 0.0
            format_info(f"Cost budget [{BOLD_PRIMARY}]unlimited[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        # Strip $ prefix if present
        cleaned = arg.lstrip("$")
        try:
            limit = float(cleaned)
        except ValueError:
            format_error(f"Invalid amount: {arg}. Use a number like 5.00 or 'off'.")
            return CommandResult(handled=True)

        if limit <= 0:
            format_error("Budget must be positive. Use 'off' to disable.")
            return CommandResult(handled=True)

        self._llm_client.max_cost_usd = limit
        format_success(f"Cost budget set to [{BOLD_PRIMARY}]${limit:.2f}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_evolve(self, args: str = "") -> CommandResult:
        """Self-evolution system commands."""
        from godspeed.evolution.registry import EvolutionRegistry

        parts = args.strip().split(None, 1)
        subcmd = parts[0] if parts else "status"

        # Use global dir for evolution storage
        evo_dir = self._cwd / ".godspeed" / "evolution"

        if subcmd == "status":
            try:
                registry = EvolutionRegistry(evo_dir)
                stats = registry.stats()
                format_info(
                    f"[{BOLD_PRIMARY}]Evolution Status[/{BOLD_PRIMARY}]\n"
                    f"  Total mutations: {stats['total_mutations']}\n"
                    f"  Active: {stats['active']}\n"
                    f"  Reverted: {stats['reverted']}\n"
                    f"  Safety passed: {stats['safety_passed']}\n"
                    f"  Safety failed: {stats['safety_failed']}\n"
                    f"  Avg fitness: {stats['avg_fitness']:.3f}"
                )
            except Exception:
                format_info(
                    "No evolution data yet. "
                    f"Run [{BOLD_PRIMARY}]/evolve run[/{BOLD_PRIMARY}] to start."
                )
            return CommandResult(handled=True)

        if subcmd == "history":
            artifact_id = parts[1] if len(parts) > 1 else ""
            if not artifact_id:
                format_error("Usage: /evolve history <artifact_id>")
                return CommandResult(handled=True)

            registry = EvolutionRegistry(evo_dir)
            history = registry.get_history(artifact_id)
            if not history:
                format_info(
                    f"No evolution history for [{BOLD_PRIMARY}]{artifact_id}[/{BOLD_PRIMARY}]"
                )
            else:
                for rec in history:
                    if rec.applied_at and not rec.reverted_at:
                        status = "active"
                    elif rec.reverted_at:
                        status = "reverted"
                    else:
                        status = "candidate"
                    format_info(
                        f"  [{DIM}]{rec.record_id}[/{DIM}] fitness={rec.fitness_overall:.3f} "
                        f"status={status} model={rec.model_used}"
                    )
            return CommandResult(handled=True)

        if subcmd == "rollback":
            record_id = parts[1] if len(parts) > 1 else ""
            if not record_id:
                format_error("Usage: /evolve rollback <record_id>")
                return CommandResult(handled=True)

            registry = EvolutionRegistry(evo_dir)
            record = registry.get_record(record_id)
            if record is None:
                format_error(f"Record not found: {record_id}")
            else:
                registry.mark_reverted(record_id)
                format_success(f"Rolled back [{BOLD_PRIMARY}]{record_id}[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        if subcmd == "review":
            registry = EvolutionRegistry(evo_dir)
            records = [
                r
                for r in registry._load_records()
                if r.safety_passed and not r.applied_at and r.requires_review
            ]
            if not records:
                format_info("No pending reviews.")
            else:
                for rec in records:
                    format_info(
                        f"  [{BOLD_PRIMARY}]{rec.record_id}[/{BOLD_PRIMARY}] "
                        f"{rec.artifact_type}:{rec.artifact_id} "
                        f"fitness={rec.fitness_overall:.3f}"
                    )
                format_info(f"\nApprove with: [{DIM}]/evolve approve <id>[/{DIM}]")
            return CommandResult(handled=True)

        if subcmd == "run":
            format_info(
                f"[{BOLD_PRIMARY}]Evolution run[/{BOLD_PRIMARY}] — analyzing traces...\n"
                f"  [{DIM}]This runs asynchronously. Results will appear when complete.[/{DIM}]"
            )
            # The actual run is kicked off by the agent loop when it sees this message
            return CommandResult(
                handled=True,
                message=(
                    "Run evolution cycle: analyze traces → mutate → evaluate → apply improvements."
                ),
            )

        format_error(
            f"Unknown subcommand: {subcmd}\n  Usage: /evolve [status|run|history|rollback|review]"
        )
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
                format_info("No checkpoints saved yet.")
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
        format_success(
            f"Checkpoint saved: [{BOLD_PRIMARY}]{name}[/{BOLD_PRIMARY}]  [{DIM}]{path}[/{DIM}]"
        )
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
        format_success(
            f"Restored checkpoint: [{BOLD_PRIMARY}]{name}[/{BOLD_PRIMARY}]"
            f" ({msg_count} messages, {token_count:,} tokens)"
        )
        return CommandResult(handled=True)

    def _cmd_pause(self, _args: str = "") -> CommandResult:
        """Pause the agent loop at the next iteration."""
        if self._pause_event is None:
            format_error("Pause/resume not available in this session.")
            return CommandResult(handled=True)

        self._pause_event.clear()
        format_warning("Agent paused. Use /resume or /guidance <msg>.")
        return CommandResult(handled=True)

    def _cmd_resume(self, _args: str = "") -> CommandResult:
        """Resume a paused agent loop."""
        if self._pause_event is None:
            format_error("Pause/resume not available in this session.")
            return CommandResult(handled=True)

        if self._pause_event.is_set():
            format_info("Agent is not paused.")
            return CommandResult(handled=True)

        self._pause_event.set()
        format_success("Agent resumed.")
        return CommandResult(handled=True)

    def _cmd_guidance(self, args: str = "") -> CommandResult:
        """Inject guidance as a user message and resume the paused agent."""
        if not args.strip():
            format_error("Usage: /guidance <your guidance message>")
            return CommandResult(handled=True)

        # Inject guidance into conversation
        self._conversation.add_user_message(f"[User guidance]: {args.strip()}")
        format_info(f"Guidance injected: {args.strip()}")

        # Resume if paused
        if self._pause_event is not None and not self._pause_event.is_set():
            self._pause_event.set()
            format_success("Agent resumed with guidance.")

        return CommandResult(handled=True)

    def _cmd_tasks(self, _args: str = "") -> CommandResult:
        """Show current task list."""
        if self._task_store is None:
            format_info("Task tracking not enabled.")
            return CommandResult()

        tasks = self._task_store.list_all()
        if not tasks:
            format_info("No tasks.")
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

    def _cmd_reindex(self, _args: str = "") -> CommandResult:
        """Rebuild the codebase search index."""
        if self._codebase_index is None:
            format_info("Codebase index not available.")
            console.print(f"  [{DIM}]Install with: pip install godspeed[index][/{DIM}]")
            return CommandResult()

        if not self._codebase_index.is_available:
            format_error(f"ChromaDB not installed. [{DIM}]pip install godspeed[index][/{DIM}]")
            return CommandResult()

        if self._codebase_index.is_building:
            format_warning("Index is already building...")
            return CommandResult()

        import asyncio

        format_info("Rebuilding codebase index...")
        asyncio.get_event_loop().create_task(self._codebase_index.build_index_async())
        format_success("Reindex started in background.")
        return CommandResult()

    def _cmd_stats(self, _args: str = "") -> CommandResult:
        """Show session statistics including token usage and estimated cost."""
        from godspeed.llm.cost import estimate_cost

        input_tokens = self._llm_client.total_input_tokens
        output_tokens = self._llm_client.total_output_tokens
        cost = estimate_cost(self._llm_client.model, input_tokens, output_tokens)

        format_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._llm_client.model,
            session_id=self._session_id,
            cost=cost if cost > 0 else None,
        )
        return CommandResult(handled=True)

    def _cmd_export(self, args: str = "") -> CommandResult:
        """Export the current conversation as a markdown file."""
        from datetime import datetime

        export_dir = self._cwd / ".godspeed" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        name = args.strip() or self._session_id[:12]
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"{name}_{timestamp}.md"

        lines = ["# Godspeed Session Export\n"]
        lines.append(f"- **Session**: {self._session_id}")
        lines.append(f"- **Model**: {self._llm_client.model}")
        lines.append(f"- **Exported**: {datetime.now(tz=UTC).isoformat()}\n")
        lines.append("---\n")

        for msg in self._conversation.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                lines.append("## System Prompt\n")
                lines.append(f"```\n{content[:500]}\n```\n")
                if len(content) > 500:
                    lines.append(f"*({len(content) - 500} chars truncated)*\n")
            elif role == "user":
                lines.append(f"## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"## Assistant\n\n{content}\n")
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        lines.append(
                            f"**Tool call**: `{func.get('name', '?')}`\n"
                            f"```json\n{func.get('arguments', '{}')}\n```\n"
                        )
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "?")
                lines.append(f"## Tool Result ({tool_id})\n")
                lines.append(f"```\n{content[:1000]}\n```\n")
                if len(content) > 1000:
                    lines.append(f"*({len(content) - 1000} chars truncated)*\n")

        export_path.write_text("\n".join(lines), encoding="utf-8")
        format_success(f"Exported to: [{DIM}]{export_path}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_quit(self, _args: str = "") -> CommandResult:
        """Exit Godspeed — session summary shown by app.py."""
        return CommandResult(handled=True, should_quit=True)
