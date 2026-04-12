"""Main TUI application for Godspeed — prompt-toolkit input, Rich output."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.audit.trail import AuditTrail
from godspeed.llm.client import LLMClient
from godspeed.security.permissions import ALLOW, ASK, PermissionDecision, PermissionEngine
from godspeed.tools.base import ToolContext
from godspeed.tools.registry import ToolRegistry
from godspeed.tui.commands import Commands
from godspeed.tui.completions import GodspeedCompleter
from godspeed.tui.output import (
    console,
    format_assistant_text,
    format_error,
    format_parallel_results,
    format_parallel_tool_calls,
    format_permission_denied,
    format_permission_prompt,
    format_session_summary,
    format_thinking,
    format_tool_call,
    format_tool_result,
    format_welcome,
)
from godspeed.tui.theme import BOLD_WARNING, DIM, ERROR, MUTED, icon_prompt

logger = logging.getLogger(__name__)


def _build_key_bindings() -> KeyBindings:
    """Build prompt-toolkit key bindings.

    - Enter: submit input
    - Escape+Enter: insert newline for multiline input
    - Ctrl+C: abort current input
    """
    bindings = KeyBindings()

    @bindings.add(Keys.Enter)
    def _submit(event: Any) -> None:
        """Enter submits the input."""
        event.current_buffer.validate_and_handle()

    @bindings.add(Keys.Escape, Keys.Enter)
    def _newline(event: Any) -> None:
        """Escape+Enter inserts a newline for multiline input."""
        event.current_buffer.insert_text("\n")

    return bindings


class TUIApp:
    """Main TUI application orchestrating input, agent loop, and output.

    Wires together: prompt-toolkit input -> slash commands / agent loop -> Rich output.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        conversation: Conversation,
        permission_engine: PermissionEngine | None,
        audit_trail: AuditTrail | None,
        session_id: str,
        skills: list[Any] | None = None,
        extra_completions: list[tuple[str, str]] | None = None,
        hook_executor: Any | None = None,
        task_store: Any | None = None,
        codebase_index: Any | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._conversation = conversation
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id

        # Pause/resume event for human-in-the-loop
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start in running state

        self._commands = Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=permission_engine,
            audit_trail=audit_trail,
            session_id=session_id,
            cwd=tool_context.cwd,
            pause_event=self._pause_event,
        )

        # Wire task store and codebase index for commands
        if task_store is not None:
            self._commands._task_store = task_store
        if codebase_index is not None:
            self._commands._codebase_index = codebase_index

        # Register skill commands
        if skills:
            from godspeed.skills.commands import register_skill_commands

            register_skill_commands(self._commands, conversation, skills)

        self._completer = GodspeedCompleter(
            cwd=tool_context.cwd,
            extra_commands=extra_completions,
        )
        self._key_bindings = _build_key_bindings()
        self._hook_executor = hook_executor

        # Patch the permission check to handle ASK interactively
        self._original_permissions = tool_context.permissions
        if permission_engine is not None:
            from godspeed.security.approval_tracker import ApprovalTracker

            self._approval_tracker = ApprovalTracker()
            tool_context.permissions = _InteractivePermissionProxy(
                permission_engine,
                approval_tracker=self._approval_tracker,
            )

    async def run(self) -> None:
        """Run the main TUI loop."""
        start_time = time.monotonic()
        tool_calls = 0
        tool_errors = 0
        tool_denied = 0

        # Collect tool names and deny rules for safety disclosure
        tool_names = [t.name for t in self._tool_registry.list_tools()]
        deny_rules = (
            [r.pattern for r in self._permission_engine.deny_rules]
            if self._permission_engine is not None
            else []
        )

        format_welcome(
            model=self._llm_client.model,
            project_dir=str(self._tool_context.cwd),
            tools=tool_names,
            deny_rules=deny_rules,
            audit_enabled=self._audit_trail is not None,
        )

        try:
            session: PromptSession[str] = PromptSession(
                completer=self._completer,
                key_bindings=self._key_bindings,
                multiline=True,
            )
        except Exception as exc:
            # prompt-toolkit fails in non-TTY contexts (piped input, CI, etc.)
            console.print(
                f"\n[{ERROR}]  Cannot create interactive session: {exc}[/{ERROR}]\n"
                f"  [{DIM}]Godspeed requires a real terminal. Run it directly in your"
                f" terminal, not through a pipe or non-interactive shell.[/{DIM}]"
            )
            return

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt(
                        HTML(icon_prompt()),
                    ),
                )
            except KeyboardInterrupt:
                console.print(f"\n  [{DIM}]Interrupted. Type /quit to exit.[/{DIM}]")
                continue
            except EOFError:
                break

            if not user_input.strip():
                continue

            # Check for slash commands
            cmd_result = self._commands.dispatch(user_input)
            if cmd_result is not None:
                if cmd_result.should_quit:
                    break
                continue

            # Parse @-mentions from input and resolve to content blocks
            effective_input = user_input
            from godspeed.tui.mentions import parse_mentions, resolve_mentions

            cleaned_text, mentions = parse_mentions(user_input)
            if mentions:
                try:
                    mention_blocks = await resolve_mentions(mentions, self._tool_context.cwd)
                    if mention_blocks:
                        # Build multimodal message: cleaned text + resolved content

                        content_blocks = [{"type": "text", "text": cleaned_text}]
                        content_blocks.extend(mention_blocks)
                        self._conversation.add_user_message(content_blocks)
                        effective_input = ""  # Already added to conversation
                except Exception as exc:
                    logger.warning("Mention resolution failed: %s", exc)
                    # Fall through with original input

            # Run agent loop with context-aware thinking indicator
            spinner = _ThinkingSpinner()

            def _track_tool_call(
                tool_name: str, args: dict[str, Any], _s: _ThinkingSpinner = spinner
            ) -> None:
                nonlocal tool_calls
                tool_calls += 1
                _s.update(tool_name, args)
                _s.stop()
                format_tool_call(tool_name, args)

            def _track_tool_result(
                tool_name: str, result: Any, _s: _ThinkingSpinner = spinner
            ) -> None:
                nonlocal tool_errors
                is_error = getattr(result, "is_error", False)
                if is_error:
                    tool_errors += 1
                _s.start()
                output = getattr(result, "output", str(result))
                error = getattr(result, "error", None)
                display_text = str(error) if is_error and error else str(output)
                _s.stop()
                format_tool_result(tool_name, display_text, is_error=is_error)

            def _track_permission_denied(
                tool_name: str, reason: str, _s: _ThinkingSpinner = spinner
            ) -> None:
                nonlocal tool_denied
                tool_denied += 1
                _s.stop()
                format_permission_denied(tool_name, reason)

            def _track_parallel_start(
                calls: list[tuple[str, dict[str, Any]]],
                _s: _ThinkingSpinner = spinner,
            ) -> None:
                _s.stop()
                format_parallel_tool_calls(calls)

            def _track_parallel_complete(
                results: list[tuple[str, str, bool]],
                _s: _ThinkingSpinner = spinner,
            ) -> None:
                format_parallel_results(results)
                _s.start()

            def _on_thinking(
                text: str,
                _s: _ThinkingSpinner = spinner,
            ) -> None:
                _s.stop()
                format_thinking(text)
                _s.start()

            try:
                spinner.start()
                await agent_loop(
                    user_input=effective_input if effective_input else user_input,
                    conversation=self._conversation,
                    llm_client=self._llm_client,
                    tool_registry=self._tool_registry,
                    tool_context=self._tool_context,
                    on_assistant_text=spinner.wrap(_on_assistant_text),
                    on_tool_call=_track_tool_call,
                    on_tool_result=_track_tool_result,
                    on_permission_denied=_track_permission_denied,
                    on_assistant_chunk=spinner.wrap(_on_assistant_chunk),
                    max_iterations=self._commands.max_iterations,
                    pause_event=self._pause_event,
                    hook_executor=self._hook_executor,
                    skip_user_message=not effective_input,
                    on_parallel_start=_track_parallel_start,
                    on_parallel_complete=_track_parallel_complete,
                    on_thinking=_on_thinking,
                )
                console.print()  # End streaming output with newline
            except KeyboardInterrupt:
                console.print(f"\n  [{DIM}]Agent interrupted.[/{DIM}]")
            except Exception as exc:
                logger.error("Agent loop error: %s", exc, exc_info=True)
                format_error(f"Agent error: {exc}")
            finally:
                spinner.stop()

        # Session summary on exit
        duration = time.monotonic() - start_time
        format_session_summary(
            duration_secs=duration,
            input_tokens=self._llm_client.total_input_tokens,
            output_tokens=self._llm_client.total_output_tokens,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            tool_denied=tool_denied,
        )

        if self._audit_trail is not None:
            self._audit_trail.record(
                event_type="session_end",
                detail={"reason": "user_quit"},
            )


# -- Thinking spinner -------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "file_read": "Reading",
    "file_write": "Writing",
    "file_edit": "Editing",
    "shell": "Running",
    "grep_search": "Searching",
    "glob_search": "Searching",
    "git": "Git",
    "repo_map": "Mapping",
}


class _ThinkingSpinner:
    """Context-aware Rich Status spinner.

    Shows what the agent is doing — "Thinking..." when waiting for LLM,
    tool-specific labels during tool execution.
    """

    def __init__(self) -> None:
        self._status: Any | None = None
        self._started = False

    def _make_label(self, text: str) -> str:
        from godspeed.tui.theme import PROMPT_ICON

        return f"[{MUTED}]{PROMPT_ICON} {text}[/{MUTED}]"

    def start(self) -> None:
        if self._started:
            return
        from rich.status import Status

        self._status = Status(
            self._make_label("Thinking..."),
            console=console,
            spinner="dots",
            spinner_style=MUTED,
        )
        self._status.start()
        self._started = True

    def update(self, tool_name: str, args: dict[str, Any]) -> None:
        """Update spinner text based on current tool call."""
        if not self._started or self._status is None:
            return
        label = _TOOL_LABELS.get(tool_name, tool_name)
        primary_arg = args.get("file_path") or args.get("command") or args.get("pattern") or ""
        if primary_arg:
            # Truncate long args
            if len(primary_arg) > 50:
                primary_arg = "..." + primary_arg[-47:]
            self._status.update(self._make_label(f"{label} {primary_arg}"))
        else:
            self._status.update(self._make_label(f"{label}..."))

    def stop(self) -> None:
        if self._started and self._status is not None:
            self._status.stop()
            self._started = False

    def wrap(self, fn: Any) -> Any:
        """Return a wrapper that stops the spinner before calling *fn*."""
        spinner = self

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            spinner.stop()
            return fn(*args, **kwargs)

        return _wrapped


# -- Callbacks for the agent loop -------------------------------------------------


def _on_assistant_chunk(text: str) -> None:
    """Callback: display streaming text chunk as it arrives."""
    console.print(text, end="")


def _on_assistant_text(text: str) -> None:
    """Callback: render assistant text as Markdown."""
    format_assistant_text(text)


class _InteractivePermissionProxy:
    """Wraps PermissionEngine to intercept ASK decisions with an interactive prompt.

    When the permission engine returns ASK, this proxy prompts the user
    via the terminal and either grants, denies, or creates a session-scoped grant.

    Optionally tracks repeated approvals via ApprovalTracker and suggests
    adding patterns as permanent allow rules after a configurable threshold.
    """

    def __init__(
        self,
        engine: PermissionEngine,
        approval_tracker: Any | None = None,
    ) -> None:
        self._engine = engine
        self._tracker = approval_tracker

    def evaluate(self, tool_call: Any) -> PermissionDecision:
        """Evaluate permissions, prompting the user for ASK decisions."""
        decision = self._engine.evaluate(tool_call)
        if decision != ASK:
            return decision

        # Show the permission prompt with contextual detail
        args = getattr(tool_call, "arguments", None) or {}
        format_permission_prompt(tool_call.tool_name, decision.reason, arguments=args)
        try:
            answer = console.input(f"[{BOLD_WARNING}]  > [/{BOLD_WARNING}]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer in ("y", "yes"):
            # Track approval for auto-permission suggestion
            pattern = tool_call.format_for_permission()
            if self._tracker is not None:
                self._tracker.record_approval(pattern)
                if self._tracker.should_suggest(pattern):
                    self._suggest_auto_permission(pattern)
            return PermissionDecision(ALLOW, "user approved")
        if answer in ("a", "always"):
            pattern = tool_call.format_for_permission()
            self._engine.grant_session_permission(pattern)
            return PermissionDecision(ALLOW, f"session grant: {pattern}")

        return PermissionDecision("deny", "user denied")

    def _suggest_auto_permission(self, pattern: str) -> None:
        """Suggest adding a pattern as a permanent allow rule."""
        # Skip if already in allow rules
        for rule in self._engine.allow_rules:
            if rule == pattern:
                return

        from godspeed.tui.theme import ACCENT, SUCCESS

        console.print(
            f"\n  [{ACCENT}]You've approved [{SUCCESS}]{pattern}"
            f"[/{SUCCESS}] multiple times.[/{ACCENT}]"
        )
        console.print(f"  [{ACCENT}]Add to permanent allow rules? (y/n)[/{ACCENT}]")
        try:
            answer = console.input(f"[{BOLD_WARNING}]  > [/{BOLD_WARNING}]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer in ("y", "yes"):
            from godspeed.config import append_allow_rule

            success = append_allow_rule(pattern)
            if success:
                # Also update engine in-memory
                self._engine.allow_rules.append(pattern)
                console.print(f"  [{SUCCESS}]Added to allow rules.[/{SUCCESS}]")
            else:
                from godspeed.tui.theme import WARNING

                console.print(
                    f"  [{WARNING}]Could not persist rule. Added for this session only.[/{WARNING}]"
                )
                self._engine.grant_session_permission(pattern)
