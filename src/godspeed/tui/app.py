"""Main TUI application for Godspeed — prompt-toolkit input, Rich output."""

from __future__ import annotations

import asyncio
import logging
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
    format_permission_denied,
    format_permission_prompt,
    format_tool_call,
    format_tool_result,
    format_welcome,
)

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
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._conversation = conversation
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id

        self._commands = Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=permission_engine,
            audit_trail=audit_trail,
            session_id=session_id,
            cwd=tool_context.cwd,
        )

        self._completer = GodspeedCompleter(cwd=tool_context.cwd)
        self._key_bindings = _build_key_bindings()

        # Patch the permission check to handle ASK interactively
        self._original_permissions = tool_context.permissions
        if permission_engine is not None:
            tool_context.permissions = _InteractivePermissionProxy(permission_engine)

    async def run(self) -> None:
        """Run the main TUI loop."""
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
                f"\n[red]  Cannot create interactive session: {exc}[/red]\n"
                "  [dim]Godspeed requires a real terminal. Run it directly in your"
                " terminal, not through a pipe or non-interactive shell.[/dim]"
            )
            return

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt(
                        HTML("<b><ansibrightblue>godspeed></ansibrightblue></b> "),
                    ),
                )
            except KeyboardInterrupt:
                console.print("\n  [dim]Interrupted. Type /quit to exit.[/dim]")
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

            # Run agent loop
            try:
                await agent_loop(
                    user_input=user_input,
                    conversation=self._conversation,
                    llm_client=self._llm_client,
                    tool_registry=self._tool_registry,
                    tool_context=self._tool_context,
                    on_assistant_text=_on_assistant_text,
                    on_tool_call=_on_tool_call,
                    on_tool_result=_on_tool_result,
                    on_permission_denied=_on_permission_denied,
                    on_assistant_chunk=_on_assistant_chunk,
                    max_iterations=self._commands.max_iterations,
                )
                console.print()  # End streaming output with newline
            except KeyboardInterrupt:
                console.print("\n  [dim]Agent interrupted.[/dim]")
            except Exception as exc:
                logger.error("Agent loop error: %s", exc, exc_info=True)
                format_error(f"Agent error: {exc}")

        # Final stats on exit
        if self._audit_trail is not None:
            self._audit_trail.record(
                event_type="session_end",
                detail={"reason": "user_quit"},
            )


# -- Callbacks for the agent loop -------------------------------------------------


def _on_assistant_chunk(text: str) -> None:
    """Callback: display streaming text chunk as it arrives."""
    console.print(text, end="")


def _on_assistant_text(text: str) -> None:
    """Callback: render assistant text as Markdown."""
    format_assistant_text(text)


def _on_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    """Callback: display tool call panel."""
    format_tool_call(tool_name, args)


def _on_tool_result(tool_name: str, result: Any) -> None:
    """Callback: display tool result."""
    is_error = getattr(result, "is_error", False)
    output = getattr(result, "output", str(result))
    error = getattr(result, "error", None)
    display_text = str(error) if is_error and error else str(output)
    format_tool_result(tool_name, display_text, is_error=is_error)


def _on_permission_denied(tool_name: str, reason: str) -> None:
    """Callback: display permission denied notice."""
    format_permission_denied(tool_name, reason)


class _InteractivePermissionProxy:
    """Wraps PermissionEngine to intercept ASK decisions with an interactive prompt.

    When the permission engine returns ASK, this proxy prompts the user
    via the terminal and either grants, denies, or creates a session-scoped grant.
    """

    def __init__(self, engine: PermissionEngine) -> None:
        self._engine = engine

    def evaluate(self, tool_call: Any) -> PermissionDecision:
        """Evaluate permissions, prompting the user for ASK decisions."""
        decision = self._engine.evaluate(tool_call)
        if decision != ASK:
            return decision

        # Show the permission prompt and get user input
        format_permission_prompt(tool_call.tool_name, decision.reason)
        try:
            answer = console.input("[bold yellow]  > [/bold yellow]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer in ("y", "yes"):
            return PermissionDecision(ALLOW, "user approved")
        if answer in ("a", "always"):
            pattern = tool_call.format_for_permission()
            self._engine.grant_session_permission(pattern)
            return PermissionDecision(ALLOW, f"session grant: {pattern}")

        return PermissionDecision("deny", "user denied")
