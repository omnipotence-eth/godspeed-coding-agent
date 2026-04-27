"""Professional Textual TUI for Godspeed.

Replaces the prompt-toolkit + Rich console output with a structured
widget-based interface: header bar, sidebars, chat log, and modal
screens for interactive prompts (permissions, diff review).

Agent loop runs in a Textual worker so the UI remains responsive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.agent.result import AgentCancelledError
from godspeed.audit.trail import AuditTrail
from godspeed.llm.client import LLMClient
from godspeed.security.permissions import ALLOW, ASK, PermissionDecision, PermissionEngine
from godspeed.tools.base import ToolContext
from godspeed.tools.registry import ToolRegistry
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    DIM,
    ERROR,
    PROMPT_ICON,
    SUCCESS,
    WARNING,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class HeaderBar(Static):
    """Top status bar showing model, cost, turns, permission mode."""

    model: reactive[str] = reactive("unknown")
    cost_usd: reactive[float] = reactive(0.0)
    turns: reactive[int] = reactive(0)
    permission_mode: reactive[str] = reactive("normal")
    context_pct: reactive[float] = reactive(0.0)
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static(id="header-content")

    def watch_model(self, value: str) -> None:
        self._update()

    def watch_cost_usd(self, value: float) -> None:
        self._update()

    def watch_turns(self, value: int) -> None:
        self._update()

    def watch_permission_mode(self, value: str) -> None:
        self._update()

    def watch_context_pct(self, value: float) -> None:
        self._update()

    def watch_input_tokens(self, value: int) -> None:
        self._update()

    def watch_output_tokens(self, value: int) -> None:
        self._update()

    def _update(self) -> None:
        content = self.query_one("#header-content", Static)
        parts = [
            "[b]Godspeed[/b] v3.4.0",
            f"model={self.model}",
            f"cost=${self.cost_usd:.4f}",
            f"turns={self.turns}",
            f"tok={self.input_tokens}/{self.output_tokens}",
            f"ctx={self.context_pct:.0f}%",
            f"mode=[{self.permission_mode}]",
        ]
        content.update("  •  ".join(parts))


class Sidebar(Vertical):
    """Left sidebar listing available tools."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        super().__init__()
        self._tool_registry = tool_registry

    def compose(self) -> ComposeResult:
        yield Static("Tools", classes="title")
        yield ListView(id="tool-list")

    def on_mount(self) -> None:
        lv = self.query_one("#tool-list", ListView)
        for tool in self._tool_registry.list_tools():
            lv.append(ListItem(Label(f"  {tool.name}")))


class ContextPanel(Vertical):
    """Right sidebar showing session context and stats."""

    def compose(self) -> ComposeResult:
        yield Static("Session", classes="title")
        yield Static(id="session-info")

    def update_session_info(self, text: str) -> None:
        self.query_one("#session-info", Static).update(text)


class ChatPanel(Vertical):
    """Main chat area with scrollable RichLog."""

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", highlight=True, markup=True)

    def write(self, text: str, style: str = "") -> None:
        log = self.query_one("#chat-log", RichLog)
        if style:
            log.write(Text.from_markup(f"[{style}]{text}[/{style}]"))
        else:
            log.write(text)

    def write_user(self, text: str) -> None:
        self.write(f"{PROMPT_ICON} {text}", BOLD_PRIMARY)

    def write_assistant(self, text: str) -> None:
        self.write(text)

    def write_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.write(f"  → {name}({args})", WARNING)

    def write_tool_result(self, name: str, result: str, is_error: bool = False) -> None:
        style = ERROR if is_error else SUCCESS
        self.write(f"  ← {name}: {result}", style)

    def write_error(self, text: str) -> None:
        self.write(f"  Error: {text}", ERROR)

    def write_system(self, text: str) -> None:
        self.write(text, DIM)


class InputBar(Horizontal):
    """Bottom input bar with text entry and submit button."""

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a command or /help...", id="user-input")
        yield Button("Submit", id="submit-btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()

    def get_value(self) -> str:
        return self.query_one("#user-input", Input).value

    def clear(self) -> None:
        self.query_one("#user-input", Input).value = ""


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class PermissionScreen(Screen[str]):
    """Modal screen for interactive permission decisions.

    Dismisses with "allow", "deny", or "always".
    """

    def __init__(
        self,
        tool_name: str,
        reason: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._reason = reason
        self._arguments = arguments or {}

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static("[b]Permission Request[/b]", classes="title")
            yield Static(f"Tool: [warning]{self._tool_name}[/warning]")
            yield Static(f"Reason: {self._reason}")
            if self._arguments:
                yield Static(f"Args: {self._arguments}")
            with Horizontal(classes="buttons"):
                yield Button("Yes (y)", id="perm-yes", variant="success")
                yield Button("No (n)", id="perm-no", variant="error")
                yield Button("Always (a)", id="perm-always")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "perm-yes": "allow",
            "perm-no": "deny",
            "perm-always": "always",
        }
        self.dismiss(mapping.get(event.button.id, "deny"))


class DiffReviewScreen(Screen[str]):
    """Modal screen for diff review before applying writes."""

    def __init__(
        self,
        tool_name: str,
        path: str,
        before: str,
        after: str,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._path = path
        self._before = before
        self._after = after

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(f"[b]Diff Review: {self._path}[/b]", classes="title")
            yield Static(f"Tool: {self._tool_name}")
            yield Static("[dim]--- before ---[/dim]")
            yield Static(self._before[:500])
            yield Static("[dim]+++ after +++[/dim]")
            yield Static(self._after[:500])
            with Horizontal(classes="buttons"):
                yield Button("Accept (y)", id="diff-yes", variant="success")
                yield Button("Reject (n)", id="diff-no", variant="error")
                yield Button("Always (a)", id="diff-always")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "diff-yes": "accept",
            "diff-no": "reject",
            "diff-always": "always",
        }
        self.dismiss(mapping.get(event.button.id, "reject"))


# ---------------------------------------------------------------------------
# Interactive proxies (adapt TUIApp pattern to Textual)
# ---------------------------------------------------------------------------


class _TextualPermissionProxy:
    """Wraps PermissionEngine to intercept ASK decisions via modal screen."""

    def __init__(
        self,
        engine: PermissionEngine,
        app: GodspeedTextualApp,
    ) -> None:
        self._engine = engine
        self._app = app

    def evaluate(self, tool_call: Any) -> PermissionDecision:
        decision = self._engine.evaluate(tool_call)
        if decision != ASK:
            return decision

        args = getattr(tool_call, "arguments", None) or {}
        screen = PermissionScreen(
            tool_name=tool_call.tool_name,
            reason=decision.reason,
            arguments=args,
        )
        # push_screen_wait blocks the worker thread until the screen is
        # dismissed.  The main event loop stays responsive.
        result = self._app.push_screen_wait(screen)
        answer = result if isinstance(result, str) else "deny"

        if answer == "allow":
            return PermissionDecision(ALLOW, "user approved")
        if answer == "always":
            pattern = tool_call.format_for_permission()
            self._engine.grant_session_permission(pattern)
            return PermissionDecision(ALLOW, f"session grant: {pattern}")
        return PermissionDecision("deny", "user denied")


class _TextualDiffReviewer:
    """Implements diff review via modal screen."""

    def __init__(self, app: GodspeedTextualApp) -> None:
        self._app = app
        self._always_accept = False

    async def review(
        self,
        *,
        tool_name: str,
        path: str,
        before: str,
        after: str,
    ) -> str:
        if self._always_accept:
            return "accept"

        screen = DiffReviewScreen(
            tool_name=tool_name,
            path=path,
            before=before,
            after=after,
        )
        # push_screen_wait blocks the calling thread.  We run it in a
        # thread-pool executor so the async agent loop can await it.
        result = await asyncio.to_thread(self._app.push_screen_wait, screen)
        answer = result if isinstance(result, str) else "reject"
        if answer == "always":
            self._always_accept = True
            return "accept"
        return answer if answer in ("accept", "reject") else "reject"


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class GodspeedTextualApp(App[None]):
    """Textual-based TUI for Godspeed.

    CSS: textual_app.css
    """

    CSS_PATH = "textual_app.css"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    turn_count: reactive[int] = reactive(0)
    running: reactive[bool] = reactive(False)

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
        correction_tracker: Any | None = None,
        session_memory: Any | None = None,
    ) -> None:
        super().__init__()
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._conversation = conversation
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id
        self._correction_tracker = correction_tracker
        self._session_memory = session_memory
        self._hook_executor = hook_executor
        self._task_store = task_store
        self._codebase_index = codebase_index
        self._skills = skills
        self._extra_completions = extra_completions

        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._cancel_event = asyncio.Event()

        self._tool_calls = 0
        self._tool_errors = 0
        self._tool_denied = 0
        self._start_time = time.monotonic()

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        yield Sidebar(self._tool_registry)
        yield ChatPanel(id="chat")
        yield ContextPanel(id="context")
        yield InputBar(id="input")

    def on_mount(self) -> None:
        self.title = "Godspeed"
        self.sub_title = self._llm_client.model
        chat = self.query_one("#chat", ChatPanel)
        chat.write_system(
            f"Welcome to Godspeed  •  model={self._llm_client.model}  "
            f"•  project={self._tool_context.cwd}"
        )
        self._update_header()
        self._update_context()
        self._wire_permissions()

    def _wire_permissions(self) -> None:
        """Replace tool_context permissions/diff_reviewer with interactive proxies."""
        if self._permission_engine is not None:
            self._tool_context.permissions = _TextualPermissionProxy(self._permission_engine, self)
        self._tool_context.diff_reviewer = _TextualDiffReviewer(self)

    def _update_header(self) -> None:
        header = self.query_one("#header", HeaderBar)
        header.model = self._llm_client.model
        header.cost_usd = self._llm_client.total_cost_usd
        header.turns = self.turn_count
        header.permission_mode = self._get_permission_mode()
        header.context_pct = (
            self._conversation.token_count / self._conversation.max_tokens * 100
            if self._conversation.max_tokens > 0
            else 0.0
        )
        header.input_tokens = self._llm_client.total_input_tokens
        header.output_tokens = self._llm_client.total_output_tokens

    def _update_context(self) -> None:
        panel = self.query_one("#context", ContextPanel)
        lines = [
            f"Session: {self._session_id[:8]}",
            f"Tools: {len(self._tool_registry.list_tools())}",
            f"Calls: {self._tool_calls}",
            f"Errors: {self._tool_errors}",
            f"Denied: {self._tool_denied}",
        ]
        panel.update_session_info("\n".join(lines))

    def _get_permission_mode(self) -> str:
        if self._permission_engine is None:
            return "normal"
        if getattr(self._permission_engine, "plan_mode", False):
            return "plan"
        deny_count = len(getattr(self._permission_engine, "deny_rules", []))
        has_wildcard = any(
            r.pattern in ("*", "Shell(*)", "FileWrite(*)", "FileEdit(*)")
            for r in getattr(self._permission_engine, "deny_rules", [])
        )
        if deny_count > 5 or has_wildcard:
            return "strict"
        ask_count = len(getattr(self._permission_engine, "ask_rules", []))
        if ask_count == 0 and deny_count == 0:
            return "yolo"
        return "normal"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "user-input":
            self._handle_input()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit-btn":
            self._handle_input()

    def _handle_input(self) -> None:
        input_bar = self.query_one("#input", InputBar)
        text = input_bar.get_value().strip()
        if not text:
            return
        input_bar.clear()

        if text.startswith("/"):
            self._dispatch_command(text)
            return

        self.turn_count += 1
        self._update_header()

        chat = self.query_one("#chat", ChatPanel)
        chat.write_user(text)

        if self._correction_tracker is not None:
            self._correction_tracker.check_for_correction(text)

        self.run_worker(self._agent_worker(text))

    def _dispatch_command(self, text: str) -> None:
        chat = self.query_one("#chat", ChatPanel)
        if text in ("/quit", "/exit"):
            chat.write_system("Goodbye.")
            self.exit()
        elif text == "/help":
            chat.write_system("Commands: /quit, /help, /pause, /resume, /cancel")
        elif text == "/pause":
            self._pause_event.clear()
            chat.write_system("Paused.")
        elif text == "/resume":
            self._pause_event.set()
            chat.write_system("Resumed.")
        elif text == "/cancel":
            self._cancel_event.set()
            chat.write_system("Cancelling...")
        else:
            chat.write_system(f"Unknown command: {text}")

    async def _agent_worker(self, user_input: str) -> None:
        """Run the agent loop inside a Textual worker."""
        self.running = True
        chat = self.query_one("#chat", ChatPanel)
        self._cancel_event.clear()

        try:
            await agent_loop(
                user_input=user_input,
                conversation=self._conversation,
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
                tool_context=self._tool_context,
                on_assistant_text=lambda t: self.call_from_thread(lambda: chat.write_assistant(t)),
                on_tool_call=lambda name, args: self.call_from_thread(
                    lambda n=name, a=args: chat.write_tool_call(n, a)
                ),
                on_tool_result=lambda name, res: self.call_from_thread(
                    lambda n=name, r=res: self._handle_tool_result(n, r)
                ),
                on_permission_denied=lambda name, reason: self.call_from_thread(
                    lambda n=name, r=reason: chat.write_system(f"  Permission denied: {n} ({r})")
                ),
                on_assistant_chunk=lambda chunk: self.call_from_thread(
                    lambda c=chunk: chat.write_assistant(c)
                ),
                max_iterations=25,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
                hook_executor=self._hook_executor,
                on_parallel_start=lambda calls: self.call_from_thread(
                    lambda c=calls: chat.write_system(f"  Parallel: {len(c)} calls")
                ),
                on_parallel_complete=lambda results: None,
                on_thinking=lambda t: self.call_from_thread(
                    lambda txt=t: chat.write_system(f"  Thinking: {txt}")
                ),
            )
        except AgentCancelledError:
            chat.write_system("Agent cancelled.")
        except Exception as exc:
            logger.error("Agent loop error: %s", exc, exc_info=True)
            chat.write_error(str(exc))
        finally:
            self.running = False
            self._update_header()
            self._update_context()

    def _handle_tool_result(self, name: str, result: Any) -> None:
        self._tool_calls += 1
        is_error = getattr(result, "is_error", False)
        if is_error:
            self._tool_errors += 1
        output = getattr(result, "output", str(result))
        error = getattr(result, "error", None)
        display = str(error) if is_error and error else str(output)
        chat = self.query_one("#chat", ChatPanel)
        chat.write_tool_result(name, display, is_error=is_error)

    def action_quit(self) -> None:
        duration = time.monotonic() - self._start_time
        chat = self.query_one("#chat", ChatPanel)
        chat.write_system(
            f"Session ended  •  {duration:.0f}s  "
            f"•  calls={self._tool_calls}  errors={self._tool_errors}"
        )
        if self._session_memory is not None:
            self._session_memory.end_session(
                self._session_id,
                summary=f"turns={self.turn_count} tools={self._tool_calls}",
            )
        if self._audit_trail is not None:
            self._audit_trail.record(
                event_type="session_end",
                detail={"reason": "user_quit"},
            )
        super().action_quit()
