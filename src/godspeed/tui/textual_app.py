"""Professional Textual TUI for Godspeed — OpenCode-inspired.

Layout:
- Left: Chat panel (messages, tool calls, streaming)
- Right: Info panel (session, usage, tools, mode, footer)
- Bottom: Input bar

Design principles:
- No emojis. Clean text-only indicators.
- Every element on screen has a function.
- Right panel shows live operational data.
- Minimal chrome, maximum information density.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from godspeed import __version__
from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.agent.result import AgentCancelledError
from godspeed.audit.trail import AuditTrail
from godspeed.llm.client import LLMClient
from godspeed.security.permissions import ALLOW, ASK, DENY, PermissionDecision, PermissionEngine
from godspeed.tools.base import ToolContext
from godspeed.tools.registry import ToolRegistry
from godspeed.tui.commands import Commands
from godspeed.tui.output import capture_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool labels (text only, no emojis)
# ---------------------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "file_read": "Read",
    "file_write": "Write",
    "file_edit": "Edit",
    "shell": "Shell",
    "grep_search": "Grep",
    "glob_search": "Glob",
    "git": "Git",
    "repo_map": "Map",
    "complexity": "Complexity",
    "system_optimizer": "System",
    "default": "Tool",
}

_COMMANDS: list[tuple[str, str, str]] = [
    ("/quit", "Exit Godspeed", "Session"),
    ("/help", "Show available commands", "Session"),
    ("/pause", "Pause agent loop", "Control"),
    ("/resume", "Resume agent loop", "Control"),
    ("/cancel", "Cancel current operation", "Control"),
    ("/clear", "Clear chat history", "Session"),
    ("/model", "Switch AI model", "Config"),
    ("/mode", "Change permission mode", "Config"),
    ("/cost", "Show session cost breakdown", "Info"),
    ("/tokens", "Show token usage", "Info"),
    ("/tools", "List available tools", "Info"),
    ("/permissions", "Show permission rules", "Info"),
    ("/audit", "Show audit trail status", "Info"),
    ("/correct", "Record a correction for memory", "Memory"),
    ("/preferences", "Show learned preferences", "Memory"),
    ("/evolve", "Run evolution cycle on tool descriptions", "Advanced"),
]


# ---------------------------------------------------------------------------
# Info Panel (right side)
# ---------------------------------------------------------------------------


class InfoPanel(Static):
    """Right-side panel showing live operational data."""

    # Reactive fields bound from the main app
    session_id: reactive[str] = reactive("")
    model: reactive[str] = reactive("")
    project_dir: reactive[str] = reactive("")
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)
    context_pct: reactive[float] = reactive(0.0)
    cost_usd: reactive[float] = reactive(0.0)
    tool_calls: reactive[int] = reactive(0)
    tool_errors: reactive[int] = reactive(0)
    tool_denied: reactive[int] = reactive(0)
    permission_mode: reactive[str] = reactive("normal")
    is_running: reactive[bool] = reactive(False)
    duration_sec: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static("Session", classes="section-title")
        yield Static("", id="info-session")

        yield Static("Usage", classes="section-title")
        yield Static("", id="info-usage")

        yield Static("Tools", classes="section-title")
        yield Static("", id="info-tools")

        yield Static("Mode", classes="section-title")
        yield Static("", id="info-mode")

        yield Static(f"Godspeed v{__version__}", classes="info-footer")

    def on_mount(self) -> None:
        self._update_all()

    def _update_all(self) -> None:
        self._update_session()
        self._update_usage()
        self._update_tools()
        self._update_mode()

    def _update_session(self) -> None:
        widget = self.query_one("#info-session", Static)
        sid = self.session_id[:10] if len(self.session_id) > 10 else self.session_id
        lines = [
            f"[info-label]ID:[/]     [info-value]{sid}[/]",
            f"[info-label]Model:[/]   [info-value]{self.model[:22]}[/]",
            f"[info-label]Dir:[/]     [info-value]{self.project_dir[:22]}[/]",
        ]
        widget.update("\n".join(lines))

    def _update_usage(self) -> None:
        widget = self.query_one("#info-usage", Static)
        total = self.input_tokens + self.output_tokens
        if self.context_pct < 50:
            ctx_color = "info-value-success"
        elif self.context_pct < 80:
            ctx_color = "info-value-warn"
        else:
            ctx_color = "info-value-error"
        lines = [
            f"[info-label]In:[/]      [info-value]{self.input_tokens:,}[/]",
            f"[info-label]Out:[/]     [info-value]{self.output_tokens:,}[/]",
            f"[info-label]Total:[/]   [info-value]{total:,}[/]",
            f"[info-label]Context:[/]  [{ctx_color}]{self.context_pct:.0f}%[/]",
            f"[info-label]Cost:[/]     [info-value]${self.cost_usd:.4f}[/]",
        ]
        widget.update("\n".join(lines))

    def _update_tools(self) -> None:
        widget = self.query_one("#info-tools", Static)
        err_color = "info-value-error" if self.tool_errors > 0 else "info-value"
        deny_color = "info-value-warn" if self.tool_denied > 0 else "info-value"
        lines = [
            f"[info-label]Calls:[/]   [info-value]{self.tool_calls}[/]",
            f"[info-label]Errors:[/]  [{err_color}]{self.tool_errors}[/]",
            f"[info-label]Denied:[/]  [{deny_color}]{self.tool_denied}[/]",
        ]
        widget.update("\n".join(lines))

    def _update_mode(self) -> None:
        widget = self.query_one("#info-mode", Static)
        mode_color = {
            "normal": "info-value-success",
            "strict": "info-value-error",
            "plan": "info-value-accent",
            "yolo": "info-value-warn",
        }.get(self.permission_mode, "info-value")
        status = "running" if self.is_running else "idle"
        lines = [
            f"[info-label]Mode:[/]    [{mode_color}]{self.permission_mode}[/]",
            f"[info-label]Status:[/]  [info-value]{status}[/]",
            f"[info-label]Time:[/]    [info-value]{self.duration_sec}s[/]",
        ]
        widget.update("\n".join(lines))

    def watch_session_id(self, _value: str) -> None:
        self._update_session()

    def watch_model(self, _value: str) -> None:
        self._update_session()

    def watch_project_dir(self, _value: str) -> None:
        self._update_session()

    def watch_input_tokens(self, _value: int) -> None:
        self._update_usage()

    def watch_output_tokens(self, _value: int) -> None:
        self._update_usage()

    def watch_context_pct(self, _value: float) -> None:
        self._update_usage()

    def watch_cost_usd(self, _value: float) -> None:
        self._update_usage()

    def watch_tool_calls(self, _value: int) -> None:
        self._update_tools()

    def watch_tool_errors(self, _value: int) -> None:
        self._update_tools()

    def watch_tool_denied(self, _value: int) -> None:
        self._update_tools()

    def watch_permission_mode(self, _value: str) -> None:
        self._update_mode()

    def watch_is_running(self, _value: bool) -> None:
        self._update_mode()

    def watch_duration_sec(self, _value: int) -> None:
        self._update_mode()


# ---------------------------------------------------------------------------
# Chat Components
# ---------------------------------------------------------------------------


class UserMessage(Static):
    """A user message."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("You", classes="msg-label")
        yield Static(self._text, classes="msg-body")


class AssistantMessage(Static):
    """An assistant message."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("Assistant", classes="msg-label")
        yield Static(self._text, classes="msg-body")


class SystemMessage(Static):
    """A system/status message."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(self._text, classes="msg-body")


class ErrorMessage(Static):
    """An error message block."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("Error", classes="msg-label")
        yield Static(self._text, classes="msg-body")


class ToolCallBlock(Static):
    """Collapsible tool call display with timing."""

    expanded: reactive[bool] = reactive(False)

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any],
        start_time: float,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args
        self._start_time = start_time
        self._end_time: float | None = None
        self._result: str | None = None
        self._is_error = False

    def compose(self) -> ComposeResult:
        label = _TOOL_LABELS.get(self._tool_name, self._tool_name)
        args_str = self._format_args()
        header_text = f"{label} {args_str}"
        yield Static(header_text, classes="tool-header")
        yield Static("", classes="tool-body")
        yield Static("", classes="tool-result")

    def _format_args(self) -> str:
        primary = (
            self._args.get("file_path")
            or self._args.get("command")
            or self._args.get("pattern")
            or ""
        )
        if primary:
            if len(primary) > 35:
                primary = "..." + primary[-32:]
            return f"  {primary}"
        return ""

    def on_click(self) -> None:
        self.expanded = not self.expanded

    def watch_expanded(self, value: bool) -> None:
        if not self.is_mounted:
            return
        body = self.query_one(".tool-body", Static)
        body.set_class(value, "tool-body-expanded")
        if value and self._result is not None:
            body.update(self._result)
        elif not value:
            body.update("")

    def set_result(self, result: str, is_error: bool = False) -> None:
        self._end_time = time.monotonic()
        self._result = result
        self._is_error = is_error
        elapsed = (self._end_time - self._start_time) * 1000
        if not self.is_mounted:
            return
        result_widget = self.query_one(".tool-result", Static)
        result_class = "tool-result-error" if is_error else "tool-result"
        result_widget.set_class(True, result_class)
        status = "ERR" if is_error else "OK"
        result_widget.update(f"{status}  {elapsed:.0f}ms  {result[:120]}")
        if self.expanded:
            body = self.query_one(".tool-body", Static)
            body.update(result)


class StreamingIndicator(Static):
    """Animated thinking indicator (text only)."""

    def __init__(self, text: str = "Thinking") -> None:
        super().__init__()
        self._base_text = text
        self._dot_count = 0

    def on_mount(self) -> None:
        self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self.update(f"{self._base_text}{dots}")


# ---------------------------------------------------------------------------
# Chat Panel
# ---------------------------------------------------------------------------


class ChatPanel(Static):
    """Main chat area with scrollable message widgets."""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="messages-scroll")

    def _container(self) -> VerticalScroll:
        return self.query_one("#messages-scroll", VerticalScroll)

    def write_user(self, text: str) -> None:
        self._container().mount(UserMessage(text))
        self._scroll_to_bottom()

    def write_assistant(self, text: str) -> None:
        self._container().mount(AssistantMessage(text))
        self._scroll_to_bottom()

    def write_system(self, text: str) -> None:
        self._container().mount(SystemMessage(text))
        self._scroll_to_bottom()

    def write_error(self, text: str) -> None:
        self._container().mount(ErrorMessage(text))
        self._scroll_to_bottom()

    def add_tool_call(self, name: str, args: dict[str, Any]) -> ToolCallBlock:
        block = ToolCallBlock(name, args, time.monotonic())
        self._container().mount(block)
        self._scroll_to_bottom()
        return block

    def add_streaming_indicator(self, text: str = "Thinking") -> StreamingIndicator:
        indicator = StreamingIndicator(text)
        self._container().mount(indicator)
        self._scroll_to_bottom()
        return indicator

    def _scroll_to_bottom(self) -> None:
        scroll = self._container()
        scroll.scroll_end(animate=False)


# ---------------------------------------------------------------------------
# Input Bar
# ---------------------------------------------------------------------------


class InputBar(Horizontal):
    """Bottom input bar."""

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Ask Godspeed anything...", id="user-input")
        yield Button("Send", id="submit-btn", variant="primary")
        yield Static("Ctrl+K  Ctrl+C", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()

    def get_value(self) -> str:
        return self.query_one("#user-input", Input).value

    def clear(self) -> None:
        self.query_one("#user-input", Input).value = ""

    def focus_input(self) -> None:
        self.query_one("#user-input", Input).focus()


# ---------------------------------------------------------------------------
# Modal Screens
# ---------------------------------------------------------------------------


class PermissionScreen(Screen[str]):
    """Modal screen for interactive permission decisions."""

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
            yield Static("Permission Required", classes="dialog-title")
            yield Static(
                f"The assistant wants to run [b]{self._tool_name}[/b]",
                classes="dialog-body",
            )
            yield Static(f"Reason: {self._reason}", classes="dialog-body")
            if self._arguments:
                yield Static(
                    f"Arguments: {self._arguments}",
                    classes="dialog-detail",
                )
            with Horizontal(classes="buttons"):
                yield Button("Yes (y)", id="btn-yes", variant="success")
                yield Button("No (n)", id="btn-no", variant="error")
                yield Button("Always (a)", id="btn-always")
            yield Static(
                "y = allow once  ·  n = deny  ·  a = allow for session",
                classes="shortcut-hint",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping: dict[str | None, str] = {
            "btn-yes": "allow",
            "btn-no": "deny",
            "btn-always": "always",
        }
        self.dismiss(mapping.get(event.button.id, "deny"))

    def on_key(self, event: Any) -> None:
        key_map = {
            "y": "allow",
            "n": "deny",
            "a": "always",
            "d": "deny",
        }
        if event.key in key_map:
            self.dismiss(key_map[event.key])


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
            yield Static(
                f"Review Changes: {self._path}",
                classes="dialog-title",
            )
            yield Static(f"Tool: {self._tool_name}", classes="dialog-body")
            yield Static("[dim]--- before ---[/dim]", classes="dialog-detail")
            yield Static(self._before[:400])
            yield Static("[dim]+++ after +++[/dim]", classes="dialog-detail")
            yield Static(self._after[:400])
            with Horizontal(classes="buttons"):
                yield Button("Accept (y)", id="btn-accept", variant="success")
                yield Button("Reject (n)", id="btn-reject", variant="error")
                yield Button("Always (a)", id="btn-always")
            yield Static(
                "y = accept  ·  n = reject  ·  a = accept all",
                classes="shortcut-hint",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping: dict[str | None, str] = {
            "btn-accept": "accept",
            "btn-reject": "reject",
            "btn-always": "always",
        }
        self.dismiss(mapping.get(event.button.id, "reject"))

    def on_key(self, event: Any) -> None:
        key_map = {
            "y": "accept",
            "n": "reject",
            "a": "always",
            "d": "reject",
        }
        if event.key in key_map:
            self.dismiss(key_map[event.key])


class CommandPaletteScreen(Screen[str | None]):
    """Command palette with fuzzy search (Ctrl+K)."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Input(placeholder="Type a command...", id="palette-input")
            yield ListView(id="palette-list")

    def on_mount(self) -> None:
        lv = self.query_one("#palette-list", ListView)
        for cmd, desc, _category in _COMMANDS:
            lv.append(
                ListItem(
                    Label(f"[bold cyan]{cmd}[/]  [dim]{desc}[/]"),
                    id=f"cmd-{cmd.lstrip('/').replace('/', '-')}",
                )
            )
        self.query_one("#palette-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        lv = self.query_one("#palette-list", ListView)
        lv.clear()
        for cmd, desc, _category in _COMMANDS:
            if query in cmd.lower() or query in desc.lower():
                lv.append(
                    ListItem(
                        Label(f"[bold cyan]{cmd}[/]  [dim]{desc}[/]"),
                    )
                )
        if lv.children:
            lv.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        text = str(event.item.render())
        cmd = text.split()[0] if text else None
        self.dismiss(cmd)

    def on_key(self, event: Any) -> None:
        if event.key == "escape":
            self.dismiss(None)


class WelcomeScreen(Screen[None]):
    """Welcome overlay shown on first launch."""

    def __init__(
        self,
        model: str,
        project_dir: str,
        tool_count: int,
    ) -> None:
        super().__init__()
        self._model = model
        self._project_dir = project_dir
        self._tool_count = tool_count

    def compose(self) -> ComposeResult:
        with Vertical(classes="welcome-container"):
            yield Static("Godspeed", classes="welcome-title")
            yield Static(
                "Security-first AI coding agent",
                classes="welcome-subtitle",
            )
            yield Static(
                f"[dim]Model:[/]     [b]{self._model}[/]",
                classes="welcome-info",
            )
            yield Static(
                f"[dim]Project:[/]   [b]{self._project_dir}[/]",
                classes="welcome-info",
            )
            yield Static(
                f"[dim]Tools:[/]     [b]{self._tool_count}[/]",
                classes="welcome-info",
            )
            yield Static(
                "Press Enter to start  ·  Ctrl+K for commands",
                classes="welcome-hint",
            )

    def on_key(self, _event: Any) -> None:
        self.dismiss()

    def on_click(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------------------
# Interactive Proxies
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

    async def evaluate(self, tool_call: Any) -> PermissionDecision:
        decision = self._engine.evaluate(tool_call)
        if decision != ASK:
            return decision

        args = getattr(tool_call, "arguments", None) or {}
        screen = PermissionScreen(
            tool_name=tool_call.tool_name,
            reason=decision.reason,
            arguments=args,
        )
        result = await asyncio.to_thread(self._app.push_screen_wait, screen)
        answer = result if isinstance(result, str) else "deny"

        if answer == "allow":
            return PermissionDecision(ALLOW, "user approved")
        if answer == "always":
            pattern = tool_call.format_for_permission()
            self._engine.grant_session_permission(pattern)
            return PermissionDecision(ALLOW, f"session grant: {pattern}")
        return PermissionDecision(DENY, "user denied")


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
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+k", "command_palette", "Commands"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+n", "new_session", "New Session"),
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

        self._tool_calls_total = 0
        self._tool_errors_total = 0
        self._tool_denied_total = 0
        self._start_time = time.monotonic()

        self._current_streaming_indicator: StreamingIndicator | None = None
        self._current_tool_block: ToolCallBlock | None = None
        self._current_assistant_msg: AssistantMessage | None = None

        # Slash command registry
        self._commands = Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=permission_engine,
            audit_trail=audit_trail,
            session_id=session_id,
            cwd=tool_context.cwd if tool_context else Path.cwd(),
            pause_event=self._pause_event,
            tool_registry=tool_registry,
        )
        Commands._task_store = task_store
        Commands._codebase_index = codebase_index

        if skills:
            from godspeed.skills.commands import register_skill_commands

            register_skill_commands(self._commands, conversation, skills)

    def compose(self) -> ComposeResult:
        with Vertical(id="main-grid"):
            yield ChatPanel(id="chat-panel")
            yield InfoPanel(id="info-panel")
            yield InputBar(id="input-bar")

    def on_mount(self) -> None:
        self.title = "Godspeed"
        self.sub_title = self._llm_client.model
        self._update_info_panel()
        self._wire_permissions()
        # Show welcome screen on first mount
        self.push_screen(
            WelcomeScreen(
                model=self._llm_client.model,
                project_dir=str(self._tool_context.cwd),
                tool_count=len(self._tool_registry.list_tools()),
            )
        )
        # Start duration timer
        self.set_interval(1.0, self._tick_duration)

    def _tick_duration(self) -> None:
        try:
            panel = self.query_one("#info-panel", InfoPanel)
        except Exception:
            return
        panel.duration_sec = int(time.monotonic() - self._start_time)

    def _wire_permissions(self) -> None:
        if self._permission_engine is not None:
            self._tool_context.permissions = _TextualPermissionProxy(self._permission_engine, self)
        self._tool_context.diff_reviewer = _TextualDiffReviewer(self)

    def _update_info_panel(self) -> None:
        try:
            panel = self.query_one("#info-panel", InfoPanel)
        except Exception:
            return
        panel.session_id = self._session_id
        panel.model = self._llm_client.model
        panel.project_dir = str(self._tool_context.cwd)
        panel.input_tokens = self._llm_client.total_input_tokens
        panel.output_tokens = self._llm_client.total_output_tokens
        panel.context_pct = (
            self._conversation.token_count / self._conversation.max_tokens * 100
            if self._conversation.max_tokens > 0
            else 0.0
        )
        panel.cost_usd = self._llm_client.total_cost_usd
        panel.tool_calls = self._tool_calls_total
        panel.tool_errors = self._tool_errors_total
        panel.tool_denied = self._tool_denied_total
        panel.permission_mode = self._get_permission_mode()
        panel.is_running = self.running

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
        if self.running:
            return
        input_bar = self.query_one("#input-bar", InputBar)
        text = input_bar.get_value().strip()
        if not text:
            return
        input_bar.clear()

        if text.startswith("/"):
            self._dispatch_command(text)
            return

        self.turn_count += 1
        self._update_info_panel()

        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_user(text)

        if self._correction_tracker is not None:
            self._correction_tracker.check_for_correction(text)

        self.run_worker(self._agent_worker(text))

    def _dispatch_command(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)

        # Textual-specific commands not in the shared Commands library
        if text == "/cancel":
            self._cancel_event.set()
            chat.write_system("Cancelling current operation...")
            return

        # Dispatch via Commands library, capturing Rich console output
        with capture_output() as sio:
            result = self._commands.dispatch(text)

        output = sio.getvalue()
        if output:
            chat.write_system(output)

        if result is None:
            chat.write_system(f"Unknown command: {text}. Type /help for available commands.")
            return

        if result.should_quit:
            self.exit()

        # Textual-specific UI side effects
        if text == "/clear":
            scroll = self.query_one("#messages-scroll", VerticalScroll)
            scroll.remove_children()

    async def _agent_worker(self, user_input: str) -> None:
        self.running = True
        self._update_info_panel()
        chat = self.query_one("#chat-panel", ChatPanel)
        self._cancel_event.clear()

        # Add streaming indicator
        self._current_streaming_indicator = chat.add_streaming_indicator("Thinking")

        try:
            await agent_loop(
                user_input=user_input,
                conversation=self._conversation,
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
                tool_context=self._tool_context,
                on_assistant_text=lambda t: self.call_from_thread(
                    lambda: self._on_assistant_text(t)
                ),
                on_tool_call=lambda name, args: self.call_from_thread(
                    lambda n=name, a=args: self._on_tool_call(n, a)
                ),
                on_tool_result=lambda name, res: self.call_from_thread(
                    lambda n=name, r=res: self._on_tool_result(n, r)
                ),
                on_permission_denied=lambda name, reason: self.call_from_thread(
                    lambda n=name, r=reason: self._on_permission_denied(n, r)
                ),
                on_assistant_chunk=lambda chunk: self.call_from_thread(
                    lambda c=chunk: self._on_assistant_chunk(c)
                ),
                max_iterations=self._commands.max_iterations or 25,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
                hook_executor=self._hook_executor,
                auto_commit=self._commands.auto_commit,
                auto_commit_threshold=self._commands.auto_commit_threshold,
                on_parallel_start=lambda calls: self.call_from_thread(
                    lambda c=calls: chat.write_system(f"Running {len(c)} tools in parallel...")
                ),
                on_parallel_complete=lambda _results: None,
                on_thinking=lambda t: self.call_from_thread(lambda txt=t: self._on_thinking(txt)),
            )
        except AgentCancelledError:
            chat.write_system("Agent cancelled.")
        except Exception as exc:
            logger.error("Agent loop error: %s", exc, exc_info=True)
            chat.write_error(str(exc))
        finally:
            self.running = False
            self._update_info_panel()
            # Remove streaming indicator
            if self._current_streaming_indicator is not None:
                self._current_streaming_indicator.remove()
                self._current_streaming_indicator = None
            # Reset streaming message ref
            self._current_assistant_msg = None
            # Focus input
            self.query_one("#input-bar", InputBar).focus_input()

    def _on_assistant_text(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is not None:
            self._current_streaming_indicator.remove()
            self._current_streaming_indicator = None
        if self._current_assistant_msg is not None:
            self._current_assistant_msg = None
            return
        chat.write_assistant(text)

    def _on_assistant_chunk(self, chunk: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is not None:
            self._current_streaming_indicator.remove()
            self._current_streaming_indicator = None
        if self._current_assistant_msg is None:
            self._current_assistant_msg = AssistantMessage(chunk)
            chat._container().mount(self._current_assistant_msg)
        else:
            self._current_assistant_msg._text += chunk
            body = self._current_assistant_msg.query_one(".msg-body", Static)
            body.update(self._current_assistant_msg._text)
        chat._scroll_to_bottom()

    def _on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is not None:
            self._current_streaming_indicator.remove()
            self._current_streaming_indicator = None
        self._current_tool_block = chat.add_tool_call(name, args)
        self._tool_calls_total += 1
        self._update_info_panel()

    def _on_tool_result(self, name: str, result: Any) -> None:
        is_error = getattr(result, "is_error", False)
        if is_error:
            self._tool_errors_total += 1
        output = getattr(result, "output", str(result))
        error = getattr(result, "error", None)
        display = str(error) if is_error and error else str(output)

        if self._current_tool_block is not None:
            self._current_tool_block.set_result(display, is_error)
            self._current_tool_block = None
        self._update_info_panel()

    def _on_permission_denied(self, name: str, reason: str) -> None:
        self._tool_denied_total += 1
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system(f"Permission denied: {name} ({reason})")
        self._update_info_panel()

    def _on_thinking(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is None:
            self._current_streaming_indicator = chat.add_streaming_indicator(f"Thinking: {text}")
        else:
            self._current_streaming_indicator.update(f"Thinking: {text}")

    def action_cancel(self) -> None:
        self._cancel_event.set()
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system("Cancelling...")

    def action_command_palette(self) -> None:
        self.push_screen(CommandPaletteScreen(), callback=self._on_palette_result)

    def _on_palette_result(self, result: str | None) -> None:
        if result is None:
            return
        self._dispatch_command(result)
        self.query_one("#input-bar", InputBar).focus_input()

    def action_clear_chat(self) -> None:
        scroll = self.query_one("#messages-scroll", VerticalScroll)
        scroll.remove_children()
        self.query_one("#chat-panel", ChatPanel).write_system("Chat cleared.")

    def action_new_session(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system("New session started.")
        self.turn_count = 0
        self._tool_calls_total = 0
        self._tool_errors_total = 0
        self._tool_denied_total = 0
        self._start_time = time.monotonic()
        self._update_info_panel()

    async def action_quit(self) -> None:
        duration = time.monotonic() - self._start_time
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system(
            f"Session ended  {duration:.0f}s  "
            f"{self._tool_calls_total} calls  "
            f"${self._llm_client.total_cost_usd:.4f}"
        )
        if self._session_memory is not None:
            self._session_memory.end_session(
                self._session_id,
                summary=f"turns={self.turn_count} tools={self._tool_calls_total}",
            )
        if self._audit_trail is not None:
            self._audit_trail.record(
                event_type="session_end",
                detail={"reason": "user_quit"},
            )
        await super().action_quit()
