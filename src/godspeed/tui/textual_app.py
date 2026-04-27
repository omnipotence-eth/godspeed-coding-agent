"""Professional Textual TUI for Godspeed — Claude Code x OpenCode inspired.

Architecture:
- 3-pane layout: tool sidebar | chat panel | context sidebar
- Top status bar with live metrics
- Bottom input bar with command hints
- Collapsible tool call blocks with timing
- Modal screens: permission, diff review, command palette
- Welcome screen on first launch
- Keyboard shortcuts throughout
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
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
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
from godspeed.tui.commands import Commands
from godspeed.tui.output import capture_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

_TOOL_ICONS: dict[str, str] = {
    "file_read": "📄",
    "file_write": "✏️",
    "file_edit": "📝",
    "shell": "⚡",
    "grep_search": "🔍",
    "glob_search": "📁",
    "git": "🌿",
    "repo_map": "🗺️",
    "complexity": "📊",
    "system_optimizer": "🔧",
    "default": "🔧",
}

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
# Sidebar Widgets
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """Top status bar — minimal, info-dense (Claude Code style)."""

    model: reactive[str] = reactive("unknown")
    cost_usd: reactive[float] = reactive(0.0)
    turns: reactive[int] = reactive(0)
    permission_mode: reactive[str] = reactive("normal")
    context_pct: reactive[float] = reactive(0.0)
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)
    is_running: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        yield Static(id="status-content")

    def _watch_all(self) -> None:
        content = self.query_one("#status-content", Static)
        mode_color = {
            "normal": "green",
            "strict": "red",
            "plan": "cyan",
            "yolo": "yellow",
        }.get(self.permission_mode, "white")
        ctx_color = "red" if self.context_pct > 90 else "yellow" if self.context_pct > 70 else "dim"
        running_indicator = "[blink]●[/] " if self.is_running else ""
        parts = [
            f"[bold cyan]{running_indicator}Godspeed[/]",
            f"model=[b]{self.model}[/]",
            f"cost=${self.cost_usd:.4f}",
            f"turns={self.turns}",
            f"tok={self.input_tokens}/{self.output_tokens}",
            f"ctx=[{ctx_color}]{self.context_pct:.0f}%[/]",
            f"mode=[{mode_color}]{self.permission_mode}[/]",
        ]
        content.update("  ·  ".join(parts))

    def watch_model(self, _value: str) -> None:
        self._watch_all()

    def watch_cost_usd(self, _value: float) -> None:
        self._watch_all()

    def watch_turns(self, _value: int) -> None:
        self._watch_all()

    def watch_permission_mode(self, _value: str) -> None:
        self._watch_all()

    def watch_context_pct(self, _value: float) -> None:
        self._watch_all()

    def watch_input_tokens(self, _value: int) -> None:
        self._watch_all()

    def watch_output_tokens(self, _value: int) -> None:
        self._watch_all()

    def watch_is_running(self, _value: bool) -> None:
        self._watch_all()


class ToolSidebar(Static):
    """Left sidebar — tool inventory (OpenCode style)."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        super().__init__()
        self._tool_registry = tool_registry

    def compose(self) -> ComposeResult:
        yield Static("Tools", classes="section-title")
        yield ListView(id="tool-list")

    def on_mount(self) -> None:
        lv = self.query_one("#tool-list", ListView)
        for tool in self._tool_registry.list_tools():
            icon = _TOOL_ICONS.get(tool.name, _TOOL_ICONS["default"])
            label = _TOOL_LABELS.get(tool.name, tool.name)
            lv.append(
                ListItem(
                    Label(f"{icon} {label}"),
                    id=f"tool-{tool.name}",
                )
            )


class ContextSidebar(Static):
    """Right sidebar — session telemetry."""

    session_id: reactive[str] = reactive("")
    tool_count: reactive[int] = reactive(0)
    tool_calls: reactive[int] = reactive(0)
    tool_errors: reactive[int] = reactive(0)
    tool_denied: reactive[int] = reactive(0)
    project_dir: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Static("Session", classes="section-title")
        yield Static(id="ctx-stats")

    def _update(self) -> None:
        panel = self.query_one("#ctx-stats", Static)
        err_color = "red" if self.tool_errors > 0 else "green"
        deny_color = "yellow" if self.tool_denied > 0 else "white"
        sid = self.session_id[:8]
        pdir = self.project_dir[:20]
        tc = self.tool_count
        tcalls = self.tool_calls
        terr = self.tool_errors
        tden = self.tool_denied
        lines = [
            f"[dim]ID:[/]    [b]{sid}[/]",
            f"[dim]Dir:[/]   [b]{pdir}[/]",
            f"[dim]Tools:[/] [b]{tc}[/]",
            f"[dim]Calls:[/] [b]{tcalls}[/]",
            f"[dim]Errors:[/] [{err_color}]{terr}[/]",
            f"[dim]Denied:[/] [{deny_color}]{tden}[/]",
        ]
        panel.update("\n".join(lines))

    def watch_session_id(self, _value: str) -> None:
        self._update()

    def watch_tool_count(self, _value: int) -> None:
        self._update()

    def watch_tool_calls(self, _value: int) -> None:
        self._update()

    def watch_tool_errors(self, _value: int) -> None:
        self._update()

    def watch_tool_denied(self, _value: int) -> None:
        self._update()

    def watch_project_dir(self, _value: str) -> None:
        self._update()


# ---------------------------------------------------------------------------
# Chat Components
# ---------------------------------------------------------------------------


class UserMessage(Static):
    """A user message bubble."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("You", classes="message-header")
        yield Static(self._text)


class AssistantMessage(Static):
    """An assistant message bubble."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("Assistant", classes="message-header")
        yield Static(self._text)


class SystemMessage(Static):
    """A system/status message."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(self._text)


class ErrorMessage(Static):
    """An error message block."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(self._text)


class ToolCallBlock(Static):
    """Collapsible tool call display with timing (Claude Code style)."""

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
        icon = _TOOL_ICONS.get(self._tool_name, _TOOL_ICONS["default"])
        label = _TOOL_LABELS.get(self._tool_name, self._tool_name)
        args_str = self._format_args()
        header_text = f"{icon} {label} {args_str}"
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
            if len(primary) > 40:
                primary = "..." + primary[-37:]
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
        result_widget.update(
            f"{'Error' if is_error else 'Result'} ({elapsed:.0f}ms): {result[:200]}"
        )
        if self.expanded:
            body = self.query_one(".tool-body", Static)
            body.update(result)


class StreamingIndicator(Static):
    """Animated thinking indicator."""

    def __init__(self, text: str = "Thinking") -> None:
        super().__init__()
        self._base_text = text
        self._dot_count = 0

    def on_mount(self) -> None:
        self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self.update(f"[blink bold cyan]{self._base_text}{dots}[/]")


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
    """Bottom input bar with submit button and hints."""

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Ask Godspeed anything...", id="user-input")
        yield Button("Send", id="submit-btn", variant="primary")
        yield Static("Ctrl+K palette · Ctrl+C cancel", classes="hint")

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
    """Command palette with fuzzy search (OpenCode Ctrl+K style)."""

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
        # Extract command from markup
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
            yield Static("⚡ Godspeed", classes="welcome-title")
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
                "Press Enter or any key to start · Ctrl+K for commands",
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

        # Slash command registry (shared with legacy command library)
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
            yield StatusBar(id="status-bar")
            yield ToolSidebar(self._tool_registry)
            yield ChatPanel(id="chat-panel")
            yield ContextSidebar(id="context-sidebar")
            yield InputBar(id="input-bar")

    def on_mount(self) -> None:
        self.title = "Godspeed"
        self.sub_title = self._llm_client.model
        self._update_status()
        self._update_context()
        self._wire_permissions()
        # Show welcome screen on first mount
        self.push_screen(
            WelcomeScreen(
                model=self._llm_client.model,
                project_dir=str(self._tool_context.cwd),
                tool_count=len(self._tool_registry.list_tools()),
            )
        )

    def _wire_permissions(self) -> None:
        if self._permission_engine is not None:
            self._tool_context.permissions = _TextualPermissionProxy(self._permission_engine, self)
        self._tool_context.diff_reviewer = _TextualDiffReviewer(self)

    def _update_status(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        bar.model = self._llm_client.model
        bar.cost_usd = self._llm_client.total_cost_usd
        bar.turns = self.turn_count
        bar.permission_mode = self._get_permission_mode()
        bar.context_pct = (
            self._conversation.token_count / self._conversation.max_tokens * 100
            if self._conversation.max_tokens > 0
            else 0.0
        )
        bar.input_tokens = self._llm_client.total_input_tokens
        bar.output_tokens = self._llm_client.total_output_tokens
        bar.is_running = self.running

    def _update_context(self) -> None:
        ctx = self.query_one("#context-sidebar", ContextSidebar)
        ctx.session_id = self._session_id
        ctx.project_dir = str(self._tool_context.cwd)
        ctx.tool_count = len(self._tool_registry.list_tools())
        ctx.tool_calls = self._tool_calls_total
        ctx.tool_errors = self._tool_errors_total
        ctx.tool_denied = self._tool_denied_total

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
        input_bar = self.query_one("#input-bar", InputBar)
        text = input_bar.get_value().strip()
        if not text:
            return
        input_bar.clear()

        if text.startswith("/"):
            self._dispatch_command(text)
            return

        self.turn_count += 1
        self._update_status()

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
            chat.write_system("⏹ Cancelling current operation...")
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
        self._update_status()
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
                max_iterations=25,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
                hook_executor=self._hook_executor,
                on_parallel_start=lambda calls: self.call_from_thread(
                    lambda c=calls: chat.write_system(f"⚡ Running {len(c)} tools in parallel...")
                ),
                on_parallel_complete=lambda _results: None,
                on_thinking=lambda t: self.call_from_thread(lambda txt=t: self._on_thinking(txt)),
            )
        except AgentCancelledError:
            chat.write_system("⏹ Agent cancelled.")
        except Exception as exc:
            logger.error("Agent loop error: %s", exc, exc_info=True)
            chat.write_error(str(exc))
        finally:
            self.running = False
            self._update_status()
            self._update_context()
            # Remove streaming indicator
            if self._current_streaming_indicator is not None:
                self._current_streaming_indicator.remove()
                self._current_streaming_indicator = None
            # Focus input
            self.query_one("#input-bar", InputBar).focus_input()

    def _on_assistant_text(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is not None:
            self._current_streaming_indicator.remove()
            self._current_streaming_indicator = None
        chat.write_assistant(text)

    def _on_assistant_chunk(self, chunk: str) -> None:
        # For streaming, we could append to a mutable message widget.
        # For now, write each chunk as it arrives.
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_assistant(chunk)

    def _on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is not None:
            self._current_streaming_indicator.remove()
            self._current_streaming_indicator = None
        self._current_tool_block = chat.add_tool_call(name, args)
        self._tool_calls_total += 1
        self._update_context()

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
        self._update_context()

    def _on_permission_denied(self, name: str, reason: str) -> None:
        self._tool_denied_total += 1
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system(f"🚫 Permission denied: {name} ({reason})")
        self._update_context()

    def _on_thinking(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._current_streaming_indicator is None:
            self._current_streaming_indicator = chat.add_streaming_indicator(f"Thinking: {text}")
        else:
            self._current_streaming_indicator.update(f"[blink bold cyan]Thinking: {text}[/]")

    def action_cancel(self) -> None:
        self._cancel_event.set()
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system("⏹ Cancelling...")

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
        chat.write_system("🔄 New session started.")
        self.turn_count = 0
        self._tool_calls_total = 0
        self._tool_errors_total = 0
        self._tool_denied_total = 0
        self._start_time = time.monotonic()
        self._update_status()
        self._update_context()

    async def action_quit(self) -> None:
        duration = time.monotonic() - self._start_time
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.write_system(
            f"Session ended · {duration:.0f}s · "
            f"{self._tool_calls_total} calls · "
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
