"""Minimal Godspeed TUI — no CSS, just compose and callbacks."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
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

_COMMANDS: list[tuple[str, str]] = [
    ("/quit", "Exit"),
    ("/help", "Show commands"),
    ("/pause", "Pause agent"),
    ("/resume", "Resume agent"),
    ("/cancel", "Cancel"),
    ("/clear", "Clear chat"),
    ("/model", "Switch model"),
    ("/cost", "Show cost"),
    ("/tokens", "Show tokens"),
    ("/permissions", "Show permissions"),
    ("/audit", "Show audit"),
    ("/evolve", "Evolve"),
]


class _TextualPermissionProxy:
    def __init__(self, engine: PermissionEngine, app: GodspeedTextualApp) -> None:
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


class PermissionScreen(Screen[str]):
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
        with Vertical():
            yield Static(f"Permission Required: {self._tool_name}")
            yield Static(f"Reason: {self._reason}")
            if self._arguments:
                yield Static(f"Args: {self._arguments}")
            with Horizontal():
                yield Button("Yes", id="btn-yes", variant="success")
                yield Button("No", id="btn-no", variant="error")
                yield Button("Always", id="btn-always")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {"btn-yes": "allow", "btn-no": "deny", "btn-always": "always"}
        btn_id = event.button.id or ""
        self.dismiss(mapping.get(btn_id, "deny"))

    def on_key(self, event: Any) -> None:
        key_map = {"y": "allow", "n": "deny", "a": "always", "d": "deny"}
        if event.key in key_map:
            self.dismiss(key_map[event.key])


class CommandPaletteScreen(Screen[str | None]):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Input(placeholder="Command...", id="palette-input")
            yield ListView(id="palette-list")

    def on_mount(self) -> None:
        lv = self.query_one("#palette-list", ListView)
        for cmd, desc in _COMMANDS:
            lv.append(ListItem(Label(f"{cmd}  {desc}")))
        self.query_one("#palette-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        lv = self.query_one("#palette-list", ListView)
        lv.clear()
        for cmd, desc in _COMMANDS:
            if query in cmd.lower() or query in desc.lower():
                lv.append(ListItem(Label(f"{cmd}  {desc}")))
        if lv.children:
            lv.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        text = str(event.item.render())
        cmd = text.split()[0] if text else None
        self.dismiss(cmd)

    def on_key(self, event: Any) -> None:
        if event.key == "escape":
            self.dismiss(None)


class GodspeedTextualApp(App[None]):
    """Minimal Textual TUI with no custom CSS."""

    CSS_PATH = None
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+k", "command_palette", "Commands"),
        Binding("ctrl+l", "clear_chat", "Clear"),
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
        with Horizontal():
            with Vertical():
                yield Static("Chat", id="chat-header")
                yield Static("", id="chat-log", expand=True)
                with Horizontal():
                    yield Input(placeholder="Ask Godspeed...", id="user-input")
                    yield Button("Send", id="submit-btn")
            with Vertical():
                yield Static(f"Godspeed v{__version__}")
                yield Static("---")
                yield Static("Session", id="info-session")
                yield Static("Usage", id="info-usage")
                yield Static("Tools", id="info-tools")
                yield Static("Mode", id="info-mode")

    def on_mount(self) -> None:
        self.title = "Godspeed"
        self.sub_title = self._llm_client.model
        chat_log = self.query_one("#chat-log", Static)
        chat_log.styles.height = "1fr"
        self._update_info_panel()
        self._wire_permissions()
        self.query_one("#user-input", Input).focus()

    def _wire_permissions(self) -> None:
        if self._permission_engine is not None:
            self._tool_context.permissions = _TextualPermissionProxy(self._permission_engine, self)
        # Diff reviewer placeholder
        self._tool_context.diff_reviewer = None

    def _update_info_panel(self) -> None:
        try:
            session = self.query_one("#info-session", Static)
            usage = self.query_one("#info-usage", Static)
            tools = self.query_one("#info-tools", Static)
            mode = self.query_one("#info-mode", Static)
        except Exception:
            return

        sid = self._session_id[:10]
        cwd = str(self._tool_context.cwd)[:30]
        session.update(f"ID: {sid}\nModel: {self._llm_client.model}\nDir: {cwd}")

        total = self._llm_client.total_input_tokens + self._llm_client.total_output_tokens
        usage.update(
            f"In: {self._llm_client.total_input_tokens:,}\n"
            f"Out: {self._llm_client.total_output_tokens:,}\n"
            f"Total: {total:,}\n"
            f"Cost: ${self._llm_client.total_cost_usd:.4f}"
        )

        tools.update(
            f"Calls: {self._tool_calls_total}\n"
            f"Errors: {self._tool_errors_total}\n"
            f"Denied: {self._tool_denied_total}"
        )

        perm = self._get_permission_mode()
        status = "running" if self.running else "idle"
        mode.update(f"Mode: {perm}\nStatus: {status}")

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
        inp = self.query_one("#user-input", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""

        if text.startswith("/"):
            self._dispatch_command(text)
            return

        self.turn_count += 1
        self._update_info_panel()

        chat = self.query_one("#chat-log", Static)
        current = str(chat.renderable or "")
        prefix = f"\n[You]: {text}\n" if current else f"[You]: {text}\n"
        chat.update(current + prefix if current else prefix)

        if self._correction_tracker is not None:
            self._correction_tracker.check_for_correction(text)

        self.run_worker(self._agent_worker(text))

    def _dispatch_command(self, text: str) -> None:
        chat = self.query_one("#chat-log", Static)

        if text == "/cancel":
            self._cancel_event.set()
            current = str(chat.renderable or "")
            chat.update(current + "\nCancelling...\n")
            return

        with capture_output() as sio:
            result = self._commands.dispatch(text)

        output = sio.getvalue()
        if output:
            current = str(chat.renderable or "")
            chat.update(current + f"\n{output}\n")

        if result is None:
            current = str(chat.renderable or "")
            chat.update(current + f"\nUnknown command: {text}\n")
            return

        if result.should_quit:
            self.exit()

        if text == "/clear":
            chat.update("")

    async def _agent_worker(self, user_input: str) -> None:
        self.running = True
        self._update_info_panel()
        self._cancel_event.clear()

        try:
            await agent_loop(
                user_input=user_input,
                conversation=self._conversation,
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
                tool_context=self._tool_context,
                on_assistant_text=lambda t: self.call_from_thread(
                    lambda: self._append_chat(f"[Assistant]: {t}")
                ),
                on_tool_call=lambda name, args: self.call_from_thread(
                    lambda n=name, a=args: self._append_chat(f"[Tool]: {n}")
                ),
                on_tool_result=lambda name, res: self.call_from_thread(lambda n=name, r=res: None),
                on_permission_denied=lambda name, reason: self.call_from_thread(
                    lambda n=name, r=reason: self._append_chat(f"Permission denied: {n} ({r})")
                ),
                max_iterations=self._commands.max_iterations or 25,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
                hook_executor=self._hook_executor,
                auto_commit=self._commands.auto_commit,
                auto_commit_threshold=self._commands.auto_commit_threshold,
            )
        except AgentCancelledError:
            self._append_chat("Agent cancelled.")
        except Exception as exc:
            logger.error("Agent loop error: %s", exc, exc_info=True)
            self._append_chat(f"Error: {exc}")
        finally:
            self.running = False
            self._update_info_panel()
            self.query_one("#user-input", Input).focus()

    def _append_chat(self, text: str) -> None:
        chat = self.query_one("#chat-log", Static)
        current = str(chat.renderable or "")
        chat.update(current + f"\n{text}\n")

    def action_cancel(self) -> None:
        self._cancel_event.set()
        self._append_chat("Cancelling...")

    def action_command_palette(self) -> None:
        self.push_screen(CommandPaletteScreen(), callback=self._on_palette_result)

    def _on_palette_result(self, result: str | None) -> None:
        if result is None:
            return
        self._dispatch_command(result)
        self.query_one("#user-input", Input).focus()

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", Static).update("")

    async def action_quit(self) -> None:
        duration = time.monotonic() - self._start_time
        self._append_chat(
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
