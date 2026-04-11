"""Click CLI entry point for Godspeed."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import click

from godspeed import __version__

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/tags"
OLLAMA_STARTUP_TIMEOUT = 15  # seconds to wait for ollama to come up


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level.

    In verbose mode, only godspeed.* loggers get DEBUG — all third-party
    libraries stay at WARNING so they don't drown the TUI output.
    Logs go to stderr to avoid interleaving with Rich's stdout streaming.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )

    # Root logger: WARNING always (catches all third-party noise)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)

    # Godspeed loggers: DEBUG in verbose mode, WARNING otherwise
    godspeed_logger = logging.getLogger("godspeed")
    godspeed_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def _is_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(OLLAMA_URL, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
            return True
    except Exception:
        return False


def _ensure_ollama(console: object | None = None) -> bool:
    """Start Ollama if it's not running. Returns True if Ollama is available.

    Args:
        console: Optional Rich Console for status output.
    """
    if _is_ollama_running():
        return True

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        if console is not None:
            console.print(  # type: ignore[union-attr]
                "[yellow]  Ollama is not installed. "
                "Install from https://ollama.com or use a cloud model: "
                "godspeed -m claude-sonnet-4-20250514[/yellow]"
            )
        return False

    # Start ollama serve as a detached background process
    if console is not None:
        console.print("[dim]  Starting Ollama...[/dim]", end="")  # type: ignore[union-attr]

    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        logger.warning("Failed to start Ollama: %s", exc)
        if console is not None:
            console.print(f" [red]failed: {exc}[/red]")  # type: ignore[union-attr]
        return False

    # Poll until it's up
    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if _is_ollama_running():
            if console is not None:
                console.print(" [green]ready[/green]")  # type: ignore[union-attr]
            return True

    if console is not None:
        console.print(  # type: ignore[union-attr]
            " [yellow]timed out. Ollama may still be starting.[/yellow]"
        )
    return False


def _build_tool_registry() -> tuple:
    """Create all tool instances and register them.

    Returns:
        (ToolRegistry, dict[str, RiskLevel]) tuple.
    """
    from godspeed.tools.base import RiskLevel
    from godspeed.tools.file_edit import FileEditTool
    from godspeed.tools.file_read import FileReadTool
    from godspeed.tools.file_write import FileWriteTool
    from godspeed.tools.registry import ToolRegistry

    registry = ToolRegistry()
    risk_levels: dict[str, RiskLevel] = {}

    from godspeed.tools.git import GitTool
    from godspeed.tools.glob_search import GlobSearchTool
    from godspeed.tools.grep_search import GrepSearchTool
    from godspeed.tools.shell import ShellTool
    from godspeed.tools.verify import VerifyTool

    tools = [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        ShellTool(),
        GlobSearchTool(),
        GrepSearchTool(),
        GitTool(),
        VerifyTool(),
    ]

    for tool in tools:
        registry.register(tool)
        risk_levels[tool.name] = tool.risk_level

    return registry, risk_levels


async def _run_app(
    model: str,
    project_dir: Path,
    verbose: bool,
    audit_dir: Path | None,
) -> None:
    """Wire up all components and launch the TUI."""
    from godspeed.agent.conversation import Conversation
    from godspeed.agent.system_prompt import build_system_prompt
    from godspeed.audit.trail import AuditTrail
    from godspeed.config import GodspeedSettings
    from godspeed.context.project_instructions import load_project_instructions
    from godspeed.llm.client import LLMClient
    from godspeed.security.permissions import PermissionEngine
    from godspeed.tools.base import ToolContext
    from godspeed.tui.app import TUIApp

    # Load config
    overrides: dict = {}
    if model:
        overrides["model"] = model
    settings = GodspeedSettings(**overrides)

    effective_model = model or settings.model
    effective_project_dir = project_dir.resolve()
    session_id = str(uuid4())

    # Tools
    registry, risk_levels = _build_tool_registry()

    # Permission engine
    permission_engine = PermissionEngine(
        deny_patterns=settings.permissions.deny,
        allow_patterns=settings.permissions.allow,
        ask_patterns=settings.permissions.ask,
        tool_risk_levels=risk_levels,
    )

    # Audit trail
    audit_trail: AuditTrail | None = None
    if settings.audit.enabled:
        effective_audit_dir = audit_dir or (settings.global_dir / "audit")
        audit_trail = AuditTrail(
            log_dir=effective_audit_dir,
            session_id=session_id,
        )
        audit_trail.record(
            event_type="session_start",
            detail={
                "model": effective_model,
                "project_dir": str(effective_project_dir),
            },
        )
        # Purge expired audit logs on startup
        audit_trail.cleanup_expired(settings.audit.retention_days)

    # Tool context
    tool_context = ToolContext(
        cwd=effective_project_dir,
        session_id=session_id,
        permissions=permission_engine,
        audit=audit_trail,
    )

    # System prompt
    project_instructions = load_project_instructions(
        effective_project_dir,
        settings.context.project_instructions,
    )
    system_prompt = build_system_prompt(
        tools=registry.list_tools(),
        project_instructions=project_instructions,
        cwd=effective_project_dir,
    )

    # Auto-start Ollama if the model needs it
    if effective_model.lower().startswith("ollama"):
        from godspeed.tui.output import console as rich_console

        _ensure_ollama(console=rich_console)

    # LLM client
    llm_client = LLMClient(
        model=effective_model,
        fallback_models=settings.fallback_models,
    )

    # Conversation
    conversation = Conversation(
        system_prompt=system_prompt,
        model=effective_model,
        max_tokens=settings.max_context_tokens,
        compaction_threshold=settings.compaction_threshold,
    )

    # Launch TUI
    app = TUIApp(
        llm_client=llm_client,
        tool_registry=registry,
        tool_context=tool_context,
        conversation=conversation,
        permission_engine=permission_engine,
        audit_trail=audit_trail,
        session_id=session_id,
    )
    await app.run()


@click.group(invoke_without_command=True)
@click.option("--model", "-m", default="", help="Model to use (e.g. claude-sonnet-4-20250514).")
@click.option(
    "--project-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Project directory (default: current directory).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--audit-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for audit logs (default: ~/.godspeed/audit).",
)
@click.pass_context
def main(
    ctx: click.Context,
    model: str,
    project_dir: Path,
    verbose: bool,
    audit_dir: Path | None,
) -> None:
    """Godspeed -- Security-first open-source coding agent."""
    _setup_logging(verbose)

    # Store params for subcommands
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["project_dir"] = project_dir
    ctx.obj["verbose"] = verbose
    ctx.obj["audit_dir"] = audit_dir

    # If no subcommand, launch the TUI
    if ctx.invoked_subcommand is None:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(_run_app(model, project_dir, verbose, audit_dir))


@main.command()
def version() -> None:
    """Show Godspeed version."""
    from rich.console import Console as RichConsole

    c = RichConsole()
    c.print(f"godspeed [bold]{__version__}[/bold]")


@main.group()
def audit() -> None:
    """Audit trail commands."""


@audit.command("verify")
@click.argument("session_id", required=False, default=None)
@click.option(
    "--audit-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing audit logs.",
)
def audit_verify(session_id: str | None, audit_dir: Path | None) -> None:
    """Verify the hash chain integrity of an audit session.

    If SESSION_ID is not provided, verifies all sessions in the audit directory.
    """
    from rich.console import Console as RichConsole

    from godspeed.audit.trail import AuditTrail
    from godspeed.config import DEFAULT_GLOBAL_DIR

    c = RichConsole()
    effective_dir = audit_dir or (DEFAULT_GLOBAL_DIR / "audit")

    if not effective_dir.exists():
        c.print(f"[red]Audit directory not found: {effective_dir}[/red]")
        sys.exit(1)

    if session_id:
        # Verify single session
        trail = AuditTrail(log_dir=effective_dir, session_id=session_id)
        if not trail.log_path.exists():
            c.print(f"[red]No audit log found for session: {session_id}[/red]")
            sys.exit(1)
        is_valid, message = trail.verify_chain()
        if is_valid:
            c.print(f"[green]VALID[/green] -- {message}")
        else:
            c.print(f"[red]BROKEN[/red] -- {message}")
            sys.exit(1)
    else:
        # Verify all sessions
        found = False
        for log_file in sorted(effective_dir.glob("*.audit.jsonl")):
            found = True
            sid = log_file.stem.replace(".audit", "")
            trail = AuditTrail(log_dir=effective_dir, session_id=sid)
            is_valid, message = trail.verify_chain()
            status = "[green]VALID[/green]" if is_valid else "[red]BROKEN[/red]"
            c.print(f"  {status}  {sid[:12]}...  {message}")

        if not found:
            c.print("[dim]No audit logs found.[/dim]")


@main.command()
def init() -> None:
    """Set up Godspeed — create ~/.godspeed/ and default settings.yaml."""

    from rich.console import Console as RichConsole

    from godspeed.config import DEFAULT_GLOBAL_DIR

    c = RichConsole()
    global_dir = DEFAULT_GLOBAL_DIR
    settings_path = global_dir / "settings.yaml"
    audit_dir = global_dir / "audit"

    global_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        c.print(f"  [dim]Settings already exist:[/dim] {settings_path}")
    else:
        # Copy the example settings
        example = Path(__file__).parent.parent.parent / "settings.yaml.example"
        if example.exists():
            settings_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # Inline minimal config if example not bundled
            settings_path.write_text(
                "# Godspeed settings — see https://github.com/omnipotence-eth/godspeed-coding-agent\n"
                "model: ollama/qwen3:4b\n"
                "fallback_models: []\n",
                encoding="utf-8",
            )
        c.print(f"  [green]Created settings:[/green] {settings_path}")

    c.print(f"  [green]Audit directory:[/green] {audit_dir}")
    c.print()
    c.print("  [bold]Next steps:[/bold]")
    c.print("    1. Install a local model: [cyan]ollama pull qwen3:4b[/cyan]")
    c.print("    2. Or set an API key:     [cyan]export ANTHROPIC_API_KEY=sk-...[/cyan]")
    c.print(f"    3. Edit your settings:    [cyan]{settings_path}[/cyan]")
    c.print("    4. Launch Godspeed:        [cyan]godspeed[/cyan]")


@main.command()
def models() -> None:
    """Show popular model options and how to configure them."""
    from rich.console import Console as RichConsole
    from rich.table import Table

    c = RichConsole()

    table = Table(title="Popular Models", border_style="blue", expand=False)
    table.add_column("Model", style="bold cyan")
    table.add_column("Provider", style="dim")
    table.add_column("Cost")
    table.add_column("API Key Env Var", style="dim")

    # Free local models
    table.add_row("ollama/qwen3:4b", "Ollama", "[green]Free[/green]", "None (local)")
    table.add_row("ollama/qwen3:8b", "Ollama", "[green]Free[/green]", "None (local)")
    table.add_row("ollama/gemma4:e4b", "Ollama", "[green]Free[/green]", "None (local)")
    table.add_row("ollama/llama3.3:8b", "Ollama", "[green]Free[/green]", "None (local)")
    table.add_row("ollama/deepseek-r1:8b", "Ollama", "[green]Free[/green]", "None (local)")
    table.add_row("ollama/mistral:7b", "Ollama", "[green]Free[/green]", "None (local)")

    # Paid cloud models
    table.add_row("claude-sonnet-4-20250514", "Anthropic", "Paid", "ANTHROPIC_API_KEY")
    table.add_row("claude-opus-4-20250514", "Anthropic", "Paid", "ANTHROPIC_API_KEY")
    table.add_row("gpt-4o", "OpenAI", "Paid", "OPENAI_API_KEY")
    table.add_row("gpt-4o-mini", "OpenAI", "Paid", "OPENAI_API_KEY")
    table.add_row("gemini/gemini-2.0-flash", "Google", "Paid", "GEMINI_API_KEY")
    table.add_row("deepseek/deepseek-chat", "DeepSeek", "Paid", "DEEPSEEK_API_KEY")

    c.print(table)
    c.print()
    c.print("  [bold]Switch models:[/bold]")
    c.print("    [dim]CLI flag:[/dim]     godspeed -m claude-sonnet-4-20250514")
    c.print("    [dim]Env var:[/dim]      GODSPEED_MODEL=gpt-4o godspeed")
    c.print("    [dim]Settings:[/dim]     Edit ~/.godspeed/settings.yaml")
    c.print("    [dim]At runtime:[/dim]   /model claude-sonnet-4-20250514")
