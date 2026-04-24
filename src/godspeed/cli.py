"""Click CLI entry point for Godspeed."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import click

from godspeed import __version__
from godspeed.config import DEFAULT_GLOBAL_DIR


def _force_utf8_stdio() -> None:
    """Make stdout/stderr survive non-ASCII output on legacy consoles.

    Default stdout encoding on Windows is often cp1252, which cannot encode
    common agent output characters (arrows, em-dashes, smart quotes) and
    crashes with UnicodeEncodeError after the agent already finished real
    work — masking success with a nonzero exit. We rewrap stdout/stderr in
    UTF-8 with errors='replace' so stray high-bit chars become '?' instead
    of crashing. Idempotent; only rewraps when the current encoding is not
    already utf-8.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "buffer"):
            continue
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc == "utf8":
            continue
        try:
            setattr(
                sys,
                name,
                io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                ),
            )
        except (AttributeError, OSError):
            continue


_force_utf8_stdio()

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/tags"
OLLAMA_STARTUP_TIMEOUT = 15  # seconds to wait for ollama to come up


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=value`` env file into a dict.

    Supports:
    - ``# ...`` comments (line must start with ``#``; inline comments aren't stripped)
    - Blank lines
    - Optional surrounding ``"..."`` or ``'...'`` quotes on values

    Malformed lines (no ``=``, empty key) are silently skipped — an env
    file should never crash startup. Returns an empty dict if the file
    is missing or unreadable.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # OSError: permissions / transient FS issue.
        # UnicodeDecodeError: someone wrote a binary blob or cp1252 file
        # into .env.local — log and skip rather than crash the CLI.
        logger.debug("Could not read env file %s: %s", path, exc)
        return result
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip a matching pair of surrounding quotes so
        # ``KEY="value with spaces"`` works as expected.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def _load_env_files(project_dir: Path | None = None) -> list[tuple[Path, list[str]]]:
    """Load env files from standard locations into ``os.environ``.

    Precedence (highest to lowest):

    1. **Shell environment** — always wins. Keys already in ``os.environ``
       are never overwritten. This keeps ``$env:NVIDIA_NIM_API_KEY = ...;
       godspeed`` one-offs working and matches every ``.env`` library's
       convention.
    2. ``<project_dir>/.godspeed/.env.local`` — project-scoped override
    3. ``<project_dir>/.godspeed/.env`` — project-scoped defaults
    4. ``~/.godspeed/.env.local`` — user-wide override
    5. ``~/.godspeed/.env`` — user-wide defaults

    Files are merged in low-to-high priority order so higher-priority
    files overwrite lower-priority ones BEFORE the single shell-env
    check. In-file semantics mirror dotenv / Vite / Next: ``.env.local``
    is the gitignored local override, ``.env`` is the checked-in default.

    Returns a list of ``(path, injected_keys)`` pairs for each file
    that contributed at least one variable — used by the caller to log
    which files the session picked up. Never raises; bad files are
    silently skipped so a malformed ``.env.local`` can't brick the CLI.
    """

    # Lowest priority first so later files overwrite earlier ones in
    # the merged dict. Shell-env wins is applied at the very end.
    candidates: list[Path] = [
        DEFAULT_GLOBAL_DIR / ".env",
        DEFAULT_GLOBAL_DIR / ".env.local",
    ]
    if project_dir is not None:
        candidates.extend(
            [
                project_dir / ".godspeed" / ".env",
                project_dir / ".godspeed" / ".env.local",
            ]
        )

    resolved: dict[str, str] = {}
    contributions: list[tuple[Path, list[str]]] = []
    for path in candidates:
        parsed = _parse_env_file(path)
        if not parsed:
            continue
        # Record which keys this file will contribute to the merged view.
        # (The actual injection into os.environ happens after the full
        # merge; shell-env wins over everything below.)
        contributions.append((path, sorted(parsed.keys())))
        resolved.update(parsed)

    loaded: list[tuple[Path, list[str]]] = []
    injected_keys: set[str] = set()
    for key, value in resolved.items():
        if key in os.environ:
            continue
        os.environ[key] = value
        injected_keys.add(key)

    # Re-project contributions onto actually-injected keys, so the log
    # reflects what the session ended up with after shell-env overrides.
    for path, keys in contributions:
        effective = [k for k in keys if k in injected_keys]
        if not effective:
            continue
        loaded.append((path, effective))
        logger.info(
            "Loaded %d env var(s) from %s: %s",
            len(effective),
            path,
            ", ".join(effective),
        )
    return loaded


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

        req = urllib.request.Request(OLLAMA_URL, method="GET")  # noqa: S310  # nosec B310
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310  # nosec B310
            return True
    except Exception:
        return False


def _ensure_ollama(console: Any | None = None) -> bool:
    """Start Ollama if it's not running. Returns True if Ollama is available.

    Args:
        console: Optional Rich Console for status output.
    """
    if _is_ollama_running():
        return True

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        if console is not None:
            from godspeed.tui.theme import WARNING

            console.print(
                f"[{WARNING}]  Ollama is not installed. "
                "Install from https://ollama.com or use a cloud model: "
                f"godspeed -m claude-sonnet-4-20250514[/{WARNING}]"
            )
        return False

    # Start ollama serve as a detached background process
    if console is not None:
        from godspeed.tui.theme import DIM

        console.print(f"[{DIM}]  Starting Ollama...[/{DIM}]", end="")
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
            from godspeed.tui.theme import ERROR

            console.print(f" [{ERROR}]failed: {exc}[/{ERROR}]")
        return False

    # Poll until it's up
    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if _is_ollama_running():
            if console is not None:
                from godspeed.tui.theme import SUCCESS

                console.print(f" [{SUCCESS}]ready[/{SUCCESS}]")
            return True

    if console is not None:
        from godspeed.tui.theme import WARNING

        console.print(f" [{WARNING}]timed out. Ollama may still be starting.[/{WARNING}]")
    return False


def _build_tool_registry(tool_set: str = "full") -> tuple:
    """Create all tool instances and register them.

    Args:
        tool_set: one of "local", "web", "full" (default). "local" hides
            web_search/web_fetch/github so weak models don't pick them
            over file_read/grep_search on local-codebase tasks. "web"
            adds the web tools back. "full" registers everything.

    Returns:
        (ToolRegistry, dict[str, RiskLevel]) tuple.
    """
    from godspeed.tools.tool_sets import get_allowed_tool_names

    allowed = get_allowed_tool_names(tool_set)
    from godspeed.tools.base import RiskLevel
    from godspeed.tools.file_edit import FileEditTool
    from godspeed.tools.batch_edit import BatchEditTool
    from godspeed.tools.file_read import FileReadTool
    from godspeed.tools.file_write import FileWriteTool
    from godspeed.tools.registry import ToolRegistry

    registry = ToolRegistry()
    risk_levels: dict[str, RiskLevel] = {}

    from godspeed.tools.background import BackgroundCheckTool
    from godspeed.tools.clarification import AskClarificationTool
    from godspeed.tools.complexity import ComplexityTool
    from godspeed.tools.coverage import CoverageTool
    from godspeed.tools.dep_audit import DepAuditTool
    from godspeed.tools.generate_tests import GenerateTestsTool
    from godspeed.tools.git import GitTool
    from godspeed.tools.glob_search import GlobSearchTool
    from godspeed.tools.grep_search import GrepSearchTool
    from godspeed.tools.notebook import NotebookEditTool
    from godspeed.tools.repo_map import RepoMapTool
    from godspeed.tools.security_scan import SecurityScanTool
    from godspeed.tools.shell import ShellTool
    from godspeed.tools.test_runner import TestRunnerTool
    from godspeed.tools.verify import VerifyTool
    from godspeed.tools.web_fetch import WebFetchTool
    from godspeed.tools.web_search import WebSearchTool

    tools: list = [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        ShellTool(),
        GlobSearchTool(),
        GrepSearchTool(),
        GitTool(),
        VerifyTool(),
        RepoMapTool(),
        TestRunnerTool(),
        CoverageTool(),
        SecurityScanTool(),
        ComplexityTool(),
        DepAuditTool(),
        GenerateTestsTool(),
        WebSearchTool(),
        WebFetchTool(),
        NotebookEditTool(),
        BackgroundCheckTool(),
        AskClarificationTool(),
    ]

    # Optional tools — register if their dependencies are available
    try:
        from godspeed.tools.image_read import ImageReadTool

        tools.append(ImageReadTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.pdf_read import PdfReadTool

        tools.append(PdfReadTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.github import GithubTool

        tools.append(GithubTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.diff_apply import DiffApplyTool

        tools.append(DiffApplyTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.batch_edit import BatchEditTool

        tools.append(BatchEditTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.session_history import SessionHistoryTool

        tools.append(SessionHistoryTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.tool_analytics import ToolAnalyticsTool

        tools.append(ToolAnalyticsTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.smart_complete import SmartCompleteTool

        tools.append(SmartCompleteTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.checkpoint import CheckpointTool

        tools.append(CheckpointTool())
    except ImportError:
        pass

    try:
        from godspeed.tools.workflow import WorkflowTool

        tools.append(WorkflowTool())
    except ImportError:
        pass

    for tool in tools:
        if allowed is not None and tool.name not in allowed:
            continue
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

    # Task tracking
    from godspeed.tools.tasks import TaskStore, TaskTool

    task_store = TaskStore()
    task_tool = TaskTool(task_store)
    registry.register(task_tool)
    risk_levels[task_tool.name] = task_tool.risk_level

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

    # Conversation logger (training data collection)
    conversation_logger = None
    if settings.log_conversations:
        from godspeed.training.conversation_logger import ConversationLogger

        training_dir = settings.global_dir / "training"
        conversation_logger = ConversationLogger(
            session_id=session_id,
            output_dir=training_dir,
        )
        logger.info("Conversation logging enabled output_dir=%s", training_dir)

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

    # LLM client with model routing
    from godspeed.llm.client import ModelRouter

    router = ModelRouter(routing=settings.routing) if settings.routing else None
    llm_client = LLMClient(
        model=effective_model,
        fallback_models=settings.fallback_models,
        router=router,
        thinking_budget=settings.thinking_budget,
        max_cost_usd=settings.max_cost_usd,
    )

    # Tool context — constructed after the LLM client so tools that need an
    # LLM (e.g. generate_tests) can invoke it via the context.
    tool_context = ToolContext(
        cwd=effective_project_dir,
        session_id=session_id,
        permissions=permission_engine,
        audit=audit_trail,
        llm_client=llm_client,  # type: ignore[arg-type]
    )

    # Kick off background codebase index if needed (non-blocking).
    from godspeed.context.auto_index import maybe_start_auto_index

    maybe_start_auto_index(effective_project_dir, settings.auto_index)

    # Conversation
    conversation = Conversation(
        system_prompt=system_prompt,
        model=effective_model,
        max_tokens=settings.max_context_tokens,
        compaction_threshold=settings.compaction_threshold,
        conversation_logger=conversation_logger,
    )

    # MCP server discovery
    from godspeed.mcp.client import MCPClient, MCPServerConfig, discover_mcp_servers
    from godspeed.mcp.tool_adapter import adapt_mcp_tools

    mcp_client = MCPClient()

    if settings.mcp_servers:
        for server_cfg in settings.mcp_servers:
            config = MCPServerConfig(
                name=server_cfg.get("name", "unknown"),
                command=server_cfg.get("command", ""),
                args=server_cfg.get("args", []),
                env=server_cfg.get("env", {}),
                transport=server_cfg.get("transport", "stdio"),
                url=server_cfg.get("url"),
                headers=server_cfg.get("headers"),
            )
            try:
                definitions = await mcp_client.connect(config)
                for tool in adapt_mcp_tools(definitions, mcp_client):
                    registry.register(tool)
                    risk_levels[tool.name] = tool.risk_level
                logger.info("MCP server %s: %d tools", config.name, len(definitions))
            except Exception as exc:
                logger.warning("MCP server %s failed: %s", config.name, exc)
    elif mcp_client.available:
        auto_servers = discover_mcp_servers(str(effective_project_dir))
        for config in auto_servers:
            try:
                definitions = await mcp_client.connect(config)
                for tool in adapt_mcp_tools(definitions, mcp_client):
                    registry.register(tool)
                    risk_levels[tool.name] = tool.risk_level
                logger.info("Auto-discovered MCP server %s: %d tools", config.name, len(definitions))
            except Exception as exc:
                logger.warning("Auto-discovered MCP server %s failed: %s", config.name, exc)

    # Sub-agent coordinator
    from godspeed.agent.coordinator import AgentCoordinator, SpawnAgentTool

    coordinator = AgentCoordinator(
        llm_client=llm_client,
        tool_registry=registry,
        tool_context=tool_context,
    )
    spawn_tool = SpawnAgentTool(coordinator)
    registry.register(spawn_tool)
    risk_levels[spawn_tool.name] = spawn_tool.risk_level

    # Codebase index (optional — requires chromadb)
    codebase_index = None
    from godspeed.context.codebase_index import CodebaseIndex

    codebase_index = CodebaseIndex(project_dir=effective_project_dir)
    if codebase_index.is_available:
        from godspeed.tools.code_search import CodeSearchTool

        code_search_tool = CodeSearchTool(codebase_index)
        registry.register(code_search_tool)
        risk_levels[code_search_tool.name] = code_search_tool.risk_level

        # Auto-reindex in background if stale
        if codebase_index.needs_reindex():
            logger.info("Codebase index is stale, rebuilding in background")
            asyncio.get_event_loop().create_task(codebase_index.build_index_async())

    # Discover skills
    from godspeed.skills.loader import discover_skills

    skill_dirs = [
        settings.global_dir / "skills",
        effective_project_dir / ".godspeed" / "skills",
    ]
    skills = discover_skills(skill_dirs)
    skill_completions = [(f"/{s.trigger}", s.description) for s in skills]

    # Hook executor
    hook_executor = None
    if settings.hooks:
        from godspeed.hooks.config import HookDefinition
        from godspeed.hooks.executor import HookExecutor

        hook_defs = [HookDefinition(**h) for h in settings.hooks]
        hook_executor = HookExecutor(
            hooks=hook_defs,
            cwd=effective_project_dir,
            session_id=session_id,
        )
        hook_executor.run_pre_session()

    # Launch TUI
    app = TUIApp(
        llm_client=llm_client,
        tool_registry=registry,
        tool_context=tool_context,
        conversation=conversation,
        permission_engine=permission_engine,
        audit_trail=audit_trail,
        session_id=session_id,
        skills=skills,
        extra_completions=skill_completions,
        hook_executor=hook_executor,
        task_store=task_store,
        codebase_index=codebase_index,
    )
    await app.run()

    # Close conversation logger
    if conversation_logger is not None:
        conversation_logger.close()

    # Post-session hooks
    if hook_executor is not None:
        hook_executor.run_post_session()


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

    # Auto-load env files so API keys in ~/.godspeed/.env.local (and the
    # project's .godspeed/.env.local) reach LiteLLM without requiring
    # per-shell env configuration. Shell env still wins — see
    # _load_env_files for the precedence rules.
    _load_env_files(project_dir=project_dir)

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

    from godspeed.tui.theme import brand

    c = RichConsole()
    c.print(brand(__version__))


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
    from godspeed.tui.theme import DIM, ERROR, SUCCESS

    c = RichConsole()
    effective_dir = audit_dir or (DEFAULT_GLOBAL_DIR / "audit")

    if not effective_dir.exists():
        c.print(f"[{ERROR}]Audit directory not found: {effective_dir}[/{ERROR}]")
        sys.exit(1)

    if session_id:
        # Verify single session
        trail = AuditTrail(log_dir=effective_dir, session_id=session_id)
        if not trail.log_path.exists():
            c.print(f"[{ERROR}]No audit log found for session: {session_id}[/{ERROR}]")
            sys.exit(1)
        is_valid, message = trail.verify_chain()
        if is_valid:
            c.print(f"[{SUCCESS}]VALID[/{SUCCESS}] -- {message}")
        else:
            c.print(f"[{ERROR}]BROKEN[/{ERROR}] -- {message}")
            sys.exit(1)
    else:
        # Verify all sessions
        found = False
        for log_file in sorted(effective_dir.glob("*.audit.jsonl")):
            found = True
            sid = log_file.stem.replace(".audit", "")
            trail = AuditTrail(log_dir=effective_dir, session_id=sid)
            is_valid, message = trail.verify_chain()
            status = f"[{SUCCESS}]VALID[/{SUCCESS}]" if is_valid else f"[{ERROR}]BROKEN[/{ERROR}]"
            c.print(f"  {status}  {sid[:12]}...  {message}")

        if not found:
            c.print(f"[{DIM}]No audit logs found.[/{DIM}]")


@main.command()
def init() -> None:
    """Set up Godspeed — create ~/.godspeed/ and default settings.yaml."""

    from rich.console import Console as RichConsole

    from godspeed.tui.theme import ACCENT, BOLD_PRIMARY, DIM, SUCCESS

    c = RichConsole()
    global_dir = DEFAULT_GLOBAL_DIR
    settings_path = global_dir / "settings.yaml"
    audit_dir = global_dir / "audit"

    global_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        c.print(f"  [{DIM}]Settings already exist:[/{DIM}] {settings_path}")
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
        c.print(f"  [{SUCCESS}]Created settings:[/{SUCCESS}] {settings_path}")

    c.print(f"  [{SUCCESS}]Audit directory:[/{SUCCESS}] {audit_dir}")
    c.print()
    c.print(f"  [{BOLD_PRIMARY}]Next steps:[/{BOLD_PRIMARY}]")
    c.print(f"    1. Install a local model: [{ACCENT}]ollama pull qwen3:4b[/{ACCENT}]")
    c.print(f"    2. Or set an API key:     [{ACCENT}]export ANTHROPIC_API_KEY=sk-...[/{ACCENT}]")
    c.print(f"    3. Edit your settings:    [{ACCENT}]{settings_path}[/{ACCENT}]")
    c.print(f"    4. Launch Godspeed:        [{ACCENT}]godspeed[/{ACCENT}]")


@main.command("run")
@click.argument("task", required=False, default="")
@click.option("--model", "-m", default="", help="Model to use.")
@click.option(
    "--project-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Project directory.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--auto-approve",
    type=click.Choice(["reads", "all", "none"]),
    default="reads",
    help="Auto-approve permission level (default: reads).",
)
@click.option("--max-iterations", type=int, default=50, help="Max agent loop iterations.")
@click.option(
    "--timeout",
    type=int,
    default=0,
    help="Wall-clock session timeout in seconds (0 = no limit).",
)
@click.option("--json-output", is_flag=True, help="Output result as JSON.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read the task from a file instead of the positional argument.",
)
def headless_run(
    task: str,
    model: str,
    project_dir: Path,
    verbose: bool,
    auto_approve: str,
    max_iterations: int,
    timeout: int,
    json_output: bool,
    prompt_file: Path | None,
) -> None:
    """Run a task non-interactively (headless/CI mode).

    Executes the agent loop without a TUI. Permissions are auto-approved
    based on --auto-approve level. Outputs the final response to stdout.

    Task input precedence:
        1. --prompt-file FILE          (read task from file)
        2. TASK positional argument    (passed as argument)
        3. stdin                       (when TASK is '-' or omitted and stdin is a pipe)

    Exit codes:
        0   success — model stopped with a final text response
        1   tool error — final response starts with "Error:"
        2   max iterations reached without the model stopping
        3   cost budget exceeded
        4   LLM provider failure (all fallbacks exhausted)
        5   invalid input (no task provided)
        6   wall-clock timeout (--timeout exceeded)
        130 keyboard interrupt (SIGINT)

    Examples:
        godspeed run "Fix the failing test in test_auth.py"
        godspeed run --prompt-file experiment.prompt --json-output
        cat task.md | godspeed run -
        godspeed run "Long running task" --timeout 1800
    """
    from godspeed.agent.result import ExitCode

    _setup_logging(verbose)

    resolved_task = _resolve_task_input(task, prompt_file)
    if not resolved_task:
        sys.stderr.write(
            "Error: No task provided. Pass a positional argument, use --prompt-file, "
            "or pipe via stdin with `godspeed run -`.\n"
        )
        sys.exit(ExitCode.INVALID_INPUT)

    try:
        exit_code = asyncio.run(
            _headless_run(
                resolved_task,
                model,
                project_dir,
                auto_approve,
                max_iterations,
                timeout,
                json_output,
            )
        )
        sys.exit(int(exit_code))
    except KeyboardInterrupt:
        sys.exit(ExitCode.INTERRUPTED)


def _resolve_task_input(task_arg: str, prompt_file: Path | None) -> str:
    """Resolve the task text from the available sources (in precedence order).

    Returns an empty string when no task is available — the caller treats
    that as INVALID_INPUT.
    """
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    if task_arg == "-":
        return sys.stdin.read().strip()
    if task_arg:
        return task_arg
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


async def _headless_run(
    task: str,
    model: str,
    project_dir: Path,
    auto_approve: str,
    max_iterations: int,
    timeout: int,
    json_output: bool,
) -> int:
    """Execute the headless agent loop.

    Returns an ExitCode-compatible integer:
        0 SUCCESS, 1 TOOL_ERROR, 2 MAX_ITERATIONS, 3 BUDGET_EXCEEDED,
        4 LLM_ERROR, 6 TIMEOUT. Never returns INTERRUPTED (130) — the
        caller translates KeyboardInterrupt.
    """
    import json as json_module

    from godspeed.agent.conversation import Conversation
    from godspeed.agent.loop import agent_loop
    from godspeed.agent.result import AgentMetrics, ExitCode, ExitReason
    from godspeed.agent.system_prompt import build_system_prompt
    from godspeed.audit.trail import AuditTrail
    from godspeed.config import GodspeedSettings
    from godspeed.context.project_instructions import load_project_instructions
    from godspeed.llm.client import LLMClient, ModelRouter
    from godspeed.security.permissions import ALLOW, PermissionDecision, PermissionEngine
    from godspeed.tools.base import RiskLevel, ToolContext

    overrides: dict = {}
    if model:
        overrides["model"] = model
    settings = GodspeedSettings(**overrides)

    effective_model = model or settings.model
    effective_project_dir = project_dir.resolve()
    session_id = str(uuid4())

    # Audit trail — headless must have an audit trail by default. A
    # security-first agent without a tamper-evident log in unattended mode
    # is the worst of both worlds.
    audit_dir = settings.global_dir / "audit"
    audit_trail = AuditTrail(log_dir=audit_dir, session_id=session_id)
    audit_trail.record(
        event_type="session_start",
        detail={
            "mode": "headless",
            "task": task[:500],  # clipped; full task goes to training logger
            "model": effective_model,
            "auto_approve": auto_approve,
        },
    )

    # Tools
    registry, risk_levels = _build_tool_registry()

    # Permission engine with headless auto-approve
    permission_engine = PermissionEngine(
        deny_patterns=settings.permissions.deny,
        allow_patterns=settings.permissions.allow,
        ask_patterns=settings.permissions.ask,
        tool_risk_levels=risk_levels,
    )

    # Wrap with headless auto-approve proxy
    class _HeadlessPermissionProxy:
        """Auto-approve permissions based on configured level."""

        def __init__(self, engine: PermissionEngine, level: str) -> None:
            self._engine = engine
            self._level = level

        def evaluate(self, tool_call: Any) -> PermissionDecision:
            decision = self._engine.evaluate(tool_call)
            if decision == "deny":
                return decision
            if self._level == "all":
                return PermissionDecision(ALLOW, "headless: auto-approved (all)")
            if self._level == "reads":
                tool_risk = risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
                if tool_risk == RiskLevel.READ_ONLY:
                    return PermissionDecision(ALLOW, "headless: auto-approved (reads)")
                return PermissionDecision(ALLOW, "headless: auto-approved (reads+low)")
            return decision

    # Auto-start Ollama if needed
    if effective_model.lower().startswith("ollama"):
        _ensure_ollama()

    project_instructions = load_project_instructions(
        effective_project_dir,
        settings.context.project_instructions,
    )
    system_prompt = build_system_prompt(
        tools=registry.list_tools(),
        project_instructions=project_instructions,
        cwd=effective_project_dir,
    )

    router = ModelRouter(routing=settings.routing) if settings.routing else None
    llm_client = LLMClient(
        model=effective_model,
        fallback_models=settings.fallback_models,
        router=router,
        thinking_budget=settings.thinking_budget,
        max_cost_usd=settings.max_cost_usd,
    )

    # Tool context — constructed after the LLM client so tools that need an
    # LLM (e.g. generate_tests) can invoke it via the context.
    tool_context = ToolContext(
        cwd=effective_project_dir,
        session_id=session_id,
        permissions=_HeadlessPermissionProxy(permission_engine, auto_approve),
        audit=audit_trail,
        llm_client=llm_client,  # type: ignore[arg-type]
    )

    # Kick off background codebase index if needed (non-blocking).
    from godspeed.context.auto_index import maybe_start_auto_index

    maybe_start_auto_index(effective_project_dir, settings.auto_index)

    # Conversation logger (training data collection)
    conversation_logger = None
    if settings.log_conversations:
        from godspeed.training.conversation_logger import ConversationLogger

        training_dir = settings.global_dir / "training"
        conversation_logger = ConversationLogger(
            session_id=session_id,
            output_dir=training_dir,
        )

    conversation = Conversation(
        system_prompt=system_prompt,
        model=effective_model,
        max_tokens=settings.max_context_tokens,
        compaction_threshold=settings.compaction_threshold,
        conversation_logger=conversation_logger,
    )

    # Callbacks — write to stderr for tool activity, keep stdout for result
    def on_tool_call(name: str, args: dict) -> None:
        logger.info("Tool call: %s", name)
        if not json_output:
            sys.stderr.write(f"[tool] {name}\n")

    def on_tool_result(name: str, result: Any) -> None:
        is_error = getattr(result, "is_error", False)
        if is_error:
            logger.warning("Tool error: %s", name)

    metrics = AgentMetrics()
    timed_out = False
    final_text: str

    # Run agent loop, optionally under a wall-clock timeout.
    loop_coro = agent_loop(
        user_input=task,
        conversation=conversation,
        llm_client=llm_client,
        tool_registry=registry,
        tool_context=tool_context,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        max_iterations=max_iterations,
        metrics=metrics,
    )
    try:
        if timeout > 0:
            final_text = await asyncio.wait_for(loop_coro, timeout=timeout)
        else:
            final_text = await loop_coro
    except TimeoutError:
        timed_out = True
        final_text = f"Error: Session exceeded wall-clock timeout of {timeout}s."
        metrics.finalize(ExitReason.TIMEOUT)

    # Resolve final exit code. TOOL_ERROR overrides STOPPED when the model's
    # final message starts with "Error:" (common pattern for tool failures
    # that the model surfaces rather than silently ignoring).
    exit_code = int(metrics.exit_code)
    if (
        not timed_out
        and metrics.exit_reason == ExitReason.STOPPED
        and final_text.startswith("Error:")
    ):
        exit_code = int(ExitCode.TOOL_ERROR)
        metrics.exit_reason = ExitReason.TOOL_ERROR

    audit_trail.record(
        event_type="session_end",
        detail={
            "exit_reason": metrics.exit_reason.value,
            "exit_code": exit_code,
            "iterations_used": metrics.iterations_used,
            "tool_call_count": metrics.tool_call_count,
            "tool_error_count": metrics.tool_error_count,
            "must_fix_injections": metrics.must_fix_injections,
            "duration_seconds": round(metrics.duration_seconds, 3),
            "cost_usd": round(llm_client.total_cost_usd, 6),
        },
        outcome="success" if exit_code == 0 else "error",
    )

    # Mirror the audit session_end into the training log so RL pipelines
    # can read exit_reason/exit_code without parsing the audit trail.
    if conversation_logger is not None:
        conversation_logger.log_session_end(
            exit_reason=metrics.exit_reason.value,
            exit_code=exit_code,
            iterations_used=metrics.iterations_used,
            tool_call_count=metrics.tool_call_count,
            tool_error_count=metrics.tool_error_count,
            must_fix_injections=metrics.must_fix_injections,
            duration_seconds=metrics.duration_seconds,
            cost_usd=llm_client.total_cost_usd,
        )
        conversation_logger.close()

    # Output result to stdout
    if json_output:
        output = {
            "task": task,
            "model": effective_model,
            "session_id": session_id,
            "response": final_text,
            "exit_reason": metrics.exit_reason.value,
            "exit_code": exit_code,
            "iterations_used": metrics.iterations_used,
            "tool_calls": [{"name": tc.name, "is_error": tc.is_error} for tc in metrics.tool_calls],
            "tool_call_count": metrics.tool_call_count,
            "tool_error_count": metrics.tool_error_count,
            "must_fix_injections": metrics.must_fix_injections,
            "duration_seconds": round(metrics.duration_seconds, 3),
            "input_tokens": llm_client.total_input_tokens,
            "output_tokens": llm_client.total_output_tokens,
            "cost_usd": round(llm_client.total_cost_usd, 6),
            "audit_log_path": str(audit_trail.log_path),
        }
        sys.stdout.write(json_module.dumps(output, indent=2) + "\n")
    else:
        sys.stdout.write(final_text + "\n")

    return exit_code


@main.command()
def models() -> None:
    """Show popular model options and how to configure them."""
    from rich.console import Console as RichConsole
    from rich.table import Table

    from godspeed.tui.theme import BOLD_PRIMARY, DIM, MUTED, SUCCESS, TABLE_BORDER

    c = RichConsole()

    table = Table(title="Popular Models", border_style=TABLE_BORDER, expand=False)
    table.add_column("Model", style=BOLD_PRIMARY)
    table.add_column("Provider", style=MUTED)
    table.add_column("Cost")
    table.add_column("API Key Env Var", style=MUTED)

    free = f"[{SUCCESS}]Free[/{SUCCESS}]"
    # Free local models
    table.add_row("ollama/qwen3:4b", "Ollama", free, "None (local)")
    table.add_row("ollama/qwen3:8b", "Ollama", free, "None (local)")
    table.add_row("ollama/qwen3:14b", "Ollama", free, "None (local, 8GB)")
    table.add_row(
        "ollama/qwen3-coder:latest",
        "Ollama",
        free,
        "None (local, 18GB, MoE 30B-A3B, coder-tuned)",
    )
    table.add_row("ollama/gemma4:e4b", "Ollama", free, "None (local)")
    table.add_row("ollama/llama3.3:8b", "Ollama", free, "None (local)")
    table.add_row("ollama/deepseek-r1:8b", "Ollama", free, "None (local)")
    table.add_row("ollama/mistral:7b", "Ollama", free, "None (local)")

    # Paid cloud models
    table.add_row("claude-sonnet-4-20250514", "Anthropic", "Paid", "ANTHROPIC_API_KEY")
    table.add_row("claude-opus-4-20250514", "Anthropic", "Paid", "ANTHROPIC_API_KEY")
    table.add_row("gpt-4o", "OpenAI", "Paid", "OPENAI_API_KEY")
    table.add_row("gpt-4o-mini", "OpenAI", "Paid", "OPENAI_API_KEY")
    table.add_row("gemini/gemini-2.0-flash", "Google", "Paid", "GEMINI_API_KEY")
    table.add_row("deepseek/deepseek-chat", "DeepSeek", "Paid", "DEEPSEEK_API_KEY")

    c.print(table)
    c.print()
    c.print(f"  [{BOLD_PRIMARY}]Switch models:[/{BOLD_PRIMARY}]")
    c.print(f"    [{DIM}]CLI flag:[/{DIM}]     godspeed -m claude-sonnet-4-20250514")
    c.print(f"    [{DIM}]Env var:[/{DIM}]      GODSPEED_MODEL=gpt-4o godspeed")
    c.print(f"    [{DIM}]Settings:[/{DIM}]     Edit ~/.godspeed/settings.yaml")
    c.print(f"    [{DIM}]At runtime:[/{DIM}]   /model claude-sonnet-4-20250514")


@main.command("export-training")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["openai", "chatml", "sharegpt"]),
    default="openai",
    help="Output format (default: openai).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("training_data.jsonl"),
    help="Output file path (default: training_data.jsonl).",
)
@click.option("--success-only", is_flag=True, help="Only export sessions with no tool errors.")
@click.option("--min-tools", type=int, default=1, help="Minimum tool calls per session.")
@click.option("--min-turns", type=int, default=2, help="Minimum user turns per session.")
@click.option("--max-sessions", type=int, default=0, help="Max sessions to export (0=all).")
@click.option(
    "--tools",
    type=str,
    default=None,
    help="Comma-separated tool names to filter by (e.g. 'file_read,file_edit').",
)
@click.option(
    "--max-tool-output",
    type=int,
    default=2000,
    help="Max chars per tool output (default: 2000).",
)
def export_training(
    fmt: str,
    output: Path,
    success_only: bool,
    min_tools: int,
    min_turns: int,
    max_sessions: int,
    tools: str | None,
    max_tool_output: int,
) -> None:
    """Export conversation logs to fine-tuning JSONL.

    Reads conversation logs from ~/.godspeed/training/ and converts them
    to the specified format for LLM fine-tuning.

    Examples:
        godspeed export-training --format openai --output training.jsonl
        godspeed export-training --format sharegpt --success-only --min-tools 3
        godspeed export-training --format chatml --tools file_read,file_edit
    """
    from rich.console import Console as RichConsole

    from godspeed.training.exporter import ExportFilters, TrainingExporter
    from godspeed.tui.theme import DIM, ERROR, SUCCESS

    c = RichConsole()
    training_dir = DEFAULT_GLOBAL_DIR / "training"

    if not training_dir.exists():
        c.print(f"[{ERROR}]No training data found at {training_dir}[/{ERROR}]")
        c.print(f"[{DIM}]Enable conversation logging in settings (log_conversations: true)[/{DIM}]")
        sys.exit(1)

    tool_list = [t.strip() for t in tools.split(",")] if tools else None

    filters = ExportFilters(
        min_tool_calls=min_tools,
        success_only=success_only,
        min_turns=min_turns,
        tools=tool_list,
        max_sessions=max_sessions,
    )

    exporter = TrainingExporter()
    stats = exporter.export_all(
        training_dir=training_dir,
        output_path=output,
        fmt=fmt,
        filters=filters,
        max_tool_output=max_tool_output,
    )

    c.print(f"[{SUCCESS}]Export complete[/{SUCCESS}]")
    c.print(f"  Format:   {fmt}")
    c.print(f"  Output:   {output}")
    c.print(f"  Sessions: {stats.sessions_exported}/{stats.sessions_scanned} exported")
    c.print(f"  Filtered: {stats.sessions_filtered}")
    c.print(f"  Messages: {stats.total_messages}")
    c.print(f"  Tool calls: {stats.total_tool_calls}")
    if stats.errors:
        c.print(f"  [{ERROR}]Errors: {len(stats.errors)}[/{ERROR}]")
        for err in stats.errors[:5]:
            c.print(f"    {err}")
