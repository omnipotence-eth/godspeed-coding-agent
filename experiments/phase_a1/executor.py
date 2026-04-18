"""Stage B — execute a blueprint against real Godspeed tools on a tmp sandbox.

For each blueprint:
  1. Create a ``tempfile.TemporaryDirectory`` with a seeded project (README,
     a few Python files, ``git init``) so tools like ``glob_search``,
     ``grep_search``, ``git``, ``repo_map`` have realistic material to work on.
  2. Build a permissive ``PermissionEngine`` (fnmatch ``*``) and a
     ``ToolContext`` pointing at the sandbox.
  3. For each ``(tool_name, args)`` in the blueprint:
       - if tool is sandbox-safe → ``registry.dispatch(ToolCall, ctx)``
       - otherwise (web_search / web_fetch / github / pdf_read / image_read /
         code_search / spawn_agent) → fixture-backed ``ToolResult`` picked
         deterministically by hash(tool_name, args)
  4. Stream user/assistant/tool events into a ``ConversationLogger``.

The logger persists to ``session_dir/{session_id}.conversation.jsonl``.
``emit.py`` later converts that to the final ``{messages, tools}`` format via
``TrainingExporter.export_session(fmt="openai")``.

Safety:
  - Each dispatch wrapped in ``asyncio.wait_for(timeout=15.0)`` — a wedged
    shell/test_runner won't hang the pipeline.
  - ``PermissionEngine`` still fires its built-in dangerous-command detection
    for shell, so ``rm -rf /``-class calls fail loudly instead of executing.
  - Fixture dispatch never touches the network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.phase_a1.registry_builder import FIXTURE_BACKED_TOOLS
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import RiskLevel, ToolCall, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry
from godspeed.training.conversation_logger import ConversationLogger

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S: float = 15.0


@dataclass
class PlannedCall:
    """One step in a blueprint: which tool to invoke with which args."""

    tool_name: str
    arguments: dict[str, Any]

    def to_tool_call(self, call_id: str) -> ToolCall:
        return ToolCall(
            tool_name=self.tool_name,
            arguments=self.arguments,
            call_id=call_id,
        )


@dataclass
class Blueprint:
    """LLM-planned sample blueprint (output of Stage A)."""

    user_intent: str
    planned_calls: list[PlannedCall]
    expected_outcome: str
    category: str
    primary_tool: str
    spec_index: int
    spec_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_intent": self.user_intent,
            "planned_calls": [dataclasses.asdict(p) for p in self.planned_calls],
            "expected_outcome": self.expected_outcome,
            "category": self.category,
            "primary_tool": self.primary_tool,
            "spec_index": self.spec_index,
            "spec_seed": self.spec_seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Blueprint:
        return cls(
            user_intent=d["user_intent"],
            planned_calls=[PlannedCall(**p) for p in d.get("planned_calls", [])],
            expected_outcome=d.get("expected_outcome", ""),
            category=d["category"],
            primary_tool=d["primary_tool"],
            spec_index=int(d["spec_index"]),
            spec_seed=int(d["spec_seed"]),
        )


@dataclass
class ExecutedStep:
    """One tool call's resolved outcome."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    output: str
    is_error: bool
    error: str | None = None
    source: str = "real"  # "real" | "fixture" | "error"


@dataclass
class SessionArtifact:
    """What executor produces per blueprint, consumed by emit.py."""

    session_id: str
    session_path: Path
    blueprint: Blueprint
    steps: list[ExecutedStep] = field(default_factory=list)
    sandbox_dir: Path | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


_FIXTURE_CACHE: dict[str, list[str]] = {}


def _load_fixtures(tool_name: str, fixtures_dir: Path) -> list[str]:
    """Return the fixture pool for ``tool_name`` as a list of output strings.

    Expects ``fixtures_dir/{tool_name}.json`` to be a JSON array of
    ``{"args_match": {...optional...}, "output": "..."}`` objects OR a plain
    JSON array of strings. Missing files yield a generic stub.
    """
    cached = _FIXTURE_CACHE.get(tool_name)
    if cached is not None:
        return cached

    path = fixtures_dir / f"{tool_name}.json"
    if not path.exists():
        stub = [
            f"[{tool_name} fixture placeholder] result_id={i} "
            f"(provide real fixture in fixtures/{tool_name}.json)"
            for i in range(3)
        ]
        _FIXTURE_CACHE[tool_name] = stub
        return stub

    raw = json.loads(path.read_text(encoding="utf-8"))
    pool: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                pool.append(item)
            elif isinstance(item, dict) and "output" in item:
                pool.append(str(item["output"]))
    if not pool:
        pool = [f"[empty fixture for {tool_name}]"]
    _FIXTURE_CACHE[tool_name] = pool
    return pool


def _pick_fixture(tool_name: str, arguments: dict[str, Any], fixtures_dir: Path) -> str:
    pool = _load_fixtures(tool_name, fixtures_dir)
    key = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True, default=str)
    digest = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return pool[digest % len(pool)]


# ---------------------------------------------------------------------------
# Sandbox setup
# ---------------------------------------------------------------------------


_SEED_FILES: dict[str, str] = {
    "README.md": (
        "# sandbox-project\n\nSynthetic project used by Godspeed Phase A1 "
        "data generation. Files here are seeded for realistic tool output.\n"
    ),
    "src/__init__.py": "",
    "src/main.py": (
        '"""Main entry point."""\n\n'
        "from __future__ import annotations\n\n"
        "import logging\n\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "def greet(name: str) -> str:\n"
        '    return f"hello {name}"\n\n\n'
        'if __name__ == "__main__":\n'
        '    logger.info(greet("world"))\n'
    ),
    "src/utils.py": (
        '"""Utility helpers."""\n\n'
        "from __future__ import annotations\n\n\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n\n"
        "def slugify(s: str) -> str:\n"
        '    return s.strip().lower().replace(" ", "-")\n'
    ),
    "tests/__init__.py": "",
    "tests/test_main.py": (
        "from src.main import greet\n\n\n"
        "def test_greet():\n"
        '    assert greet("world") == "hello world"\n'
    ),
    "tests/test_utils.py": (
        "from src.utils import add, slugify\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n\n"
        "def test_slugify():\n"
        '    assert slugify("Hello World") == "hello-world"\n'
    ),
    "pyproject.toml": (
        '[project]\nname = "sandbox-project"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
    ),
    ".gitignore": "__pycache__/\n*.pyc\n.venv/\ndist/\n",
}


def _seed_sandbox(root: Path) -> None:
    """Populate ``root`` with files and a git repo."""
    for rel, content in _SEED_FILES.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")

    try:
        subprocess.run(
            ["git", "init", "-q", "-b", "main"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=5,
        )
        subprocess.run(
            ["git", "-c", "user.email=ci@local", "-c", "user.name=CI", "add", "."],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=5,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=ci@local",
                "-c",
                "user.name=CI",
                "commit",
                "-q",
                "-m",
                "seed",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("git seed failed (continuing without vcs): %s", e)


def _build_permission_engine(registry: ToolRegistry) -> PermissionEngine:
    """Fully-permissive engine for sandbox generation.

    Dangerous-shell detection inside ``PermissionEngine.evaluate`` is still
    active — it's not a permission rule, it's a hard guard. That's the one
    safety floor we keep.
    """
    risk_levels = {t.name: t.risk_level for t in registry.list_tools()}
    # Ensure DESTRUCTIVE tools also get through; we're executing in a tmp dir
    # so no production state is at risk.
    for name, rl in list(risk_levels.items()):
        if rl == RiskLevel.DESTRUCTIVE:
            risk_levels[name] = RiskLevel.LOW
    return PermissionEngine(
        allow_patterns=["*"],
        tool_risk_levels=risk_levels,
    )


# ---------------------------------------------------------------------------
# Blueprint execution
# ---------------------------------------------------------------------------


async def execute_blueprint(
    blueprint: Blueprint,
    registry: ToolRegistry,
    *,
    output_dir: Path,
    fixtures_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SessionArtifact:
    """Run one blueprint end-to-end and return the captured session artifact.

    For ``no_tool`` category samples the blueprint has zero planned calls;
    only user/assistant messages are logged.
    """
    session_id = f"a1-{blueprint.spec_index:06d}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = output_dir / f"{session_id}.conversation.jsonl"
    conv = ConversationLogger(session_id=session_id, output_dir=output_dir)

    # Canonical Godspeed-style system prompt. Executor does not synthesize
    # assistant prose; narrator.py does. For now we log a minimal system +
    # user message and then each tool call / result pair.
    conv.log_system(_SYSTEM_PROMPT)
    conv.log_user(blueprint.user_intent)

    artifact = SessionArtifact(
        session_id=session_id,
        session_path=session_path,
        blueprint=blueprint,
    )

    with tempfile.TemporaryDirectory(prefix="a1-sbx-") as tmp:
        sandbox = Path(tmp)
        _seed_sandbox(sandbox)
        artifact.sandbox_dir = sandbox

        permissions = _build_permission_engine(registry)
        ctx = ToolContext(
            cwd=sandbox,
            session_id=session_id,
            permissions=permissions,
            audit=None,
            llm_client=None,
        )

        for i, planned in enumerate(blueprint.planned_calls):
            call_id = f"call_{i:02d}"
            tool_call = planned.to_tool_call(call_id)

            # Log the assistant's tool-call request. Narrator may later wrap
            # this with reasoning text; logger accepts either shape.
            conv.log_assistant(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": planned.tool_name,
                            "arguments": json.dumps(planned.arguments, ensure_ascii=False),
                        },
                    }
                ],
            )

            if planned.tool_name in FIXTURE_BACKED_TOOLS:
                output = _pick_fixture(planned.tool_name, planned.arguments, fixtures_dir)
                step = ExecutedStep(
                    tool_name=planned.tool_name,
                    arguments=planned.arguments,
                    call_id=call_id,
                    output=output,
                    is_error=False,
                    source="fixture",
                )
            else:
                try:
                    result: ToolResult = await asyncio.wait_for(
                        registry.dispatch(tool_call, ctx), timeout=timeout_s
                    )
                    step = ExecutedStep(
                        tool_name=planned.tool_name,
                        arguments=planned.arguments,
                        call_id=call_id,
                        output=result.output or "",
                        is_error=bool(result.is_error),
                        error=result.error,
                        source="real",
                    )
                except TimeoutError:
                    step = ExecutedStep(
                        tool_name=planned.tool_name,
                        arguments=planned.arguments,
                        call_id=call_id,
                        output="",
                        is_error=True,
                        error=f"tool timed out after {timeout_s}s",
                        source="error",
                    )
                except Exception as e:
                    logger.warning(
                        "tool dispatch failed tool=%s: %s",
                        planned.tool_name,
                        e,
                        exc_info=True,
                    )
                    step = ExecutedStep(
                        tool_name=planned.tool_name,
                        arguments=planned.arguments,
                        call_id=call_id,
                        output="",
                        is_error=True,
                        error=f"dispatch error: {e}",
                        source="error",
                    )

            conv.log_tool_result(
                tool_call_id=call_id,
                tool_name=planned.tool_name,
                content=step.output if not step.is_error else (step.error or "unknown error"),
                is_error=step.is_error,
            )
            artifact.steps.append(step)

    return artifact


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are Godspeed, a security-first open-source coding agent. "
    "You have access to a set of tools for reading, writing, searching, "
    "executing code, managing version control, and browsing the web. "
    "Use the tools when the user's request requires inspecting or modifying "
    "their project. For conceptual or conversational questions, answer "
    "directly without calling a tool."
)


# ---------------------------------------------------------------------------
# Convenience / self-test
# ---------------------------------------------------------------------------


async def _self_test() -> int:
    """Run a tiny blueprint end-to-end to validate plumbing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from experiments.phase_a1.registry_builder import build_registry

    registry = build_registry()
    blueprint = Blueprint(
        user_intent="Read the main entry point of this project.",
        planned_calls=[
            PlannedCall(tool_name="file_read", arguments={"file_path": "src/main.py"}),
        ],
        expected_outcome="Shows src/main.py contents with the greet() function.",
        category="single_tool",
        primary_tool="file_read",
        spec_index=0,
        spec_seed=42,
    )
    out_dir = Path("experiments/phase_a1/data/_selftest_sessions")
    fixtures_dir = Path("experiments/phase_a1/fixtures")
    artifact = await execute_blueprint(
        blueprint, registry, output_dir=out_dir, fixtures_dir=fixtures_dir
    )
    logger.info("session=%s steps=%d", artifact.session_id, len(artifact.steps))
    for s in artifact.steps:
        logger.info(
            "  tool=%s source=%s is_error=%s output_len=%d",
            s.tool_name,
            s.source,
            s.is_error,
            len(s.output),
        )
    logger.info("session_path=%s exists=%s", artifact.session_path, artifact.session_path.exists())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(_self_test()))
