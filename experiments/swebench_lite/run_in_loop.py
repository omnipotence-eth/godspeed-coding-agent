"""In-process per-instance agent-in-loop runner for SWE-Bench Lite.

Unlike ``run.py``'s subprocess path (which spawns ``godspeed run`` per
instance), this module drives ``godspeed.agent.loop.agent_loop()`` in
the same Python process so a per-instance ``SWEBenchVerifyTool`` can be
registered on the tool registry. The agent then has an oracle it can
call mid-trajectory:

    agent edits -> calls swebench_verify_patch -> sees resolved=False + tail
                -> revises -> calls swebench_verify_patch again -> done

The setup copies the subset of ``godspeed.cli._headless_run`` that's
appropriate for batch benchmarking runs: Conversation + audit + permission
proxy + LLMClient + ModelRouter. It intentionally omits MCP servers,
coordinator / sub-agents, skills, codebase auto-indexing, and the
auto-commit helper — those add cost without helping the oracle loop.

Invoked from ``run.py`` via ``run_one(...)``, which returns a payload
shaped like the subprocess path's output so the outer loop is uniform.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

# Tool module is a sibling script file — make it importable either as
# ``experiments.swebench_lite.docker_test_tool`` (when imported as a
# package) or as a bare module (when run.py is invoked as a script).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from docker_test_tool import SWEBenchVerifyTool  # noqa: E402

logger = logging.getLogger(__name__)


async def _run_one_async(
    *,
    model: str,
    prompt: str,
    project_dir: Path,
    instance_id: str,
    split: str,
    timeout_s: int,
    verify_workdir: Path,
    max_iterations: int,
) -> dict:
    """Run one agent-in-loop session; return metrics payload.

    Mirrors the setup in ``godspeed.cli._headless_run`` but scoped to
    a single SWE-Bench instance with the oracle tool registered.
    """
    # Imports kept local so this module stays cheap to import from run.py
    from godspeed.agent.conversation import Conversation
    from godspeed.agent.loop import agent_loop
    from godspeed.agent.result import AgentMetrics, ExitReason
    from godspeed.agent.system_prompt import build_system_prompt
    from godspeed.audit.trail import AuditTrail
    from godspeed.cli import _build_tool_registry, _ensure_ollama
    from godspeed.config import GodspeedSettings
    from godspeed.llm.client import LLMClient, ModelRouter
    from godspeed.security.permissions import (
        ALLOW,
        PermissionDecision,
        PermissionEngine,
    )
    from godspeed.tools.base import ToolContext

    overrides: dict = {}
    if model:
        overrides["model"] = model
    settings = GodspeedSettings(**overrides)
    effective_model = model or settings.model
    session_id = str(uuid4())

    audit_dir = settings.global_dir / "audit"
    audit_trail = AuditTrail(log_dir=audit_dir, session_id=session_id)
    audit_trail.record(
        event_type="session_start",
        detail={
            "mode": "swebench_in_loop",
            "instance_id": instance_id,
            "split": split,
            "model": effective_model,
        },
    )

    registry, risk_levels = _build_tool_registry()

    # Register the per-instance oracle tool AFTER the default registry.
    verify_tool = SWEBenchVerifyTool(
        instance_id=instance_id,
        model_name=effective_model,
        workdir=verify_workdir.resolve(),
        split=split,
        timeout_s=timeout_s,
    )
    registry.register(verify_tool)
    risk_levels[verify_tool.name] = verify_tool.risk_level

    permission_engine = PermissionEngine(
        deny_patterns=settings.permissions.deny,
        allow_patterns=settings.permissions.allow,
        ask_patterns=settings.permissions.ask,
        tool_risk_levels=risk_levels,
    )

    class _AutoApproveAll:
        """Headless auto-approve for benchmark runs (equivalent to CLI --auto-approve all).

        Always respects an explicit ``deny`` decision from the engine; otherwise
        returns ALLOW so the agent can work without interactive prompts.
        """

        def evaluate(self, tool_call: Any) -> PermissionDecision:
            decision = permission_engine.evaluate(tool_call)
            if decision == "deny":
                return decision
            return PermissionDecision(ALLOW, "swebench_in_loop: auto-approved")

    if effective_model.lower().startswith("ollama"):
        _ensure_ollama()

    system_prompt = build_system_prompt(
        tools=registry.list_tools(),
        project_instructions="",  # SWE-Bench repos are fresh checkouts
        cwd=project_dir,
    )

    router = ModelRouter(routing=settings.routing) if settings.routing else None
    llm_client = LLMClient(
        model=effective_model,
        fallback_models=settings.fallback_models,
        router=router,
        thinking_budget=settings.thinking_budget,
        max_cost_usd=settings.max_cost_usd,
    )

    tool_context = ToolContext(
        cwd=project_dir,
        session_id=session_id,
        permissions=_AutoApproveAll(),
        audit=audit_trail,
        llm_client=llm_client,  # type: ignore[arg-type]
    )

    conversation = Conversation(
        system_prompt=system_prompt,
        model=effective_model,
        max_tokens=settings.max_context_tokens,
        compaction_threshold=settings.compaction_threshold,
    )

    verify_call_count = 0

    def on_tool_call(name: str, _args: dict) -> None:
        nonlocal verify_call_count
        if name == "swebench_verify_patch":
            verify_call_count += 1
            logger.info("[%s] swebench_verify_patch call #%d", instance_id, verify_call_count)

    metrics = AgentMetrics()
    timed_out = False
    final_text = ""

    loop_coro = agent_loop(
        user_input=prompt,
        conversation=conversation,
        llm_client=llm_client,
        tool_registry=registry,
        tool_context=tool_context,
        on_tool_call=on_tool_call,
        max_iterations=max_iterations,
        metrics=metrics,
    )
    try:
        if timeout_s > 0:
            final_text = await asyncio.wait_for(loop_coro, timeout=timeout_s)
        else:
            final_text = await loop_coro
    except TimeoutError:
        timed_out = True
        final_text = f"(session exceeded wall-clock timeout of {timeout_s}s)"
        metrics.finalize(ExitReason.TIMEOUT)

    audit_trail.record(
        event_type="session_end",
        detail={
            "mode": "swebench_in_loop",
            "instance_id": instance_id,
            "exit_reason": metrics.exit_reason.value,
            "iterations_used": metrics.iterations_used,
            "tool_call_count": metrics.tool_call_count,
            "tool_error_count": metrics.tool_error_count,
            "verify_call_count": verify_call_count,
            "duration_seconds": round(metrics.duration_seconds, 3),
            "cost_usd": round(llm_client.total_cost_usd, 6),
        },
        outcome="success" if not timed_out else "error",
    )

    return {
        "final_text": final_text,
        "exit_reason": metrics.exit_reason.value,
        "timed_out": timed_out,
        "iterations_used": metrics.iterations_used,
        "tool_call_count": metrics.tool_call_count,
        "tool_error_count": metrics.tool_error_count,
        "verify_call_count": verify_call_count,
        "duration_seconds": round(metrics.duration_seconds, 3),
        "cost_usd": round(llm_client.total_cost_usd, 6),
        "input_tokens": llm_client.total_input_tokens,
        "output_tokens": llm_client.total_output_tokens,
    }


def run_one(
    *,
    instance_id: str,
    model: str,
    prompt: str,
    project_dir: Path,
    split: str,
    timeout_s: int,
    verify_workdir: Path,
    max_iterations: int = 40,
) -> dict:
    """Synchronous entry point for ``run.py``'s main loop.

    Returns a payload dict shaped to be a drop-in replacement for
    ``_run_godspeed()``'s output: the caller reads ``_wall_s``,
    ``_shell_exit_code``, ``tool_call_count``, ``cost_usd``,
    ``output_tokens``. Extras (``verify_call_count``, ``iterations_used``,
    ``exit_reason``) are included for the metrics JSONL line.
    """
    t0 = time.monotonic()
    try:
        payload = asyncio.run(
            _run_one_async(
                model=model,
                prompt=prompt,
                project_dir=project_dir,
                instance_id=instance_id,
                split=split,
                timeout_s=timeout_s,
                verify_workdir=verify_workdir,
                max_iterations=max_iterations,
            )
        )
        payload["_shell_exit_code"] = 0 if not payload.get("timed_out") else 6
    except Exception as exc:
        logger.exception("run_in_loop.run_one failed for %s", instance_id)
        payload = {
            "_shell_exit_code": 4,  # matches ExitCode.LLM_ERROR
            "_error": str(exc)[:400],
            "final_text": "",
            "tool_call_count": 0,
            "tool_error_count": 0,
            "verify_call_count": 0,
            "iterations_used": 0,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_seconds": 0.0,
            "timed_out": False,
            "exit_reason": "llm_error",
        }
    payload["_wall_s"] = round(time.monotonic() - t0, 1)
    return payload
