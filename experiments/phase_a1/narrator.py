"""Stage C — LLM fills assistant-role narration around the REAL tool I/O.

After ``executor.execute_blueprint`` runs the blueprint on the sandbox we have
a deterministic transcript of tool calls and their real outputs. What the
session log lacks is the natural-language reasoning a human assistant would
produce around each call.

``narrate_session`` reads the session JSONL, groups assistant tool-calls with
their tool-result responses, and asks the secondary-tier LLM to produce:
  - a short ``pre_call`` reasoning string before each assistant tool-call turn
  - a short ``final`` summary after the last tool result (or as the whole
    response for ``no_tool`` samples)

The narration is written back into the session's assistant events via the
``ConversationLogger``'s append mechanism — we rewrite the JSONL in place,
preserving event order but populating the assistant ``content`` fields that
were empty after ``executor`` ran.

We DO NOT ask the narrator to invent tool calls or tool outputs — those come
from the real execution. The narrator only adds reasoning prose.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.phase_a1.executor import Blueprint
from experiments.phase_a1.providers import LLMResponse, ProviderRouter

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 2


_SYSTEM_PROMPT = """You are generating natural-language assistant content for
a training transcript of a coding agent. You will see a planned user intent,
the sequence of tool calls the agent issued, and the REAL tool outputs the
sandbox produced.

Your job: emit a JSON object with EXACTLY these keys:

{
  "pre_call": ["<reasoning before call 0>", "<before call 1>", ...],
  "final": "<final assistant response to the user after all tool results>"
}

Rules:
1. Output ONLY valid JSON, no fences, no prose before or after.
2. ``pre_call`` is a list with EXACTLY ONE STRING PER PLANNED TOOL CALL, IN
   ORDER. If the transcript has N tool-call events (N > 0), ``pre_call`` MUST
   be a list of length N — not N-1, not N+1. For each index i, the string is
   the brief natural-language reasoning the assistant would state immediately
   before invoking call i. Typical forms:
     "I'll read src/main.py to see the entry point."
     "Let me search for all callers of slugify first."
   1-2 sentences each. First person. Professional and concise. Do NOT restate
   the tool name verbatim or the arguments verbatim.
3. ``final`` is the assistant's natural-language summary to the user AFTER all
   tool results are in. 1-3 sentences. Reference concrete details you saw in
   the tool outputs. For error_recovery samples, briefly note what went wrong
   on the first attempt and how you corrected it.
4. For no_tool samples: ``pre_call`` MUST be an empty list and ``final`` MUST
   be the ENTIRE assistant answer to the user (2-5 sentences; answer conceptual
   or meta questions directly without inventing tool results).
5. Keep a consistent first-person voice across pre_call + final. Do not claim
   to have run tools you did not run; do not speculate beyond the tool outputs.
6. Never include raw tool output in ``final``; summarize or cite it naturally.

ANTI-HALLUCINATION RULES (violations cause the sample to be rejected):
  A. EVERY factual claim in ``final`` must be directly supported by the
     EXECUTED TRANSCRIPT in the user payload. If the tool output shows the
     PR is closed, DO NOT say it's open. If the tool output shows zero
     issues, DO NOT say there are three. If grep returned no matches, DO
     NOT invent matches.
  B. If a tool result was an ERROR or empty, SAY SO in ``final`` — "no
     results were found" / "the file does not exist" / "the request
     failed" — do not pretend the call succeeded.
  C. If the tool output is insufficient to answer the user's question,
     acknowledge that in ``final`` rather than fabricating a complete
     answer. It is better to say "I couldn't find X in the provided
     outputs" than to make up X.
  D. Do not reference files, functions, URLs, PR numbers, issue numbers,
     or commits that do NOT appear in the executed transcript.
"""


def _load_session(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_session(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_llm_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    return json.loads(stripped)


def _build_user_payload(blueprint: Blueprint, records: list[dict[str, Any]]) -> str:
    """Render the executed session into an inspectable prompt for the LLM."""
    lines: list[str] = []
    lines.append(f"user_intent: {blueprint.user_intent}")
    lines.append(f"category: {blueprint.category}")
    lines.append(f"primary_tool: {blueprint.primary_tool}")
    lines.append(f"expected_outcome: {blueprint.expected_outcome}")
    lines.append("")
    lines.append("EXECUTED TRANSCRIPT (tool calls and real tool outputs):")

    call_idx = 0
    for rec in records:
        role = rec.get("role")
        if role == "assistant":
            tool_calls = rec.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {})
                lines.append(
                    f"  [call {call_idx}] tool={fn.get('name')} args={fn.get('arguments')}"
                )
                call_idx += 1
        elif role == "tool":
            content = rec.get("content", "")
            is_err = rec.get("is_error", False)
            tag = "ERROR" if is_err else "ok"
            # truncate very long tool output for the LLM prompt
            excerpt = content if len(content) <= 800 else content[:800] + " ...[truncated]"
            lines.append(f"  [result {tag}] {excerpt}")
    lines.append("")
    lines.append(
        'Emit the JSON object with "pre_call" and "final" keys now.'
        " Match the rules in the system prompt exactly."
    )
    return "\n".join(lines)


def _expected_pre_call_count(blueprint: Blueprint) -> int:
    return 0 if blueprint.category == "no_tool" else len(blueprint.planned_calls)


def _validate_narration(data: dict[str, Any], blueprint: Blueprint) -> list[str]:
    errs: list[str] = []
    if not isinstance(data, dict):
        return ["top-level must be a JSON object"]
    if "pre_call" not in data or not isinstance(data["pre_call"], list):
        errs.append("missing pre_call list")
    if "final" not in data or not isinstance(data["final"], str) or not data["final"].strip():
        errs.append("missing non-empty final string")
    if errs:
        return errs

    expected = _expected_pre_call_count(blueprint)
    actual = len(data["pre_call"])
    if actual != expected:
        errs.append(f"pre_call length {actual} != expected {expected}")
    for i, s in enumerate(data["pre_call"]):
        if not isinstance(s, str) or not s.strip():
            errs.append(f"pre_call[{i}] must be a non-empty string")
    return errs


async def narrate_session(
    blueprint: Blueprint,
    session_path: Path,
    router: ProviderRouter,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> LLMResponse:
    """Ask the LLM for narration; rewrite the session JSONL with prose added.

    Returns the raw LLMResponse (for cost accounting in orchestrate.py).
    Raises on validation failure so orchestrate.py can discard the sample.

    Retries up to ``max_retries`` times on invalid JSON or validation failure,
    bumping temperature each attempt. Mirrors the blueprint retry pattern —
    the most common narrator slip (pre_call length off by one) resolves on
    retry far more often than it repeats.
    """
    records = _load_session(session_path)
    user = _build_user_payload(blueprint, records)
    last_error: str = ""
    narration: dict[str, Any] | None = None
    resp: LLMResponse | None = None

    for attempt in range(max_retries + 1):
        effective_temp = min(1.0, temperature + 0.1 * attempt)
        resp = await router.complete(
            tier="secondary",
            system=_SYSTEM_PROMPT,
            user=user,
            max_tokens=max_tokens,
            temperature=effective_temp,
            json_mode=True,
        )

        try:
            candidate = _parse_llm_json(resp.text)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON: {e}. text={resp.text[:200]!r}"
            logger.info(
                "narrator retry spec#%d attempt %d/%d: %s",
                blueprint.spec_index,
                attempt + 1,
                max_retries + 1,
                last_error,
            )
            continue

        errs = _validate_narration(candidate, blueprint)
        if not errs:
            narration = candidate
            break

        last_error = str(errs)
        logger.info(
            "narrator retry spec#%d attempt %d/%d: %s",
            blueprint.spec_index,
            attempt + 1,
            max_retries + 1,
            last_error,
        )

    if narration is None or resp is None:
        msg = (
            f"narration validation failed for spec#{blueprint.spec_index} "
            f"after {max_retries + 1} attempts: {last_error}"
        )
        raise ValueError(msg)

    # Inject content. For no_tool samples, append a single assistant record
    # with `final` as content. For tool-using samples, populate the empty
    # assistant records (those with tool_calls) with pre_call[i] content, and
    # append a trailing assistant record with `final`.
    pre_calls: list[str] = narration["pre_call"]
    final_text: str = narration["final"]

    updated: list[dict[str, Any]] = []
    assistant_idx = 0
    if blueprint.category == "no_tool":
        # Session records are: system, user (no assistant/tool events). Append
        # one final assistant turn.
        updated = list(records)
        sid = records[0].get("session_id", "") if records else ""
        updated.append(_assistant_record(final_text, blueprint.spec_index, session_id=sid))
    else:
        for rec in records:
            if rec.get("role") == "assistant" and rec.get("tool_calls"):
                rec = dict(rec)
                if assistant_idx < len(pre_calls):
                    rec["content"] = pre_calls[assistant_idx]
                assistant_idx += 1
            updated.append(rec)
        # Trailing final assistant message after the last tool result
        sid = records[0].get("session_id", "") if records else ""
        updated.append(_assistant_record(final_text, blueprint.spec_index, session_id=sid))

    _write_session(session_path, updated)
    return resp


def _assistant_record(content: str, spec_index: int | None, session_id: str) -> dict[str, Any]:

    return {
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "spec_index": spec_index,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import asyncio

    from experiments.phase_a1.executor import Blueprint, PlannedCall, execute_blueprint
    from experiments.phase_a1.providers import default_router
    from experiments.phase_a1.registry_builder import build_registry

    async def _demo() -> None:
        router = default_router()
        registry = build_registry()
        bp = Blueprint(
            user_intent="Read src/main.py and tell me what it does.",
            planned_calls=[PlannedCall("file_read", {"file_path": "src/main.py"})],
            expected_outcome="Shows src/main.py contents.",
            category="single_tool",
            primary_tool="file_read",
            spec_index=0,
            spec_seed=42,
        )
        out_dir = Path("experiments/phase_a1/data/_narrator_test")
        fixtures_dir = Path("experiments/phase_a1/fixtures")
        artifact = await execute_blueprint(
            bp, registry, output_dir=out_dir, fixtures_dir=fixtures_dir
        )
        resp = await narrate_session(bp, artifact.session_path, router)
        logger.info(
            "narrator: provider=%s model=%s in=%d out=%d latency=%.2fs",
            resp.provider,
            resp.model,
            resp.input_tokens,
            resp.output_tokens,
            resp.latency_s,
        )
        logger.info("session content after narration:")
        for line in artifact.session_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            logger.info(
                "  role=%s content=%r",
                d.get("role"),
                (d.get("content") or "")[:120],
            )

    asyncio.run(_demo())
