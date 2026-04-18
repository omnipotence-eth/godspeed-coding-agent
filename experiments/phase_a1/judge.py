"""Stage D — LLM judge for synthetic samples.

Given a final ``{messages, tools}`` training record, ask the judge-tier LLM
(GLM-4.5-Flash by default) to rate four dimensions on a 1-5 scale:

1. ``tool_correctness`` — Did the assistant pick the right tool for the user's
   intent? Would a careful engineer have made the same choice?
2. ``arg_correctness`` — Are the tool call arguments well-formed, complete, and
   appropriate for the tool's schema and the user's intent?
3. ``realism`` — Does the conversation look like something a real Godspeed user
   would send, and the real tool would emit? No boilerplate, no placeholders.
4. ``coherence`` — Do the assistant's textual responses follow from the actual
   tool outputs? No drift, hallucinated results, or contradictions.

A sample is ``passed`` when every dimension ≥ ``threshold`` (default 4).

Few-shot examples come from the Opus-hand-authored anchor set in
``data/anchor_opus_50.jsonl`` once it exists; judge still runs without them.

The judge is deliberately strict — better to drop a borderline sample than
train on noise. Expected drop rate on an honest run: 10-25%.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.phase_a1.providers import LLMResponse, ProviderRouter

logger = logging.getLogger(__name__)


DIMENSIONS: tuple[str, ...] = (
    "tool_correctness",
    "arg_correctness",
    "realism",
    "coherence",
)

DEFAULT_THRESHOLD: int = 4


@dataclass
class JudgeResult:
    """Per-sample verdict with 4 dimension scores."""

    scores: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    raw_text: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error or not self.scores:
            return False
        return all(self.scores.get(d, 0) >= DEFAULT_THRESHOLD for d in DIMENSIONS)

    def min_score(self) -> int:
        if not self.scores:
            return 0
        return min(self.scores.get(d, 0) for d in DIMENSIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "reason": self.reason,
            "passed": self.passed,
            "min_score": self.min_score(),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


_SYSTEM = (
    "You are a strict expert judge of tool-calling training samples for a "
    "coding agent named Godspeed. For each sample you will score four "
    "dimensions on an integer 1-5 scale and briefly explain the lowest score. "
    "Be demanding: 5 means publication-quality, 4 means good, 3 means "
    "noticeable issue, 2 means broken in meaningful ways, 1 means unusable. "
    "Output only valid JSON that matches the requested schema."
)


_RUBRIC = """Rubric (all four scores are integers 1-5):

tool_correctness: Did the assistant pick the right tool for the user's
intent? 5 = obvious correct choice; 4 = defensible choice; 3 = reasonable but
not the best tool; 2 = wrong tool that happens to work; 1 = irrelevant tool.

arg_correctness: Are the tool arguments well-formed, complete, and appropriate
for the schema and the user's intent? 5 = exactly right; 4 = minor nit (e.g.,
overly broad glob); 3 = missing a helpful optional field; 2 = missing a
required field or wrong type; 1 = argument shape contradicts the schema.

realism: Does the conversation read like a genuine Godspeed interaction?
5 = indistinguishable from a real user + real tool; 4 = plausible with minor
stiffness; 3 = noticeably synthetic; 2 = placeholder/stubby content; 1 = the
tool output is obviously fabricated or the user intent is absurd.

coherence: Does the assistant's narration follow from the real tool output?
5 = perfectly grounded; 4 = grounded with negligible paraphrase drift; 3 =
minor drift (extra detail not in output); 2 = claims content the tool did
not return; 1 = contradicts the tool output or ignores it entirely.

Required output shape (JSON, no commentary, no code fences):
{
  "tool_correctness": 1-5,
  "arg_correctness": 1-5,
  "realism": 1-5,
  "coherence": 1-5,
  "reason": "one sentence citing the lowest-scoring dimension and why"
}"""


def _format_sample_for_judge(record: dict[str, Any]) -> str:
    """Render a training record as a readable transcript for the judge.

    Condenses base64 payloads and very long tool outputs so we stay under the
    judge model's context window. Full fidelity isn't needed to rate quality;
    structure and coherence are.
    """
    messages = record.get("messages", [])
    tools = record.get("tools", [])

    lines: list[str] = []
    tool_names = sorted({t.get("function", {}).get("name", "?") for t in tools})
    lines.append(f"Available tools ({len(tool_names)}): {', '.join(tool_names)}")
    lines.append("")

    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if role == "system":
            lines.append(f"[system] {_truncate(content, 300)}")
        elif role == "user":
            lines.append(f"[user] {_truncate(content, 600)}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if content:
                lines.append(f"[assistant] {_truncate(content, 600)}")
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                args_s = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                lines.append(f"[assistant:tool_call] {name}({_truncate(args_s, 240)})")
        elif role == "tool":
            name = msg.get("name", "?")
            lines.append(f"[tool:{name}] {_truncate(content, 800)}")
        else:
            lines.append(f"[{role}] {_truncate(content, 400)}")

    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _render_user(record: dict[str, Any], few_shots: list[dict[str, Any]] | None) -> str:
    parts: list[str] = [_RUBRIC, ""]
    if few_shots:
        parts.append("Reference calibration examples (Opus-authored gold):")
        for fs in few_shots:
            parts.append("----")
            parts.append(_format_sample_for_judge(fs))
            parts.append(
                '(this is gold; would score {"tool_correctness":5,"arg_correctness":5,'
                '"realism":5,"coherence":5})'
            )
        parts.append("----")
        parts.append("")
    parts.append("Sample to judge:")
    parts.append(_format_sample_for_judge(record))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_scores(text: str) -> tuple[dict[str, int], str]:
    """Extract the 4 dim scores + reason from the judge's JSON response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"judge returned non-object: {type(data).__name__}")

    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        value = data.get(dim)
        if not isinstance(value, int):
            raise ValueError(f"missing/non-integer score for '{dim}': {value!r}")
        if not 1 <= value <= 5:
            raise ValueError(f"{dim} out of range 1-5: {value}")
        scores[dim] = value

    reason = str(data.get("reason", "")).strip()
    return scores, reason


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def judge_sample(
    record: dict[str, Any],
    router: ProviderRouter,
    *,
    few_shots: list[dict[str, Any]] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score one sample on the 4 rubric dimensions.

    Returns a :class:`JudgeResult`. Errors (invalid JSON, out-of-range scores,
    provider exhaustion) populate ``result.error`` and set ``passed=False``.
    Does not raise on judge failure — the caller decides whether to retry
    or drop the sample.
    """
    user = _render_user(record, few_shots)
    try:
        resp: LLMResponse = await router.complete(
            tier="judge",
            system=_SYSTEM,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    except Exception as e:
        logger.warning("judge provider call failed: %s", e, exc_info=True)
        return JudgeResult(error=f"provider_error: {e}")

    try:
        scores, reason = _parse_scores(resp.text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("judge response parse failed: %s | raw=%r", e, resp.text[:300])
        return JudgeResult(
            error=f"parse_error: {e}",
            raw_text=resp.text,
        )

    return JudgeResult(scores=scores, reason=reason, raw_text=resp.text)


def load_few_shots(path: Path, *, limit: int = 3) -> list[dict[str, Any]]:
    """Load up to ``limit`` anchor samples for judge calibration."""
    if not path.exists():
        return []
    shots: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            shots.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(shots) >= limit:
            break
    return shots


# ---------------------------------------------------------------------------
# CLI — judge every record in a JSONL and emit a sidecar verdict file
# ---------------------------------------------------------------------------


async def _judge_file(
    input_path: Path,
    output_path: Path,
    *,
    anchor_path: Path | None,
    limit: int | None,
) -> None:
    from experiments.phase_a1.providers import default_router

    router = default_router()
    few_shots = load_few_shots(anchor_path) if anchor_path else []

    records: list[dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    if limit is not None:
        records = records[:limit]

    kept = dropped = errored = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for i, rec in enumerate(records):
            verdict = await judge_sample(rec, router, few_shots=few_shots)
            line = json.dumps(
                {"index": i, **verdict.to_dict()},
                ensure_ascii=False,
            )
            fp.write(line + "\n")
            if verdict.error:
                errored += 1
            elif verdict.passed:
                kept += 1
            else:
                dropped += 1
            logger.info(
                "#%04d %-8s min=%d reason=%s",
                i,
                "pass" if verdict.passed else ("err" if verdict.error else "drop"),
                verdict.min_score(),
                verdict.reason[:80] or verdict.error or "",
            )

    logger.info(
        "judge complete  kept=%d dropped=%d errored=%d  out=%s",
        kept,
        dropped,
        errored,
        output_path,
    )


if __name__ == "__main__":
    import argparse
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Judge phase-A1 JSONL samples.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/phase_a1/data/phase_a1_smoke.jsonl"),
        help="JSONL of {messages, tools} records to judge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase_a1/data/judge_verdicts.jsonl"),
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path("experiments/phase_a1/data/anchor_opus_50.jsonl"),
        help="Anchor JSONL for few-shot calibration (optional, skipped if missing).",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(
        _judge_file(
            args.input,
            args.output,
            anchor_path=args.anchor if args.anchor.exists() else None,
            limit=args.limit,
        )
    )
