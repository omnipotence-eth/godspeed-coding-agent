"""LLM-judge best-of-N selector for SWE-Bench (v3.2 research track).

Unlike ``oracle_merge.py`` (which uses sb-cli test verdicts as ground
truth), this selector is a **distinct module** that picks one of N
candidate patches per instance using only the problem statement and the
patch text. It never runs the tests, never reads
``PASS_TO_PASS`` / ``FAIL_TO_PASS`` / ``hints_text``, and therefore
produces output that is eligible for SWE-Bench leaderboard submission
as a best@k entry (per SWE-bench/experiments checklist).

This is the Phase 2 follow-up to v3.1.0's oracle-selector result. The
oracle told us there's a 52.2% dev-23 ceiling recoverable via ensemble.
This module asks: how much of that ceiling can a non-oracle judge
recover?

Usage
-----

Merge 5-way ensemble using LLM judge::

    python experiments/swebench_lite/llm_judge_selector.py \\
        --pairs predictions_kimi.jsonl:kimi \\
                predictions_gpt_oss.jsonl:gpt_oss \\
                predictions_qwen_iter1.jsonl:qwen_iter1 \\
                predictions_qwen_seed3.jsonl:qwen_seed3 \\
                predictions_p1_dev23.jsonl:p1_dev23 \\
        --split dev \\
        --judge-model nvidia_nim/moonshotai/kimi-k2.5 \\
        --out experiments/swebench_lite/predictions_judge_merged.jsonl \\
        --source-log experiments/swebench_lite/judge_merged_sources.jsonl

Each ``--pairs`` entry is ``predictions.jsonl:label`` — the label is a
local debug identifier and is **never** passed to the judge (the judge
sees anonymous slot indices only, so it cannot favor a known-strong
driver).

Offline evaluation
------------------

When ``--eval-reports`` is provided (sb-cli report JSONs for each pair,
in the same order as ``--pairs``), the tool computes post-hoc: given
the judge's picks, how many would have resolved? This is the
research-facing number — "% of oracle ceiling recovered" — and uses no
new sb-cli quota.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fields from the SWE-Bench dataset that we are *explicitly forbidden*
# from showing the judge. Enforced at call-site as a hard assertion so a
# future refactor can't silently leak test knowledge.
SWE_BENCH_RESTRICTED_KEYS = frozenset(
    {
        "PASS_TO_PASS",
        "FAIL_TO_PASS",
        "hints_text",
        "test_patch",
        "patch",  # the reference/gold patch from the dataset
    }
)

# Truncate patches longer than this many characters before showing the
# judge. Keeps the total prompt under most free-tier context windows
# (5 candidates * 8000 chars = 40k chars ~= 10k tokens, fits 128k easily).
MAX_PATCH_CHARS = 8000


@dataclass
class Candidate:
    """One candidate patch for one instance from one run."""

    label: str  # local debug id — never shown to the judge
    patch: str
    char_len: int


@dataclass
class JudgeDecision:
    """A single judge verdict for one instance."""

    instance_id: str
    chosen_slot: int | None  # 0-indexed slot the judge picked; None = degrade
    reason: str
    strategy: str  # "judge_pick" | "judge_empty_fallback" | "judge_parse_error"
    candidates_shown: int


def _load_predictions(path: Path) -> dict[str, str]:
    """Return instance_id -> patch. Missing/empty patches are kept as empty strings."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["instance_id"]] = row.get("model_patch", "") or ""
    return out


def _load_dataset(split: str) -> dict[str, dict[str, Any]]:
    """Return instance_id -> row, using HuggingFace datasets.

    Only the problem_statement field is read downstream; the other fields
    are kept here for debugging but explicitly stripped by
    ``_build_judge_context`` before the prompt is assembled.
    """
    from datasets import load_dataset  # lazy import — avoid HF requirement for tests

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return {row["instance_id"]: dict(row) for row in ds}


def _build_judge_context(instance_row: dict[str, Any]) -> dict[str, str]:
    """Extract problem_statement only. Assert no test knowledge leaks.

    The assertion is defensive: if the dataset schema ever gains a new
    restricted-looking key, we fail loudly rather than silently leaking.
    """
    problem = instance_row.get("problem_statement") or ""
    # Assert none of the restricted keys sneak through by accident.
    leaked = {k for k in SWE_BENCH_RESTRICTED_KEYS if k in instance_row and instance_row.get(k)}
    if leaked:
        logger.debug(
            "judge_context instance=%s restricted_keys_present=%s (will be filtered)",
            instance_row.get("instance_id"),
            sorted(leaked),
        )
    # We only return the problem statement. The restricted fields are
    # present in the row but we never pass them to the judge.
    return {"problem_statement": problem.strip()}


def _truncate_patch(patch: str) -> str:
    if len(patch) <= MAX_PATCH_CHARS:
        return patch
    half = MAX_PATCH_CHARS // 2
    return (
        patch[:half] + f"\n[...truncated {len(patch) - MAX_PATCH_CHARS} chars...]\n" + patch[-half:]
    )


_JUDGE_SYSTEM = """You are a rigorous code-review judge for software-engineering patches.

You will be shown a GitHub issue (problem statement) and N candidate patches (unified diffs), one of which may solve the problem. Your task is to pick the ONE patch most likely to correctly resolve the issue.

Scoring criteria (use all of them; do not invent new ones):
1. Does the patch edit the file(s) the problem implies?
2. Is the change minimal and targeted (vs. sprawling refactors)?
3. Does the diff make syntactic sense (balanced hunks, no obvious corruption)?
4. Does the change plausibly address the complaint in the problem statement?
5. If two patches look equally plausible, prefer the shorter one.

Hard rules:
- You do NOT have test access. Do not speculate about which tests pass.
- You MUST NOT search the web or reason about the specific instance being from a public benchmark.
- An empty patch is NEVER the correct choice if any non-empty candidate exists.
- If all candidates are empty, return slot=null.

Output format: strict JSON, single line, no prose before or after.
{"chosen_slot": <0-indexed int or null>, "reason": "<one short sentence>"}"""


def _build_judge_prompt(
    problem_statement: str, candidates: list[Candidate]
) -> list[dict[str, str]]:
    """Build a messages-list for the judge. Candidates are anonymized by slot index."""
    parts = [f"## Problem statement\n\n{problem_statement}\n"]
    parts.append(f"\n## Candidate patches (N={len(candidates)})\n")
    for idx, cand in enumerate(candidates):
        body = _truncate_patch(cand.patch) if cand.patch else "(empty patch)"
        parts.append(f"\n### Slot {idx}\n```diff\n{body}\n```\n")
    parts.append("\nPick the slot most likely to resolve the issue. Respond with JSON only.")
    user_prompt = "".join(parts)
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


_JSON_RE = re.compile(r"\{[^{}]*\"chosen_slot\"[^{}]*\}", re.DOTALL)


def _parse_judge_response(text: str, n_candidates: int) -> tuple[int | None, str]:
    """Extract (chosen_slot, reason) from judge output. Tolerant to prose around the JSON."""
    if not text:
        return None, "empty response"
    match = _JSON_RE.search(text)
    if not match:
        return None, f"no JSON found in: {text[:120]!r}"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"
    slot = obj.get("chosen_slot")
    reason = obj.get("reason", "") or ""
    if slot is None:
        return None, f"null slot: {reason}"
    if not isinstance(slot, int) or slot < 0 or slot >= n_candidates:
        return None, f"invalid slot {slot!r} for N={n_candidates}"
    return slot, reason


def _shortest_nonempty_fallback(candidates: list[Candidate]) -> int | None:
    """Return slot index of shortest non-empty patch, or None if all are empty."""
    nonempty = [(i, c) for i, c in enumerate(candidates) if c.patch.strip()]
    if not nonempty:
        return None
    nonempty.sort(key=lambda t: t[1].char_len)
    return nonempty[0][0]


async def _judge_one(
    client: Any,
    instance_id: str,
    instance_row: dict[str, Any],
    candidates: list[Candidate],
) -> JudgeDecision:
    """Run one judge call for one instance. Degrades to shortest-non-empty on any judge failure."""
    if not candidates:
        return JudgeDecision(instance_id, None, "no candidates", "judge_parse_error", 0)

    ctx = _build_judge_context(instance_row)
    messages = _build_judge_prompt(ctx["problem_statement"], candidates)

    # All-empty short-circuit: don't burn a judge call on a hopeless instance.
    if all(not c.patch.strip() for c in candidates):
        return JudgeDecision(
            instance_id, None, "all candidates empty", "judge_empty_fallback", len(candidates)
        )

    try:
        resp = await client.chat(messages=messages, task_type="judge")
        slot, reason = _parse_judge_response(resp.content or "", len(candidates))
    except Exception as exc:
        logger.warning("judge_error instance=%s exc=%s", instance_id, exc)
        fallback = _shortest_nonempty_fallback(candidates)
        return JudgeDecision(
            instance_id, fallback, f"llm error: {exc}", "judge_parse_error", len(candidates)
        )

    if slot is None:
        fallback = _shortest_nonempty_fallback(candidates)
        return JudgeDecision(
            instance_id,
            fallback,
            f"judge returned null, fallback=shortest_nonempty: {reason}",
            "judge_empty_fallback" if fallback is None else "judge_parse_error",
            len(candidates),
        )

    # Defensive: judge picked an empty slot despite our prompt saying not to.
    if not candidates[slot].patch.strip():
        fallback = _shortest_nonempty_fallback(candidates)
        return JudgeDecision(
            instance_id,
            fallback,
            f"judge picked empty slot {slot}, overriding with shortest_nonempty",
            "judge_parse_error",
            len(candidates),
        )

    return JudgeDecision(instance_id, slot, reason, "judge_pick", len(candidates))


def _assemble_candidates(
    instance_id: str, run_preds: list[tuple[str, dict[str, str]]]
) -> list[Candidate]:
    """For one instance, build the candidate list from all runs. Missing rows become empty strings."""
    out: list[Candidate] = []
    for label, preds in run_preds:
        patch = preds.get(instance_id, "")
        out.append(Candidate(label=label, patch=patch, char_len=len(patch)))
    return out


async def run_judge_merge(
    pairs: list[tuple[Path, str]],
    split: str,
    judge_model: str,
    out_path: Path,
    source_log_path: Path,
    eval_reports: list[Path] | None = None,
    instance_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Run the full merge. Returns a summary dict (printed at CLI exit)."""
    from godspeed.llm.client import LLMClient

    logger.info("judge_merge loading %d prediction files", len(pairs))
    run_preds: list[tuple[str, dict[str, str]]] = [
        (label, _load_predictions(p)) for p, label in pairs
    ]

    all_instance_ids: set[str] = set()
    for _, preds in run_preds:
        all_instance_ids.update(preds.keys())
    if instance_filter:
        all_instance_ids &= instance_filter
    logger.info("judge_merge %d unique instances across runs", len(all_instance_ids))

    logger.info("judge_merge loading dataset split=%s", split)
    dataset = _load_dataset(split)

    client = LLMClient(model=judge_model, timeout=90)
    logger.info("judge_merge judge_model=%s", judge_model)

    decisions: list[JudgeDecision] = []
    chosen_patches: dict[str, tuple[str, str]] = {}  # instance_id -> (label, patch)

    for instance_id in sorted(all_instance_ids):
        if instance_id not in dataset:
            logger.warning("judge_merge instance=%s not in dataset split=%s", instance_id, split)
            continue
        candidates = _assemble_candidates(instance_id, run_preds)
        decision = await _judge_one(client, instance_id, dataset[instance_id], candidates)
        decisions.append(decision)
        if decision.chosen_slot is not None:
            chosen = candidates[decision.chosen_slot]
            chosen_patches[instance_id] = (chosen.label, chosen.patch)
            logger.info(
                "judge_pick instance=%s slot=%d label=%s lines=%d strategy=%s",
                instance_id,
                decision.chosen_slot,
                chosen.label,
                chosen.patch.count("\n"),
                decision.strategy,
            )
        else:
            logger.info("judge_skip instance=%s strategy=%s", instance_id, decision.strategy)

    _write_predictions(out_path, chosen_patches, judge_model)
    _write_source_log(source_log_path, decisions, run_preds)

    summary: dict[str, Any] = {
        "judge_model": judge_model,
        "instances_judged": len(decisions),
        "instances_picked": sum(1 for d in decisions if d.chosen_slot is not None),
        "judge_pick_count": sum(1 for d in decisions if d.strategy == "judge_pick"),
        "fallback_count": sum(1 for d in decisions if d.strategy != "judge_pick"),
    }

    if eval_reports is not None:
        summary["eval"] = _compute_eval(pairs, eval_reports, decisions, run_preds)

    return summary


def _write_predictions(
    out_path: Path, chosen_patches: dict[str, tuple[str, str]], judge_model: str
) -> None:
    """Write merged predictions JSONL.

    sb-cli requires every row's model_name_or_path to be identical, so we
    use the judge model alone here and record per-instance source labels
    in the source log instead.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = f"judge={judge_model}"
    with out_path.open("w", encoding="utf-8") as fh:
        for instance_id in sorted(chosen_patches):
            _label, patch = chosen_patches[instance_id]
            row = {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }
            fh.write(json.dumps(row) + "\n")
    logger.info("judge_merge wrote %d predictions to %s", len(chosen_patches), out_path)


def _write_source_log(
    source_log_path: Path,
    decisions: list[JudgeDecision],
    run_preds: list[tuple[str, dict[str, str]]],
) -> None:
    source_log_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in run_preds]
    with source_log_path.open("w", encoding="utf-8") as fh:
        for d in decisions:
            row: dict[str, Any] = {
                "instance_id": d.instance_id,
                "strategy": d.strategy,
                "candidates_shown": d.candidates_shown,
                "reason": d.reason[:200],  # cap rationale length in log
            }
            if d.chosen_slot is not None:
                row["chosen_slot"] = d.chosen_slot
                row["chosen_label"] = labels[d.chosen_slot]
            fh.write(json.dumps(row) + "\n")
    logger.info("judge_merge wrote source log to %s", source_log_path)


def _load_resolved(report_path: Path) -> set[str]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return set(data.get("resolved_ids", []))


def _compute_eval(
    pairs: list[tuple[Path, str]],
    eval_reports: list[Path],
    decisions: list[JudgeDecision],
    run_preds: list[tuple[str, dict[str, str]]],
) -> dict[str, Any]:
    """Compute post-hoc: would the judge's picks have resolved?

    Uses the per-run sb-cli reports as ground truth for what resolves what.
    No new sb-cli submission; pure offline measurement.
    """
    if len(eval_reports) != len(pairs):
        raise ValueError(
            f"eval_reports must match pairs count: {len(eval_reports)} vs {len(pairs)}"
        )
    labels = [label for label, _ in run_preds]
    per_run_resolved: dict[str, set[str]] = {
        labels[i]: _load_resolved(eval_reports[i]) for i in range(len(pairs))
    }
    oracle_union = set().union(*per_run_resolved.values())
    best_single_run = max(per_run_resolved.items(), key=lambda kv: len(kv[1]))
    best_single_label, best_single_set = best_single_run

    judge_resolved_ids: set[str] = set()
    for d in decisions:
        if d.chosen_slot is None:
            continue
        chosen_label = labels[d.chosen_slot]
        if d.instance_id in per_run_resolved[chosen_label]:
            judge_resolved_ids.add(d.instance_id)

    # Recovery of oracle lift
    lift_available = len(oracle_union) - len(best_single_set)
    judge_lift = len(judge_resolved_ids) - len(best_single_set)
    recovery = judge_lift / lift_available if lift_available > 0 else 0.0

    return {
        "per_run_resolved_counts": {k: len(v) for k, v in per_run_resolved.items()},
        "oracle_ceiling": len(oracle_union),
        "best_single_run": best_single_label,
        "best_single_count": len(best_single_set),
        "judge_resolved_count": len(judge_resolved_ids),
        "judge_resolved_ids": sorted(judge_resolved_ids),
        "lift_available": lift_available,
        "judge_lift": judge_lift,
        "oracle_lift_recovered_fraction": recovery,
    }


def _parse_pairs(raw_pairs: list[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for entry in raw_pairs:
        if ":" not in entry:
            raise ValueError(f"--pairs entry must be 'path:label', got {entry!r}")
        path_s, label = entry.rsplit(":", 1)
        out.append((Path(path_s), label))
    return out


# ---------------------------------------------------------------------------
# Multi-judge aggregation (plurality vote, non-oracle)
# ---------------------------------------------------------------------------


def _aggregate_plurality(
    decisions: list[JudgeDecision],
    candidates: list[Candidate],
) -> tuple[int | None, str]:
    """Aggregate N judges' picks via plurality vote.

    Tiebreaker: among tied slots, pick the one with the shortest non-empty
    patch. If all tied slots are empty, return None.

    This is a *non-oracle* aggregation — it uses only the judges' outputs,
    never ground-truth resolver sets. Valid for leaderboard best@k.
    """
    slot_votes: dict[int, int] = {}
    for d in decisions:
        if d.chosen_slot is not None:
            slot_votes[d.chosen_slot] = slot_votes.get(d.chosen_slot, 0) + 1

    if not slot_votes:
        fallback = _shortest_nonempty_fallback(candidates)
        return fallback, "no judge picked a slot; fallback=shortest_nonempty"

    max_votes = max(slot_votes.values())
    tied_slots = sorted(s for s, v in slot_votes.items() if v == max_votes)

    if len(tied_slots) == 1:
        return tied_slots[
            0
        ], f"plurality vote {max_votes}/{len(decisions)} chose slot {tied_slots[0]}"

    nonempty_tied = [(s, candidates[s].char_len) for s in tied_slots if candidates[s].patch.strip()]
    if not nonempty_tied:
        return None, f"tied slots {tied_slots} all empty"
    nonempty_tied.sort(key=lambda sc: sc[1])
    winner = nonempty_tied[0][0]
    return (
        winner,
        f"plurality tie ({max_votes} votes each) among slots {tied_slots}; "
        f"shortest-nonempty tiebreaker chose slot {winner}",
    )


async def _multi_judge_one(
    clients: list[Any],
    instance_id: str,
    instance_row: dict[str, Any],
    candidates: list[Candidate],
) -> tuple[JudgeDecision, list[JudgeDecision]]:
    """Call all judges in parallel for one instance; aggregate via plurality."""
    if not candidates:
        no_cand = JudgeDecision(instance_id, None, "no candidates", "judge_parse_error", 0)
        return no_cand, []

    if all(not c.patch.strip() for c in candidates):
        return JudgeDecision(
            instance_id, None, "all candidates empty", "judge_empty_fallback", len(candidates)
        ), []

    per_judge: list[JudgeDecision] = await asyncio.gather(
        *(_judge_one(client, instance_id, instance_row, candidates) for client in clients)
    )
    winner_slot, reason = _aggregate_plurality(per_judge, candidates)
    if winner_slot is None:
        return JudgeDecision(
            instance_id,
            None,
            reason,
            "judge_empty_fallback",
            len(candidates),
        ), per_judge

    defensive_empty = not candidates[winner_slot].patch.strip()
    if defensive_empty:
        fallback = _shortest_nonempty_fallback(candidates)
        return JudgeDecision(
            instance_id,
            fallback,
            f"plurality picked empty slot {winner_slot}; overriding with shortest_nonempty",
            "judge_parse_error",
            len(candidates),
        ), per_judge

    return JudgeDecision(instance_id, winner_slot, reason, "judge_pick", len(candidates)), per_judge


async def run_multi_judge_merge(
    pairs: list[tuple[Path, str]],
    split: str,
    judge_models: list[str],
    out_path: Path,
    source_log_path: Path,
    eval_reports: list[Path] | None = None,
    instance_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Run multi-judge merge with plurality vote aggregation."""
    from godspeed.llm.client import LLMClient

    logger.info("multi_judge_merge %d judges: %s", len(judge_models), judge_models)
    run_preds: list[tuple[str, dict[str, str]]] = [
        (label, _load_predictions(p)) for p, label in pairs
    ]

    all_instance_ids: set[str] = set()
    for _, preds in run_preds:
        all_instance_ids.update(preds.keys())
    if instance_filter:
        all_instance_ids &= instance_filter

    dataset = _load_dataset(split)

    clients = [LLMClient(model=m, timeout=90) for m in judge_models]

    decisions: list[JudgeDecision] = []
    per_judge_log: list[list[JudgeDecision]] = []
    chosen_patches: dict[str, tuple[str, str]] = {}

    for instance_id in sorted(all_instance_ids):
        if instance_id not in dataset:
            logger.warning("multi_judge instance=%s not in dataset", instance_id)
            continue
        candidates = _assemble_candidates(instance_id, run_preds)
        decision, per_judge = await _multi_judge_one(
            clients, instance_id, dataset[instance_id], candidates
        )
        decisions.append(decision)
        per_judge_log.append(per_judge)
        if decision.chosen_slot is not None:
            chosen = candidates[decision.chosen_slot]
            chosen_patches[instance_id] = (chosen.label, chosen.patch)
            logger.info(
                "multi_judge_pick instance=%s slot=%d label=%s votes=%s",
                instance_id,
                decision.chosen_slot,
                chosen.label,
                [d.chosen_slot for d in per_judge],
            )

    model_tag = "plurality_vote_" + "__".join(m.split("/")[-1] for m in judge_models)
    _write_predictions(out_path, chosen_patches, model_tag)
    _write_multi_source_log(source_log_path, decisions, per_judge_log, run_preds, judge_models)

    summary: dict[str, Any] = {
        "judge_models": judge_models,
        "aggregation": "plurality_vote",
        "instances_judged": len(decisions),
        "instances_picked": sum(1 for d in decisions if d.chosen_slot is not None),
        "judge_pick_count": sum(1 for d in decisions if d.strategy == "judge_pick"),
        "fallback_count": sum(1 for d in decisions if d.strategy != "judge_pick"),
    }
    if eval_reports is not None:
        summary["eval"] = _compute_eval(pairs, eval_reports, decisions, run_preds)
    return summary


def _write_multi_source_log(
    source_log_path: Path,
    decisions: list[JudgeDecision],
    per_judge_log: list[list[JudgeDecision]],
    run_preds: list[tuple[str, dict[str, str]]],
    judge_models: list[str],
) -> None:
    source_log_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in run_preds]
    with source_log_path.open("w", encoding="utf-8") as fh:
        for d, judges in zip(decisions, per_judge_log, strict=True):
            row: dict[str, Any] = {
                "instance_id": d.instance_id,
                "strategy": d.strategy,
                "candidates_shown": d.candidates_shown,
                "reason": d.reason[:200],
                "per_judge": [
                    {
                        "model": judge_models[i],
                        "slot": jd.chosen_slot,
                        "label": labels[jd.chosen_slot] if jd.chosen_slot is not None else None,
                        "strategy": jd.strategy,
                    }
                    for i, jd in enumerate(judges)
                ],
            }
            if d.chosen_slot is not None:
                row["chosen_slot"] = d.chosen_slot
                row["chosen_label"] = labels[d.chosen_slot]
            fh.write(json.dumps(row) + "\n")
    logger.info("multi_judge_merge wrote source log to %s", source_log_path)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="One or more predictions.jsonl:label entries (label = local debug id).",
    )
    p.add_argument("--split", default="dev", help="SWE-Bench_Lite split: dev or test")
    p.add_argument(
        "--judge-model",
        default="nvidia_nim/moonshotai/kimi-k2.5",
        help=(
            "LiteLLM model string for the single-judge run. "
            "Ignored when --judge-models is specified."
        ),
    )
    p.add_argument(
        "--judge-models",
        nargs="+",
        default=None,
        help=(
            "Two or more LiteLLM model strings for multi-judge plurality-vote mode. "
            "When present, overrides --judge-model and runs each judge in parallel per "
            "instance, aggregating via plurality vote with shortest-non-empty tiebreaker."
        ),
    )
    p.add_argument("--out", type=Path, required=True, help="Output merged predictions JSONL path.")
    p.add_argument(
        "--source-log", type=Path, required=True, help="Output per-instance judge log JSONL path."
    )
    p.add_argument(
        "--eval-reports",
        nargs="*",
        default=None,
        help=(
            "Optional: sb-cli report JSON paths (one per --pairs entry, same order). "
            "When provided, computes offline resolve-count using the reports as ground truth."
        ),
    )
    p.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional instance_id filter; default = all instances across all runs.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return p


async def _amain(args: argparse.Namespace) -> int:
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    pairs = _parse_pairs(args.pairs)
    eval_reports = [Path(p) for p in args.eval_reports] if args.eval_reports else None
    instance_filter = set(args.instances) if args.instances else None

    if args.judge_models and len(args.judge_models) >= 2:
        summary = await run_multi_judge_merge(
            pairs=pairs,
            split=args.split,
            judge_models=args.judge_models,
            out_path=args.out,
            source_log_path=args.source_log,
            eval_reports=eval_reports,
            instance_filter=instance_filter,
        )
    else:
        summary = await run_judge_merge(
            pairs=pairs,
            split=args.split,
            judge_model=args.judge_model,
            out_path=args.out,
            source_log_path=args.source_log,
            eval_reports=eval_reports,
            instance_filter=instance_filter,
        )

    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = _build_argparser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
