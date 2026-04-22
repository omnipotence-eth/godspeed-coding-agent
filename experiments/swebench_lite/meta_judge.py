"""Meta-judge: resolve 2-judge disagreements with a third LLM that sees
both rationales + both candidate patches.

Rationale
---------

v3.2 + v3.3 established:
- Solo judges (Kimi K2.5, GPT-OSS-120B) each hit 10/23 = 43.5%.
- They agree on 19/23 instances; disagree on 4.
- Plurality vote on disagreements guesses blindly (shortest-non-empty
  tiebreak) and regresses to 9/23.
- Apply-check doesn't break ties — all contested candidates apply.

This module asks: on the disagreement instances only, can a 3rd LLM
call — given both judges' picks + rationales + both candidate patches
+ the problem statement — pick the better one?

It is non-oracle: the meta-judge sees only the same problem_statement
the primary judges saw, plus the primary judges' rationales. No test
access, no `PASS_TO_PASS` / `FAIL_TO_PASS` / `hints_text` / `patch`.

Cost is small: only disagreement instances get a meta-judge call. On
our 23-instance set with 2-judge disagreement count = 4, that's ≤4
meta-judge calls.

Usage
-----

    python experiments/swebench_lite/meta_judge.py \\
        --judge-sources \\
            experiments/swebench_lite/judge_merged_5way_sources.jsonl:kimi_k2_5 \\
            experiments/swebench_lite/judge_gpt_oss_5way_sources.jsonl:gpt_oss_120b \\
        --pairs \\
            experiments/swebench_lite/predictions_e1_kimi.jsonl:e1_kimi \\
            ... \\
        --eval-reports \\
            experiments/swebench_lite/reports/...kimi-k2_5.json \\
            ... \\
        --split dev \\
        --meta-judge-model nvidia_nim/moonshotai/kimi-k2.5 \\
        --out experiments/swebench_lite/meta_judge_picks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_META_JUDGE_SYSTEM = """You are a tiebreaker for a panel of code-review judges.

Each judge has picked a patch for the same GitHub issue. They disagree. You
will see the problem, each judge's choice, each judge's rationale, and the
full diffs of their chosen patches. Your job: pick the ONE patch most likely
to correctly fix the issue.

Scoring criteria (use all; invent none):
1. Which patch edits the file(s) the problem implies?
2. Which is most minimal and targeted?
3. Which makes syntactic sense (balanced hunks, no obvious corruption)?
4. Which most directly addresses the specific complaint in the problem?
5. Weigh the judges' rationales — when one judge gives a specific
   code-level reason, that's stronger than an abstract preference.

Hard rules:
- You do NOT have test access. Do not speculate about tests.
- You MUST NOT search the web or reason about this being from a public benchmark.
- Pick exactly one of the presented patches.

Output: strict JSON, one line, no prose before or after.
{"chosen_slot": <0-indexed int among the presented options>, "reason": "<one short sentence>"}"""


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_predictions(path: Path) -> dict[str, str]:
    return {r["instance_id"]: r.get("model_patch", "") or "" for r in _load_jsonl(path)}


def _load_problem_statements(split: str) -> dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return {row["instance_id"]: row.get("problem_statement", "") or "" for row in ds}


def _truncate(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... truncated {len(text) - limit} chars ...]\n" + text[-half:]


def _build_meta_prompt(
    problem: str,
    judge_a_name: str,
    judge_a_reason: str,
    patch_a: str,
    judge_b_name: str,
    judge_b_reason: str,
    patch_b: str,
) -> list[dict[str, str]]:
    body = (
        f"## Problem statement\n\n{problem.strip()}\n\n"
        f"## Option 0 — preferred by judge `{judge_a_name}`\n\n"
        f"Rationale: {judge_a_reason}\n\n"
        f"```diff\n{_truncate(patch_a)}\n```\n\n"
        f"## Option 1 — preferred by judge `{judge_b_name}`\n\n"
        f"Rationale: {judge_b_reason}\n\n"
        f"```diff\n{_truncate(patch_b)}\n```\n\n"
        "Pick slot 0 or slot 1. JSON only.\n"
    )
    return [
        {"role": "system", "content": _META_JUDGE_SYSTEM},
        {"role": "user", "content": body},
    ]


async def _meta_judge_one(
    client: Any,
    instance_id: str,
    problem: str,
    judge_a_name: str,
    judge_a_reason: str,
    patch_a: str,
    judge_b_name: str,
    judge_b_reason: str,
    patch_b: str,
) -> tuple[int | None, str]:
    """Call meta-judge for one disagreement case. Returns (0 | 1 | None, reason)."""
    messages = _build_meta_prompt(
        problem, judge_a_name, judge_a_reason, patch_a, judge_b_name, judge_b_reason, patch_b
    )
    try:
        resp = await client.chat(messages=messages, task_type="meta_judge")
        text = resp.content or ""
    except Exception as exc:
        return None, f"llm error: {exc}"
    # Simple JSON extraction
    import re

    match = re.search(r'\{[^{}]*"chosen_slot"[^{}]*\}', text, re.DOTALL)
    if not match:
        return None, f"no JSON in: {text[:120]!r}"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"
    slot = obj.get("chosen_slot")
    if slot not in (0, 1):
        return None, f"invalid slot {slot!r}"
    reason = (obj.get("reason") or "")[:200]
    return int(slot), reason


def _parse_source_pairs(raw: list[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"--judge-sources entry must be 'path:name', got {entry!r}")
        path_s, name = entry.rsplit(":", 1)
        out.append((Path(path_s), name))
    return out


def _parse_pred_pairs(raw: list[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"--pairs entry must be 'path:label', got {entry!r}")
        path_s, label = entry.rsplit(":", 1)
        out.append((Path(path_s), label))
    return out


def _load_resolved(report_path: Path) -> set[str]:
    return set(json.loads(report_path.read_text(encoding="utf-8")).get("resolved_ids", []))


async def run_meta_judge(
    judge_sources: list[tuple[Path, str]],
    pairs: list[tuple[Path, str]],
    split: str,
    meta_judge_model: str,
    out_path: Path,
    eval_reports: list[Path] | None = None,
) -> dict[str, Any]:
    """For each 2-judge disagreement, call meta-judge + record pick.

    Writes one JSONL row per disagreement instance with keys:
    instance_id, judges (list), meta_choice (slot 0/1 of the two disagreed
    judges), final_label (the run label the meta-judge chose from), reason.
    """
    from godspeed.llm.client import LLMClient

    if len(judge_sources) != 2:
        raise SystemExit("meta_judge currently supports exactly 2 primary judges")

    (src_a_path, src_a_name), (src_b_path, src_b_name) = judge_sources
    src_a = {d["instance_id"]: d for d in _load_jsonl(src_a_path)}
    src_b = {d["instance_id"]: d for d in _load_jsonl(src_b_path)}

    labels = [label for _, label in pairs]
    preds = {label: _load_predictions(path) for path, label in pairs}

    problems = _load_problem_statements(split)

    client = LLMClient(model=meta_judge_model, timeout=90)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_iids = sorted(set(src_a) & set(src_b))
    results: list[dict] = []
    disagreement_count = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for iid in all_iids:
            a = src_a[iid]
            b = src_b[iid]
            a_lbl = a.get("chosen_label")
            b_lbl = b.get("chosen_label")
            if not a_lbl or not b_lbl or a_lbl == b_lbl:
                continue  # agreement or missing — nothing to meta-judge
            disagreement_count += 1
            patch_a = preds.get(a_lbl, {}).get(iid, "")
            patch_b = preds.get(b_lbl, {}).get(iid, "")
            if not patch_a.strip() or not patch_b.strip():
                # One side is empty — nothing to meta-judge
                continue
            problem = problems.get(iid, "")
            slot, reason = await _meta_judge_one(
                client,
                iid,
                problem,
                src_a_name,
                a.get("reason", "")[:200],
                patch_a,
                src_b_name,
                b.get("reason", "")[:200],
                patch_b,
            )
            final_label = a_lbl if slot == 0 else (b_lbl if slot == 1 else None)
            row = {
                "instance_id": iid,
                "judges": [
                    {"name": src_a_name, "label": a_lbl},
                    {"name": src_b_name, "label": b_lbl},
                ],
                "meta_slot": slot,
                "final_label": final_label,
                "reason": reason,
            }
            results.append(row)
            fh.write(json.dumps(row) + "\n")
            logger.info(
                "meta_judge instance=%s a=%s b=%s chose=%s",
                iid,
                a_lbl,
                b_lbl,
                final_label,
            )

    summary: dict[str, Any] = {
        "meta_judge_model": meta_judge_model,
        "disagreement_count": disagreement_count,
        "meta_judge_calls": len(results),
        "rows": results,
    }

    if eval_reports is not None:
        if len(eval_reports) != len(pairs):
            raise ValueError("--eval-reports count must match --pairs count")
        resolved = {labels[i]: _load_resolved(eval_reports[i]) for i in range(len(pairs))}
        # Project what a "solo-A + meta-on-disagreement" selector would resolve.
        a_picks = {iid: d.get("chosen_label") for iid, d in src_a.items()}
        meta_by_iid = {r["instance_id"]: r["final_label"] for r in results if r["final_label"]}
        a_plus_meta_resolved = 0
        b_picks = {iid: d.get("chosen_label") for iid, d in src_b.items()}
        b_plus_meta_resolved = 0
        for iid in all_iids:
            # A+meta
            pick = meta_by_iid.get(iid, a_picks.get(iid))
            if pick and iid in resolved.get(pick, set()):
                a_plus_meta_resolved += 1
            # B+meta
            pick = meta_by_iid.get(iid, b_picks.get(iid))
            if pick and iid in resolved.get(pick, set()):
                b_plus_meta_resolved += 1
        summary["a_solo_plus_meta_resolved"] = a_plus_meta_resolved
        summary["b_solo_plus_meta_resolved"] = b_plus_meta_resolved

    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--judge-sources", nargs=2, required=True, help="Exactly 2 path:name pairs")
    p.add_argument("--pairs", nargs="+", required=True, help="predictions.jsonl:label pairs")
    p.add_argument("--split", default="dev")
    p.add_argument("--meta-judge-model", default="nvidia_nim/moonshotai/kimi-k2.5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-reports", nargs="*", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    judge_sources = _parse_source_pairs(args.judge_sources)
    pairs = _parse_pred_pairs(args.pairs)
    eval_reports = [Path(p) for p in args.eval_reports] if args.eval_reports else None

    summary = await run_meta_judge(
        judge_sources=judge_sources,
        pairs=pairs,
        split=args.split,
        meta_judge_model=args.meta_judge_model,
        out_path=args.out,
        eval_reports=eval_reports,
    )
    # Strip row payloads for clean stdout
    display = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(display, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_amain(_build_argparser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
