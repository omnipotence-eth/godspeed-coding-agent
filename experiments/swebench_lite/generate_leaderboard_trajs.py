"""Generate SWE-bench/experiments leaderboard `trajs/` files.

SWE-bench requires one reasoning trace per task instance. For a best@k
system like this one, the reasoning trace must "reflect all rollouts +
the mechanism for selecting which solution was used" (quote from
SWE-bench/experiments README § Reasoning Traces).

For each instance we emit a single Markdown file at
``trajs/<instance_id>.md`` containing:

1. The problem statement (same verbatim text the judge saw — already
   public in the dataset).
2. All N candidate slot diffs (anonymized by slot index, matching what
   the judge was shown — the judge never saw run labels).
3. The judge's decision and rationale.
4. The final selected patch.

No test knowledge leaks through: the same SWE_BENCH_RESTRICTED_KEYS
guard that governs the live judge applies here (we only read
problem_statement from the dataset). The candidate diffs are the public
model outputs from the five already-submitted constituent runs.

Usage
-----

    python experiments/swebench_lite/generate_leaderboard_trajs.py \\
        --source-log experiments/swebench_lite/judge_merged_5way_sources.jsonl \\
        --predictions experiments/swebench_lite/predictions_judge_merged_5way.jsonl \\
        --pair-files \\
            experiments/swebench_lite/predictions_e1_kimi.jsonl \\
            experiments/swebench_lite/predictions_gpt_oss.jsonl \\
            experiments/swebench_lite/predictions_iter1.jsonl \\
            experiments/swebench_lite/predictions_seed3.jsonl \\
            experiments/swebench_lite/predictions_p1_dev23_v3.jsonl \\
        --pair-labels e1_kimi gpt_oss iter1 seed3 p1_dev23_v3 \\
        --out-dir experiments/swebench_lite/leaderboard_submission/evaluation/lite/20260421_godspeed_v3_2_judge/trajs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_problem_statements(split: str) -> dict[str, str]:
    """Map instance_id -> problem_statement. Lazy HF import (heavy dep)."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return {row["instance_id"]: row.get("problem_statement", "") or "" for row in ds}


def _write_traj(
    out_dir: Path,
    instance_id: str,
    problem: str,
    slot_diffs: list[tuple[str, str]],  # (label, diff) per slot, in slot order
    decision: dict,
    chosen_patch: str,
) -> None:
    """Write trajs/<instance_id>.md."""
    lines: list[str] = []
    lines.append(f"# {instance_id}\n")
    lines.append("## Problem statement\n")
    lines.append(problem.strip() + "\n")

    lines.append(f"## Candidate patches (N={len(slot_diffs)})\n")
    lines.append(
        "> Candidates are anonymized by slot index; the judge does not see run labels. "
        "Labels in parentheses are for post-hoc inspection only.\n"
    )
    for i, (label, diff) in enumerate(slot_diffs):
        lines.append(f"### Slot {i} (post-hoc label: `{label}`)\n")
        if diff.strip():
            lines.append("```diff")
            lines.append(diff.rstrip())
            lines.append("```\n")
        else:
            lines.append("_(empty patch — this constituent run did not produce an edit)_\n")

    lines.append("## Judge decision\n")
    lines.append(f"- **Strategy:** `{decision.get('strategy', '?')}`")
    chosen_slot = decision.get("chosen_slot")
    chosen_label = decision.get("chosen_label", "?")
    if chosen_slot is not None:
        lines.append(f"- **Chosen slot:** `{chosen_slot}` (post-hoc label: `{chosen_label}`)")
    else:
        lines.append("- **Chosen slot:** `None` (degraded to fallback / all candidates empty)")
    lines.append(f"- **Judge rationale:** {decision.get('reason', '')!r}\n")

    lines.append("## Final selected patch\n")
    if chosen_patch.strip():
        lines.append("```diff")
        lines.append(chosen_patch.rstrip())
        lines.append("```\n")
    else:
        lines.append("_(empty — selector could not find a non-empty candidate)_\n")

    out_path = out_dir / f"{instance_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--source-log", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--pair-files", nargs="+", type=Path, required=True)
    ap.add_argument("--pair-labels", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if len(args.pair_files) != len(args.pair_labels):
        raise ValueError("--pair-files and --pair-labels must be same length")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_run_preds = [
        {row["instance_id"]: row.get("model_patch", "") or "" for row in _load_jsonl(p)}
        for p in args.pair_files
    ]
    labels = list(args.pair_labels)

    final_preds = {
        row["instance_id"]: row.get("model_patch", "") or ""
        for row in _load_jsonl(args.predictions)
    }

    decisions_by_iid = {row["instance_id"]: row for row in _load_jsonl(args.source_log)}

    logger.info("loading problem_statements from SWE-bench_Lite %s", args.split)
    problems = _load_problem_statements(args.split)

    written = 0
    for iid in sorted(decisions_by_iid):
        slot_diffs = [(labels[i], per_run_preds[i].get(iid, "")) for i in range(len(labels))]
        _write_traj(
            args.out_dir,
            iid,
            problems.get(iid, "(problem statement unavailable)"),
            slot_diffs,
            decisions_by_iid[iid],
            final_preds.get(iid, ""),
        )
        written += 1

    logger.info("wrote %d trajectory files to %s", written, args.out_dir)
    print(f"wrote {written} trajectory files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
