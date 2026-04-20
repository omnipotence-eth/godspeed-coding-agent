"""Pick the best-of-N patch per instance across multiple run directories.

Strategy: for each instance, look at the patches produced across all
supplied prediction files and pick one per these rules:

  1. If at least one non-empty patch exists, pick the shortest non-empty
     patch. Rationale: shortest is typically most minimal, which
     correlates with correctness in SWE-Bench-style bug fixes.
  2. If only empty patches exist, emit an empty patch.

The selected patch's model_name_or_path is set to "best_of_<N>" with the
source run recorded in a sidecar JSONL.

This exploits run-to-run variance in our favor: if a correct fix appears
in any seed, we submit it. This is legitimate for benchmark comparison
if disclosed as its own configuration ("best-of-N with transparent
selector") — both Aider and mini-swe-agent published numbers include
such techniques.

Usage:
    python experiments/swebench_lite/best_of_n.py \\
        --preds experiments/swebench_lite/predictions_iter1.jsonl \\
                experiments/swebench_lite/predictions_seed2.jsonl \\
                experiments/swebench_lite/predictions.jsonl \\
        --labels iter1 seed2 seed3 \\
        --out experiments/swebench_lite/predictions_best_of_3.jsonl \\
        --source-log experiments/swebench_lite/best_of_3_sources.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["instance_id"]] = row.get("model_patch", "") or ""
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preds",
        nargs="+",
        type=Path,
        required=True,
        help="Two or more predictions.jsonl files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Short labels for each predictions file (same order as --preds)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, default=None)
    parser.add_argument(
        "--model-name",
        default=None,
        help="model_name_or_path for the combined predictions file (defaults to best_of_<N>)",
    )
    args = parser.parse_args()

    if len(args.preds) < 2:
        raise SystemExit("Need at least 2 --preds files")
    labels = args.labels or [f"run{i + 1}" for i in range(len(args.preds))]
    if len(labels) != len(args.preds):
        raise SystemExit("--labels must match --preds count")

    runs = [(_load(p), label) for p, label in zip(args.preds, labels, strict=True)]

    all_ids = sorted({iid for run, _ in runs for iid in run})
    model_name = args.model_name or f"best_of_{len(runs)}"

    picks: list[dict] = []
    sources: list[dict] = []

    for iid in all_ids:
        # Gather (patch, label) pairs where the instance is present
        variants = [(run[iid], label) for run, label in runs if iid in run]
        non_empty = [(p, lbl) for p, lbl in variants if p.strip()]
        if non_empty:
            # Shortest non-empty patch
            best_patch, best_label = min(non_empty, key=lambda pl: len(pl[0]))
            source_strategy = "shortest_non_empty"
        else:
            best_patch, best_label = "", "none"
            source_strategy = "all_empty"

        picks.append(
            {
                "instance_id": iid,
                "model_name_or_path": model_name,
                "model_patch": best_patch,
            }
        )
        sources.append(
            {
                "instance_id": iid,
                "strategy": source_strategy,
                "chosen_from": best_label,
                "chosen_lines": len(best_patch.splitlines()) if best_patch else 0,
                "variants": {lbl: len(p.splitlines()) if p.strip() else 0 for p, lbl in variants},
            }
        )

    with args.out.open("w", encoding="utf-8") as f:
        for row in picks:
            f.write(json.dumps(row) + "\n")

    if args.source_log:
        with args.source_log.open("w", encoding="utf-8") as f:
            for row in sources:
                f.write(json.dumps(row) + "\n")

    # Summary table
    print(f"{'instance_id':<45} {'chosen':<8} lines per run")
    print("-" * 90)
    for s in sources:
        per = "  ".join(f"{lbl}={n:>3}" for lbl, n in s["variants"].items())
        print(f"{s['instance_id']:<45} {s['chosen_from']:<8} {per}")

    n_nonempty = sum(1 for p in picks if p["model_patch"].strip())
    print()
    print(f"Selected {n_nonempty}/{len(picks)} non-empty patches across {len(runs)} runs")
    print(f"Wrote {args.out}")
    if args.source_log:
        print(f"Wrote source log {args.source_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
