"""Summarize run_metrics.jsonl + predictions.jsonl into a failure-mode table.

Usage:
    python experiments/swebench_lite/analyze.py \\
        --predictions experiments/swebench_lite/predictions.jsonl \\
        --metrics experiments/swebench_lite/run_metrics.jsonl

Reports:
  - count by status (ok / agent_exit_2 / clone_error / etc.)
  - empty-patch rate
  - wall-time percentiles
  - tool-call distribution
  - per-repo breakdown
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p * (len(xs) - 1))))
    return xs[k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    preds = {p["instance_id"]: p for p in _load_jsonl(args.predictions)}
    metrics = _load_jsonl(args.metrics)
    if not metrics:
        print("no metrics rows")
        return 1

    by_status = Counter(m.get("status", "?") for m in metrics)
    empty_patch = sum(1 for m in metrics if not m.get("patch_nonempty", False))
    wall_times = [float(m["wall_s"]) for m in metrics if m.get("wall_s") is not None]
    tool_calls = [
        int(m["tool_call_count"]) for m in metrics if m.get("tool_call_count") is not None
    ]

    by_repo: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "empty": 0})
    for m in metrics:
        repo = m.get("repo", "?")
        by_repo[repo]["total"] += 1
        if not m.get("patch_nonempty", False):
            by_repo[repo]["empty"] += 1

    print(f"Instances analyzed: {len(metrics)}")
    print()
    print("-- Status distribution --")
    for status, n in by_status.most_common():
        print(f"  {n:>3}  {status}")
    print()
    print(
        f"Empty-patch rate: {empty_patch}/{len(metrics)} ({100 * empty_patch / len(metrics):.1f}%)"
    )
    print()
    if wall_times:
        print(
            f"Wall time (s): p50={_pct(wall_times, 0.5):.1f} "
            f"p90={_pct(wall_times, 0.9):.1f} max={max(wall_times):.1f}"
        )
    if tool_calls:
        print(
            f"Tool calls: mean={statistics.mean(tool_calls):.1f} "
            f"p90={_pct([float(x) for x in tool_calls], 0.9):.0f} max={max(tool_calls)}"
        )
    print()
    print("-- Per-repo --")
    for repo, stats in sorted(by_repo.items(), key=lambda x: -x[1]["total"]):
        pct = 100 * stats["empty"] / stats["total"] if stats["total"] else 0
        print(f"  {stats['empty']:>2}/{stats['total']:>2} empty  ({pct:>5.1f}%)  {repo}")

    preds_in_file = len(preds)
    if preds_in_file != len(metrics):
        print()
        print(
            f"WARNING: predictions.jsonl has {preds_in_file} rows but metrics.jsonl has {len(metrics)}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
