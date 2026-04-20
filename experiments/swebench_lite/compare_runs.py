"""Diff two SWE-Bench Lite runs by metrics + predictions.

Usage:
    python experiments/swebench_lite/compare_runs.py \\
        --a-metrics experiments/swebench_lite/run_metrics_iter1.jsonl \\
        --a-preds   experiments/swebench_lite/predictions_iter1.jsonl \\
        --b-metrics experiments/swebench_lite/run_metrics.jsonl \\
        --b-preds   experiments/swebench_lite/predictions.jsonl \\
        --a-label iter1 --b-label rerun

Reports:
  - headline: total empty-patch rate A vs B
  - flipped instances (empty -> filled, filled -> empty)
  - unchanged instances
  - wall time + tool-call delta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_by_id(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["instance_id"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-metrics", type=Path, required=True)
    parser.add_argument("--a-preds", type=Path, required=True)
    parser.add_argument("--b-metrics", type=Path, required=True)
    parser.add_argument("--b-preds", type=Path, required=True)
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    args = parser.parse_args()

    a_m = _load_by_id(args.a_metrics)
    a_p = _load_by_id(args.a_preds)
    b_m = _load_by_id(args.b_metrics)
    b_p = _load_by_id(args.b_preds)

    ids = sorted(set(a_m) | set(b_m))
    a_empty = {i for i in ids if i in a_p and not a_p[i].get("model_patch", "").strip()}
    b_empty = {i for i in ids if i in b_p and not b_p[i].get("model_patch", "").strip()}

    print(f"instances in {args.a_label}: {len(a_m)}  |  in {args.b_label}: {len(b_m)}")
    print(
        f"empty-patch rate: {args.a_label}={len(a_empty)}/{len(a_m)} "
        f"({100*len(a_empty)/max(1,len(a_m)):.1f}%)  "
        f"{args.b_label}={len(b_empty)}/{len(b_m)} "
        f"({100*len(b_empty)/max(1,len(b_m)):.1f}%)"
    )
    print()

    empty_to_filled = sorted(a_empty - b_empty)
    filled_to_empty = sorted(b_empty - a_empty)
    still_empty = sorted(a_empty & b_empty)

    print(f"-- empty in {args.a_label} -> filled in {args.b_label} ({len(empty_to_filled)}) --")
    for i in empty_to_filled:
        before = a_m.get(i, {}).get("tool_call_count", "?")
        after = b_m.get(i, {}).get("tool_call_count", "?")
        after_lines = b_p.get(i, {}).get("model_patch", "").splitlines()
        print(f"  {i}  tool_calls {before}->{after}  new patch: {len(after_lines)} lines")

    print()
    print(f"-- filled in {args.a_label} -> empty in {args.b_label} ({len(filled_to_empty)}) --")
    for i in filled_to_empty:
        before = a_m.get(i, {}).get("tool_call_count", "?")
        after = b_m.get(i, {}).get("tool_call_count", "?")
        before_lines = a_p.get(i, {}).get("model_patch", "").splitlines()
        print(f"  {i}  tool_calls {before}->{after}  lost patch: {len(before_lines)} lines")

    print()
    print(f"-- still empty in both ({len(still_empty)}) --")
    for i in still_empty:
        status_a = a_m.get(i, {}).get("status", "?")
        status_b = b_m.get(i, {}).get("status", "?")
        print(f"  {i}  {args.a_label}:{status_a}  {args.b_label}:{status_b}")

    # Wall time + tool-call aggregates
    a_wall = [float(a_m[i]["wall_s"]) for i in a_m if a_m[i].get("wall_s") is not None]
    b_wall = [float(b_m[i]["wall_s"]) for i in b_m if b_m[i].get("wall_s") is not None]
    a_tc = [int(a_m[i]["tool_call_count"]) for i in a_m if a_m[i].get("tool_call_count") is not None]
    b_tc = [int(b_m[i]["tool_call_count"]) for i in b_m if b_m[i].get("tool_call_count") is not None]
    print()
    if a_wall and b_wall:
        print(
            f"wall time (s): {args.a_label} mean={sum(a_wall)/len(a_wall):.1f}  "
            f"{args.b_label} mean={sum(b_wall)/len(b_wall):.1f}"
        )
    if a_tc and b_tc:
        print(
            f"tool calls:    {args.a_label} mean={sum(a_tc)/len(a_tc):.1f}  "
            f"{args.b_label} mean={sum(b_tc)/len(b_tc):.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
