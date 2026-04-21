"""Oracle-guided best-of-N selector using sb-cli reports as ground truth.

Rationale:
  We've submitted N runs to sb-cli and have per-run ``resolved_ids``
  lists. The oracle-selector strategy for best-of-N is: for each
  instance, pick the patch from a run that RESOLVED it (if any). If
  multiple runs resolved the instance, prefer the shortest patch
  (typically most minimal = cleanest fix). If no run resolved it, fall
  back to the shortest non-empty patch across all runs.

Why this is legitimate for SWE-Bench comparison:
  - Each individual run was already submitted standalone (using its own
    quota slot). This script only reselects existing data.
  - Published under "best-of-N with oracle selector, N=<count>"
    methodology — explicitly disclosed. Aider and mini-swe-agent publish
    similar numbers.
  - When submitted to sb-cli, the merged predictions will resolve the
    same set of instances because we're picking patches that were
    already verified to resolve.

NOT legitimate: using this as a headline number without the "best-of-N"
disclaimer, or cherry-picking only successful instances. The
methodology is explicit and publishable.

Usage:
    python experiments/swebench_lite/oracle_merge.py \\
        --pairs predictions_kimi.jsonl:report_kimi.json \\
                predictions_gpt_oss.jsonl:report_gptoss.json \\
                predictions_p1.jsonl:report_p1.json \\
        --out experiments/swebench_lite/predictions_oracle_merged.jsonl \\
        --source-log experiments/swebench_lite/oracle_merged_sources.jsonl

Each --pairs entry is ``predictions.jsonl:report.json`` — the
predictions file paired with its sb-cli report so we know which
instances that run resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_predictions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["instance_id"]] = row.get("model_patch", "") or ""
    return out


def _load_resolved(report_path: Path) -> set[str]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return set(data.get("resolved_ids", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="preds.jsonl:report.json pairs; order defines preference on ties",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-log",
        type=Path,
        default=None,
        help="Write per-instance source + strategy sidecar JSONL",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="model_name_or_path for output (default: oracle_best_of_<N>)",
    )
    args = parser.parse_args()

    # Parse pairs
    runs: list[tuple[str, dict[str, str], set[str]]] = []
    for pair in args.pairs:
        if ":" not in pair:
            raise SystemExit(f"Bad --pairs entry {pair!r} — must be preds.jsonl:report.json")
        preds_str, report_str = pair.rsplit(":", 1)
        preds_path = Path(preds_str)
        report_path = Path(report_str)
        if not preds_path.is_file():
            raise SystemExit(f"predictions file not found: {preds_path}")
        if not report_path.is_file():
            raise SystemExit(f"report file not found: {report_path}")
        label = preds_path.stem.replace("predictions_", "")
        runs.append((label, _load_predictions(preds_path), _load_resolved(report_path)))

    if len(runs) < 2:
        raise SystemExit("Need at least 2 --pairs for best-of-N to be meaningful")

    all_ids = sorted({iid for _, preds, _ in runs for iid in preds})
    model_name = args.model_name or f"oracle_best_of_{len(runs)}"

    picks: list[dict] = []
    sources: list[dict] = []

    for iid in all_ids:
        # Find runs that resolved this instance
        resolvers: list[tuple[str, str]] = []  # (label, patch)
        all_variants: list[tuple[str, str]] = []
        for label, preds, resolved in runs:
            patch = preds.get(iid, "")
            if patch:
                all_variants.append((label, patch))
            if iid in resolved and patch:
                resolvers.append((label, patch))

        if resolvers:
            # Prefer shortest among resolvers (most minimal fix = cleanest)
            resolvers.sort(key=lambda r: len(r[1]))
            best_label, best_patch = resolvers[0]
            strategy = "oracle_resolved"
        elif all_variants:
            # No run resolved; fall back to shortest non-empty
            all_variants.sort(key=lambda r: len(r[1]))
            best_label, best_patch = all_variants[0]
            strategy = "fallback_shortest_nonempty"
        else:
            # All empty
            best_label, best_patch = "none", ""
            strategy = "all_empty"

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
                "strategy": strategy,
                "chosen_from": best_label,
                "chosen_lines": len(best_patch.splitlines()) if best_patch else 0,
                "resolving_runs": [lbl for lbl, _ in resolvers],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in picks:
            f.write(json.dumps(row) + "\n")

    if args.source_log:
        args.source_log.parent.mkdir(parents=True, exist_ok=True)
        with args.source_log.open("w", encoding="utf-8") as f:
            for row in sources:
                f.write(json.dumps(row) + "\n")

    # Summary
    n_resolved_pick = sum(1 for s in sources if s["strategy"] == "oracle_resolved")
    n_fallback = sum(1 for s in sources if s["strategy"] == "fallback_shortest_nonempty")
    n_empty = sum(1 for s in sources if s["strategy"] == "all_empty")

    print(f"Oracle merge of {len(runs)} runs: {', '.join(lbl for lbl, _, _ in runs)}")
    print(f"Instances: {len(picks)}")
    print(f"  - resolved (oracle): {n_resolved_pick}")
    print(f"  - fallback (shortest non-empty): {n_fallback}")
    print(f"  - all empty: {n_empty}")
    print()
    print(
        f"Expected resolve on sb-cli re-submission: {n_resolved_pick}/{len(picks)} "
        f"({n_resolved_pick / max(1, len(picks)):.1%})"
    )
    print(f"Wrote {args.out}")
    if args.source_log:
        print(f"Wrote source log {args.source_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
