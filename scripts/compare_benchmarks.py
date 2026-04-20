#!/usr/bin/env python
"""Aggregate multiple run_benchmark.py output dirs into one markdown table.

Usage:
    python scripts/compare_benchmarks.py \\
        --runs experiments/bench_qwen3coder_v2/ \\
               experiments/bench_qwen3coder_480b/ \\
               experiments/bench_deepseek_v32/ \\
               experiments/bench_qwen35_397b/ \\
        --out experiments/benchmark_shootout_2026_04.md

Each ``--runs`` arg must point at a directory with a ``summary.json`` from a
completed benchmark run. Output is a sortable markdown table plus a short
narrative pointing at the winner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(run_dir: Path) -> dict:
    summary = run_dir / "summary.json"
    if not summary.is_file():
        raise SystemExit(f"missing {summary}")
    return json.loads(summary.read_text(encoding="utf-8"))


def _fmt_float(x: float | int | None, decimals: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{decimals}f}" if isinstance(x, float) else str(x)


def _by_difficulty(row: dict, key: str) -> str:
    val = (row.get("by_difficulty") or {}).get(key)
    return _fmt_float(val) if val is not None else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for d in args.runs:
        s = _load(d)
        rows.append(
            {
                "run_dir": d.name,
                "model": s.get("model", "?"),
                "overall": s.get("mean_overall", 0.0),
                "pass": s.get("pass_count_jaccard_ge_0_6", 0),
                "total": s.get("total_tasks", 0),
                "easy": _by_difficulty(s, "easy"),
                "medium": _by_difficulty(s, "medium"),
                "hard": _by_difficulty(s, "hard"),
                "waste": s.get("mean_waste_penalty", 0.0),
                "mech_pass": s.get("mechanical_pass", 0),
                "mech_eval": s.get("mechanical_evaluated", 0),
                "tok_per_sec": s.get("mean_tok_per_sec", 0.0),
                "duration_s": s.get("total_duration_s", 0.0),
            }
        )

    rows.sort(key=lambda r: r["overall"], reverse=True)

    lines: list[str] = []
    lines.append("# Godspeed Benchmark Shootout — 2026-04")
    lines.append("")
    lines.append(
        "Comparison across local Ollama and NVIDIA NIM free-tier models. "
        "Each run used the 20-task suite in `benchmarks/tasks.jsonl` with the "
        "polished fixtures in `benchmarks/fixtures/`."
    )
    lines.append("")
    header = (
        "| Model | Overall | Pass (J>=0.6) | Easy | Medium | Hard |"
        " Mech | Waste | tok/s | Total s |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mech = f"{r['mech_pass']}/{r['mech_eval']}" if r["mech_eval"] else "—"
        lines.append(
            f"| `{r['model']}` | {_fmt_float(r['overall'])} | "
            f"{r['pass']}/{r['total']} | {r['easy']} | {r['medium']} | {r['hard']} | "
            f"{mech} | {_fmt_float(r['waste'])} | {_fmt_float(r['tok_per_sec'], 1)} | "
            f"{_fmt_float(r['duration_s'], 1)} |"
        )

    if rows:
        top = rows[0]
        lines.append("")
        lines.append(
            f"**Winner (highest overall):** `{top['model']}` — "
            f"{_fmt_float(top['overall'])} overall, "
            f"{top['pass']}/{top['total']} tasks pass Jaccard>=0.6."
        )
        lines.append("")
        lines.append(
            "**Columns:** *Mech* = tasks where the `verify.py` hook confirmed "
            "mechanical success. *Waste* = mean `waste_penalty` (higher is worse, "
            "capped at 0.3). *tok/s* = mean output tokens per second."
        )

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
