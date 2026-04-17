#!/usr/bin/env python
"""Run the Godspeed 20-task benchmark against a target model.

Shells out to ``godspeed run ... --json-output`` per task, parses the
returned JSON (tool_calls list + tool_call_count + exit_code + duration
+ token counts), builds a BenchmarkResult, and scores it with
``score_result`` from ``godspeed.training.benchmark``.

Not a shipped CLI command — a measurement script for Stage A of the
Track B plan. Results feed ``experiments/<run>/notes.md``.

Usage:
    python scripts/run_benchmark.py \\
        --model ollama/qwen3.6:35b-a3b-q4_k_m \\
        --tasks benchmarks/tasks.jsonl \\
        --out experiments/qwen3.6-smoke/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

from godspeed.training.benchmark import (
    BenchmarkResult,
    aggregate_scores,
    load_tasks,
    score_result,
)

RUN_TIMEOUT_S = 180
MAX_ITERATIONS = 20


def run_one_task(model: str, prompt: str, project_dir: Path) -> tuple[dict, float]:
    """Shell out to `godspeed run` and return the parsed JSON + wall time."""
    cmd = [
        "godspeed",
        "run",
        prompt,
        "-m",
        model,
        "-d",
        str(project_dir),
        "--json-output",
        "--auto-approve",
        "all",
        "--max-iterations",
        str(MAX_ITERATIONS),
        "--timeout",
        str(RUN_TIMEOUT_S),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S + 30,
    )
    elapsed = time.monotonic() - t0
    # godspeed run writes JSON to stdout on --json-output; stderr has progress.
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {"_parse_error": True, "_stdout": proc.stdout, "_stderr": proc.stderr[-500:]}
    payload["_shell_exit_code"] = proc.returncode
    return payload, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Model id (e.g. ollama/qwen3.6:35b-a3b-q4_k_m)"
    )
    parser.add_argument("--tasks", type=Path, required=True, help="Path to tasks.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project dir godspeed runs against (default: cwd)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    summary_path = args.out / "summary.json"

    tasks = load_tasks(args.tasks)
    print(f"[run_benchmark] model={args.model} tasks={len(tasks)} out={args.out}", file=sys.stderr)

    scores = []
    per_task_rows: list[dict] = []
    total_tokens = 0
    total_duration = 0.0

    with open(results_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task.task_id}", file=sys.stderr, flush=True)
            try:
                payload, elapsed = run_one_task(args.model, task.prompt, args.project_dir)
            except subprocess.TimeoutExpired:
                payload = {"_timeout": True}
                elapsed = float(RUN_TIMEOUT_S + 30)

            tool_calls = payload.get("tool_calls") or []
            tools_used = [
                tc["name"] for tc in tool_calls if isinstance(tc, dict) and tc.get("name")
            ]

            result = BenchmarkResult(
                task_id=task.task_id,
                tools_used=tools_used,
                tool_sequence=tools_used,
                completed=(payload.get("exit_code") == 0),
                error=None
                if payload.get("exit_code") == 0
                else str(payload.get("exit_reason", "")),
            )
            score = score_result(task, result)
            scores.append(score)

            out_tokens = int(payload.get("output_tokens") or 0)
            tok_per_sec = out_tokens / elapsed if elapsed > 0 else 0.0
            total_tokens += out_tokens
            total_duration += elapsed

            row = {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "expected_tools": task.expected_tools,
                "tools_used": tools_used,
                "exit_code": payload.get("exit_code"),
                "exit_reason": payload.get("exit_reason"),
                "tool_call_count": payload.get("tool_call_count", len(tool_calls)),
                "duration_s": round(elapsed, 2),
                "output_tokens": out_tokens,
                "tok_per_sec": round(tok_per_sec, 1),
                "cost_usd": payload.get("cost_usd", 0.0),
                "score": dataclasses.asdict(score),
            }
            per_task_rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

    suite = aggregate_scores(scores)
    pass_count = sum(1 for s in scores if s.tool_selection >= 0.6)
    mean_tok_per_sec = (total_tokens / total_duration) if total_duration > 0 else 0.0

    summary = {
        "model": args.model,
        "total_tasks": suite.total_tasks,
        "pass_count_jaccard_ge_0_6": pass_count,
        "mean_tool_selection": suite.mean_tool_selection,
        "mean_sequence_quality": suite.mean_sequence_quality,
        "mean_overall": suite.mean_overall,
        "by_difficulty": suite.by_difficulty,
        "mean_tok_per_sec": round(mean_tok_per_sec, 1),
        "total_duration_s": round(total_duration, 1),
        "total_output_tokens": total_tokens,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSummary: {json.dumps(summary, indent=2)}", file=sys.stderr)
    return 0 if pass_count >= 14 else 1


if __name__ == "__main__":
    sys.exit(main())
