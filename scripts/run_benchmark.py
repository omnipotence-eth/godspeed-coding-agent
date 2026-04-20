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
import shutil
import subprocess
import sys
import tempfile
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
FIXTURES_DIR = Path(__file__).parent.parent / "benchmarks" / "fixtures"


def _resolve_workspace(task_id: str, fallback: Path) -> tuple[Path, Path | None, Path | None]:
    """Return (workspace_dir, tempdir_to_cleanup, verify_script).

    If ``benchmarks/fixtures/<task_id>/`` exists, copy it to a temp dir and
    run the agent there so every run starts from identical state. If a
    ``_setup.py`` is present in the fixture, run it first (with cwd set to
    the workspace) so tasks that need git state or other runtime setup can
    stage it deterministically. Any ``verify.py`` is preserved and returned
    separately so it can run post-agent to determine mechanical success.
    """
    fixture = FIXTURES_DIR / task_id
    if not fixture.is_dir():
        return fallback, None, None

    tmp_root = Path(tempfile.mkdtemp(prefix=f"godspeed-bench-{task_id}-"))
    workspace = tmp_root / "workspace"
    shutil.copytree(fixture, workspace)

    setup = workspace / "_setup.py"
    if setup.is_file():
        try:
            subprocess.run(
                [sys.executable, str(setup)],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"[warn] {task_id}: _setup.py timed out", file=sys.stderr)

    verify = workspace / "verify.py"
    return workspace, tmp_root, (verify if verify.is_file() else None)


def _run_verify(verify_script: Path, workspace: Path) -> bool | None:
    """Run the task's verify.py and return True/False. None if it errors."""
    try:
        proc = subprocess.run(
            [sys.executable, str(verify_script)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    return proc.returncode == 0


def run_one_task(model: str, prompt: str, project_dir: Path) -> tuple[dict, float, bool | None]:
    """Shell out to `godspeed run`; also run verify.py if provided.

    Returns (payload, elapsed, mechanical_success) where mechanical_success
    is True/False when the task has a verify.py, None otherwise.
    """
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
    # Mechanical verification happens on the workspace the agent just touched.
    verify_script = project_dir / "verify.py"
    mechanical = _run_verify(verify_script, project_dir) if verify_script.is_file() else None
    return payload, elapsed, mechanical


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
    mechanical_pass = 0
    mechanical_evaluated = 0

    with open(results_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task.task_id}", file=sys.stderr, flush=True)
            workspace, tmp_root, _ = _resolve_workspace(task.task_id, args.project_dir)
            try:
                try:
                    payload, elapsed, mechanical = run_one_task(args.model, task.prompt, workspace)
                except subprocess.TimeoutExpired:
                    payload = {"_timeout": True}
                    elapsed = float(RUN_TIMEOUT_S + 30)
                    mechanical = None
            finally:
                if tmp_root is not None:
                    shutil.rmtree(tmp_root, ignore_errors=True)

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
            if mechanical is not None:
                mechanical_evaluated += 1
                if mechanical:
                    mechanical_pass += 1

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
                "mechanical_success": mechanical,
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
        "mean_waste_penalty": suite.mean_waste_penalty,
        "mechanical_pass": mechanical_pass,
        "mechanical_evaluated": mechanical_evaluated,
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
