# Benchmark Fixtures

One directory per task in `benchmarks/tasks.jsonl`. Each fixture provides
deterministic starter state so benchmark runs are reproducible across
models and invocations.

## Contract

For a task with `task_id: "<id>"`:

- `benchmarks/fixtures/<id>/` is copied to a temp workspace before each run.
- The agent runs `--project-dir <workspace>`; all edits are scoped to that copy.
- If `verify.py` is present, it runs after the agent with cwd at the workspace
  and exits non-zero to signal failure. Its result is recorded as
  `mechanical_success` in `results.jsonl` and aggregated in `summary.json`.
- Tasks without a `verify.py` still benefit from deterministic state — only
  tool-call metrics are scored mechanically.

## What NOT to put in a fixture

- Large files (>50KB). Keep starter code minimal.
- Real secrets, credentials, or large binaries.
- Network fixtures — tasks that need the internet (`web_search`, `web_fetch`)
  skip fixture mocking; their score is Jaccard/LCS only.
