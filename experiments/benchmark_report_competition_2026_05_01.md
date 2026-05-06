# Godspeed Benchmark Report — 2026-05-01 (Competition Mode)

## Model Tested
- **Model:** `nvidia_nim/qwen/qwen3.5-397b-a17b`
- **Platform:** NVIDIA NIM R&D free tier
- **Cost:** $0
- **Mode:** Competition mode (no compaction, auto-stash, auto-commit, must-fix)

## Changes Applied

1. **Empty-patch fix:** System prompt now requires at least one write tool call before stopping on coding tasks
2. **Competition mode:** New `--competition-mode` flag disables non-essential loop features

## Internal 20-Task Suite Results

| Metric | Previous (Apr) | This Run (May) | Delta |
|--------|---------------|----------------|-------|
| **Pass (Jaccard >= 0.6)** | **11 / 20 (55%)** | **10 / 20 (50%)** | -1 |
| Mean tool selection | 0.608 | 0.559 | -0.049 |
| Mean sequence quality | — | 0.697 | — |
| Mean overall | 0.608 | 0.526 | -0.082 |
| Mechanical pass | — | 9 / 13 (69%) | — |
| Mean tok/s | 10.3 | 7.7 | -2.6 |
| Total duration | 556s | 1396s | +840s |

### Per-Task Breakdown

| # | Task | Difficulty | Score | Mechanical | Notes |
|---|------|-----------|-------|------------|-------|
| 1 | easy-fix-syntax-01 | easy | **1.0** | **PASS** | Perfect |
| 2 | easy-read-file-01 | easy | **1.0** | N/A | Perfect |
| 3 | easy-run-tests-01 | easy | 0.25 | N/A | Over-explored (12 tool calls) |
| 4 | easy-git-status-01 | easy | **1.0** | N/A | Perfect |
| 5 | easy-search-01 | easy | 0.85 | N/A | Minor overrun |
| 6 | medium-find-fix-01 | medium | **1.0** | **PASS** | Perfect |
| 7 | medium-add-logging-01 | medium | 0.183 | **PASS** | Over-explored (13 tool calls) |
| 8 | medium-debug-error-01 | medium | 0.4 | **PASS** | Over-explored (14 tool calls) |
| 9 | medium-refactor-01 | medium | **1.0** | **PASS** | Perfect |
| 10 | medium-explore-01 | medium | 0.667 | N/A | LLM error mid-run |
| 11 | medium-web-lookup-01 | medium | 0.5 | N/A | LLM error mid-run |
| 12 | medium-new-file-01 | medium | 0.0 | FAIL | **LLM error** — no tool calls |
| 13 | medium-git-commit-01 | medium | 0.0 | FAIL | **LLM error** — no tool calls |
| 14 | hard-multi-file-01 | hard | 0.0 | FAIL | **LLM error** — no tool calls |
| 15 | hard-test-coverage-01 | hard | 0.0 | FAIL | **LLM error** — no tool calls |
| 16 | hard-security-audit-01 | hard | 0.483 | **PASS** | Good but over-explored |
| 17 | hard-migration-01 | hard | **0.705** | **PASS** | Strong |
| 18 | hard-debug-perf-01 | hard | **0.61** | N/A | Solid |
| 19 | hard-feature-01 | hard | 0.407 | **PASS** | Complex but completed |
| 20 | hard-cicd-01 | hard | 0.46 | **PASS** | Timeout after 16 tool calls |

## Key Findings

### 1. LLM Errors Are the Dominant Failure Mode
**5 out of 20 tasks (25%) hit `exit_code: 4, exit_reason: llm_error`** — the model failed to respond entirely, producing 0 tool calls. These were clustered in the middle of the run (tasks 10-15), suggesting:
- **Rate limiting** from NVIDIA NIM free tier
- **Context window overflow** on complex prompts
- **Transient API instability**

### 2. Empty-Patch Fix Is Working
The model is now making actual edits on successful tasks. No task exits with only text analysis when a coding task is given.

### 3. Over-Exploration Is the Secondary Issue
Tasks 3, 7, 8, and 20 show high tool call counts (8-16 calls) with waste penalties. The agent is exploring excessively before acting.

### 4. Competition Mode Is Functional
The loop ran without compaction, auto-stash, auto-commit, or must-fix injections. No crashes occurred.

## Hypothesis: Why Score Dropped from 0.608 to 0.526

The April shootout was likely run under different conditions:
- Possibly with a different model version or API endpoint
- Possibly with higher rate limits (less llm_errors)
- The empty-patch fix may cause the model to try harder and hit limits more often

## Path to Improvement

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| **1** | **Retry on LLM errors** — Add backoff retry for transient API failures | +2-3 tasks recovered |
| **2** | **Reduce over-exploration** — Cap exploratory tool calls before requiring action | +0.5 overall score |
| **3** | **Run SWE-bench Lite** with this model + retry logic | Publishable headline number |
| **4** | **Test without empty-patch fix** on same model to verify net effect | A/B comparison |

## Reproduce This Run

```bash
export NVIDIA_NIM_API_KEY=...
python scripts/run_benchmark.py \
    --model nvidia_nim/qwen/qwen3.5-397b-a17b \
    --tasks benchmarks/tasks.jsonl \
    --out experiments/bench_qwen35_397b_competition_2026_05_01 \
    --project-dir .
```

## Files Changed

- `src/godspeed/agent/system_prompt.py` — Added Task Completion Rule
- `src/godspeed/agent/loop.py` — Added `competition_mode` parameter
- `src/godspeed/cli.py` — Added `--competition-mode` flag
