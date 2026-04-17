# Stage A post-mortem — Qwen3.6 / Qwen3-Coder local integration smoke

**Date**: 2026-04-17
**Branch**: `feat/stage-a-qwen3.6-integration`
**Plan**: `~/.claude/plans/validated-moseying-graham.md`
**Verdict**: **Benchmark gate failed (1/20 vs. ≥ 14/20 required).** Integration and infrastructure work shipped regardless.

---

## What we validated

| Claim | Status | Evidence |
|---|---|---|
| Qwen3.6-35B-A3B is runnable on this rig via Ollama | **No** — Ollama 0.20.7 errors on manifest finalization | ~22 GB of layers download cleanly, then `Error: 400` — architecture unknown to Ollama as of 2026-04-17 |
| Fallback `qwen3-coder:latest` (30B-A3B, MoE, Q4_K_M) runs | Yes | 18 GB on disk, architecture `qwen3moe`, native `tools` capability advertised |
| Inference speed meets ≥ 30 tok/s gate | **Yes** — 53 tok/s raw generation | Measured via `/api/generate` with 300-token completion; end-to-end with tool overhead drops to ~11 tok/s |
| Godspeed integrates with the model | **Partially** — needed a parser shim | Ollama's built-in parser doesn't extract Qwen3-Coder's `<function=...>` XML into `tool_calls`; shim added |
| 20-task benchmark passes ≥ 14/20 | **No** — 1/20 pass | Mean Jaccard 0.098, mean tok/s 11.3 (with tool overhead), total 17.3 min |
| Exit codes + audit trail + headless JSON | **Yes** | All structured outputs work correctly |

---

## What we built (and what ships regardless of the gate failure)

1. **`src/godspeed/llm/qwen3_coder_parser.py`** — regex-based shim that extracts Qwen3-Coder's `<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>` XML into OpenAI-shaped tool_calls. Hooked into `LLMClient._call()` as a no-op for any response that already has structured tool_calls. 20 unit tests covering type coercion (bool/int/float/JSON/string), multi-call responses, malformed input, and uniqueness of synthesized IDs.

2. **`scripts/run_benchmark.py`** — measurement runner that shells out `godspeed run` per task, parses the JSON output, and scores via the existing `training.benchmark` library. Not a shipped CLI command; a tool for running this and future stages.

3. **Model table rows** — added `ollama/qwen3:14b` and `ollama/qwen3-coder:latest` to the `/models` CLI + `settings.yaml.example`.

4. **`.gitignore`** extensions for `.godspeed/training/`, `.godspeed/checkpoints/`, `.godspeed/memory.db*` — session artifacts shouldn't follow into repos.

---

## Benchmark failure analysis

### Numbers

```json
{
  "pass_count_jaccard_ge_0_6": 1,
  "mean_tool_selection": 0.098,
  "mean_sequence_quality": 0.163,
  "by_difficulty": {"easy": 0.25, "medium": 0.125, "hard": 0.031},
  "mean_tok_per_sec": 11.3,
  "total_duration_s": 1040.4
}
```

### Root cause — the model is web-biased

Qwen3-Coder-30B-A3B-Instruct consistently reached for `web_search`, `github`, `pdf_read`, `background_check` on tasks where the expected tools were `file_read`, `file_edit`, `grep_search`, `shell`. The one pass (`medium-web-lookup-01`) was the one task where `web_search` was actually correct.

Failure patterns by category:

- **Local-file tasks** → `web_search`: `easy-fix-syntax-01` reached for `web_search` on "there's a syntax error in app.py line 15" instead of `file_read`+`file_edit`.
- **Git tasks** → `background_check` / `github`: `easy-git-status-01` called `background_check` on "check git status"; `medium-git-commit-01` called `github` + `background_check` on local commit work.
- **Exploration tasks** → `shell` loops: `medium-explore-01` called `shell` three times instead of `repo_map` / `glob_search`.
- **Multi-step hard tasks** → runaway `web_search` loops: `hard-security-audit-01` made 38 tool calls, mostly `web_search`. `medium-refactor-01` made 15 `web_search` calls in a row.
- **Tool-name hallucination**: `read_file` (vs actual `file_read`), `grep` (vs `grep_search`), `glob` (vs `glob_search`), `example_function_name`.

The model has strong agentic priors, but those priors are biased toward web+API workflows rather than local codebase workflows. Training data composition, not a bug.

---

## Why the parser shim still matters

Even though this benchmark failed, the parser shim is a genuinely useful shipped artifact:

- Any user who `ollama pull qwen3-coder` today and tries to use it with Godspeed hits the same silent-failure (tool calls in the content, empty `tool_calls` field). The shim fixes that.
- When Qwen3.6-35B-A3B eventually gets Ollama architecture support (the layers already cache, only the manifest needs an upstream fix), the shim likely applies to it too (same model family, same tool-call format).
- The shim is isolated: 100 lines in `llm/qwen3_coder_parser.py`, called only when `tool_calls` is empty AND content has the `<function=` fingerprint. Zero impact on other models.

---

## Options forward (not decisions)

Per the plan, Stage A failure means stop + post-mortem. This document is the post-mortem. The user decides what happens next; here are the options I see with tradeoffs:

### Option 1: Land Stage A's artifacts, defer the local-model story

Ship the parser shim + benchmark runner + model table additions as a minor release (v2.10.0 / patch v2.9.1). Do not promote Qwen3-Coder to default. Revisit when:
- Ollama ships architecture support for Qwen3.6-35B-A3B, or
- A Godspeed-tuned variant (Stage B output) exists, or
- An instruction-tuned, locally-biased open-model lands.

Cost: zero new engineering. Benefit: infrastructure improvement. Drawback: Godspeed's local story stays at `ollama/qwen3:4b` — functional but not frontier.

### Option 2: Change Godspeed's system prompt to bias against web tools

Add an explicit directive like *"Default to local file tools (file_read, grep_search, shell). Only use web_search when the task explicitly mentions external information."* to `QUALITY_PROMPT` or a new section.

Cost: 15 min + re-run benchmark. Risk: helps small open models, may hurt larger frontier models that had good priors already. Testable.

### Option 3: Introduce a tool-filtering mode

Godspeed could support `--tool-set local` / `--tool-set web` / `--tool-set full` to hide tools the model doesn't need. Small open models often choose the wrong tool because there ARE too many — constraining the set could help.

Cost: ~2 hours. Benefit: helps weak models AND token economy (fewer tool schemas = shorter prompt).

### Option 4: Use this data for Stage B (the fine-tune)

The 20 tasks × actual-tool-choices × expected-tool-choices is effectively a tool-selection preference dataset. Fine-tune Qwen3-Coder-30B-A3B on exactly this: "when the user says 'fix the syntax error in app.py,' call file_read + file_edit, not web_search." This is what Phase 3 ORPO was designed to do — we now have a concrete, measured, expensive-to-fix weakness to target.

Cost: $25-50 cloud + ~2 weeks per the original Stage B scope. Benefit: could turn this failure into the strongest portfolio story — "I diagnosed a frontier model's weakness on my specific workflow and fixed it with ORPO."

### Option 5: Try different models not yet explored

- `qwen3:32b` (dense) — may fit on 16GB at Q3_K_M, expected to have better tool-selection priors than 8B/14B. Untested.
- `gpt-oss:20b` (if available) — OpenAI's open model family from 2025; unknown coding priors.
- `glm-4` / `devstral` / `granite-code` — other open coding models.

Cost: 1-2 hours per model. Could find a better local baseline before deciding on options 1-4.

---

## My read (offered, not acted on)

**Option 1 + Option 4.** Ship the infrastructure we built as a small release today. Use the failure data (this post-mortem + results.jsonl) as the Stage B ORPO training target. The value is real: we now know *exactly* what fine-tuning needs to fix. Option 2 and 3 are cheaper but less informative — they paper over a model weakness rather than measure and correct it.

Option 5 is tempting but unbounded — could spend days comparing models without a clear winner.

Waiting on your direction.

---

## Raw data

- **Per-task**: `experiments/qwen3.6-smoke/results.jsonl` (20 rows)
- **Summary**: `experiments/qwen3.6-smoke/summary.json`
- **Environment**: Windows 11, RTX 5070 Ti 16GB + 96GB DDR5, Ollama 0.20.7, Godspeed main@e46cc35, qwen3-coder:latest (ID 06c1097efce0, Q4_K_M, 18 GB)

## Sources consulted during Stage A

- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — confirmed `qwen3_coder` tool-parser format, 262K native context, SWE-bench 73.4%
- [Unsloth Qwen3.6 GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — inference-only, no fine-tune recipe as of 2026-04-17
- [TurboQuant TQ3_4S variant](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-TQ3_4S) — considered and ruled out (custom runtime, no tool-call support, aggressive 2-bit experts)
