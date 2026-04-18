# Phase A1 — Next Session Handoff

**Last touch**: 2026-04-17, end of session. All green.

## What's running today

End-to-end synthetic data pipeline for Godspeed training corpus. Produces
`{messages, tools}` OpenAI-format JSONL consumable by ml-lab `messages_raw`
reader. Free-tier only (Cerebras Qwen-3-235B + Z.ai GLM-4.7-flash primary,
Groq Llama-3.3-70B overflow, Ollama local fallback). $0 budget.

## Current state (verified live at end of session)

Ran `python -m experiments.phase_a1.orchestrate --limit 3 --reset`:
- 3 samples produced, 0 failed, 19.3s wall-clock, 8.9K tokens across providers
- Rate-limit cascade exercised and works
- Output: `data/phase_a1_smoke.jsonl` (48KB, 3 lines)
- Metrics: `data/run_metrics.jsonl`

## Files already built (all pass ruff)

| File | Purpose | Status |
|------|---------|--------|
| `providers.py` | 4-backend async router + SQLite quota tracker | DONE |
| `registry_builder.py` | Headless 21-tool `ToolRegistry` | DONE |
| `specs.py` | Stratified spec generator (6200 → ≥295/tool) | DONE |
| `executor.py` | Real tool exec on tmp sandbox + fixtures | DONE |
| `emit.py` | `ConversationLogger` → `TrainingExporter(fmt="openai")` | DONE |
| `blueprints.py` | Stage A LLM call (Cerebras primary) | DONE |
| `narrator.py` | Stage C LLM call (Z.ai secondary) | DONE |
| `orchestrate.py` | Composed runner, resumable | DONE (minimal; no judge/validate yet) |

## Files still to build

1. **Real fixtures** — `fixtures/{web_search,web_fetch,github,pdf_read,image_read,code_search,spawn_agent}.json`.
   20 realistic responses per tool. Opus-authored. Current state: placeholder
   strings (`"[<tool> fixture placeholder]..."`). Samples using these tools
   work but have visibly stubby tool output.
2. **`judge.py`** — GLM-Flash 4-dimension rubric (tool correctness, arg
   correctness, realism, multi-turn coherence). Drop samples with any
   dim < 4. Few-shot from anchor samples.
3. **`validate.py`** — Schema + tool-name + arg-shape + coverage checks.
   Load each final JSONL record, assert `{messages, tools}` shape, assert
   every `tool_calls[].function.name` is in registry, per-tool arg validators
   (file_read path absolute, shell whitelist, etc.).
4. **`anchor_opus.py`** — 50 Opus-hand-authored gold samples. Batched 10-15
   at a time during authoring. Output: `data/anchor_opus_50.jsonl`. Used as
   judge few-shots + held-out eval set.
5. **`swesmith_distill.py`** — TF-IDF k=30 cluster existing
   `ml-lab/.../phase2_swesmith.jsonl` (24K), diversity-sample 1,500. Cap
   shell-only cluster at 800 to avoid monoculture.
6. **`augment.py`** — Param-shuffle + intent paraphrase for 200 samples
   targeting under-represented tools.
7. **`tests/test_*.py`** — Unit tests for providers (mocked HTTP), executor
   (blueprint→sandbox roundtrip), judge (prompt render + threshold), validate
   (arg shapes vs registry).
8. **Makefile** — `a1-smoke`, `a1-anchor`, `a1-run`, `a1-validate` targets.

## How to resume next session

```bash
# Load env + verify everything still works
cd "$HOME/Documents/Project Portfolio/godspeed"

# 1. Confirm pipeline still healthy (3-sample smoke, ~20s, $0)
/c/Users/ttimm/miniconda3/envs/mlenv/python.exe -m experiments.phase_a1.orchestrate --limit 3 --reset

# 2. Inspect the 3 samples
/c/Users/ttimm/miniconda3/envs/mlenv/python.exe -c "
import json
for line in open('experiments/phase_a1/data/phase_a1_smoke.jsonl'):
    d = json.loads(line)
    print(len(d['messages']), 'msgs;', len(d['tools']), 'tools')
"

# 3. Pick up the highest-leverage remaining piece
#    Recommended next: fixtures (Opus-authored, unblocks judge quality)
#    Then: judge.py (enables quality filter before full run)
#    Then: validate.py (enables gate to Phase A2)
#    Then: swesmith_distill + anchor_opus + augment (parallel streams)
#    Finally: full make a1-run
```

## Keys

Stored in `experiments/phase_a1/.env.local` (gitignored via repo's `.env.*`).
Loaded automatically by `providers.py::_load_env_local`. **Rotate after the
6200-sample run completes** — they've been in chat transcripts.

## Provider routing (locked in after live testing)

| Tier | Primary | Fallback chain |
|------|---------|----------------|
| primary | Cerebras `qwen-3-235b-a22b-instruct-2507` | Z.ai glm-4.7-flash → Groq → Ollama |
| secondary | Z.ai `glm-4.7-flash` | Cerebras qwen → Groq → Ollama |
| judge | Z.ai `glm-4.5-flash` | Cerebras llama3.1-8b → Groq |
| overflow | Groq `llama-3.3-70b-versatile` | Ollama → Cerebras → Z.ai |
| ollama_only | Ollama `qwen2.5-coder:3b` (smoke tests) | — |

Z.ai flash models have reasoning mode disabled via `extra_body={"thinking":{"type":"disabled"}}` (fixed in `_ZAIBackend._call`).

## Known quality gaps

- Fixtures are placeholders → non-sandbox tool samples look stubby
- Narrator occasionally drifts from real tool output (e.g. claims "mypy"
  when `verify` tool used ruff). Judge will catch this once it exists.
- No per-tool arg validators yet → LLM can emit args that real tool
  gracefully rejects (acceptable; caught at execution time).

## Plan file

`~/.claude/plans/modular-watching-simon.md` — approved Ultraplan v2.

## Reference implementations in the ml-lab experiment

`C:\Users\ttimm\Documents\Project Portfolio\ml-lab\experiments\2026-04-godspeed-coder\`
- `data/phase2_swesmith.jsonl` — source corpus for swesmith_distill
- `data/smoke_test_10.jsonl` — hand-crafted format reference
- `data/godspeed_tools.json` — 21-tool OpenAI schemas
- `src/data.py` — training's `messages_raw` reader (target format)
