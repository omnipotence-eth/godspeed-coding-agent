# Stage B revised — weakness-targeted ORPO on Qwen3-Coder-30B-A3B-Instruct

**Status**: Draft. Gated on user approval + cloud-budget confirmation.
**Companion docs**: `post_mortem.md` (Stage A results), `results.jsonl`
(20-task benchmark raw data).

## Why revise

The original Track B plan's Stage B used Qwen3-Coder-30B-A3B-Instruct as
a stretch target, with a 3-phase curriculum (SFT tool-calling → SFT
SWE-smith → ORPO When2Call + custom negatives). That plan was built
before we had measured data. It used generic preference pairs
(NVIDIA When2Call) to teach abstract "when to call tools" patterns.

Stage A measured the exact failure: on our 20-task Godspeed suite, the
model picks the wrong tool on 19/20 tasks. Specifically:

- **Local-file tasks → `web_search`**: instead of `file_read`+`file_edit`
- **Git tasks → `background_check` / `github`**: instead of `git`
- **Exploration → `shell` loops**: instead of `repo_map`/`glob_search`
- **Hard multi-step → runaway `web_search` loops** (up to 38 calls)
- **Tool-name hallucination**: `read_file`/`grep`/`glob` (wrong names)

That's precision training signal. Far better than generic preference data.

## Revised recipe (single phase)

Replace the 3-phase curriculum with a single ORPO pass targeting the
measured weakness. Rationale: Qwen3-Coder's base is already strong on
code (SWE-bench 73.4%); SFT on SWE-smith would be redundant. The actual
weakness is tool-prior, not coding capability.

### Base model
`Qwen/Qwen3-Coder-30B-A3B-Instruct` (not Qwen3.6-35B — Ollama
architecture support landed, but Unsloth QLoRA fine-tuning for Qwen3.6
is unverified as of 2026-04-17; Qwen3-Coder has published recipes).

### Training method
**ORPO only.** No SFT. Skip Phase 1 and 2 from the original plan.

- Rank 32, alpha 32, dropout 0, target modules: attention only
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Experts remain frozen —
  what we're correcting is attention-level tool-selection priors, not
  the expert knowledge.
- ORPO beta 0.1, learning rate 5e-6, 1 epoch.
- QLoRA 4-bit, bf16, seq 2048.
- Fits A100-80GB with headroom; dry run needed first to confirm.

### Why ORPO (not DPO / PPO / GRPO)

- ORPO doesn't require a reference-model copy → fits A100-80GB with
  ~16 GB headroom. DPO needs 2× model in VRAM → tight on 80GB for 30B MoE.
- Preference pairs are the natural data shape for our measured failure
  (chosen = expected tool sequence, rejected = actual wrong sequence).
- PPO/GRPO need a reward model or verifier; we have one already via
  `scripts/run_benchmark.py` + `training.benchmark.score_result`, but
  the implementation cost is higher for uncertain incremental win.

## Data strategy — grow 20 tasks → ~500 pairs

Stage A's 20 tasks give us 19 preference pairs immediately (one task
already passed). That's too few for ORPO to generalize. Four expansion
strategies:

### Expansion 1 — prompt phrasing variants (~100 pairs)

For each of the 20 tasks, generate 4-5 prompt phrasings of the same
intent. "Fix the syntax error in app.py" → "app.py has a syntax bug,
fix it" / "There's a broken import on line 15 of app.py" / etc. The
expected tool sequence stays the same; the prompt surface varies.

Script: `ml-lab/experiments/2026-04-godspeed-coder/scripts/phrase_variants.py`
(~30 lines, uses LLM to generate paraphrases; validates tool mapping
stays intact).

### Expansion 2 — audit-log mining (~100-200 pairs)

Every `godspeed run` writes a session to `~/.godspeed/audit/<id>.audit.jsonl`.
For any session where `must_fix_injections > 0` OR `tool_error_count > 0`
OR final exit_reason was TOOL_ERROR / MAX_ITERATIONS, we have a
rejected trajectory. Pairing that with a hand-labeled "correct"
sequence gives realistic preference pairs from actual usage.

Script: `mine_godspeed_negatives_from_audit.py`. Humans review and
approve each pair (quality > quantity for ORPO).

### Expansion 3 — per-tool synthetic correctness (~200 pairs)

For each of Godspeed's 23 tools, generate 8-10 canonical scenarios where
that tool is correct. E.g., for `grep_search`: "find all TODO comments",
"locate where X is defined", "count uses of logger". Paired with a
common wrong-tool scenario (web_search / shell looking for the same
thing). Synthetic but grounded in the tool descriptions.

Source of truth: each tool's `description` and `get_schema()` output in
`src/godspeed/tools/`. Script walks the registry, feeds each schema to
an LLM to generate scenarios, validates by running the scenarios
through the base Qwen3-Coder to confirm it gets them wrong
(otherwise the pair isn't useful).

### Expansion 4 — hallucination-correction pairs (~50 pairs)

The model invented `read_file` (actual: `file_read`), `grep` (actual:
`grep_search`), `example_function_name`, and others. Pair each
hallucinated name with the correct one for simple read / search / edit
tasks. Teaches exact tool-name priors.

### Total target
~500-600 pairs. 19 measured + 100 phrases + 150 audit + 200 synthetic +
50 hallucination = 519.

## Training infrastructure (reuse from original plan)

- `ml-lab/experiments/2026-04-godspeed-coder/` scaffold — copy to a new
  sibling: `2026-04-qwen3-coder-priors/`
- `make cloud-train EXP=2026-04-qwen3-coder-priors PROVIDER=runpod`
- W&B online (Linux, no Device Guard). Project:
  `ml-lab-qwen3-coder-priors`.

## Evaluation

Same as Stage A for direct comparability:

1. `python scripts/run_benchmark.py --model <tuned> --tasks benchmarks/tasks.jsonl --out experiments/qwen3-coder-tuned/`
2. Compare tuned `pass_count_jaccard_ge_0_6` vs base's 1/20.
3. Base retention: `lm-eval arc_easy hellaswag`, tuned within 97% of base.

### Success criteria

- Tuned model passes ≥ 12/20 (from 1/20 baseline — 60% pass rate). A
  12-point absolute gain on the same benchmark is a strong, publishable
  result. Weaker than the original plan's 15/20 target because we're
  starting from a lower floor (1/20 vs Qwen2.5's projected 4-5/20).
- Base retention ≥ 97% on arc_easy + hellaswag.
- Tuned model loadable via vLLM + Ollama GGUF (quantize after training).

### Failure mode

If tuned ≤ 2/20 pass rate, the weakness is architectural (the model
can't be taught to prefer local tools within ORPO's capacity at this
scale), not data-driven. Post-mortem; revert to Qwen3-Coder-base as the
production model; consider larger-scale RL (GRPO with the benchmark
runner as verifier).

## Cost + timeline

| Phase | Work | Cost | Wall time |
|---|---|---|---|
| B.0 — data build | Scripts + human review of mined pairs | $0 | 2-3 days |
| B.1 — cloud dry run | 100 steps on A100-80GB, measure VRAM + loss | $2-3 | 1 hour |
| B.2 — full training | ORPO 1 epoch × 500 pairs ≈ 250 steps | $8-15 | 3-4 hours |
| B.3 — eval + GGUF quant | Benchmark + quantize + push to HF | $2 | 2 hours |
| B.4 — writeup | Comparison doc + blog post draft | $0 | 1 day |
| **Total** | | **$12-20** | **~1 week** |

Down from the original Stage B's $25-50 and ~2 weeks.

## Open decisions

1. **Budget cap**: default $20 on RunPod. OK?
2. **Which data-expansion strategies to implement**: all 4, or start
   with the 2 cheapest (audit-log mining + hallucination-correction)
   and see if the initial ~150 pairs move the needle before investing
   in phrase variants + synthetic?
3. **LoRA targets**: start conservative (attention only) or go broader
   (include expert projections)? Conservative is safer and fits in
   VRAM; broader has higher capacity but risks destabilizing the
   expert knowledge the base already has.
4. **Skip Stage C entirely, or keep it as a cheaper parallel run?**
   Stage C (Qwen2.5-14B training on cloud, $8-15) was originally a
   pipeline-validation step. Now that we have a concrete target
   (Qwen3-Coder), Stage C is less useful — the pipeline gets validated
   by B.1 dry run anyway. My read: skip it; save the dollars.

## What gets shipped at the end

On success:

- `omnipotence-eth/godspeed-coder-qwen3-30b-priors-v0.1` on HF Hub
  (tuned LoRA adapter + merged weights + GGUF quant)
- `experiments/qwen3-coder-tuned/comparison.md` (base 1/20 → tuned N/20
  on the same benchmark; raw numbers; per-task breakdown)
- W&B run link
- Blog post / portfolio artifact: "I diagnosed and fixed a frontier
  open coding model's tool-selection weakness with 500 preference pairs"

This is a more specific, measured, defensible story than the original
plan's "I fine-tuned Qwen3-Coder" — because we're the only ones with
the measured weakness (Stage A) that motivated the fix.
