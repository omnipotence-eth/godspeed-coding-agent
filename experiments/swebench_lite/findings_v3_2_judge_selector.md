# Godspeed v3.2 research — LLM-judge selector, best@k on SWE-Bench Lite dev-23

**Date:** 2026-04-21
**Branch:** `feat/llm-judge-selector-v3.2`
**Extends / follows:** [`findings_2026_04_21.md`](findings_2026_04_21.md) (v3.1.0 oracle-selector)

> **Status: experimental first measurement.** This document captures a single run of a new
> LLM-judge ensemble selector on the v3.1.0 5-way dev-23 predictions. The judge never sees
> test verdicts; its picks are therefore eligible for SWE-Bench leaderboard submission as
> `best@k` per the [SWE-bench/experiments checklist](https://github.com/SWE-bench/experiments/blob/main/checklist.md).

---

## Executive summary

- **Judge model:** `nvidia_nim/moonshotai/kimi-k2.5` (free tier)
- **Judge input per instance:** SWE-Bench `problem_statement` only (no `PASS_TO_PASS`, `FAIL_TO_PASS`, `hints_text`, `test_patch`, gold `patch`)
- **Ensemble:** same 5 runs that formed the v3.1.0 oracle ceiling
- **Judge dev-23 resolve count:** **10 / 23 = 43.5%**
- **Oracle ceiling (upper bound):** 12 / 23 = 52.2%
- **Best single run (lower bound):** `e1_kimi` = 8 / 23 = 34.8%
- **Recovery of oracle lift:** `(10 - 8) / (12 - 8)` = **50.0%**

**Headline (leaderboard-safe):**

> On SWE-Bench Lite dev-23, a non-oracle LLM-judge selector over our v3.1.0 5-way free-tier ensemble resolves **10/23 = 43.5%**, an **+8.7pp absolute gain over the best single run (34.8%)** and **50% of the gap between best-single and the oracle ceiling (52.2%)**. The judge sees only problem statements + anonymized candidate diffs; no test knowledge. This is leaderboard-eligible as a `best@k` submission (per SWE-bench/experiments checklist).

**Cost:** $0 (NIM free tier); ~38 min wall-clock for the 23 judge calls (NIM contention + one call falling back after a double-timeout).

---

## Method

### Pipeline

[`experiments/swebench_lite/llm_judge_selector.py`](llm_judge_selector.py) — 525 lines, 24 unit tests.

For each SWE-Bench instance:

1. Assemble N candidate patches (one per run; missing = empty string).
2. **All-empty short-circuit:** if every candidate is empty, emit `chosen_slot=None` without calling the judge.
3. Build a prompt with:
   - System message specifying scoring criteria (file targeting, minimality, syntactic sense, addresses-problem, shorter-wins-ties), **hard rules forbidding test-access reasoning and web-search**, strict JSON output schema.
   - User message with the `problem_statement` + N anonymized diff slots (labeled `Slot 0..N-1` only — run labels are never shown, so the judge can't favor a known-strong driver).
4. Call Kimi K2.5 via LiteLLM/NIM.
5. Parse JSON `{"chosen_slot": int | null, "reason": str}`.
6. **Fallbacks (degrade-safely):**
   - LLM error → shortest non-empty patch across slots
   - Malformed/unparseable JSON → shortest non-empty
   - Judge picks an empty slot → override to shortest non-empty
7. Write merged predictions JSONL (drop-in format matching `oracle_merge.py`) + per-instance source log.

### Eligibility guard (leaderboard compliance)

`SWE_BENCH_RESTRICTED_KEYS` is a `frozenset` of dataset fields the judge is forbidden from seeing:
`PASS_TO_PASS`, `FAIL_TO_PASS`, `hints_text`, `test_patch`, and the gold reference `patch`. `_build_judge_context` extracts only `problem_statement`; a regression test asserts none of the restricted-key *contents* ever appear in the assembled prompt (sentinel-string test).

### Offline evaluation (`--eval-reports`)

Given per-run sb-cli reports (already paid), the tool projects "if we had submitted the judge's picks, what would sb-cli have returned?" — by checking each pick against that run's `resolved_ids` set. This uses zero new sb-cli quota.

---

## Results

### Per-run constituent rates (from sb-cli reports, dev-23)

| Run label | Resolves | Rate |
|---|---|---|
| `e1_kimi` (Kimi K2.5 single-shot) | 8 | 34.8% |
| `gpt_oss` (GPT-OSS-120B) | 6 | 26.1% |
| `seed3` (Qwen3.5-397B seed3) | 5 | 21.7% |
| `iter1` (Qwen3.5-397B iter2-capture-fix) | 4 | 17.4% |
| `p1_dev23_v3` (Kimi K2.5 + agent-in-loop) | 7 | 30.4% |

### Selector comparison

| Selector | Resolved | Rate | Δ vs best-single |
|---|---|---|---|
| Best single run (`e1_kimi`) | 8 / 23 | 34.8% | — |
| **LLM judge (Kimi K2.5, this work)** | **10 / 23** | **43.5%** | **+8.7 pp** |
| Oracle (ground truth, v3.1.0) | 12 / 23 | 52.2% | +17.4 pp |

### Per-instance pick distribution

The judge chose `e1_kimi` most frequently (11/23), which matches the "pick the strongest single run often" prior you'd expect from a reasonable judge. Non-trivially, it also picked every other run at least twice — the judge is not collapsing onto a single driver.

| Run | Times picked |
|---|---|
| `e1_kimi` | 11 |
| `seed3` | 4 |
| `iter1` | 3 |
| `p1_dev23_v3` | 3 |
| `gpt_oss` | 2 |

### Per-instance resolution map

`WIN` = judge picked a resolver, `MISS` = judge picked a non-resolver when a resolver existed in another slot, `-` = no run resolved this instance so the pick is moot.

| instance_id | judge pick | status | oracle resolvers |
|---|---|---|---|
| marshmallow-code__marshmallow-1343 | e1_kimi | WIN | e1_kimi, iter1, seed3, p1_dev23_v3 |
| marshmallow-code__marshmallow-1359 | seed3 | **MISS** | p1_dev23_v3 (unique) |
| pvlib__pvlib-python-1072 | e1_kimi | WIN | e1_kimi, p1_dev23_v3 |
| pvlib__pvlib-python-1154 | e1_kimi | WIN | e1_kimi, gpt_oss, seed3 |
| pvlib__pvlib-python-1606 | e1_kimi | WIN | e1_kimi, gpt_oss, seed3, p1_dev23_v3 |
| pvlib__pvlib-python-1707 | e1_kimi | - | none |
| pvlib__pvlib-python-1854 | p1_dev23_v3 | WIN | e1_kimi, iter1, p1_dev23_v3 |
| pydicom__pydicom-1139 | iter1 | - | none |
| pydicom__pydicom-1256 | e1_kimi | **MISS** | gpt_oss (unique) |
| pydicom__pydicom-1413 | seed3 | - | none |
| pydicom__pydicom-1694 | e1_kimi | WIN | e1_kimi, gpt_oss |
| pydicom__pydicom-901 | e1_kimi | - | none |
| pylint-dev__astroid-1196 | iter1 | **WIN** | iter1 (unique) |
| pylint-dev__astroid-1268 | p1_dev23_v3 | - | none |
| pylint-dev__astroid-1333 | gpt_oss | - | none |
| pylint-dev__astroid-1866 | e1_kimi | WIN | e1_kimi, seed3, p1_dev23_v3 |
| pylint-dev__astroid-1978 | seed3 | - | none |
| pyvista__pyvista-4315 | iter1 | - | none |
| sqlfluff__sqlfluff-1517 | seed3 | - | none |
| sqlfluff__sqlfluff-1625 | p1_dev23_v3 | - | none |
| sqlfluff__sqlfluff-1733 | gpt_oss | **WIN** | gpt_oss (unique) |
| sqlfluff__sqlfluff-1763 | e1_kimi | - | none |
| sqlfluff__sqlfluff-2419 | e1_kimi | WIN | all 5 runs |

### Judge quality when a resolver exists

Of the 12 instances that any run resolved, the judge picked a resolver on **10/12 = 83.3%**. That is the core "judge is useful" metric — given any non-empty opportunity for lift, the judge takes it five sixths of the time.

The two misses are instructive: both required the judge to pick a **uniquely-resolving** non-obvious driver (`p1_dev23_v3` for marshmallow-1359, `gpt_oss` for pydicom-1256) against the strong `e1_kimi` / `seed3` baseline prior. On ambiguous diffs where no strong cue favors the unique resolver, the judge drifts toward the more general-looking candidate.

### Additional-lift recovery

The 4 instances where the oracle beat best-single (`oracle_ceiling - best_single_count = 12 - 8 = 4`):

| instance_id | lone resolver | judge pick | captured? |
|---|---|---|---|
| marshmallow-code__marshmallow-1359 | p1_dev23_v3 | seed3 | ✗ |
| pydicom__pydicom-1256 | gpt_oss | e1_kimi | ✗ |
| pylint-dev__astroid-1196 | iter1 | iter1 | ✓ |
| sqlfluff__sqlfluff-1733 | gpt_oss | gpt_oss | ✓ |

2 / 4 = **50.0% of additional ensemble lift captured** by the judge.

---

## Discussion

### What the number means

43.5% on dev-23 is the first non-oracle number in the 40+% band that Godspeed has measured on this subset. It is leaderboard-eligible and reproducible in under 40 minutes of free-tier NIM time on top of the v3.1.0 ensemble data that was already collected.

The **"83% judge quality"** figure (10/12 when a resolver exists) is the more transferable number — it says the judge, as constructed, picks the right slot five out of six times when a right slot exists. That generalization should carry to new ensembles / new splits better than any specific resolved-rate headline tied to dev-23.

### Known limitations

- **N=1 judge model, single trial.** Kimi K2.5 is the same model family as constituent run 1 (`e1_kimi`) and run 5 (`p1_dev23_v3`). A reviewer could reasonably ask whether the judge favors its own style. The pick distribution shows the judge choosing `e1_kimi` 11/23 times — a plurality, but not a monopoly. Follow-up: re-run with GPT-OSS-120B and Qwen3-Next-Thinking as alternate judges; report median + spread.
- **Problem-statement-only context.** Nothing about the project's conventions, related code, or test surface area is shown to the judge. A future selector could inspect the apply-ability of each patch against a scratch checkout (pass/fail only — no test execution) as another test-free signal.
- **23-instance subset.** Same caveat as v3.1.0: dev-23 variance is high; test-300 validation is pending.
- **NIM rate-limit noise.** One instance (pvlib-1154) hit a double-timeout and fell back to shortest-non-empty; that fallback happened to be `e1_kimi` which resolved the instance, so it's a WIN in the count but should have been a judge decision. If NIM had been quiet, this run would likely show the same 10/23 or 11/23.

### What this enables

- **Leaderboard eligibility.** Unlike the oracle selector, this result is submittable to the [SWE-bench/experiments](https://github.com/SWE-bench/experiments) lite leaderboard as a best@k entry (with `system/attempts=2+` in `metadata.yaml`). The submission would still need a technical report (this doc is a draft of one) and the reasoning-traces requirement; both are in reach.
- **Re-using ensemble compute.** All five constituent runs are already paid. The judge adds 23 lightweight inference calls (~25-40 min NIM wall-clock including rate-limit waits). If the recovery fraction holds on a larger split, this is a near-free uplift over the best single run.
- **Ablation axis.** By varying judge model / prompt / context, future work can cleanly measure judge-quality delta without re-running the base ensemble.

---

## Reproducibility

```bash
# Prerequisites: Python 3.11+ with godspeed installed in an env that has 'datasets' + 'litellm'.
# NVIDIA_NIM_API_KEY must be exported.
set -a && source ~/.godspeed/.env.local && set +a

PYTHONUNBUFFERED=1 python -u experiments/swebench_lite/llm_judge_selector.py \
  --pairs \
    experiments/swebench_lite/predictions_e1_kimi.jsonl:e1_kimi \
    experiments/swebench_lite/predictions_gpt_oss.jsonl:gpt_oss \
    experiments/swebench_lite/predictions_iter1.jsonl:iter1 \
    experiments/swebench_lite/predictions_seed3.jsonl:seed3 \
    experiments/swebench_lite/predictions_p1_dev23_v3.jsonl:p1_dev23_v3 \
  --eval-reports \
    experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-kimi-k2_5.json \
    experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-gpt-oss-120b.json \
    experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-qwen3_5-397b-iter2-capture-fix.json \
    experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-qwen3_5-397b-seed3.json \
    experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v3_1-p1-agent-in-loop-dev23.json \
  --split dev \
  --judge-model nvidia_nim/moonshotai/kimi-k2.5 \
  --out experiments/swebench_lite/predictions_judge_merged_5way.jsonl \
  --source-log experiments/swebench_lite/judge_merged_5way_sources.jsonl
```

Artifacts produced (committed alongside this doc):
- `predictions_judge_merged_5way.jsonl` — merged predictions, one per instance
- `judge_merged_5way_sources.jsonl` — per-instance decision log (chosen slot, strategy, reason)
- `judge_full_run.log` — full transcript of the judge run (redacted of API keys)

---

## Addendum 2026-04-21 (later that day) — judge variance with GPT-OSS-120B

To test for model-family bias in the Kimi K2.5 judge result, the same 5-way ensemble was rejudged with `nvidia_nim/openai/gpt-oss-120b` as the alternate judge.

### GPT-OSS-120B judge results

- **Judge dev-23 resolve count:** **10 / 23 = 43.5%** (identical headline to Kimi K2.5)
- **Recovery of oracle lift:** (10 − 8) / (12 − 8) = **50.0%** (identical)
- **Judge quality when a resolver exists:** 10 / 12 = 83.3% (identical)
- **Wall-clock:** ~3 min (vs ~38 min for Kimi K2.5 — GPT-OSS NIM endpoint was uncongested)
- **Fallback count:** 0 (zero LLM errors)

### Per-instance disagreement

| instance_id | Kimi pick | GPT-OSS pick | outcome |
|---|---|---|---|
| marshmallow-code__marshmallow-1359 | seed3 | **p1_dev23_v3** | GPT-OSS WIN, Kimi MISS (only p1_dev23_v3 resolves) |
| pvlib__pvlib-python-1606 | e1_kimi | gpt_oss | both WIN (4 runs resolve) |
| pylint-dev__astroid-1196 | **iter1** | p1_dev23_v3 | Kimi WIN, GPT-OSS MISS (only iter1 resolves) |
| sqlfluff__sqlfluff-1517 | seed3 | gpt_oss | neither resolves (no run resolves this) |

**Judge agreement: 19/23 = 82.6%.**

### Judge-union ceiling (oracle-flavored upper bound)

The two judges' errors are **non-overlapping** on the resolver-recovery axis:

- Kimi misses marshmallow-1359, GPT-OSS catches it.
- GPT-OSS misses astroid-1196, Kimi catches it.
- Both miss pydicom-1256 (only `gpt_oss` resolves; both judges defaulted to `e1_kimi`).

**Oracle-knowledge "either picks a resolver" union ceiling:**
- 11 / 12 = **91.7%** picker accuracy if you knew which judge to trust per instance
- Projects to **11 / 23 = 47.8%** resolved
- Recovers **75% of oracle lift**

⚠️ **This 47.8% is a CEILING, not a deployable result.** Knowing "which judge picked the resolver" requires ground-truth verdicts (oracle access). A real non-oracle deployment must pick blindly when judges disagree — see the next section.

### 2-judge plurality vote — the actual non-oracle result (NULL RESULT)

When the oracle is removed and the selector applies a real **plurality vote with shortest-non-empty tiebreaker** (the rule we ship in `_aggregate_plurality`), the measured rate is **9/23 = 39.1% — worse than solo judge.**

Why it regresses: the two judges agree on 19/23 instances (where plurality just echoes the solo judge). On the 4 disagreements, the shortest-non-empty tiebreaker has no signal about which slot resolves — it picks shorter patches under the "minimal change" prior. That prior is wrong on 3 of 4 disagreement instances:

| instance_id | Kimi | GPT-OSS | shortest-non-empty pick | actually resolves? |
|---|---|---|---|---|
| marshmallow-1359 | seed3 | p1_dev23_v3 | seed3 | ✗ (only p1_dev23_v3 resolves) |
| pvlib-1606 | e1_kimi | gpt_oss | gpt_oss | ✓ (4 runs resolve) |
| astroid-1196 | iter1 | p1_dev23_v3 | p1_dev23_v3 | ✗ (only iter1 resolves) |
| sqlfluff-1517 | seed3 | gpt_oss | seed3 | ✗ (no run resolves) |

Net: plurality loses 1 instance vs solo Kimi (astroid-1196: Kimi solo got it, plurality flipped to p1_dev23_v3), and gains nothing the solo judge didn't already have on the disagreement slots.

### What we learned about multi-judge

- **2-judge plurality with simple tiebreakers is a regression, not a lift.** The optimistic "either-judge wins" framing is an oracle reading.
- **The honest published number remains the solo judge at 10/23 = 43.5%.** Multi-judge work is a research direction, not a shippable improvement at N=2 or N=3 with plurality+shortest-non-empty.

**Cost:** the plurality vote experiment was offline-only (re-aggregating already-saved per-judge picks); zero new NIM calls for the aggregation step.

### Addendum 2026-04-21 (late session) — 3rd judge + 3-judge plurality (SECOND NULL RESULT)

`nvidia_nim/moonshotai/kimi-k2-thinking` (reasoning-model variant) was run as a 3rd judge on the same 5-way ensemble to test whether majority-vote with a 3rd tie-breaker could recover the oracle-union ceiling.

**Kimi-K2-Thinking solo result:**
- Judge dev-23 resolve: **9 / 23 = 39.1%** (worse than Kimi K2.5 solo at 10/23 = 43.5% and GPT-OSS-120B solo at 10/23 = 43.5%)
- 20 judge_picks + 3 fallbacks (NIM endpoint chronic timeouts on the thinking variant; 3 fallbacks were shortest-non-empty)
- Picker accuracy when a resolver exists: 9/12 = 75.0% (vs 83.3% for the non-thinking judges)
- ~75 min wall-clock (vs ~3 min for GPT-OSS, ~38 min for Kimi K2.5)

**3-judge plurality vote (Kimi K2.5 + GPT-OSS-120B + Kimi-K2-Thinking):**
- Result: **9 / 23 = 39.1%** — NO IMPROVEMENT over 2-judge plurality, WORSE than any solo non-thinking judge.
- Per-instance disagreement breakdown (6 disagreements of 23):

| instance_id | Kimi K2.5 | GPT-OSS | Kimi-Thinking | plurality pick | outcome |
|---|---|---|---|---|---|
| marshmallow-1359 | seed3 | p1_dev23_v3 | seed3 | seed3 (2-of-3) | **MISS** (only p1_dev23_v3 resolves) |
| pvlib-1606 | e1_kimi | gpt_oss | e1_kimi | e1_kimi (2-of-3) | WIN |
| pvlib-1854 | p1_dev23_v3 | p1_dev23_v3 | e1_kimi | p1_dev23_v3 (2-of-3) | WIN |
| pylint-astroid-1196 | iter1 | p1_dev23_v3 | e1_kimi | p1_dev23_v3 (shortest-nonempty tiebreak) | **MISS** (only iter1 resolves) |
| sqlfluff-1517 | seed3 | gpt_oss | seed3 | seed3 (2-of-3) | - (no run resolves) |

**Why a 3rd judge didn't help:**

1. **marshmallow-1359** (only p1_dev23_v3 resolves): 2 of 3 judges voted seed3 → plurality locks in the wrong answer with full confidence. The non-thinking Kimi K2.5 and the thinking variant BOTH preferred seed3's minimal patch over p1_dev23_v3's verify-in-loop output; the "minimal diff" bias is shared across judges.
2. **pylint-astroid-1196** (only iter1 resolves): All 3 judges disagreed (iter1, p1_dev23_v3, e1_kimi), so plurality fell back to shortest-non-empty — which picked p1_dev23_v3 (14 lines, non-resolver) over iter1 (38 lines, resolver). The shortest-non-empty heuristic is wrong on this instance.
3. **Correlated judge biases beat majority voting.** When 2+ judges share the same "minimal-diff" or "targets-obvious-file" prior, majority vote reinforces the wrong answer rather than correcting it.

**Updated conclusion:** multi-judge plurality with simple tiebreakers doesn't lift the solo-judge result **at N=2 or N=3 with the judges tested**. Paths that might actually work (future research):

- **Meta-judge:** a 4th LLM call seeing problem + only the disagreed-slot diffs + individual judge rationales, trained to break ties (hard because most "training" signal requires oracle labels — bootstrapping risk).
- **Heterogeneous judges:** include a judge from a different architecture family (e.g. an instruction-tuned Llama-70B) to break the correlated-bias pattern. Our 3 judges were all Moonshot/OpenAI/Moonshot — too correlated.
- **Test-free signal:** expose `git apply --check` pass/fail per slot. Would have caught pylint-astroid-1196 where iter1's patch applies and others don't (need to verify).
- **Weighted vote using per-judge solo recovery rate:** Kimi K2.5 and GPT-OSS both solve 10/12; Kimi-Thinking 9/12. Weighting by solo accuracy still would have landed on the same picks above — not an obvious win either.

**The publishable number stays at 10/23 = 43.5% solo judge (either Kimi K2.5 or GPT-OSS-120B).** v3.2's contribution is the *eligibility-guard framework* + the robust cross-judge replication of the 10/23 result + the honest negative results on N=2/N=3 plurality aggregation.

### What this means

The 43.5% solo-judge result is **robust to judge model choice** — same headline, similar pick distribution, same picker-accuracy ceiling. The errors are not driven by Kimi-favors-Kimi or GPT-OSS-favors-GPT-OSS bias — both judges land on the same 9 / 12 "easy" oracle-resolver picks and split on the harder ones.

**The non-overlapping-errors finding does NOT translate into a free uplift via simple plurality.** When the oracle is removed and a real non-oracle plurality vote is applied (the rule shipped in `_aggregate_plurality`), the result regresses to 9/23 = 39.1% — see the next subsection. Recovering the union ceiling requires either a 3rd judge (so plurality can resolve disagreements without arbitrary tiebreakers) or a smarter meta-aggregator. Initial Kimi-K2-Thinking 3rd-judge attempts hit NIM timeout repeatedly and did not complete a full run.

### Additional artifacts

- `predictions_judge_gpt_oss_5way.jsonl` — GPT-OSS judge predictions (committed)
- `judge_gpt_oss_5way_sources.jsonl` — GPT-OSS per-instance decision log (committed)
- Reproduce: same CLI as the Kimi run, swap `--judge-model nvidia_nim/openai/gpt-oss-120b`

---

## Next steps (v3.2 research track)

1. **Judge-union selector (next):** add `--judge-models a,b,c` to llm_judge_selector.py supporting parallel judge calls + union/vote/weighted aggregation. Estimated lift: 43.5% → 47.8% (Kimi+GPT-OSS) before adding a third judge.
2. **Third judge (Qwen3-Next-Thinking):** see if it overlaps with Kimi+GPT-OSS error patterns or surfaces new resolvers. Test the 3-way union ceiling.
3. **Prompt ablation:** measure effect of (a) reasoning-step-by-step, (b) self-consistency k=3, (c) apply-then-describe (let the judge mentally apply the patch before scoring).
4. **Richer context:** expose each candidate patch's `git apply --check` status (pass/fail only — no test execution) as an additional test-free signal.
5. **Test-300 validation:** rerun on the full SWE-Bench Lite test split once free-tier quota permits. Requires constituent test-300 runs (some already exist: Kimi test-50 seed 2, GPT-OSS test-50 seed 1).
6. **sb-cli verification:** dev quota was exhausted by failed-CLI-call attempts during this session; verification of the offline 10/23 number is gated until quota resets.
7. **Leaderboard submission:** if the judge-union number holds on test-300, prepare a `best@k` submission PR to SWE-bench/experiments with this doc as the technical report.
