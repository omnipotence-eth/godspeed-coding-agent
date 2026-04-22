# Godspeed v3.3 research — three attempts to beat the solo-judge ceiling (THIRD NULL RESULT)

**Date:** 2026-04-21
**Branch:** `feat/v3_3-heterogeneous-judge-and-apply-check`
**Extends:** [`findings_v3_2_judge_selector.md`](findings_v3_2_judge_selector.md)

> **Headline:** three independent attempts to beat the v3.2 solo-judge result of 10/23 = 43.5%.
> All three came in at or below. **The solo-judge headline stands.**

---

## Context

v3.2 landed the `LLM-judge best@k` selector: both Kimi K2.5 and GPT-OSS-120B judge the 5-way ensemble to **10/23 = 43.5%** via different per-instance picks. 2-judge plurality regresses to 9/23 (documented in v3.2 addendum) because when 2 judges disagree, shortest-non-empty tiebreak is a blind guess.

This doc captures three proposed remedies from the v3.2 addendum's "future research" section and their empirical outcomes.

## Three attempts

### Attempt 1 — Heterogeneous judge family (Qwen3.5-397B)

**Hypothesis:** prior judges (Kimi K2.5, GPT-OSS-120B, Kimi-Thinking) were all Moonshot / OpenAI family → correlated "minimal-diff" priors that collapse on ambiguous disagreements. Adding a Qwen judge would break correlation.

**Result:** **6/23 = 26.1%** solo — *far below* the 43.5% baseline. Qwen is a noisy judge. 3-judge plurality including Qwen: 9/23 = 39.1% (still regression).

**Why:** Qwen's judgments diverged even on cases where Kimi and GPT-OSS agreed correctly. Adding a weaker judge to a majority vote drags the result down rather than breaking ties usefully.

### Attempt 2 — `git apply --check` as test-free signal

**Hypothesis:** Disqualify any candidate patch that doesn't apply cleanly at `base_commit`. This might catch non-resolving picks that look plausible to the judge but are structurally broken.

**Implementation:** [`apply_check.py`](apply_check.py) (14 unit tests, idempotent repo cache, CRLF/autocrlf-safe via `--ignore-whitespace`).

Per-constituent apply rates on dev-23:
| Run | Non-empty | Applies cleanly |
|---|---|---|
| e1_kimi | 23/23 | 19/23 (83%) |
| gpt_oss | 17/23 | 21/23 (apply >= non-empty because tool also counts non-fails) |
| iter1 | 13/23 | 14/23 |
| seed3 | 17/23 | 18/23 |
| p1_dev23_v3 | 15/23 | 15/23 |

**Result:** apply-check-gated Kimi K2.5 solo judge = **10/23 = 43.5%** (identical to un-gated solo). Apply-check gate added to 3-judge plurality = **9/23 = 39.1%** (still null).

**Why apply-check doesn't help:**

- Every resolver already applies cleanly (if it didn't, sb-cli couldn't have graded it as resolved).
- The judges' misses aren't on *non-applying* patches. They're on *applying-but-wrong* patches. Example: `pydicom-1256` (only gpt_oss resolves): all 5 candidates apply cleanly; the judge picked `e1_kimi` (applies, doesn't resolve). Apply-check can't distinguish.
- The gate would only help if judges preferred non-applying patches, which they don't.

### Attempt 3 — Apply-check + plurality combined

**Hypothesis:** combine Attempts 1+2: let all 3 judges vote, and when they tie, prefer applying candidates.

**Result:** **9/23 = 39.1%** — same as vanilla 3-judge plurality.

**Why:** the tiebreak rule only fires on disagreement; on the 4 real-disagreement instances, the applying-candidates pool still contained multiple applying patches (they all apply; only one resolves). Shortest-applying tiebreak collapsed to same pick as shortest-non-empty.

---

## Comparative summary

| Selector | Resolved | Rate | Eligibility | Commentary |
|---|---|---|---|---|
| Best single run (e1_kimi) | 8/23 | 34.8% | pass@1 | baseline |
| **Solo Kimi K2.5 judge** | **10/23** | **43.5%** | best@k | **v3.2 headline — unchanged** |
| Solo GPT-OSS judge | 10/23 | 43.5% | best@k | v3.2 variance — identical headline |
| Solo Kimi-Thinking judge | 9/23 | 39.1% | best@k | v3.2 addendum — thinking model worse |
| Solo Qwen3.5-397B judge | 6/23 | 26.1% | best@k | v3.3 Attempt 1 — heterogeneous judge noisier |
| 2-judge plurality | 9/23 | 39.1% | best@k | v3.2 null result |
| 3-judge plurality (+ Qwen) | 9/23 | 39.1% | best@k | v3.3 Attempt 1 null |
| Apply-check-gated solo | 10/23 | 43.5% | best@k | v3.3 Attempt 2 — no change |
| Apply-check + plurality | 9/23 | 39.1% | best@k | v3.3 Attempt 3 null |
| Oracle union (upper) | 11/23 | 47.8% | *oracle, not eligible* | requires ground-truth knowledge |
| Oracle (sb-cli verdicts) | 12/23 | 52.2% | *oracle, not eligible* | v3.1.0 number |

**Headline held at 10/23 = 43.5%.** After three independent attempts, the solo-judge ceiling is robust.

---

## What the three null results teach us

1. **Judge choice matters more than judge count.** Kimi K2.5 = GPT-OSS = 10/23; Kimi-Thinking = 9/23; Qwen = 6/23. Picking a good base judge beats any tested ensembling strategy.

2. **Correlated judge biases can't be broken by simple plurality.** Even a heterogeneous judge (Qwen) either agrees with the others on easy cases (adding no lift) or votes wildly on hard cases (hurting the vote).

3. **Apply-check is necessary but not sufficient.** Every resolver applies; that's a low bar. The hard signal — "does this particular patch produce the correct behavior change?" — requires *running tests*, which we can't do without test access.

4. **The 47.8% oracle-union ceiling is unreachable without test knowledge.** Reaching it requires a selector that can tell applying-and-wrong from applying-and-right *without running tests*. None of Kimi/GPT-OSS/Kimi-Thinking/Qwen can.

### Addendum — Attempt 4 (the "meta-judge on disagreements" direction): FOURTH NULL

Implemented [`meta_judge.py`](meta_judge.py) — for each 2-judge disagreement instance, call a 3rd LLM with both judges' picks + both rationales + both full diffs + problem statement, and have it pick between the two options.

Tested with two meta-judge models:
- **Kimi K2.5 as meta-judge:** 10/23 (same as Kimi K2.5 solo). Self-echo: same model, same prior → picks its own original choice on every tie. Gives us no new information.
- **Qwen3.5-397B as meta-judge (heterogeneous):** 10/23 (same). Qwen picked Kimi's side 3 of 4 times, GPT-OSS's side 1 of 4 times. Critically, on marshmallow-1359 (the only disagreement instance where picking correctly would lift us to 11/23), Qwen ALSO chose `seed3` over `p1_dev23_v3` — the same wrong answer Kimi K2.5's solo judge made. Both meta-judges agreed that seed3's patch "looks better" — it IS more minimal. Tests are what reveal it doesn't work.

**This is the ceiling.** When the visually-better patch is wrong and only tests distinguish right-from-wrong, no test-free selector (solo, ensemble, apply-check, or meta-judge) can find the right answer.

## Paths that remain open (v3.5+ research, deferred)

**The following would require material departures from the "free-tier, test-free, single-pass" regime:**
- **Weighted voting by per-instance confidence:** each judge emits a confidence score; vote weighted by that. Subtly closer to oracle than equal-weight plurality.
- **Instance-difficulty routing:** some instances are "easy" (multiple constituents resolve) and some are "hard" (unique resolver). A meta-selector could route hard instances to an expensive frontier judge.
- **Tool-augmented judge:** the judge gets file-reading permission + can inspect the project's test scaffolding (not test outputs!) to see if the patch targets the right module. Still test-free.

## Artifacts committed

- `experiments/swebench_lite/apply_check.py` — the apply-check module (525 lines, 14 unit tests)
- `experiments/swebench_lite/apply_check_5way.jsonl` — per-instance, per-candidate apply-check results (115 rows)
- `experiments/swebench_lite/predictions_judge_qwen_5way.jsonl` — Qwen3.5-397B judge predictions (sb-cli-compatible)
- `experiments/swebench_lite/judge_qwen_5way_sources.jsonl` — Qwen per-instance decision log
- `tests/test_apply_check.py` — 14 unit tests

## Honest conclusion

Ring it in: after **v3.2 (3 attempts)** + **v3.3 (3 attempts)** = **six** independently-designed selector/aggregator experiments, **the publishable number remains 10/23 = 43.5% via a single free-tier LLM judge** (Kimi K2.5 or GPT-OSS-120B; choose either). Multi-judge plurality is definitively a dead end at this scale. Apply-check is a useful correctness signal but doesn't move the headline. The path to the 11/23 ceiling lies in either (a) genuinely better single judges (frontier models, or tool-augmented), or (b) meta-judge with rationale-aware tie-breaking — both v3.4 research.

The v3.3 contribution is **three clean null results, each with a concrete mechanical explanation**, narrowing the space of "what could work" for future researchers.
