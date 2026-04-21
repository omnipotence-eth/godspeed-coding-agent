# Godspeed v3.2 — LLM-judge best@5 on SWE-Bench Lite (dev-23 measurement)

**System:** Godspeed open-source coding agent (v3.2 research track) with an LLM-judge best@k selector.
**Submission date:** 2026-04-21
**Repo:** https://github.com/omnipotence-eth/godspeed-coding-agent
**Technical report:** [findings_v3_2_judge_selector.md](https://github.com/omnipotence-eth/godspeed-coding-agent/blob/main/experiments/swebench_lite/findings_v3_2_judge_selector.md)

## SWE-Bench submission checklist

- [x] Is a `best@k` submission (N=5 candidates, distinct LLM-judge module selects one per instance)
- [x] Does not use SWE-bench test knowledge (`PASS_TO_PASS`, `FAIL_TO_PASS`, `test_patch`, gold `patch`, `hints_text`) — enforced structurally via `SWE_BENCH_RESTRICTED_KEYS` frozenset + sentinel-string regression test
- [x] Does not use the `hints_text` field
- [x] Does not have web-browsing access during benchmark runs (`--allow-web-search=False` flag, `tool_set="local"` excludes web_search / web_fetch / github from the agent registry)
- [x] `system/attempts: "2+"` set in `metadata.yaml`

## Results

**Headline:** 10 / 23 = 43.5% resolved on SWE-Bench Lite dev-23 via a non-oracle LLM-judge selector over a 5-way free-tier ensemble.

**Note on split:** this submission measures dev-23 (a 23-instance subset of the 300-instance dev split). The full Lite test-300 measurement is pending free-tier quota resets.

### Selector comparison on dev-23

| Selector | Resolved | Rate | Δ vs best-single |
|---|---|---|---|
| Best single run (Kimi K2.5 free tier) | 8 / 23 | 34.8% | — |
| **LLM-judge best@5 (this submission)** | **10 / 23** | **43.5%** | **+8.7 pp** |
| Oracle (sb-cli verdicts as selector; *not* leaderboard-eligible) | 12 / 23 | 52.2% | +17.4 pp |

### Constituent runs (N=5)

| Label | Model | Agent mode | Resolved |
|---|---|---|---|
| `e1_kimi` | `nvidia_nim/moonshotai/kimi-k2.5` | single-shot | 8 / 23 |
| `gpt_oss` | `nvidia_nim/openai/gpt-oss-120b` | single-shot | 6 / 23 |
| `iter1` | `nvidia_nim/qwen/qwen3.5-397b-a17b` | single-shot (iter2-capture-fix) | 4 / 23 |
| `seed3` | `nvidia_nim/qwen/qwen3.5-397b-a17b` | single-shot (seed=3) | 5 / 23 |
| `p1_dev23_v3` | `nvidia_nim/moonshotai/kimi-k2.5` | agent-in-loop with Docker swebench_verify tool | 7 / 23 |

### Judge

| Field | Value |
|---|---|
| Judge model | `nvidia_nim/moonshotai/kimi-k2.5` (NVIDIA NIM free tier) |
| Judge input | SWE-Bench `problem_statement` + N anonymized candidate diffs (labeled `Slot 0..4`) |
| Judge output | Strict JSON `{"chosen_slot": int, "reason": str}` |
| Aggregation | Single judge; plurality-vote aggregation across multiple judges supported and tested |
| Eligibility guard | `SWE_BENCH_RESTRICTED_KEYS = {PASS_TO_PASS, FAIL_TO_PASS, hints_text, test_patch, patch}` frozenset, structurally enforced in `_build_judge_context`; sentinel-string regression test (`test_judge_prompt_never_contains_restricted_keys`) asserts none of these fields ever reach the prompt |

### Judge quality

Of the 12 dev-23 instances resolved by any constituent run, the judge picked a resolver on **10/12 = 83.3%** of them — that is the transferable "judge is useful" metric. See technical report for the 2 misses (both required picking a uniquely-resolving non-obvious driver).

## Methodology (short form)

For each SWE-Bench instance:

1. Gather N=5 candidate patches (one per constituent run; missing rows = empty string).
2. Build a prompt containing only `problem_statement` + N anonymized slot diffs.
3. Call the judge LLM with strict JSON output schema.
4. Parse `chosen_slot` and write the selected patch as the final prediction.
5. Fallbacks (degrade-safely): LLM error → shortest non-empty; malformed JSON → shortest non-empty; judge picks empty slot → override to shortest non-empty; all-empty → skip.

Candidates are anonymized by slot index (the judge never sees run labels), preventing "favor the strongest known run" bias.

## Reproducibility

Full command, artifacts, and constituent-run details are in the technical report. The reproducer requires:
- NVIDIA NIM free-tier API key (`NVIDIA_NIM_API_KEY`)
- Python 3.11+ with `godspeed>=3.2.x` + `datasets` + `litellm`
- ~40 min NIM wall-clock for the judge step (5 constituent runs already paid in v3.1.0)

Code reference: [`experiments/swebench_lite/llm_judge_selector.py`](https://github.com/omnipotence-eth/godspeed-coding-agent/blob/main/experiments/swebench_lite/llm_judge_selector.py) (525 lines, 32 unit tests).

## Cost

- API: **$0** (all constituents + judge via NVIDIA NIM free tier)
- Wall-clock: ~6 hours for the 5 constituent runs (v3.1.0) + ~40 min for this judge step
- Hardware: consumer laptop (Windows 11 + RTX 5070 Ti, 96 GB RAM)

## Known limitations (disclosed)

- **dev-23 is a 23-instance subset of the 300-instance Lite dev split.** Variance on small subsets is high; test-300 replication is pending free-tier quota.
- **Single judge model on the headline number.** A follow-up variance run with GPT-OSS-120B as alternate judge (reported in the tech report) produces the identical headline (10/23 = 43.5%) via different per-instance picks, with 82.6% inter-judge agreement on the 23 instances.
- **sb-cli cloud verification of the 10/23 number is pending** — dev quota was exhausted by unrelated CLI glitches during the session. Per-constituent sb-cli reports are authoritative; the offline projection computes each pick against the committed per-run `resolved_ids` sets.
- **Three constituent models are free-tier only.** Results may shift modestly with deterministic modes or paid-tier variants.

## Authors & contact

Tremayne Timms (omnipotence-eth) — independent researcher, Dallas-Fort Worth TX.
GitHub: https://github.com/omnipotence-eth
Repo issues welcome at the Godspeed repo.

## Co-authors

This submission was prepared with substantial assistance from Claude Code (Anthropic Claude Opus 4.7).
