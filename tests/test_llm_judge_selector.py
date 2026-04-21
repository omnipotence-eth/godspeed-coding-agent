"""Tests for experiments/swebench_lite/llm_judge_selector.py.

Covers the non-LLM machinery: context building (eligibility guard),
response parsing, fallback logic, and the eval computation. The judge
call itself is mocked — no live API traffic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments" / "swebench_lite"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from llm_judge_selector import (  # noqa: E402
    MAX_PATCH_CHARS,
    SWE_BENCH_RESTRICTED_KEYS,
    Candidate,
    JudgeDecision,
    _aggregate_plurality,
    _assemble_candidates,
    _build_judge_context,
    _build_judge_prompt,
    _compute_eval,
    _judge_one,
    _multi_judge_one,
    _parse_judge_response,
    _parse_pairs,
    _shortest_nonempty_fallback,
    _truncate_patch,
)

# ---------------------------------------------------------------------------
# Eligibility guard: judge must never see test knowledge
# ---------------------------------------------------------------------------


def test_build_judge_context_returns_only_problem_statement() -> None:
    row = {
        "instance_id": "foo-1",
        "problem_statement": "Null pointer in Schema.dump",
        "PASS_TO_PASS": ["tests/test_schema.py::test_a"],
        "FAIL_TO_PASS": ["tests/test_schema.py::test_bug"],
        "hints_text": "maintainer hint that would leak the fix",
        "test_patch": "diff --git a/tests/test_schema.py ...",
        "patch": "the gold reference patch",
    }
    ctx = _build_judge_context(row)
    assert set(ctx.keys()) == {"problem_statement"}
    assert ctx["problem_statement"] == "Null pointer in Schema.dump"


def test_restricted_keys_frozenset_covers_all_disallowed_fields() -> None:
    # If this set shrinks, the eligibility argument gets weaker.
    # Test is load-bearing against accidental deletion.
    for key in ("PASS_TO_PASS", "FAIL_TO_PASS", "hints_text", "test_patch", "patch"):
        assert key in SWE_BENCH_RESTRICTED_KEYS


def test_judge_prompt_never_contains_restricted_keys() -> None:
    """The assembled user prompt must not contain any restricted field's CONTENTS."""
    candidates = [Candidate(label="a", patch="diff --git a/x b/x\n+foo", char_len=24)]
    messages = _build_judge_prompt(problem_statement="Something is broken", candidates=candidates)
    joined = "\n".join(m["content"] for m in messages)
    # None of these sentinel strings should ever appear — they would indicate
    # that the caller somehow fed restricted content into the prompt.
    for sentinel in ("PASS_TO_PASS", "FAIL_TO_PASS", "hints_text", "test_patch"):
        assert sentinel not in joined, f"{sentinel} leaked into judge prompt"


# ---------------------------------------------------------------------------
# Patch truncation
# ---------------------------------------------------------------------------


def test_truncate_patch_leaves_short_patches_alone() -> None:
    small = "diff --git a/x b/x\n" + "+foo\n" * 10
    assert _truncate_patch(small) == small


def test_truncate_patch_caps_long_patches_at_max() -> None:
    huge = "x" * (MAX_PATCH_CHARS * 3)
    result = _truncate_patch(huge)
    assert len(result) < len(huge)
    assert "truncated" in result


# ---------------------------------------------------------------------------
# Response parsing — tolerant to prose around the JSON
# ---------------------------------------------------------------------------


def test_parse_judge_response_happy_path() -> None:
    raw = '{"chosen_slot": 2, "reason": "minimal, targets the right file"}'
    slot, reason = _parse_judge_response(raw, n_candidates=5)
    assert slot == 2
    assert "minimal" in reason


def test_parse_judge_response_with_prose_around_json() -> None:
    raw = 'After careful review: {"chosen_slot": 0, "reason": "shortest"} -- end.'
    slot, reason = _parse_judge_response(raw, n_candidates=3)
    assert slot == 0
    assert reason == "shortest"


def test_parse_judge_response_null_slot() -> None:
    raw = '{"chosen_slot": null, "reason": "all candidates look wrong"}'
    slot, reason = _parse_judge_response(raw, n_candidates=4)
    assert slot is None
    assert "all candidates" in reason


def test_parse_judge_response_out_of_range_slot_rejected() -> None:
    raw = '{"chosen_slot": 99, "reason": "..."}'
    slot, reason = _parse_judge_response(raw, n_candidates=3)
    assert slot is None
    assert "invalid slot" in reason


def test_parse_judge_response_non_int_slot_rejected() -> None:
    raw = '{"chosen_slot": "one", "reason": "..."}'
    slot, _reason = _parse_judge_response(raw, n_candidates=3)
    assert slot is None


def test_parse_judge_response_no_json_degrades() -> None:
    slot, reason = _parse_judge_response("I think slot two is best.", n_candidates=3)
    assert slot is None
    assert "no JSON" in reason


def test_parse_judge_response_empty_string_degrades() -> None:
    slot, reason = _parse_judge_response("", n_candidates=3)
    assert slot is None
    assert reason == "empty response"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_shortest_nonempty_fallback_picks_shortest() -> None:
    cands = [
        Candidate(label="a", patch="", char_len=0),
        Candidate(label="b", patch="x" * 500, char_len=500),
        Candidate(label="c", patch="x" * 100, char_len=100),
        Candidate(label="d", patch="x" * 200, char_len=200),
    ]
    assert _shortest_nonempty_fallback(cands) == 2  # label=c


def test_shortest_nonempty_fallback_returns_none_when_all_empty() -> None:
    cands = [
        Candidate(label="a", patch="", char_len=0),
        Candidate(label="b", patch="  ", char_len=2),
    ]
    assert _shortest_nonempty_fallback(cands) is None


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------


def test_assemble_candidates_pulls_from_each_run() -> None:
    run_preds = [
        ("run_a", {"inst-1": "patch_a"}),
        ("run_b", {"inst-1": "patch_b", "inst-2": "patch_b2"}),
        ("run_c", {"inst-2": "patch_c"}),  # missing inst-1 on purpose
    ]
    cands = _assemble_candidates("inst-1", run_preds)
    assert len(cands) == 3
    assert cands[0].patch == "patch_a"
    assert cands[1].patch == "patch_b"
    assert cands[2].patch == ""  # missing becomes empty


# ---------------------------------------------------------------------------
# _judge_one — mocked LLM, covers all branches
# ---------------------------------------------------------------------------


class _MockResp:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_judge_one_picks_slot_happy_path() -> None:
    client = AsyncMock()
    client.chat = AsyncMock(return_value=_MockResp('{"chosen_slot": 1, "reason": "minimal fix"}'))
    cands = [
        Candidate(label="a", patch="diff1", char_len=5),
        Candidate(label="b", patch="diff2", char_len=5),
    ]
    row = {"instance_id": "inst-1", "problem_statement": "bug in foo"}
    decision = await _judge_one(client, "inst-1", row, cands)
    assert decision.chosen_slot == 1
    assert decision.strategy == "judge_pick"


@pytest.mark.asyncio
async def test_judge_one_all_empty_short_circuit() -> None:
    client = AsyncMock()
    client.chat = AsyncMock()  # should not be called
    cands = [
        Candidate(label="a", patch="", char_len=0),
        Candidate(label="b", patch="   ", char_len=3),
    ]
    row = {"instance_id": "inst-1", "problem_statement": "bug"}
    decision = await _judge_one(client, "inst-1", row, cands)
    assert decision.chosen_slot is None
    assert decision.strategy == "judge_empty_fallback"
    client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_judge_one_falls_back_on_llm_error() -> None:
    client = AsyncMock()
    client.chat = AsyncMock(side_effect=RuntimeError("rate limit"))
    cands = [
        Candidate(label="a", patch="", char_len=0),
        Candidate(label="b", patch="nonempty" * 5, char_len=40),
        Candidate(label="c", patch="x" * 200, char_len=200),
    ]
    row = {"instance_id": "inst-1", "problem_statement": "bug"}
    decision = await _judge_one(client, "inst-1", row, cands)
    assert decision.strategy == "judge_parse_error"
    assert decision.chosen_slot == 1  # shortest non-empty
    assert "rate limit" in decision.reason


@pytest.mark.asyncio
async def test_judge_one_overrides_empty_slot_pick() -> None:
    """If the judge picks slot 0 but slot 0 is empty, we fall back."""
    client = AsyncMock()
    client.chat = AsyncMock(
        return_value=_MockResp('{"chosen_slot": 0, "reason": "picking slot 0"}')
    )
    cands = [
        Candidate(label="a", patch="", char_len=0),
        Candidate(label="b", patch="nonempty_patch", char_len=14),
    ]
    row = {"instance_id": "inst-1", "problem_statement": "bug"}
    decision = await _judge_one(client, "inst-1", row, cands)
    assert decision.chosen_slot == 1  # overridden to non-empty
    assert decision.strategy == "judge_parse_error"


# ---------------------------------------------------------------------------
# Offline eval — the research-facing number
# ---------------------------------------------------------------------------


def test_compute_eval_recovery_math(tmp_path: Path) -> None:
    # 3 runs; oracle union = {i1, i2, i3}, best single = run_b (resolves 2)
    report_a = tmp_path / "a.json"
    report_a.write_text(json.dumps({"resolved_ids": ["i1"]}))
    report_b = tmp_path / "b.json"
    report_b.write_text(json.dumps({"resolved_ids": ["i1", "i2"]}))
    report_c = tmp_path / "c.json"
    report_c.write_text(json.dumps({"resolved_ids": ["i3"]}))

    pairs = [
        (tmp_path / "preds_a.jsonl", "run_a"),
        (tmp_path / "preds_b.jsonl", "run_b"),
        (tmp_path / "preds_c.jsonl", "run_c"),
    ]
    run_preds = [
        ("run_a", {"i1": "pa1", "i2": "pa2", "i3": "pa3"}),
        ("run_b", {"i1": "pb1", "i2": "pb2", "i3": "pb3"}),
        ("run_c", {"i1": "pc1", "i2": "pc2", "i3": "pc3"}),
    ]
    # Judge picks: i1 -> run_b (resolves), i2 -> run_b (resolves), i3 -> run_c (resolves)
    decisions = [
        JudgeDecision("i1", 1, "", "judge_pick", 3),
        JudgeDecision("i2", 1, "", "judge_pick", 3),
        JudgeDecision("i3", 2, "", "judge_pick", 3),
    ]
    eval_ = _compute_eval(pairs, [report_a, report_b, report_c], decisions, run_preds)
    # Perfect judge: recovers full oracle ceiling from best-single=2 up to 3.
    assert eval_["oracle_ceiling"] == 3
    assert eval_["best_single_count"] == 2
    assert eval_["judge_resolved_count"] == 3
    assert eval_["lift_available"] == 1
    assert eval_["judge_lift"] == 1
    assert eval_["oracle_lift_recovered_fraction"] == 1.0


def test_compute_eval_mismatched_inputs_raises(tmp_path: Path) -> None:
    pairs = [(tmp_path / "a.jsonl", "a")]
    with pytest.raises(ValueError, match="must match pairs count"):
        _compute_eval(pairs, [], [], [])


# ---------------------------------------------------------------------------
# CLI pair parsing
# ---------------------------------------------------------------------------


def test_parse_pairs_happy_path() -> None:
    pairs = _parse_pairs(["preds.jsonl:kimi", "other.jsonl:qwen"])
    assert len(pairs) == 2
    assert pairs[0] == (Path("preds.jsonl"), "kimi")
    assert pairs[1] == (Path("other.jsonl"), "qwen")


def test_parse_pairs_rejects_missing_colon() -> None:
    with pytest.raises(ValueError, match="path:label"):
        _parse_pairs(["badinput"])


def test_parse_pairs_handles_windows_paths() -> None:
    """Windows paths contain colons after the drive letter."""
    pairs = _parse_pairs([r"C:\preds\kimi.jsonl:kimi"])
    assert pairs[0][1] == "kimi"
    assert "kimi.jsonl" in str(pairs[0][0])


# ---------------------------------------------------------------------------
# Multi-judge plurality aggregation
# ---------------------------------------------------------------------------


def _cands(slots: list[str]) -> list[Candidate]:
    """Helper: build candidates from patch strings."""
    return [Candidate(label=f"r{i}", patch=p, char_len=len(p)) for i, p in enumerate(slots)]


def test_plurality_unanimous_agreement() -> None:
    cands = _cands(["diff_a", "diff_b", "diff_c"])
    decisions = [
        JudgeDecision("x", 1, "r", "judge_pick", 3),
        JudgeDecision("x", 1, "r", "judge_pick", 3),
        JudgeDecision("x", 1, "r", "judge_pick", 3),
    ]
    slot, reason = _aggregate_plurality(decisions, cands)
    assert slot == 1
    assert "3/3" in reason


def test_plurality_majority_wins() -> None:
    cands = _cands(["diff_a", "diff_b", "diff_c"])
    decisions = [
        JudgeDecision("x", 2, "", "judge_pick", 3),
        JudgeDecision("x", 0, "", "judge_pick", 3),
        JudgeDecision("x", 2, "", "judge_pick", 3),
    ]
    slot, reason = _aggregate_plurality(decisions, cands)
    assert slot == 2
    assert "2/3" in reason


def test_plurality_tie_breaks_by_shortest_nonempty() -> None:
    cands = _cands(["short", "longer_patch", "mid"])
    decisions = [
        JudgeDecision("x", 0, "", "judge_pick", 3),
        JudgeDecision("x", 1, "", "judge_pick", 3),
    ]
    slot, reason = _aggregate_plurality(decisions, cands)
    assert slot == 0
    assert "tie" in reason.lower()


def test_plurality_tie_ignores_empty_slots() -> None:
    cands = _cands(["", "nonempty_patch"])
    decisions = [
        JudgeDecision("x", 0, "", "judge_pick", 2),
        JudgeDecision("x", 1, "", "judge_pick", 2),
    ]
    slot, _reason = _aggregate_plurality(decisions, cands)
    assert slot == 1


def test_plurality_all_judges_skipped_falls_back() -> None:
    cands = _cands(["a", "b"])
    decisions = [
        JudgeDecision("x", None, "", "judge_empty_fallback", 2),
        JudgeDecision("x", None, "", "judge_empty_fallback", 2),
    ]
    slot, reason = _aggregate_plurality(decisions, cands)
    assert slot == 0
    assert "fallback" in reason.lower()


def test_plurality_three_way_tie_shortest_wins() -> None:
    cands = _cands(["longest_patch_here", "mid_sz", "s"])
    decisions = [
        JudgeDecision("x", 0, "", "judge_pick", 3),
        JudgeDecision("x", 1, "", "judge_pick", 3),
        JudgeDecision("x", 2, "", "judge_pick", 3),
    ]
    slot, _reason = _aggregate_plurality(decisions, cands)
    assert slot == 2


@pytest.mark.asyncio
async def test_multi_judge_one_calls_all_and_aggregates() -> None:
    client_a = AsyncMock()
    client_a.chat = AsyncMock(return_value=_MockResp('{"chosen_slot": 1, "reason": "judge A"}'))
    client_b = AsyncMock()
    client_b.chat = AsyncMock(return_value=_MockResp('{"chosen_slot": 1, "reason": "judge B"}'))
    cands = _cands(["patch_a", "patch_b"])
    row = {"instance_id": "x", "problem_statement": "bug"}
    decision, per_judge = await _multi_judge_one([client_a, client_b], "x", row, cands)
    assert decision.chosen_slot == 1
    assert decision.strategy == "judge_pick"
    assert len(per_judge) == 2


@pytest.mark.asyncio
async def test_multi_judge_one_all_empty_short_circuit() -> None:
    client_a = AsyncMock()
    client_a.chat = AsyncMock()
    cands = [
        Candidate(label="a", patch="", char_len=0),
        Candidate(label="b", patch="  ", char_len=2),
    ]
    row = {"instance_id": "x", "problem_statement": "bug"}
    decision, per_judge = await _multi_judge_one([client_a], "x", row, cands)
    assert decision.chosen_slot is None
    assert decision.strategy == "judge_empty_fallback"
    assert per_judge == []
    client_a.chat.assert_not_called()
