"""Tests for experiments/swebench_lite/generate_leaderboard_trajs.py.

The generator has one network-touching function (_load_problem_statements,
which hits HuggingFace) — we do not test that here. Everything else is
pure file IO and markdown formatting, which we cover with tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments" / "swebench_lite"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from generate_leaderboard_trajs import _load_jsonl, _write_traj  # noqa: E402

# ---------------------------------------------------------------------------
# _load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    p.write_text(
        json.dumps({"instance_id": "a", "model_patch": "diff1"})
        + "\n"
        + json.dumps({"instance_id": "b", "model_patch": "diff2"})
        + "\n",
        encoding="utf-8",
    )
    rows = _load_jsonl(p)
    assert len(rows) == 2
    assert rows[0]["instance_id"] == "a"
    assert rows[1]["model_patch"] == "diff2"


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    p.write_text(
        json.dumps({"instance_id": "a"}) + "\n\n  \n" + json.dumps({"instance_id": "b"}) + "\n",
        encoding="utf-8",
    )
    rows = _load_jsonl(p)
    assert len(rows) == 2


def test_load_jsonl_empty_file_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert _load_jsonl(p) == []


# ---------------------------------------------------------------------------
# _write_traj — markdown structure + content
# ---------------------------------------------------------------------------


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_write_traj_contains_all_sections(tmp_path: Path) -> None:
    _write_traj(
        tmp_path,
        instance_id="foo__bar-1",
        problem="Null pointer in Schema.dump",
        slot_diffs=[
            ("run_a", "diff --git a/x b/x\n+foo"),
            ("run_b", ""),
        ],
        decision={
            "chosen_slot": 0,
            "chosen_label": "run_a",
            "strategy": "judge_pick",
            "reason": "minimal and targeted",
        },
        chosen_patch="diff --git a/x b/x\n+foo",
    )
    content = _read(tmp_path / "foo__bar-1.md")
    assert "# foo__bar-1" in content
    assert "## Problem statement" in content
    assert "Null pointer in Schema.dump" in content
    assert "## Candidate patches (N=2)" in content
    assert "### Slot 0" in content
    assert "### Slot 1" in content
    assert "## Judge decision" in content
    assert "## Final selected patch" in content
    assert "minimal and targeted" in content


def test_write_traj_anonymizes_slots_labels_in_parens(tmp_path: Path) -> None:
    """Post-hoc labels should only appear in 'post-hoc label:' markers."""
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="problem",
        slot_diffs=[("run_a", "diff_a"), ("run_b", "diff_b")],
        decision={
            "chosen_slot": 1,
            "chosen_label": "run_b",
            "strategy": "judge_pick",
            "reason": "",
        },
        chosen_patch="diff_b",
    )
    content = _read(tmp_path / "i-1.md")
    assert "### Slot 0 (post-hoc label: `run_a`)" in content
    assert "### Slot 1 (post-hoc label: `run_b`)" in content


def test_write_traj_empty_candidate_is_labeled(tmp_path: Path) -> None:
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="problem",
        slot_diffs=[("run_a", ""), ("run_b", "something")],
        decision={
            "chosen_slot": 1,
            "chosen_label": "run_b",
            "strategy": "judge_pick",
            "reason": "",
        },
        chosen_patch="something",
    )
    content = _read(tmp_path / "i-1.md")
    assert "_(empty patch — this constituent run did not produce an edit)_" in content


def test_write_traj_null_chosen_slot(tmp_path: Path) -> None:
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="p",
        slot_diffs=[("run_a", ""), ("run_b", "")],
        decision={
            "chosen_slot": None,
            "strategy": "judge_empty_fallback",
            "reason": "all empty",
        },
        chosen_patch="",
    )
    content = _read(tmp_path / "i-1.md")
    assert "**Chosen slot:** `None`" in content
    assert "_(empty — selector could not find a non-empty candidate)_" in content


def test_write_traj_escapes_nothing_that_would_break_markdown(tmp_path: Path) -> None:
    """Confidence test: realistic unified diff content survives unchanged."""
    real_patch = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "index aaa..bbb 100644\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,3 +1,4 @@\n"
        " def foo():\n"
        "+    raise ValueError('nope')\n"
        '     return "ok"\n'
    )
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="problem text",
        slot_diffs=[("run_a", real_patch)],
        decision={
            "chosen_slot": 0,
            "chosen_label": "run_a",
            "strategy": "judge_pick",
            "reason": "",
        },
        chosen_patch=real_patch,
    )
    content = _read(tmp_path / "i-1.md")
    assert "ValueError" in content
    assert "```diff" in content
    assert "@@ -1,3 +1,4 @@" in content
    # Patch body preserved inside fences
    start = content.index("```diff")
    end = content.index("```", start + 7)
    fenced = content[start:end]
    assert 'return "ok"' in fenced


def test_write_traj_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "i-1.md"
    target.write_text("stale content", encoding="utf-8")
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="new problem",
        slot_diffs=[("run_a", "d")],
        decision={
            "chosen_slot": 0,
            "chosen_label": "run_a",
            "strategy": "judge_pick",
            "reason": "",
        },
        chosen_patch="d",
    )
    content = _read(target)
    assert "stale content" not in content
    assert "new problem" in content


# ---------------------------------------------------------------------------
# End-to-end smoke on mini inputs (mocks _load_problem_statements)
# ---------------------------------------------------------------------------


def test_traj_shows_judge_rationale(tmp_path: Path) -> None:
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="p",
        slot_diffs=[("run_a", "diff")],
        decision={
            "chosen_slot": 0,
            "chosen_label": "run_a",
            "strategy": "judge_pick",
            "reason": "this patch targets the exact file mentioned in the issue",
        },
        chosen_patch="diff",
    )
    content = _read(tmp_path / "i-1.md")
    assert "this patch targets the exact file mentioned in the issue" in content


def test_traj_strategy_fallback_documented(tmp_path: Path) -> None:
    """A degraded decision (LLM error) should still produce a complete trajectory."""
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="p",
        slot_diffs=[("a", "diff")],
        decision={
            "chosen_slot": 0,
            "chosen_label": "a",
            "strategy": "judge_parse_error",
            "reason": "llm error: rate limit",
        },
        chosen_patch="diff",
    )
    content = _read(tmp_path / "i-1.md")
    assert "`judge_parse_error`" in content
    assert "rate limit" in content


@pytest.mark.parametrize("n_slots", [1, 3, 5, 8])
def test_traj_n_slots_rendered(tmp_path: Path, n_slots: int) -> None:
    slot_diffs = [(f"r{i}", f"d{i}") for i in range(n_slots)]
    _write_traj(
        tmp_path,
        instance_id="i-1",
        problem="p",
        slot_diffs=slot_diffs,
        decision={"chosen_slot": 0, "chosen_label": "r0", "strategy": "judge_pick", "reason": ""},
        chosen_patch="d0",
    )
    content = _read(tmp_path / "i-1.md")
    for i in range(n_slots):
        assert f"### Slot {i}" in content
    assert f"N={n_slots}" in content
