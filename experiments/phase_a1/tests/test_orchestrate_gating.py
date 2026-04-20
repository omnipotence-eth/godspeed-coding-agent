"""Tests for the inline quality gates added to ``orchestrate.py`` and the
anchor-driven blueprint few-shot injection in ``blueprints.py``.

We don't run the full async pipeline end-to-end (that would hit live LLM
providers); instead we test the helper functions and the rendering layer
that the gates depend on. The gating *behavior* itself is exercised in
the smoke runs.
"""

from __future__ import annotations

import json

from experiments.phase_a1.blueprints import (
    _format_anchor_for_few_shot,
    _render_few_shot_block,
    _render_prompts,
)
from experiments.phase_a1.orchestrate import (
    _infer_category_for_anchor,
    _pick_blueprint_few_shots,
)
from experiments.phase_a1.specs import GenerationSpec

# ---------------------------------------------------------------------------
# Anchor projection / few-shot rendering
# ---------------------------------------------------------------------------


def _anchor(user: str, calls: list[tuple[str, dict]]) -> dict:
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": user},
    ]
    if calls:
        msgs.append(
            {
                "role": "assistant",
                "content": "doing",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for i, (name, args) in enumerate(calls)
                ],
            }
        )
    else:
        msgs.append({"role": "assistant", "content": "no tools"})
    return {"messages": msgs, "tools": []}


def test_format_anchor_extracts_user_intent_and_calls() -> None:
    a = _anchor("read the config", [("file_read", {"file_path": "/c.yaml"})])
    projected = _format_anchor_for_few_shot(a)
    assert projected["user_intent"] == "read the config"
    assert projected["planned_calls"] == [
        {"tool_name": "file_read", "arguments": {"file_path": "/c.yaml"}}
    ]


def test_format_anchor_returns_none_when_no_user_message() -> None:
    a = {"messages": [{"role": "system", "content": "sys"}]}
    assert _format_anchor_for_few_shot(a) is None


def test_render_few_shot_block_empty_when_no_shots() -> None:
    assert _render_few_shot_block(None) == ""
    assert _render_few_shot_block([]) == ""


def test_render_few_shot_block_includes_each_anchor() -> None:
    shots = [
        _anchor("read x", [("file_read", {"file_path": "/x"})]),
        _anchor("write y", [("file_write", {"file_path": "/y", "content": "z"})]),
    ]
    block = _render_few_shot_block(shots)
    assert "Reference gold blueprints" in block
    assert "read x" in block
    assert "file_read" in block
    assert "write y" in block
    assert "file_write" in block


def test_render_prompts_with_few_shots_embeds_block() -> None:
    spec = GenerationSpec(
        index=0, primary_tool="file_read", category="single_tool", seed=1, stream="synthetic"
    )
    shots = [_anchor("example", [("file_read", {"file_path": "/x.py"})])]
    _, user_with = _render_prompts(spec, few_shots=shots)
    _, user_without = _render_prompts(spec, few_shots=None)
    assert "Reference gold blueprints" in user_with
    assert "Reference gold blueprints" not in user_without
    assert len(user_with) > len(user_without)


# ---------------------------------------------------------------------------
# Anchor category bucketing + selection
# ---------------------------------------------------------------------------


def test_infer_category_for_anchor_no_tool() -> None:
    a = _anchor("explain GIL", calls=[])
    assert _infer_category_for_anchor(a) == "no_tool"


def test_infer_category_for_anchor_single_and_multi() -> None:
    single = _anchor("read x", [("file_read", {"file_path": "/x"})])
    multi = _anchor(
        "find then read",
        [("grep_search", {"pattern": "x"}), ("file_read", {"file_path": "/y"})],
    )
    assert _infer_category_for_anchor(single) == "single_tool"
    assert _infer_category_for_anchor(multi) == "multi_turn"


def test_pick_blueprint_few_shots_prefers_matching_category() -> None:
    bucket = {
        "single_tool": [_anchor("a", [("file_read", {"file_path": "/x"})])],
        "multi_turn": [
            _anchor(
                "b",
                [("grep_search", {"pattern": "p"}), ("file_read", {"file_path": "/y"})],
            )
        ],
    }
    picked = _pick_blueprint_few_shots(bucket, "single_tool", spec_seed=42, n=1)
    assert len(picked) == 1
    assert picked[0]["messages"][1]["content"] == "a"


def test_pick_blueprint_few_shots_falls_back_to_any_category() -> None:
    bucket = {
        "single_tool": [_anchor("a", [("file_read", {"file_path": "/x"})])],
    }
    # error_recovery isn't in the bucket — should fall back rather than crash.
    picked = _pick_blueprint_few_shots(bucket, "error_recovery", spec_seed=42, n=1)
    assert len(picked) == 1


def test_pick_blueprint_few_shots_returns_empty_when_no_anchors() -> None:
    assert _pick_blueprint_few_shots({}, "single_tool", spec_seed=42, n=1) == []


def test_pick_blueprint_few_shots_is_deterministic_per_seed() -> None:
    bucket = {
        "single_tool": [
            _anchor(f"prompt {i}", [("file_read", {"file_path": f"/{i}"})]) for i in range(10)
        ],
    }
    a = _pick_blueprint_few_shots(bucket, "single_tool", spec_seed=99, n=2)
    b = _pick_blueprint_few_shots(bucket, "single_tool", spec_seed=99, n=2)
    assert a == b


def test_pick_blueprint_few_shots_caps_at_bucket_size() -> None:
    bucket = {"single_tool": [_anchor("only", [("file_read", {"file_path": "/x"})])]}
    picked = _pick_blueprint_few_shots(bucket, "single_tool", spec_seed=1, n=5)
    assert len(picked) == 1
