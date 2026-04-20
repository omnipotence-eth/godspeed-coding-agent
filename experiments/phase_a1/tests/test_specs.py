"""Tests for experiments.phase_a1.specs — focused on the guarantees the
generator relies on: per-tool floor, category mix, determinism, and the
rerouting of edit-tool + single_tool combos to multi_turn (which a prod
smoke run showed was a drop-rate hotspot)."""

from __future__ import annotations

from collections import Counter

from experiments.phase_a1.specs import (
    DEFAULT_CATEGORY_MIX,
    EDIT_TOOLS_REQUIRING_CONTEXT,
    build_specs,
)


def _counts_by_category_for(specs: list, tool: str) -> Counter:
    return Counter(s.category for s in specs if s.primary_tool == tool)


def test_edit_tools_never_receive_single_tool_category() -> None:
    """file_edit, diff_apply, and notebook_edit should have zero single_tool
    specs — the builder must reroute that quota into multi_turn so the
    blueprint planner can insert a grounding read before the edit."""
    specs = build_specs(total=6200, floor_per_tool=200)
    for tool in EDIT_TOOLS_REQUIRING_CONTEXT:
        by_cat = _counts_by_category_for(specs, tool)
        assert by_cat["single_tool"] == 0, (
            f"{tool} still has {by_cat['single_tool']} single_tool specs "
            f"(expected 0 after edit-tool reroute)"
        )
        # All categories combined must still meet the per-tool floor.
        assert sum(by_cat.values()) >= 200, f"{tool} total dropped below floor: {dict(by_cat)}"


def test_non_edit_tools_still_use_the_default_mix() -> None:
    """Rerouting must NOT change the category distribution for tools that
    don't need grounding context — those should still follow the default
    mix (~70/15/10/5)."""
    specs = build_specs(total=6200, floor_per_tool=200)
    for tool in ("file_read", "grep_search", "git"):
        by_cat = _counts_by_category_for(specs, tool)
        total = sum(by_cat.values())
        single_ratio = by_cat["single_tool"] / total
        # Allow some wiggle since the largest-remainder method rounds.
        expected = DEFAULT_CATEGORY_MIX["single_tool"]
        assert abs(single_ratio - expected) < 0.05, (
            f"{tool} single_tool ratio {single_ratio:.2f} drifted from "
            f"{expected:.2f}: {dict(by_cat)}"
        )


def test_total_is_preserved_after_reroute() -> None:
    specs = build_specs(total=6200, floor_per_tool=200)
    assert len(specs) == 6200


def test_builder_is_deterministic() -> None:
    a = build_specs(total=6200, floor_per_tool=200, seed=42)
    b = build_specs(total=6200, floor_per_tool=200, seed=42)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
