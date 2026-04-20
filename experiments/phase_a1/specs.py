"""Stratified generation-spec builder for Phase A1.

Produces a deterministic list of ``GenerationSpec`` records describing what to
generate: one per final sample. Guarantees:

  - Per-tool floor: every tool gets at least ``floor_per_tool`` samples.
  - Category mix: overall ratio honors ``DEFAULT_CATEGORY_MIX`` within ±1%.
  - Deterministic: output is a pure function of ``(total, floor, seed)``.
  - Resumable: resume by filtering specs whose ``index`` is already produced.

The ``no_tool`` category is assigned a synthetic primary_tool chosen uniformly
(the value is recorded for bookkeeping; the narrator is instructed to not emit
any tool call regardless).
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.phase_a1.registry_builder import ALL_TOOLS

logger = logging.getLogger(__name__)


CATEGORIES: tuple[str, ...] = (
    "single_tool",
    "multi_turn",
    "no_tool",
    "error_recovery",
)

DEFAULT_CATEGORY_MIX: dict[str, float] = {
    "single_tool": 0.70,
    "multi_turn": 0.15,
    "no_tool": 0.10,
    "error_recovery": 0.05,
}

# Tools whose blueprints consistently need prior context in the session to
# produce a valid call — old_string for file_edit must match real file
# content, diff_apply needs a hunk that applies to a known file, notebook_edit
# needs to see the cells first. When these are paired with single_tool the
# blueprint planner can't ground the arguments, so the real tool errors out
# and the judge fails the sample on arg_correctness or coherence. The spec
# builder reroutes every such (tool, single_tool) cell to (tool, multi_turn)
# so the first call can be a file_read that grounds the subsequent edit.
EDIT_TOOLS_REQUIRING_CONTEXT: frozenset[str] = frozenset(
    {"file_edit", "diff_apply", "notebook_edit"}
)


@dataclass(frozen=True)
class GenerationSpec:
    """One sample to produce.

    Attributes:
        index: zero-based global position in the generation queue
        primary_tool: the tool the sample primarily exercises (bookkeeping only
            for ``no_tool`` category, which must not emit any tool call)
        category: one of ``CATEGORIES``
        seed: per-sample PRNG seed for blueprint + narrator variability
        stream: which stream produces this spec (``synthetic`` | ``swesmith`` |
            ``augment``) — orchestrator uses this to route the work
    """

    index: int
    primary_tool: str
    category: str
    seed: int
    stream: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "primary_tool": self.primary_tool,
            "category": self.category,
            "seed": self.seed,
            "stream": self.stream,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationSpec:
        return cls(
            index=int(d["index"]),
            primary_tool=str(d["primary_tool"]),
            category=str(d["category"]),
            seed=int(d["seed"]),
            stream=str(d.get("stream", "synthetic")),
        )


@dataclass
class SpecBudget:
    """Computed per-tool x per-category target counts."""

    total: int
    per_tool: dict[str, int]
    per_category: dict[str, int]
    grid: dict[tuple[str, str], int] = field(default_factory=dict)


def _balanced_category_for_tool(
    tool: str,
    tool_total: int,
    mix: dict[str, float],
) -> dict[str, int]:
    """Distribute ``tool_total`` across categories matching ``mix``.

    Uses largest-remainder method so the sum equals ``tool_total`` exactly.
    """
    raw = {cat: tool_total * frac for cat, frac in mix.items()}
    floors = {cat: int(v) for cat, v in raw.items()}
    remainder = tool_total - sum(floors.values())
    if remainder > 0:
        # Distribute leftover by largest fractional part
        ranked = sorted(mix.keys(), key=lambda c: raw[c] - floors[c], reverse=True)
        for cat in ranked[:remainder]:
            floors[cat] += 1
    return floors


def _compute_budget(
    total: int,
    floor_per_tool: int,
    mix: dict[str, float],
) -> SpecBudget:
    """Plan how many samples each (tool, category) cell gets.

    Floor guarantees each tool ≥ floor_per_tool samples. Remaining budget is
    distributed uniformly across tools.
    """
    if total < len(ALL_TOOLS) * floor_per_tool:
        msg = (
            f"total={total} < {len(ALL_TOOLS)}*floor={floor_per_tool} — "
            f"cannot satisfy per-tool floor"
        )
        raise ValueError(msg)

    base_per_tool = {t: floor_per_tool for t in ALL_TOOLS}
    leftover = total - len(ALL_TOOLS) * floor_per_tool

    # Even distribution of leftover across tools (remainder resolves deterministically)
    per_tool_extra = leftover // len(ALL_TOOLS)
    leftover_rem = leftover - per_tool_extra * len(ALL_TOOLS)
    per_tool = {t: base_per_tool[t] + per_tool_extra for t in ALL_TOOLS}
    for t in ALL_TOOLS[:leftover_rem]:
        per_tool[t] += 1

    grid: dict[tuple[str, str], int] = {}
    for t in ALL_TOOLS:
        cat_split = _balanced_category_for_tool(t, per_tool[t], mix)
        for cat, n in cat_split.items():
            grid[(t, cat)] = n

    # Reroute (edit_tool, single_tool) → (edit_tool, multi_turn). file_edit /
    # diff_apply / notebook_edit can't be grounded in a single-call sample,
    # so they burn judge drops otherwise. Moving the quota to multi_turn
    # lets the blueprint planner insert a grounding file_read before the edit.
    for tool in EDIT_TOOLS_REQUIRING_CONTEXT:
        single_key = (tool, "single_tool")
        stolen = grid.get(single_key, 0)
        if stolen:
            grid[single_key] = 0
            grid[(tool, "multi_turn")] = grid.get((tool, "multi_turn"), 0) + stolen

    per_category: Counter[str] = Counter()
    for (_t, cat), n in grid.items():
        per_category[cat] += n

    return SpecBudget(
        total=sum(per_tool.values()),
        per_tool=per_tool,
        per_category=dict(per_category),
        grid=grid,
    )


def build_specs(
    total: int = 6200,
    floor_per_tool: int = 200,
    mix: dict[str, float] | None = None,
    seed: int = 42,
    stream: str = "synthetic",
) -> list[GenerationSpec]:
    """Produce a full list of ``GenerationSpec`` records.

    The specs are shuffled with a deterministic RNG so adjacent samples exercise
    different tools / categories — this keeps live generation diverse if we
    stop early.
    """
    effective_mix = mix or DEFAULT_CATEGORY_MIX
    if abs(sum(effective_mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"category mix must sum to 1.0; got {effective_mix}")
    unknown = set(effective_mix) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")

    budget = _compute_budget(total, floor_per_tool, effective_mix)

    rng = random.Random(seed)
    unshuffled: list[tuple[str, str]] = []
    for (tool, cat), n in budget.grid.items():
        unshuffled.extend([(tool, cat)] * n)

    rng.shuffle(unshuffled)

    specs: list[GenerationSpec] = []
    for idx, (tool, cat) in enumerate(unshuffled):
        specs.append(
            GenerationSpec(
                index=idx,
                primary_tool=tool,
                category=cat,
                seed=rng.randint(0, 2**31 - 1),
                stream=stream,
            )
        )
    return specs


def persist_specs(specs: list[GenerationSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps(spec.to_dict()) + "\n")
    logger.info("wrote %d specs to %s", len(specs), path)


def load_specs(path: Path) -> list[GenerationSpec]:
    specs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(GenerationSpec.from_dict(json.loads(line)))
    return specs


def summarize(specs: list[GenerationSpec]) -> dict[str, Any]:
    """Return a summary dict (per-tool, per-category counts) for a spec list."""
    tools = Counter(s.primary_tool for s in specs)
    cats = Counter(s.category for s in specs)
    return {
        "total": len(specs),
        "per_tool": dict(sorted(tools.items())),
        "per_category": dict(sorted(cats.items())),
        "tool_floor_met": min(tools.values()) if tools else 0,
        "streams": dict(Counter(s.stream for s in specs)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    p = argparse.ArgumentParser(description="Build Phase A1 generation specs")
    p.add_argument("--total", type=int, default=6200)
    p.add_argument("--floor", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stream", type=str, default="synthetic")
    p.add_argument("--out", type=str, default="experiments/phase_a1/data/specs.jsonl")
    args = p.parse_args()

    specs = build_specs(
        total=args.total,
        floor_per_tool=args.floor,
        seed=args.seed,
        stream=args.stream,
    )
    summary = summarize(specs)
    logger.info("summary: %s", json.dumps(summary, indent=2))
    persist_specs(specs, Path(args.out))
