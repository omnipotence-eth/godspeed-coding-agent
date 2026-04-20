"""Tests for the Opus-authored anchor sample builder.

The anchor samples are the gold reference set for judge calibration and
held-out eval. We assert structural properties so the file can't silently
regress: every sample is schema-valid, every tool is covered, every
``tool_call_id`` is unique within its record, and the on-disk JSONL
round-trips cleanly back to the same content.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.phase_a1.anchor_opus import (
    _BUILDERS,
    build_anchor_samples,
    write_anchor_jsonl,
)
from experiments.phase_a1.registry_builder import ALL_TOOLS
from experiments.phase_a1.validate import validate_record


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return build_anchor_samples()


def test_exactly_50_samples(samples: list[dict]) -> None:
    assert len(samples) == 50
    assert len(_BUILDERS) == 50


def test_every_sample_has_messages_and_tools(samples: list[dict]) -> None:
    for i, rec in enumerate(samples):
        assert set(rec.keys()) == {"messages", "tools"}, f"sample {i} extra/missing keys"
        assert isinstance(rec["messages"], list) and rec["messages"], f"sample {i} messages empty"
        assert isinstance(rec["tools"], list) and rec["tools"], f"sample {i} tools empty"


def test_tools_field_is_canonical_21(samples: list[dict]) -> None:
    expected = set(ALL_TOOLS)
    for i, rec in enumerate(samples):
        names = {t["function"]["name"] for t in rec["tools"]}
        missing = expected - names
        extra = names - expected
        assert names == expected, f"sample {i} tools mismatch: missing={missing} extra={extra}"


def test_every_sample_validates_against_schema(samples: list[dict]) -> None:
    for i, rec in enumerate(samples):
        errs, _, _ = validate_record(rec)
        assert not errs, f"sample {i} failed validation: {errs[:3]}"


def test_first_message_is_system(samples: list[dict]) -> None:
    for i, rec in enumerate(samples):
        assert rec["messages"][0]["role"] == "system", (
            f"sample {i} first role is {rec['messages'][0]['role']!r}"
        )


def test_tool_call_ids_unique_within_record(samples: list[dict]) -> None:
    for i, rec in enumerate(samples):
        ids: list[str] = []
        for msg in rec["messages"]:
            for tc in msg.get("tool_calls") or []:
                ids.append(tc["id"])
        assert len(ids) == len(set(ids)), f"sample {i} duplicate tool_call ids: {ids}"


def test_every_tool_response_has_matching_call(samples: list[dict]) -> None:
    """Each `role=tool` message must reference a prior assistant tool_call."""
    for i, rec in enumerate(samples):
        seen_call_ids: set[str] = set()
        for msg in rec["messages"]:
            if msg["role"] == "assistant":
                for tc in msg.get("tool_calls") or []:
                    seen_call_ids.add(tc["id"])
            elif msg["role"] == "tool":
                tcid = msg["tool_call_id"]
                assert tcid in seen_call_ids, (
                    f"sample {i} tool message refs unknown call id {tcid!r}"
                )


def test_tool_call_arguments_are_valid_json(samples: list[dict]) -> None:
    for i, rec in enumerate(samples):
        for msg in rec["messages"]:
            for j, tc in enumerate(msg.get("tool_calls") or []):
                args = tc["function"]["arguments"]
                assert isinstance(args, str), f"sample {i} tc[{j}] args not a string"
                json.loads(args)  # raises if invalid


def test_every_tool_covered_at_least_twice(samples: list[dict]) -> None:
    usage: Counter[str] = Counter()
    for rec in samples:
        for msg in rec["messages"]:
            for tc in msg.get("tool_calls") or []:
                usage[tc["function"]["name"]] += 1
    for tool in ALL_TOOLS:
        assert usage[tool] >= 2, f"tool {tool!r} covered only {usage[tool]} times (min 2)"


def test_category_mix_in_target_range(samples: list[dict]) -> None:
    """Documented mix: ~30 single, ~10 multi, ~5 no-tool, ~5 error_recovery."""
    categories: Counter[str] = Counter()
    for rec in samples:
        n_calls = sum(len(msg.get("tool_calls") or []) for msg in rec["messages"])
        if n_calls == 0:
            categories["no_tool"] += 1
        elif n_calls == 1:
            categories["single_tool"] += 1
        else:
            categories["multi_turn"] += 1
    assert categories["no_tool"] >= 2, "need at least 2 no-tool conversational samples"
    assert categories["single_tool"] >= 20, "need a strong single-tool baseline"
    assert categories["multi_turn"] >= 10, "need at least 10 multi-turn samples"


def test_no_assistant_message_is_empty_when_no_tool_call(samples: list[dict]) -> None:
    """A leaf assistant message (no tool_calls) must have non-empty content."""
    for i, rec in enumerate(samples):
        for j, msg in enumerate(rec["messages"]):
            if msg["role"] != "assistant":
                continue
            if msg.get("tool_calls"):
                continue
            assert msg.get("content"), f"sample {i} msg[{j}] empty assistant content"


def test_write_and_reload_round_trip(samples: list[dict], tmp_path: Path) -> None:
    out = tmp_path / "anchor.jsonl"
    summary = write_anchor_jsonl(out)
    assert summary["samples"] == 50
    assert summary["missing_tools"] == []

    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(loaded) == 50
    assert loaded == samples, "reloaded JSONL diverges from in-memory samples"


def test_write_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "anchor.jsonl"
    write_anchor_jsonl(nested)
    assert nested.exists()
    assert sum(1 for _ in nested.read_text(encoding="utf-8").splitlines() if _.strip()) == 50
