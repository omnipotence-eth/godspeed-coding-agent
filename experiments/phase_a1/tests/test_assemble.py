"""Tests for ``assemble.py``.

The final assembly is the most consequential single step in Phase A1 —
the file it produces is what training reads. These tests pin: validate
gating, dedup logic, shuffle determinism, source-priority ordering, and
the per-source / per-tool stats output.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.phase_a1.assemble import (
    DEFAULT_SEED,
    _extract_user_prompt,
    _infer_category,
    _prompt_hash,
    assemble,
)

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {"name": name, "description": "x", "parameters": {"type": "object"}},
    }
    for name in (
        "file_read",
        "file_write",
        "file_edit",
        "diff_apply",
        "glob_search",
        "grep_search",
        "code_search",
        "repo_map",
        "shell",
        "test_runner",
        "verify",
        "background_check",
        "git",
        "github",
        "web_search",
        "web_fetch",
        "image_read",
        "pdf_read",
        "notebook_edit",
        "tasks",
        "spawn_agent",
    )
]


def _make_record(user_prompt: str, tool: str = "file_read", args: dict | None = None) -> dict:
    """Build a minimal valid {messages, tools} record."""
    args = args or {"file_path": "/x.py"}
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": "doing it",
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c0", "content": "result"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": SAMPLE_TOOLS,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_extract_user_prompt_returns_first_user_content() -> None:
    rec = _make_record("hello there")
    assert _extract_user_prompt(rec) == "hello there"


def test_extract_user_prompt_empty_when_missing() -> None:
    assert _extract_user_prompt({"messages": []}) == ""


def test_prompt_hash_normalizes_whitespace_and_case() -> None:
    a = _prompt_hash("Hello   World\n")
    b = _prompt_hash("hello world")
    assert a == b


def test_prompt_hash_differs_for_different_text() -> None:
    assert _prompt_hash("a") != _prompt_hash("b")


def test_infer_category_counts_tool_calls() -> None:
    rec0 = {"messages": [{"role": "assistant", "content": "x"}]}
    rec1 = _make_record("x")
    rec2 = {
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "x"}},
                    {"id": "b", "type": "function", "function": {"name": "y"}},
                ],
            },
        ]
    }
    assert _infer_category(rec0) == "no_tool"
    assert _infer_category(rec1) == "single_tool"
    assert _infer_category(rec2) == "multi_turn"


# ---------------------------------------------------------------------------
# Assembly behavior
# ---------------------------------------------------------------------------


def test_assemble_drops_invalid_records(tmp_path: Path) -> None:
    valid = _make_record("good prompt")
    invalid = {"messages": [{"role": "user", "content": "no system"}], "tools": []}
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", [valid, invalid])

    out = tmp_path / "final.jsonl"
    summary = assemble(tmp_path, out)
    assert summary["per_source"]["anchor"]["loaded"] == 2
    assert summary["per_source"]["anchor"]["invalid"] == 1
    assert summary["per_source"]["anchor"]["kept"] == 1


def test_assemble_dedups_across_sources(tmp_path: Path) -> None:
    """Same user prompt in two sources keeps only the first-seen (anchor wins)."""
    shared = _make_record("identical prompt")
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", [shared])
    _write_jsonl(tmp_path / "phase_a1_swesmith_distilled.jsonl", [shared])

    out = tmp_path / "final.jsonl"
    summary = assemble(tmp_path, out)
    assert summary["per_source"]["anchor"]["kept"] == 1
    assert summary["per_source"]["distill"]["duplicates"] == 1
    assert summary["per_source"]["distill"]["kept"] == 0
    assert summary["total_kept"] == 1


def test_assemble_shuffle_is_deterministic_for_seed(tmp_path: Path) -> None:
    records = [_make_record(f"prompt {i}") for i in range(12)]
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", records)

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    assemble(tmp_path, out_a, seed=7)
    assemble(tmp_path, out_b, seed=7)
    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def test_assemble_different_seed_changes_order(tmp_path: Path) -> None:
    records = [_make_record(f"prompt {i}") for i in range(12)]
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", records)

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    assemble(tmp_path, out_a, seed=7)
    assemble(tmp_path, out_b, seed=99)
    assert out_a.read_text(encoding="utf-8") != out_b.read_text(encoding="utf-8")


def test_assemble_skips_missing_source_files(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", [_make_record("only anchor")])
    summary = assemble(tmp_path, tmp_path / "final.jsonl")
    assert summary["per_source"]["augment"]["loaded"] == 0
    assert summary["per_source"]["distill"]["loaded"] == 0


def test_assemble_writes_stats_sidecar(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", [_make_record("a"), _make_record("b")])
    out = tmp_path / "final.jsonl"
    assemble(tmp_path, out)

    stats_path = out.with_suffix(".stats.json")
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["total_kept"] == 2
    assert "final_tool_usage" in stats
    assert "final_category_mix" in stats


def test_assemble_records_are_valid_after_round_trip(tmp_path: Path) -> None:
    from experiments.phase_a1.validate import validate_record

    records = [_make_record(f"prompt {i}") for i in range(5)]
    _write_jsonl(tmp_path / "anchor_opus_50.jsonl", records)

    out = tmp_path / "final.jsonl"
    assemble(tmp_path, out)

    for line in out.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        errs, _, _ = validate_record(rec)
        assert not errs, errs


def test_assemble_preserves_default_seed_value() -> None:
    """Document that DEFAULT_SEED is 42 — match the rest of Phase A1."""
    assert DEFAULT_SEED == 42
