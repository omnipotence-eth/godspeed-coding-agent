"""Tests for ``swesmith_distill.py``.

Cover the four pieces independently so a regression in one stage doesn't
silently corrupt the rest:

  1. XML transport parsing (``_parse_record``, ``_split_assistant_content``,
     ``_extract_observation``).
  2. Per-pattern observation → canonical-tool inference
     (``infer_canonical_call``).
  3. Cluster-aware diversity sampling (``diversity_sample``) — determinism,
     shell-cap behavior.
  4. End-to-end on a tiny in-memory corpus, then validation against
     ``validate.validate_record``.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.phase_a1.swesmith_distill import (
    DistilledRecord,
    Turn,
    _distill_one,
    _extract_observation,
    _parse_record,
    _split_assistant_content,
    diversity_sample,
    infer_canonical_call,
    render_to_openai,
    write_distilled_jsonl,
)
from experiments.phase_a1.validate import validate_record

# ---------------------------------------------------------------------------
# Helpers — synthesize swesmith-shaped records on the fly
# ---------------------------------------------------------------------------


def _wrap_call(name: str, args: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _wrap_obs(text: str) -> str:
    body = json.dumps({"name": "tool", "content": "OBSERVATION:\n" + text})
    return f"<tool_response>\n{body}\n</tool_response>"


def _make_swesmith_record(user_prompt: str, turns: list[tuple[str, dict, str]]) -> dict:
    """Build one swesmith-format record from (assistant_text, raw_call, obs) triples."""
    msgs: list[dict] = [
        {"role": "system", "content": "swesmith preamble"},
        {
            "role": "user",
            "content": (
                "<uploaded_files>/testbed</uploaded_files>\n"
                f"<pr_description>{user_prompt}</pr_description>"
            ),
        },
    ]
    for text, call, obs in turns:
        body = text + "\n\n" + _wrap_call(call["name"], call["arguments"])
        msgs.append({"role": "assistant", "content": body})
        msgs.append({"role": "tool", "content": _wrap_obs(obs)})
    return {"messages": msgs}


# ---------------------------------------------------------------------------
# 1. XML parsing
# ---------------------------------------------------------------------------


def test_split_assistant_content_extracts_tool_call() -> None:
    body = "Reasoning here.\n\n" + _wrap_call("file_read", {"file_path": "/x.py"})
    text, call = _split_assistant_content(body)
    assert text == "Reasoning here."
    assert call == {"name": "file_read", "arguments": {"file_path": "/x.py"}}


def test_split_assistant_content_no_tool_call() -> None:
    text, call = _split_assistant_content("plain prose only")
    assert text == "plain prose only"
    assert call is None


def test_split_assistant_content_handles_malformed_json() -> None:
    # Well-formed JSON braces but invalid content — regex matches, parser fails.
    body = 'Reasoning.\n\n<tool_call>\n{"name": broken}\n</tool_call>'
    text, call = _split_assistant_content(body)
    assert text == "Reasoning."
    assert call is None


def test_extract_observation_strips_leading_label() -> None:
    obs = _extract_observation(_wrap_obs("Here's the result of running `cat -n` on /x.py"))
    assert obs.startswith("Here's the result")
    assert "OBSERVATION" not in obs


def test_parse_record_returns_none_for_too_short() -> None:
    assert _parse_record({"messages": [{"role": "user", "content": "x"}]}) is None


def test_parse_record_extracts_user_prompt_and_turns() -> None:
    rec = _make_swesmith_record(
        "Fix the off-by-one in pagination.",
        [
            (
                "Reading the file.",
                {"name": "shell", "arguments": {"command": "# inferred from context"}},
                "Here's the result of running `cat -n` on /testbed/app/pag.py:\n  1\timport x",
            ),
        ],
    )
    parsed = _parse_record(rec)
    assert parsed is not None
    assert parsed.user_prompt == "Fix the off-by-one in pagination."
    assert len(parsed.turns) == 1
    assert parsed.turns[0].observation.startswith("Here's the result")


# ---------------------------------------------------------------------------
# 2. Inference patterns
# ---------------------------------------------------------------------------


def _turn(text: str, raw_name: str, obs: str) -> Turn:
    return Turn(assistant_text=text, raw_call={"name": raw_name, "arguments": {}}, observation=obs)


def test_infer_file_read_from_cat_n() -> None:
    obs = "Here's the result of running `cat -n` on /testbed/x.py:\n  1\tx"
    t = _turn("Reading.", "shell", obs)
    name, args = infer_canonical_call(t)
    assert name == "file_read"
    assert args == {"file_path": "/testbed/x.py"}


def test_infer_file_write_from_created_message() -> None:
    t = _turn("Writing.", "shell", "File created successfully at: /testbed/repro.py")
    name, args = infer_canonical_call(t)
    assert name == "file_write"
    assert args["file_path"] == "/testbed/repro.py"
    assert args["content"] == ""


def test_infer_test_runner_from_pytest_session() -> None:
    obs = "============================= test session starts =============================\n"
    t = _turn("Running `pytest tests/`.", "shell", obs)
    name, args = infer_canonical_call(t)
    assert name == "test_runner"
    assert "path" in args


def test_infer_glob_search_from_path_list() -> None:
    obs = "/testbed/a.py\n/testbed/b.py\n/testbed/c.py"
    t = _turn("Looking for `**/*.py`.", "shell", obs)
    name, args = infer_canonical_call(t)
    assert name == "glob_search"
    assert args["pattern"]


def test_infer_shell_for_pip_install() -> None:
    obs = "Successfully installed pkg-1.0\n"
    t = _turn("Running `pip install pkg`.", "shell", obs)
    name, args = infer_canonical_call(t)
    assert name == "shell"
    assert "pip" in args["command"]


def test_infer_returns_none_when_no_pattern_matches() -> None:
    t = _turn("idk", "file_read", "totally unstructured noise that matches nothing")
    assert infer_canonical_call(t) is None


def test_distill_one_drops_record_with_no_inferable_turns() -> None:
    parsed = _parse_record(
        _make_swesmith_record(
            "Some PR.",
            [("uh", {"name": "shell", "arguments": {}}, "totally noise")],
        )
    )
    assert parsed is not None
    assert _distill_one(parsed) is None


def test_distill_one_keeps_when_inference_ratio_met() -> None:
    parsed = _parse_record(
        _make_swesmith_record(
            "PR.",
            [
                (
                    "Reading.",
                    {"name": "shell", "arguments": {}},
                    "Here's the result of running `cat -n` on /a.py:\n  1\tx",
                ),
                (
                    "Reading.",
                    {"name": "shell", "arguments": {}},
                    "Here's the result of running `cat -n` on /b.py:\n  1\tx",
                ),
            ],
        )
    )
    assert parsed is not None
    distilled = _distill_one(parsed)
    assert distilled is not None
    assert len(distilled.canonical_calls) == 2
    assert {n for n, _ in distilled.canonical_calls} == {"file_read"}


# ---------------------------------------------------------------------------
# 3. Diversity sampling
# ---------------------------------------------------------------------------


def _make_distilled(prompt: str, name: str = "file_read") -> DistilledRecord:
    return DistilledRecord(
        user_prompt=prompt,
        turns=[Turn(assistant_text="t", raw_call={"name": name}, observation="obs")],
        canonical_calls=[(name, {"file_path": "/x.py"})],
    )


def test_diversity_sample_is_deterministic() -> None:
    records = [_make_distilled(f"prompt about topic {i % 5}") for i in range(40)]
    a = diversity_sample(records, target=10, k=3, shell_cap=10, seed=42)
    b = diversity_sample(records, target=10, k=3, shell_cap=10, seed=42)
    assert [r.user_prompt for r in a] == [r.user_prompt for r in b]


def test_diversity_sample_respects_target() -> None:
    records = [_make_distilled(f"unique prompt {i}") for i in range(50)]
    chosen = diversity_sample(records, target=12, k=4, shell_cap=99, seed=1)
    assert len(chosen) == 12


def test_diversity_sample_caps_shell_only() -> None:
    shell_records = [_make_distilled(f"prompt {i}", name="shell") for i in range(30)]
    chosen = diversity_sample(shell_records, target=20, k=3, shell_cap=5, seed=0)
    assert sum(r.is_shell_only for r in chosen) <= 5


def test_diversity_sample_handles_empty_input() -> None:
    assert diversity_sample([], target=10, k=3, shell_cap=5, seed=0) == []


# ---------------------------------------------------------------------------
# 4. Render to OpenAI shape + end-to-end validation
# ---------------------------------------------------------------------------


def test_render_to_openai_produces_valid_record() -> None:
    from experiments.phase_a1.registry_builder import get_tool_schemas

    distilled = _make_distilled("fix the bug")
    record = render_to_openai(distilled, get_tool_schemas())

    assert set(record.keys()) == {"messages", "tools"}
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "user"

    errs, _, _ = validate_record(record)
    assert not errs, errs


def test_render_links_tool_call_id_to_tool_message() -> None:
    from experiments.phase_a1.registry_builder import get_tool_schemas

    distilled = _make_distilled("fix")
    record = render_to_openai(distilled, get_tool_schemas())

    assistant_with_calls = next(
        m for m in record["messages"] if m["role"] == "assistant" and m.get("tool_calls")
    )
    tool_msg = next(m for m in record["messages"] if m["role"] == "tool")
    assert assistant_with_calls["tool_calls"][0]["id"] == tool_msg["tool_call_id"]


def test_end_to_end_on_tiny_corpus(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    out = tmp_path / "out.jsonl"

    records = [
        _make_swesmith_record(
            f"PR #{i}: address bug in subsystem {i % 3}",
            [
                (
                    "Reading.",
                    {"name": "shell", "arguments": {}},
                    f"Here's the result of running `cat -n` on /testbed/x{i}.py:\n  1\tx",
                ),
                (
                    "Writing fix.",
                    {"name": "shell", "arguments": {}},
                    f"File created successfully at: /testbed/fix{i}.py",
                ),
            ],
        )
        for i in range(20)
    ]
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    summary = write_distilled_jsonl(
        src, out, target=10, k=3, shell_cap=10, seed=7
    )
    assert summary["written"] == 10

    for line in out.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        errs, _, _ = validate_record(rec)
        assert not errs, errs
