"""Unit tests for experiments.phase_a1.judge.

Covers pure logic (prompt rendering, response parsing, pass/fail threshold).
The network call to the provider router is mocked \u2014 we don't exercise real
GLM here; that's done by the live smoke test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from experiments.phase_a1.judge import (
    DEFAULT_THRESHOLD,
    DIMENSIONS,
    JudgeResult,
    _format_sample_for_judge,
    _parse_scores,
    _render_user,
    judge_sample,
    load_few_shots,
)

# ---------------------------------------------------------------------------
# JudgeResult semantics
# ---------------------------------------------------------------------------


def test_result_with_all_threshold_scores_passes() -> None:
    scores = {d: DEFAULT_THRESHOLD for d in DIMENSIONS}
    assert JudgeResult(scores=scores).passed is True


def test_result_with_one_subthreshold_fails() -> None:
    scores = {d: 5 for d in DIMENSIONS}
    scores["realism"] = DEFAULT_THRESHOLD - 1
    assert JudgeResult(scores=scores).passed is False


def test_result_with_error_fails_regardless_of_scores() -> None:
    scores = {d: 5 for d in DIMENSIONS}
    assert JudgeResult(scores=scores, error="provider_error").passed is False


def test_empty_result_fails() -> None:
    assert JudgeResult().passed is False
    assert JudgeResult().min_score() == 0


def test_min_score_computes_correctly() -> None:
    scores = {"tool_correctness": 5, "arg_correctness": 2, "realism": 4, "coherence": 3}
    assert JudgeResult(scores=scores).min_score() == 2


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _raw(tool: int = 5, arg: int = 5, realism: int = 5, coherence: int = 5) -> str:
    return json.dumps(
        {
            "tool_correctness": tool,
            "arg_correctness": arg,
            "realism": realism,
            "coherence": coherence,
            "reason": "gold",
        }
    )


def test_parse_scores_happy_path() -> None:
    scores, reason = _parse_scores(_raw())
    assert scores == {d: 5 for d in DIMENSIONS}
    assert reason == "gold"


def test_parse_scores_strips_code_fences() -> None:
    text = f"```json\n{_raw()}\n```"
    scores, _ = _parse_scores(text)
    assert scores["tool_correctness"] == 5


def test_parse_scores_rejects_missing_dimension() -> None:
    text = json.dumps({"tool_correctness": 5, "arg_correctness": 5, "realism": 5, "reason": ""})
    with pytest.raises(ValueError, match="coherence"):
        _parse_scores(text)


def test_parse_scores_rejects_non_integer() -> None:
    text = json.dumps(
        {
            "tool_correctness": "five",
            "arg_correctness": 5,
            "realism": 5,
            "coherence": 5,
            "reason": "",
        }
    )
    with pytest.raises(ValueError, match="non-integer"):
        _parse_scores(text)


def test_parse_scores_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _parse_scores(_raw(tool=6))
    with pytest.raises(ValueError, match="out of range"):
        _parse_scores(_raw(tool=0))


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _sample_record() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "you are Godspeed"},
            {"role": "user", "content": "read main.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": json.dumps({"file_path": "main.py"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c0", "content": "print('hi')"},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": [
            {"type": "function", "function": {"name": "file_read", "description": ""}},
        ],
    }


def test_format_sample_includes_all_roles() -> None:
    rendered = _format_sample_for_judge(_sample_record())
    assert "[system]" in rendered
    assert "[user]" in rendered
    assert "[assistant:tool_call] file_read" in rendered
    assert "[tool:" in rendered
    assert "[assistant] Done." in rendered


def test_format_sample_truncates_long_content() -> None:
    rec = _sample_record()
    rec["messages"][3]["content"] = "x" * 5000
    rendered = _format_sample_for_judge(rec)
    assert "truncated" in rendered
    assert len(rendered) < 5000


def test_render_user_without_fewshots_omits_calibration_block() -> None:
    rendered = _render_user(_sample_record(), few_shots=None)
    assert "calibration" not in rendered
    assert "Sample to judge:" in rendered


def test_render_user_with_fewshots_includes_calibration_block() -> None:
    rendered = _render_user(_sample_record(), few_shots=[_sample_record()])
    assert "calibration" in rendered
    assert rendered.count("[system]") >= 2  # fewshot + sample


# ---------------------------------------------------------------------------
# load_few_shots
# ---------------------------------------------------------------------------


def test_load_few_shots_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_few_shots(tmp_path / "missing.jsonl") == []


def test_load_few_shots_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    rec = _sample_record()
    with path.open("w", encoding="utf-8") as f:
        for _ in range(10):
            f.write(json.dumps(rec) + "\n")
    shots = load_few_shots(path, limit=3)
    assert len(shots) == 3


def test_load_few_shots_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_sample_record()) + "\n")
        f.write("not json\n")
        f.write(json.dumps(_sample_record()) + "\n")
    shots = load_few_shots(path, limit=5)
    assert len(shots) == 2


# ---------------------------------------------------------------------------
# judge_sample (mocked router)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    text: str


class _FakeRouter:
    """Minimal async stand-in for ProviderRouter used only by tests."""

    def __init__(self, *, text: str | None = None, raise_exc: Exception | None = None) -> None:
        self._text = text
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        assert self._text is not None
        return _FakeResponse(text=self._text)


@pytest.mark.asyncio
async def test_judge_sample_passes_when_scores_high() -> None:
    router = _FakeRouter(text=_raw())
    result = await judge_sample(_sample_record(), router)  # type: ignore[arg-type]
    assert result.passed is True
    assert result.min_score() == 5
    assert router.calls[0]["tier"] == "judge"


@pytest.mark.asyncio
async def test_judge_sample_drops_when_any_dim_below_threshold() -> None:
    router = _FakeRouter(text=_raw(realism=DEFAULT_THRESHOLD - 1))
    result = await judge_sample(_sample_record(), router)  # type: ignore[arg-type]
    assert result.passed is False
    assert result.error is None


@pytest.mark.asyncio
async def test_judge_sample_captures_parse_error() -> None:
    router = _FakeRouter(text="not json at all")
    result = await judge_sample(_sample_record(), router)  # type: ignore[arg-type]
    assert result.passed is False
    assert result.error is not None
    assert "parse_error" in result.error


@pytest.mark.asyncio
async def test_judge_sample_captures_provider_error() -> None:
    router = _FakeRouter(raise_exc=RuntimeError("all quotas exhausted"))
    result = await judge_sample(_sample_record(), router)  # type: ignore[arg-type]
    assert result.passed is False
    assert result.error is not None
    assert "provider_error" in result.error


@pytest.mark.asyncio
async def test_judge_sample_passes_few_shots_into_prompt() -> None:
    router = _FakeRouter(text=_raw())
    few = [_sample_record()]
    await judge_sample(_sample_record(), router, few_shots=few)  # type: ignore[arg-type]
    assert "calibration" in router.calls[0]["user"]


@pytest.mark.asyncio
async def test_judge_sample_temperature_is_zero_by_default() -> None:
    router = _FakeRouter(text=_raw())
    await judge_sample(_sample_record(), router)  # type: ignore[arg-type]
    assert router.calls[0]["temperature"] == 0.0
