"""Tests for ``narrator.py`` — retry loop + anti-hallucination prompt guards.

Tests the fixes for the narrate_error + coherence judge-drops seen in the
Apr 18 preflight: pre_call length off by one, narrator hallucinating PR
state not in the tool output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase_a1.executor import Blueprint, PlannedCall
from experiments.phase_a1.narrator import _SYSTEM_PROMPT, narrate_session
from experiments.phase_a1.providers import LLMResponse


class _FakeRouter:
    """Async ProviderRouter stand-in returning canned responses in order."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if not self._texts:
            raise RuntimeError("FakeRouter exhausted: no more canned responses")
        text = self._texts.pop(0)
        return LLMResponse(
            text=text,
            provider="fake",
            model="fake-model",
            input_tokens=10,
            output_tokens=10,
            latency_s=0.01,
        )


def _multi_turn_blueprint() -> Blueprint:
    return Blueprint(
        user_intent="audit the github issues",
        planned_calls=[
            PlannedCall("github", {"action": "list_issues"}),
            PlannedCall("github", {"action": "get_issue", "issue_number": 1}),
        ],
        expected_outcome="issues audited",
        category="multi_turn",
        primary_tool="github",
        spec_index=0,
        spec_seed=42,
    )


def _write_session(path: Path, blueprint: Blueprint) -> None:
    """Write a minimal session matching ``blueprint`` (one assistant turn per call)."""
    records: list[dict] = [
        {"role": "system", "content": "sys", "session_id": "test-sess"},
        {"role": "user", "content": blueprint.user_intent, "session_id": "test-sess"},
    ]
    for i, call in enumerate(blueprint.planned_calls):
        records.append(
            {
                "role": "assistant",
                "content": "",
                "session_id": "test-sess",
                "tool_calls": [
                    {
                        "id": f"call-{i}",
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                ],
            }
        )
        records.append(
            {
                "role": "tool",
                "content": f"tool output {i}",
                "session_id": "test-sess",
                "tool_call_id": f"call-{i}",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


_VALID_2CALL_NARRATION = json.dumps(
    {
        "pre_call": [
            "I'll list the open issues first.",
            "Now I'll fetch the details on issue 1.",
        ],
        "final": "Issue 1 is open and titled 'Bug report'.",
    }
)

_BAD_WRONG_LENGTH_NARRATION = json.dumps(
    {
        "pre_call": ["Only one pre_call even though we have two calls."],
        "final": "Done.",
    }
)


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narrate_retries_after_wrong_precall_count(tmp_path: Path) -> None:
    """Regression: preflight spec#0 spawn_agent multi_turn narrate_error."""
    bp = _multi_turn_blueprint()
    session_path = tmp_path / "session.jsonl"
    _write_session(session_path, bp)

    router = _FakeRouter([_BAD_WRONG_LENGTH_NARRATION, _VALID_2CALL_NARRATION])
    resp = await narrate_session(bp, session_path, router, max_retries=2)

    assert resp.provider == "fake"
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_narrate_retries_on_invalid_json(tmp_path: Path) -> None:
    bp = _multi_turn_blueprint()
    session_path = tmp_path / "session.jsonl"
    _write_session(session_path, bp)

    router = _FakeRouter(["not json", _VALID_2CALL_NARRATION])
    await narrate_session(bp, session_path, router, max_retries=2)
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_narrate_bumps_temperature_on_retry(tmp_path: Path) -> None:
    bp = _multi_turn_blueprint()
    session_path = tmp_path / "session.jsonl"
    _write_session(session_path, bp)

    router = _FakeRouter([_BAD_WRONG_LENGTH_NARRATION, _VALID_2CALL_NARRATION])
    await narrate_session(bp, session_path, router, temperature=0.7, max_retries=2)
    assert router.calls[0]["temperature"] == 0.7
    assert router.calls[1]["temperature"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_narrate_raises_after_exhausting_retries(tmp_path: Path) -> None:
    bp = _multi_turn_blueprint()
    session_path = tmp_path / "session.jsonl"
    _write_session(session_path, bp)

    router = _FakeRouter([_BAD_WRONG_LENGTH_NARRATION] * 3)
    with pytest.raises(ValueError, match="after 3 attempts"):
        await narrate_session(bp, session_path, router, max_retries=2)
    assert len(router.calls) == 3


@pytest.mark.asyncio
async def test_narrate_succeeds_on_first_try_no_retry(tmp_path: Path) -> None:
    bp = _multi_turn_blueprint()
    session_path = tmp_path / "session.jsonl"
    _write_session(session_path, bp)

    router = _FakeRouter([_VALID_2CALL_NARRATION])
    await narrate_session(bp, session_path, router, max_retries=2)
    assert len(router.calls) == 1


@pytest.mark.asyncio
async def test_narrate_injects_content_into_session(tmp_path: Path) -> None:
    bp = _multi_turn_blueprint()
    session_path = tmp_path / "session.jsonl"
    _write_session(session_path, bp)

    router = _FakeRouter([_VALID_2CALL_NARRATION])
    await narrate_session(bp, session_path, router, max_retries=0)

    with open(session_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Assistant turns with tool_calls should have their `content` populated.
    assistant_with_calls = [
        r for r in records if r.get("role") == "assistant" and r.get("tool_calls")
    ]
    assert len(assistant_with_calls) == 2
    assert assistant_with_calls[0]["content"] == "I'll list the open issues first."
    assert assistant_with_calls[1]["content"] == "Now I'll fetch the details on issue 1."

    # Trailing assistant record with `final` should be appended.
    final_record = records[-1]
    assert final_record["role"] == "assistant"
    assert "Issue 1 is open" in final_record["content"]


# ---------------------------------------------------------------------------
# Anti-hallucination prompt content
# ---------------------------------------------------------------------------


def test_system_prompt_contains_anti_hallucination_block() -> None:
    assert "ANTI-HALLUCINATION RULES" in _SYSTEM_PROMPT
    prompt_lower = _SYSTEM_PROMPT.lower()
    # Lexical markers covering the concrete failures seen in preflight.
    assert "invent" in prompt_lower  # "DO NOT invent matches" clause
    assert "EXACTLY ONE STRING PER PLANNED TOOL CALL" in _SYSTEM_PROMPT
    # PR state hallucination guardrail mentioned explicitly (the preflight's
    # spec#2 failure was "assistant claims PR is open but tool output shows closed").
    assert "closed" in prompt_lower
