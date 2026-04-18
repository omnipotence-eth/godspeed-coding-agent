"""Tests for ``blueprints.py`` — per-tool arg validation + retry loop.

These tests exercise the fixes for the three failure modes seen in the
Apr 18 smoke-run metrics (github.action=None, grep_search.pattern="",
multi_turn with only 1 call). They use a fake router so no live LLM is hit.
"""

from __future__ import annotations

import json

import pytest

from experiments.phase_a1.blueprints import (
    _SYSTEM_TEMPLATE,
    _validate_blueprint,
    generate_blueprint,
)
from experiments.phase_a1.providers import LLMResponse
from experiments.phase_a1.specs import GenerationSpec

# ---------------------------------------------------------------------------
# FakeRouter
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Async ProviderRouter stand-in that returns canned responses in order."""

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


def _spec(category: str, primary_tool: str, index: int = 0, seed: int = 42) -> GenerationSpec:
    return GenerationSpec(index=index, primary_tool=primary_tool, category=category, seed=seed)


def _bp(calls: list[dict]) -> dict:
    return {
        "user_intent": "do something",
        "planned_calls": calls,
        "expected_outcome": "it worked",
    }


# ---------------------------------------------------------------------------
# _validate_blueprint — per-tool arg schema (regression guard)
# ---------------------------------------------------------------------------


def test_validate_rejects_github_action_none() -> None:
    """Regression: spec#2 prod smoke had github.action=None."""
    bp = _bp([{"tool_name": "github", "arguments": {"action": None}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "github"))
    assert errs, "expected github.action=None to fail blueprint validation"
    assert any("github.action invalid" in e for e in errs)


def test_validate_rejects_grep_search_empty_pattern() -> None:
    """Regression: spec#3 prod smoke had grep_search.pattern='' in turn 2."""
    bp = _bp(
        [
            {"tool_name": "web_search", "arguments": {"query": "docstring style"}},
            {"tool_name": "grep_search", "arguments": {"pattern": ""}},
        ]
    )
    errs = _validate_blueprint(bp, _spec("multi_turn", "web_search"))
    assert errs
    assert any("grep_search.pattern must be a non-empty string" in e for e in errs)


def test_validate_accepts_well_formed_github_call() -> None:
    bp = _bp([{"tool_name": "github", "arguments": {"action": "list_issues"}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "github"))
    assert errs == []


def test_validate_rejects_multi_turn_with_one_call() -> None:
    """Regression: spec#0 prod smoke had multi_turn with only 1 planned_call."""
    bp = _bp([{"tool_name": "spawn_agent", "arguments": {"task": "do it"}}])
    errs = _validate_blueprint(bp, _spec("multi_turn", "spawn_agent"))
    assert any("multi_turn must have 2-4 calls" in e for e in errs)


def test_validate_skips_arg_check_for_unknown_tool() -> None:
    """Unknown tool names are flagged once; per-tool validator is not invoked."""
    bp = _bp([{"tool_name": "nonexistent_tool", "arguments": {}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "nonexistent_tool"))
    # Exactly one error about the unknown tool — no cascade from the per-tool
    # validator (which would fail hard on unknown names).
    tool_errs = [e for e in errs if "not in registry" in e]
    assert len(tool_errs) == 1


# ---------------------------------------------------------------------------
# Retry loop in generate_blueprint
# ---------------------------------------------------------------------------


_VALID_BLUEPRINT = json.dumps(
    {
        "user_intent": "list open issues on the repo",
        "planned_calls": [{"tool_name": "github", "arguments": {"action": "list_issues"}}],
        "expected_outcome": "issues listed",
    }
)

_BAD_BLUEPRINT_NONE_ACTION = json.dumps(
    {
        "user_intent": "list open issues",
        "planned_calls": [{"tool_name": "github", "arguments": {"action": None}}],
        "expected_outcome": "issues listed",
    }
)


@pytest.mark.asyncio
async def test_generate_blueprint_retries_after_bad_action_then_succeeds() -> None:
    router = _FakeRouter([_BAD_BLUEPRINT_NONE_ACTION, _VALID_BLUEPRINT])
    bp, _resp = await generate_blueprint(_spec("single_tool", "github"), router, max_retries=2)
    assert bp.planned_calls[0].tool_name == "github"
    assert bp.planned_calls[0].arguments == {"action": "list_issues"}
    # Two provider calls — the first was retried because args failed validation.
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_generate_blueprint_bumps_temperature_on_retry() -> None:
    router = _FakeRouter([_BAD_BLUEPRINT_NONE_ACTION, _VALID_BLUEPRINT])
    await generate_blueprint(_spec("single_tool", "github"), router, temperature=0.8, max_retries=2)
    assert router.calls[0]["temperature"] == 0.8
    assert router.calls[1]["temperature"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_generate_blueprint_raises_after_exhausting_retries() -> None:
    router = _FakeRouter([_BAD_BLUEPRINT_NONE_ACTION] * 3)
    with pytest.raises(ValueError, match="after 3 attempts"):
        await generate_blueprint(_spec("single_tool", "github"), router, max_retries=2)
    assert len(router.calls) == 3


@pytest.mark.asyncio
async def test_generate_blueprint_retries_on_invalid_json() -> None:
    router = _FakeRouter(["this is not json", _VALID_BLUEPRINT])
    bp, _resp = await generate_blueprint(_spec("single_tool", "github"), router, max_retries=2)
    assert bp.planned_calls[0].arguments == {"action": "list_issues"}
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_generate_blueprint_succeeds_on_first_try_no_retry() -> None:
    router = _FakeRouter([_VALID_BLUEPRINT])
    bp, _resp = await generate_blueprint(_spec("single_tool", "github"), router, max_retries=2)
    assert bp.planned_calls[0].arguments == {"action": "list_issues"}
    assert len(router.calls) == 1


# ---------------------------------------------------------------------------
# Prompt carries the required-args cheatsheet
# ---------------------------------------------------------------------------


def test_system_prompt_lists_required_args_for_error_prone_tools() -> None:
    assert "REQUIRED ARGUMENTS" in _SYSTEM_TEMPLATE
    # The tools that produced failures in the Apr 18 smoke must be present:
    assert "github" in _SYSTEM_TEMPLATE
    assert "grep_search" in _SYSTEM_TEMPLATE
    assert "action" in _SYSTEM_TEMPLATE
    # git actions should be enumerated
    assert "status, diff, add, commit" in _SYSTEM_TEMPLATE
    # github actions should be enumerated
    assert "list_prs" in _SYSTEM_TEMPLATE
