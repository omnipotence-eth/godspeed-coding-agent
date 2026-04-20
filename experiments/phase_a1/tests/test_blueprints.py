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


def test_validate_rejects_tasks_add_action() -> None:
    """Regression: prod run judge flagged `tasks action='add'` (should be 'create')."""
    bp = _bp([{"tool_name": "tasks", "arguments": {"action": "add", "title": "x"}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "tasks"))
    assert any("tasks.action invalid: 'add'" in e for e in errs)
    # Sanity: the valid action passes.
    ok_bp = _bp([{"tool_name": "tasks", "arguments": {"action": "create", "title": "x"}}])
    assert _validate_blueprint(ok_bp, _spec("single_tool", "tasks")) == []


def test_validate_rejects_notebook_edit_invalid_action() -> None:
    bp = _bp(
        [
            {
                "tool_name": "notebook_edit",
                "arguments": {"file_path": "n.ipynb", "action": "replace_cell"},
            }
        ]
    )
    errs = _validate_blueprint(bp, _spec("single_tool", "notebook_edit"))
    assert any("notebook_edit.action invalid" in e for e in errs)


def test_validate_rejects_notebook_edit_with_legacy_notebook_path() -> None:
    """Regression: Apr 18 prod run had 9 drops for notebook_edit args using
    the wrong parameter name. The real tool requires 'file_path', not
    'notebook_path', so the blueprint prompt + validator were bringing us a
    schema mismatch that wasted a full generation cycle per drop."""
    bp = _bp(
        [
            {
                "tool_name": "notebook_edit",
                "arguments": {"notebook_path": "n.ipynb", "action": "edit_cell"},
            }
        ]
    )
    errs = _validate_blueprint(bp, _spec("single_tool", "notebook_edit"))
    assert any("notebook_edit.file_path" in e for e in errs)


def test_validate_rejects_background_check_invalid_action() -> None:
    bp = _bp([{"tool_name": "background_check", "arguments": {"action": "restart"}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "background_check"))
    assert any("background_check.action invalid" in e for e in errs)


def test_validate_rejects_background_check_kill_with_string_id() -> None:
    bp = _bp(
        [
            {
                "tool_name": "background_check",
                "arguments": {"action": "kill", "id": "proc-42"},
            }
        ]
    )
    errs = _validate_blueprint(bp, _spec("single_tool", "background_check"))
    assert any("background_check.id must be an integer" in e for e in errs)


def test_validate_rejects_git_add_action() -> None:
    """Regression: Godspeed's git tool only accepts {status, diff, commit, log,
    undo, stash, stash_pop}. The old blueprint prompt advertised 'add',
    'branch', etc., which produced runtime failures and coherence drops."""
    bp = _bp([{"tool_name": "git", "arguments": {"action": "add"}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "git"))
    assert any("git.action invalid" in e for e in errs)


def test_validate_rejects_git_commit_without_message() -> None:
    bp = _bp([{"tool_name": "git", "arguments": {"action": "commit"}}])
    errs = _validate_blueprint(bp, _spec("single_tool", "git"))
    assert any("git.message must be a non-empty string" in e for e in errs)


def test_validate_rejects_diff_apply_without_hunk_headers() -> None:
    bp = _bp(
        [
            {
                "tool_name": "diff_apply",
                "arguments": {"diff": "--- a/x.py\nno hunk header here"},
            }
        ]
    )
    errs = _validate_blueprint(bp, _spec("single_tool", "diff_apply"))
    assert any("'---'/'+++' file headers" in e for e in errs)


def test_system_prompt_documents_diff_apply_format_exactly() -> None:
    """Contract test: the blueprint system prompt MUST document all three
    required markers (---, +++, @@) for diff_apply.

    The original R1 prompt said '@@ or ---' which caused the LLM to omit
    '+++' and the validator to reject every diff_apply blueprint (see
    RESEARCH_LOG Anomaly F1 addendum). This pins the fix."""
    assert "'--- a/" in _SYSTEM_TEMPLATE or "--- a/" in _SYSTEM_TEMPLATE
    assert "'+++ b/" in _SYSTEM_TEMPLATE or "+++ b/" in _SYSTEM_TEMPLATE
    assert "'@@'" in _SYSTEM_TEMPLATE or "@@" in _SYSTEM_TEMPLATE
    # Prompt must not claim '@@ or ---' is sufficient — the validator rejects that.
    assert '"@@" or "---"' not in _SYSTEM_TEMPLATE
    assert "'@@' or '---'" not in _SYSTEM_TEMPLATE


def test_validate_accepts_error_recovery_3_calls_for_edit_tools() -> None:
    """error_recovery samples that primary on file_edit or diff_apply may
    insert a file_read between the failing and corrected attempt — that's
    3 calls, and the validator must accept it."""
    bp = _bp(
        [
            {
                "tool_name": "file_edit",
                "arguments": {
                    "file_path": "src/main.py",
                    "old_string": "def greet(name):",
                    "new_string": "def greet(name: str) -> str:",
                },
            },
            {
                "tool_name": "file_read",
                "arguments": {"file_path": "src/main.py"},
            },
            {
                "tool_name": "file_edit",
                "arguments": {
                    "file_path": "src/main.py",
                    "old_string": 'return f"hello {name}"',
                    "new_string": 'return f"hello {name.strip()}"',
                },
            },
        ]
    )
    errs = _validate_blueprint(bp, _spec("error_recovery", "file_edit"))
    assert not any("error_recovery" in e for e in errs), f"unexpected errors: {errs}"


def test_validate_rejects_error_recovery_4_calls_for_edit_tools() -> None:
    call = {
        "tool_name": "file_edit",
        "arguments": {"file_path": "a", "old_string": "x", "new_string": "y"},
    }
    bp = _bp([call] * 4)
    errs = _validate_blueprint(bp, _spec("error_recovery", "file_edit"))
    assert any("error_recovery" in e and ("2 or 3" in e or "2 calls" in e) for e in errs)


def test_validate_still_rejects_error_recovery_3_calls_for_non_edit_tools() -> None:
    bp = _bp(
        [
            {"tool_name": "grep_search", "arguments": {"pattern": "a"}},
            {"tool_name": "grep_search", "arguments": {"pattern": "b"}},
            {"tool_name": "grep_search", "arguments": {"pattern": "c"}},
        ]
    )
    errs = _validate_blueprint(bp, _spec("error_recovery", "grep_search"))
    assert any("error_recovery must have exactly 2 calls" in e for e in errs)


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
    # git actions should be enumerated — the prompt MUST only advertise the
    # 7 actions the real Godspeed git tool implements. Regressions where
    # 'add'/'branch' were advertised caused runtime errors and coherence
    # drops in the Apr 18 prod run.
    assert "status, diff, commit, log, undo, stash," in _SYSTEM_TEMPLATE
    assert "NO add" in _SYSTEM_TEMPLATE
    # github actions should be enumerated
    assert "list_prs" in _SYSTEM_TEMPLATE
    # tasks actions (regression: judge flagged action='add' in prod run)
    assert "create, update, list, complete" in _SYSTEM_TEMPLATE
    assert 'NOT "add"' in _SYSTEM_TEMPLATE
    # notebook_edit + background_check actions enumerated. The formatter
    # wraps long lines, so check each token is present rather than the
    # comma-separated sequence.
    for token in ("edit_cell", "add_cell", "delete_cell", "move_cell"):
        assert token in _SYSTEM_TEMPLATE
    assert "status, output, kill" in _SYSTEM_TEMPLATE
