"""Unit tests for experiments.phase_a1.validate.

Covers:
  * Record-level shape errors (missing keys, wrong types)
  * tool_calls name/arg validation against the 21-tool registry
  * Per-tool argument validators (catching the failure modes we saw in the
    first live smoke run — e.g. github with no action, spawn_agent empty task)
  * tool_call_id \u2192 tool-message linkage
  * Coverage-floor enforcement across a JSONL file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase_a1.registry_builder import ALL_TOOLS
from experiments.phase_a1.validate import (
    DEFAULT_MIN_COVERAGE,
    EXPECTED_TOOL_COUNT,
    validate_file,
    validate_record,
)


def _make_tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": {}},
    }


def _all_schemas() -> list[dict]:
    return [_make_tool_schema(n) for n in ALL_TOOLS]


def _tool_call(name: str, arguments: dict, *, call_id: str = "call_00") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _valid_record(
    tool_name: str = "file_read",
    args: dict | None = None,
    call_id: str = "call_00",
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "please read src/main.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(tool_name, args or {"file_path": "src/main.py"}, call_id=call_id)
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "file contents"},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": _all_schemas(),
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_single_tool_record_has_no_errors() -> None:
    errs, usage, category = validate_record(_valid_record())
    assert errs == []
    assert usage["file_read"] == 1
    assert category == "single_tool"


def test_no_tool_record_inferred_as_no_tool() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi!"},
        ],
        "tools": _all_schemas(),
    }
    errs, usage, category = validate_record(record)
    assert errs == []
    assert not usage
    assert category == "no_tool"


# ---------------------------------------------------------------------------
# Shape errors
# ---------------------------------------------------------------------------


def test_missing_tools_field_is_flagged() -> None:
    record = _valid_record()
    record.pop("tools")
    errs, _, _ = validate_record(record)
    assert any("tools must be a list" in e for e in errs)


def test_wrong_tool_count_is_flagged() -> None:
    record = _valid_record()
    record["tools"] = record["tools"][:10]
    errs, _, _ = validate_record(record)
    assert any(f"{EXPECTED_TOOL_COUNT} entries" in e for e in errs)


def test_unknown_tool_in_schemas_is_flagged() -> None:
    record = _valid_record()
    record["tools"] = [*record["tools"][:-1], _make_tool_schema("fake_tool")]
    errs, _, _ = validate_record(record)
    assert any("unknown names" in e for e in errs)


def test_empty_messages_is_flagged() -> None:
    record = {"messages": [], "tools": _all_schemas()}
    errs, _, _ = validate_record(record)
    assert any("messages must be a non-empty list" in e for e in errs)


# ---------------------------------------------------------------------------
# tool_calls validation — the failures we actually saw in the live pipeline
# ---------------------------------------------------------------------------


def test_github_missing_action_is_flagged() -> None:
    """Regression: live smoke test produced github calls with no action."""
    record = _valid_record("github", {"body": "..."})
    errs, _, _ = validate_record(record)
    assert any("github.action invalid" in e for e in errs)


def test_spawn_agent_empty_task_is_flagged() -> None:
    """Regression: live smoke test produced spawn_agent with empty task."""
    record = _valid_record("spawn_agent", {"task": ""})
    errs, _, _ = validate_record(record)
    assert any("spawn_agent.task must be a non-empty string" in e for e in errs)


def test_file_write_missing_content_is_flagged() -> None:
    record = _valid_record("file_write", {"file_path": "a.py"})
    errs, _, _ = validate_record(record)
    assert any("file_write.content must be a string" in e for e in errs)


def test_shell_dangerous_pattern_blocked() -> None:
    record = _valid_record("shell", {"command": "rm -rf /"})
    errs, _, _ = validate_record(record)
    assert any("dangerous pattern" in e for e in errs)


def test_unknown_tool_name_in_tool_call_is_flagged() -> None:
    record = _valid_record("file_read")
    record["messages"][2]["tool_calls"][0]["function"]["name"] = "bogus_tool"
    errs, _, _ = validate_record(record)
    assert any("not in registry" in e for e in errs)


def test_tool_call_arguments_must_be_valid_json() -> None:
    record = _valid_record("file_read")
    record["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not json"
    errs, _, _ = validate_record(record)
    assert any("not valid JSON" in e for e in errs)


def test_dict_arguments_are_accepted() -> None:
    """Some exporters emit args as an already-decoded object. That's fine."""
    record = _valid_record("file_read")
    record["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"file_path": "a.py"}
    errs, _, _ = validate_record(record)
    assert errs == []


# ---------------------------------------------------------------------------
# tool_call_id linkage
# ---------------------------------------------------------------------------


def test_orphan_tool_message_is_flagged() -> None:
    record = _valid_record()
    record["messages"][3]["tool_call_id"] = "no_such_call"
    errs, _, _ = validate_record(record)
    assert any("has no prior assistant tool_call" in e for e in errs)


def test_unfulfilled_tool_call_is_flagged() -> None:
    record = _valid_record()
    # Drop the tool response
    record["messages"] = [m for m in record["messages"] if m.get("role") != "tool"]
    errs, _, _ = validate_record(record)
    assert any("unfulfilled tool_call" in e for e in errs)


# ---------------------------------------------------------------------------
# File-level / coverage
# ---------------------------------------------------------------------------


def test_file_with_mixed_records_reports_correctly(tmp_path: Path) -> None:
    good = _valid_record()
    bad = _valid_record("github", {})  # missing action
    path = tmp_path / "samples.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(good) + "\n")
        f.write(json.dumps(bad) + "\n")

    report = validate_file(path, min_coverage=0)
    assert report.total_records == 2
    assert report.valid_records == 1
    assert len(report.errors) >= 1
    assert not report.ok  # any error \u2192 not ok


def test_coverage_floor_violation_reported(tmp_path: Path) -> None:
    """Single-tool corpus fails coverage floor for the other 20 tools."""
    path = tmp_path / "samples.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(3):
            rec = _valid_record(call_id=f"call_{i:02d}")
            f.write(json.dumps(rec) + "\n")
    report = validate_file(path, min_coverage=1)
    # file_read appears 3 times \u2192 OK; the other 20 tools appear 0 \u2192 violations
    assert len(report.coverage_violations) == len(ALL_TOOLS) - 1
    assert not report.ok


def test_coverage_zero_disables_floor_check(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    rec = _valid_record()
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    report = validate_file(path, min_coverage=0)
    assert not report.coverage_violations
    assert report.ok


def test_fail_fast_stops_after_first_error(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for _ in range(5):
            f.write(json.dumps(_valid_record("github", {})) + "\n")
    report = validate_file(path, min_coverage=0, fail_fast=True)
    assert report.total_records == 1


def test_missing_file_is_flagged(tmp_path: Path) -> None:
    report = validate_file(tmp_path / "nope.jsonl", min_coverage=DEFAULT_MIN_COVERAGE)
    assert not report.ok
    assert any("does not exist" in e.message for e in report.errors)


@pytest.mark.parametrize(
    "tool_name,bad_args,expected_substr",
    [
        ("file_read", {"file_path": ""}, "non-empty string"),
        (
            "file_edit",
            {"file_path": "a.py", "old_string": 5, "new_string": "x"},
            "must be a string",
        ),
        ("web_fetch", {"url": "ftp://example.com"}, "http(s)://"),
        ("glob_search", {"pattern": ""}, "non-empty"),
        ("git", {"action": "mainframe_destruct"}, "git.action invalid"),
        # Record-level lenient validator still catches the hard regressions:
        ("git", {"action": "commit"}, "git.message must be a non-empty string"),
        (
            "background_check",
            {"action": "kill", "id": "proc-42"},
            "background_check.id must be an integer",
        ),
        (
            "background_check",
            {"action": "output"},
            "background_check.id must be an integer",
        ),
        (
            "diff_apply",
            {"diff": "something with @@ but no headers"},
            "'---'/'+++' file headers",
        ),
        (
            "diff_apply",
            {"diff": ""},
            "non-empty unified-diff string",
        ),
    ],
)
def test_per_tool_validator_rejects_bad_args(
    tool_name: str, bad_args: dict, expected_substr: str
) -> None:
    record = _valid_record(tool_name, bad_args)
    errs, _, _ = validate_record(record)
    assert any(expected_substr in e for e in errs), f"no error matched {expected_substr!r}: {errs}"


def test_verify_with_empty_file_path_is_flagged() -> None:
    """Regression: smoke #11 had verify({'file_path': ''}) which the real
    verify tool rejects. Empty string must be caught at validation time."""
    record = _valid_record("verify", {"file_path": ""})
    errs, _, _ = validate_record(record)
    assert any("verify.file_path must be a non-empty string" in e for e in errs)


def test_verify_with_no_args_passes() -> None:
    """verify with no args scans the whole repo — that's a valid invocation."""
    record = _valid_record("verify", {})
    errs, _, _ = validate_record(record)
    assert not any("verify" in e for e in errs), f"unexpected errors: {errs}"


@pytest.mark.parametrize(
    "tool_name,bad_args,expected_substr",
    [
        # git legacy aliases pass lenient but must fail strict (blueprint-gen)
        ("git", {"action": "add"}, "git.action invalid for real Godspeed tool"),
        ("git", {"action": "branch"}, "git.action invalid for real Godspeed tool"),
        ("git", {"action": "push"}, "git.action invalid for real Godspeed tool"),
        # notebook_edit legacy param name passes lenient but must fail strict
        (
            "notebook_edit",
            {"notebook_path": "nb.ipynb", "action": "edit_cell"},
            "use 'file_path', NOT 'notebook_path'",
        ),
        # notebook_edit without action fails strict (action is required on the
        # real tool; leniency only exists to keep the one broken anchor valid)
        (
            "notebook_edit",
            {"file_path": "nb.ipynb"},
            "notebook_edit.action invalid",
        ),
    ],
)
def test_strict_validator_rejects_legacy_aliases(
    tool_name: str, bad_args: dict, expected_substr: str
) -> None:
    from experiments.phase_a1.validate import validate_tool_call_args

    errs = validate_tool_call_args(tool_name, bad_args, strict=True)
    assert any(expected_substr in e for e in errs), (
        f"strict mode did not flag {tool_name} {bad_args}: {errs}"
    )
    lenient_errs = validate_tool_call_args(tool_name, bad_args, strict=False)
    assert not any(expected_substr in e for e in lenient_errs), (
        f"lenient mode was supposed to tolerate the alias but flagged it: {lenient_errs}"
    )


def test_git_commit_with_message_passes() -> None:
    record = _valid_record(
        "git",
        {"action": "commit", "message": "refactor: rename slugify helper"},
    )
    errs, _, _ = validate_record(record)
    assert not any("git" in e for e in errs), f"unexpected git errors: {errs}"


def test_notebook_edit_with_file_path_and_action_passes() -> None:
    record = _valid_record(
        "notebook_edit",
        {"file_path": "notebooks/analysis.ipynb", "action": "add_cell", "content": "print(1)"},
    )
    errs, _, _ = validate_record(record)
    assert not any("notebook_edit" in e for e in errs), f"unexpected errors: {errs}"


def test_background_check_status_without_id_passes() -> None:
    record = _valid_record("background_check", {"action": "status"})
    errs, _, _ = validate_record(record)
    assert not any("background_check" in e for e in errs), f"unexpected errors: {errs}"


def test_diff_apply_with_real_unified_diff_passes() -> None:
    diff = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-def greet(name):\n"
        "+def greet(name: str) -> str:\n"
        '     return f"hello {name}"\n'
    )
    record = _valid_record("diff_apply", {"diff": diff})
    errs, _, _ = validate_record(record)
    assert not any("diff_apply" in e for e in errs), f"unexpected errors: {errs}"
