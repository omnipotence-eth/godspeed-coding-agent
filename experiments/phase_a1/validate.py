"""Stage E — schema + coverage validator for the final JSONL.

Checks every ``{messages, tools}`` record against:

  * Top-level shape: ``messages`` list, ``tools`` list of length 21
  * Message role sequence: starts with ``system`` then ``user``; every
    ``tool_call`` references a later ``tool`` message via matching id; no
    orphan ``tool`` messages
  * ``tool_calls[].function.name`` is in the canonical 21-tool registry
  * Per-tool arg validators — catch the obvious mistakes that slip past the
    blueprint validator (missing required args, wrong type, destructive shell
    patterns)

Coverage: ensure every registered tool appears at least ``--min-coverage``
times across the corpus (default 50). This surfaces a monoculture before
training begins.

Exits non-zero when any record-level error is found OR coverage floor is
violated. Used as the gate between Phase A1 and downstream training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.phase_a1.registry_builder import ALL_TOOLS

logger = logging.getLogger(__name__)


EXPECTED_TOOL_COUNT: int = 21
DEFAULT_MIN_COVERAGE: int = 50

_ALL_TOOLS_SET: frozenset[str] = frozenset(ALL_TOOLS)


@dataclass
class RecordError:
    """One validation failure bound to a record index (and optional message index)."""

    record_index: int
    message: str
    message_index: int | None = None

    def __str__(self) -> str:
        if self.message_index is None:
            return f"record #{self.record_index}: {self.message}"
        return f"record #{self.record_index} msg#{self.message_index}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregate result of validating a whole JSONL file."""

    total_records: int = 0
    valid_records: int = 0
    errors: list[RecordError] = field(default_factory=list)
    tool_usage: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    coverage_violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.coverage_violations

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "error_count": len(self.errors),
            "coverage_violations": self.coverage_violations,
            "tool_usage": dict(self.tool_usage),
            "category_counts": dict(self.category_counts),
        }


# ---------------------------------------------------------------------------
# Per-tool argument validators
# ---------------------------------------------------------------------------


_DANGEROUS_SHELL = re.compile(
    r"(?:\brm\s+-rf?\s+/(?:\s|$)|:\(\)\{\s*:\|:&\s*\};:|\bmkfs\.|\bdd\s+if=/dev/(?:zero|random|urandom))",
)


def _v_file_read(args: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    fp = args.get("file_path")
    if not isinstance(fp, str) or not fp.strip():
        errs.append("file_read.file_path must be a non-empty string")
    return errs


def _v_file_write(args: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if not isinstance(args.get("file_path"), str) or not args.get("file_path"):
        errs.append("file_write.file_path must be a non-empty string")
    if "content" not in args or not isinstance(args["content"], str):
        errs.append("file_write.content must be a string")
    return errs


def _v_file_edit(args: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for field_name in ("file_path", "old_string", "new_string"):
        if not isinstance(args.get(field_name), str):
            errs.append(f"file_edit.{field_name} must be a string")
    return errs


def _v_diff_apply(args: dict[str, Any]) -> list[str]:
    diff = args.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        return ["diff_apply.diff must be a non-empty unified-diff string"]
    # Real unified diffs need both file headers AND at least one hunk header.
    # The earlier check ("@@" in diff OR "---" in diff) accepted malformed
    # fragments that the real diff_apply tool then rejected at runtime, which
    # cascaded into judge coherence drops.
    if "---" not in diff or "+++" not in diff or "@@" not in diff:
        return [
            "diff_apply.diff must be a unified diff with '---'/'+++' file "
            "headers and at least one '@@' hunk header"
        ]
    return []


def _v_glob_search(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("pattern"), str) or not args.get("pattern"):
        return ["glob_search.pattern must be a non-empty string"]
    return []


def _v_grep_search(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("pattern"), str) or not args.get("pattern"):
        return ["grep_search.pattern must be a non-empty string"]
    return []


def _v_code_search(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("query"), str) or not args.get("query"):
        return ["code_search.query must be a non-empty string"]
    return []


def _v_repo_map(_args: dict[str, Any]) -> list[str]:
    return []


def _v_shell(args: dict[str, Any]) -> list[str]:
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return ["shell.command must be a non-empty string"]
    if _DANGEROUS_SHELL.search(cmd):
        return [f"shell.command matches dangerous pattern: {cmd[:80]!r}"]
    return []


def _v_test_runner(_args: dict[str, Any]) -> list[str]:
    return []


def _v_verify(args: dict[str, Any]) -> list[str]:
    # file_path is optional on verify (the tool can scan the whole repo) but
    # if it's provided it MUST be a non-empty string. A live prod sample was
    # dropped because the generator emitted verify({"file_path": ""}) which
    # the real tool rejects.
    if "file_path" in args:
        fp = args["file_path"]
        if not isinstance(fp, str) or not fp.strip():
            return ["verify.file_path must be a non-empty string when provided"]
    return []


_BACKGROUND_CHECK_ACTIONS = {"status", "output", "kill"}


def _v_background_check(args: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    action = args.get("action")
    # Missing action is tolerated at validation time (the real tool will
    # reject it at dispatch; anchor samples use variant shapes that we
    # don't want to retroactively invalidate). When action IS given it
    # must be one of the enum values.
    if action is not None and action not in _BACKGROUND_CHECK_ACTIONS:
        errs.append(
            f"background_check.action invalid: {action!r} "
            f"(expected one of {sorted(_BACKGROUND_CHECK_ACTIONS)})"
        )
    if action in ("output", "kill"):
        pid = args.get("id")
        if not isinstance(pid, int) or isinstance(pid, bool):
            errs.append(
                "background_check.id must be an integer process id "
                "when action is 'output' or 'kill'"
            )
    return errs


# The real Godspeed GitTool only accepts the 7 TRUE_ACTIONS. The wider
# _GIT_ACTIONS set includes historical aliases (add/branch/push/...) that
# Opus-authored anchor samples used; accepting them here keeps legacy gold
# anchors valid. Fresh synthetic generations are steered by the blueprint
# prompt toward the 7 real actions only.
_GIT_TRUE_ACTIONS = {"status", "diff", "commit", "log", "undo", "stash", "stash_pop"}
_GIT_LEGACY_ALIASES = {
    "add",
    "branch",
    "checkout",
    "show",
    "restore",
    "push",
    "pull",
    "fetch",
    "merge",
    "rebase",
    "reset",
    "tag",
}
_GIT_ACTIONS = _GIT_TRUE_ACTIONS | _GIT_LEGACY_ALIASES


def _v_git(args: dict[str, Any]) -> list[str]:
    action = args.get("action")
    if action not in _GIT_ACTIONS:
        return [f"git.action invalid: {action!r} (expected one of {sorted(_GIT_ACTIONS)})"]
    if action == "commit":
        msg = args.get("message")
        if not isinstance(msg, str) or not msg.strip():
            return ["git.message must be a non-empty string when action is 'commit'"]
    return []


def _v_git_strict(args: dict[str, Any]) -> list[str]:
    """Blueprint-gen variant: rejects legacy-alias actions that the real
    Godspeed GitTool doesn't implement. Used to force the LLM to retry when
    it emits 'add'/'branch'/etc. Full record validator stays lenient so that
    historic anchor samples using those aliases remain accepted."""
    action = args.get("action")
    if action not in _GIT_TRUE_ACTIONS:
        return [
            f"git.action invalid for real Godspeed tool: {action!r} "
            f"(expected one of {sorted(_GIT_TRUE_ACTIONS)})"
        ]
    if action == "commit":
        msg = args.get("message")
        if not isinstance(msg, str) or not msg.strip():
            return ["git.message must be a non-empty string when action is 'commit'"]
    return []


def _v_notebook_edit_strict(args: dict[str, Any]) -> list[str]:
    """Blueprint-gen variant: requires the real param name ``file_path`` and
    the enumerated action set."""
    errs: list[str] = []
    if not isinstance(args.get("file_path"), str) or not args.get("file_path"):
        errs.append(
            "notebook_edit.file_path must be a non-empty string "
            "(use 'file_path', NOT 'notebook_path')"
        )
    action = args.get("action")
    if action not in _NOTEBOOK_EDIT_ACTIONS:
        errs.append(
            f"notebook_edit.action invalid: {action!r} "
            f"(expected one of {sorted(_NOTEBOOK_EDIT_ACTIONS)})"
        )
    return errs


_GH_ACTIONS = {
    "list_prs",
    "get_pr",
    "create_pr",
    "list_issues",
    "get_issue",
    "create_issue",
    "comment_issue",
    "comment_pr",
}


def _v_github(args: dict[str, Any]) -> list[str]:
    action = args.get("action")
    if action not in _GH_ACTIONS:
        return [f"github.action invalid: {action!r} (expected one of {sorted(_GH_ACTIONS)})"]
    return []


def _v_web_search(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("query"), str) or not args.get("query"):
        return ["web_search.query must be a non-empty string"]
    return []


def _v_web_fetch(args: dict[str, Any]) -> list[str]:
    url = args.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ["web_fetch.url must be http(s):// URL"]
    return []


def _v_image_read(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("file_path"), str) or not args.get("file_path"):
        return ["image_read.file_path must be a non-empty string"]
    return []


def _v_pdf_read(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("file_path"), str) or not args.get("file_path"):
        return ["pdf_read.file_path must be a non-empty string"]
    return []


_NOTEBOOK_EDIT_ACTIONS = {"edit_cell", "add_cell", "delete_cell", "move_cell"}


def _v_notebook_edit(args: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    # The real NotebookEditTool requires ``file_path`` (the legacy anchor set
    # mistakenly used ``notebook_path`` — still accepted here to avoid
    # invalidating existing anchors, but the blueprint prompt now tells
    # fresh synthetic samples to use ``file_path``).
    path = args.get("file_path") or args.get("notebook_path")
    if not isinstance(path, str) or not path.strip():
        errs.append("notebook_edit.file_path must be a non-empty string")
    action = args.get("action")
    if action is not None and action not in _NOTEBOOK_EDIT_ACTIONS:
        errs.append(
            f"notebook_edit.action invalid: {action!r} "
            f"(expected one of {sorted(_NOTEBOOK_EDIT_ACTIONS)})"
        )
    if "content" in args and not isinstance(args["content"], str):
        errs.append("notebook_edit.content must be a string if present")
    if "new_source" in args and not isinstance(args["new_source"], str):
        errs.append("notebook_edit.new_source must be a string if present")
    return errs


_TASKS_ACTIONS = {"create", "update", "list", "complete"}


def _v_tasks(args: dict[str, Any]) -> list[str]:
    action = args.get("action")
    if action is not None and action not in _TASKS_ACTIONS:
        return [f"tasks.action invalid: {action!r} (expected one of {sorted(_TASKS_ACTIONS)})"]
    return []


def _v_spawn_agent(args: dict[str, Any]) -> list[str]:
    if not isinstance(args.get("task"), str) or not args.get("task"):
        return ["spawn_agent.task must be a non-empty string"]
    return []


_VALIDATORS = {
    "file_read": _v_file_read,
    "file_write": _v_file_write,
    "file_edit": _v_file_edit,
    "diff_apply": _v_diff_apply,
    "glob_search": _v_glob_search,
    "grep_search": _v_grep_search,
    "code_search": _v_code_search,
    "repo_map": _v_repo_map,
    "shell": _v_shell,
    "test_runner": _v_test_runner,
    "verify": _v_verify,
    "background_check": _v_background_check,
    "git": _v_git,
    "github": _v_github,
    "web_search": _v_web_search,
    "web_fetch": _v_web_fetch,
    "image_read": _v_image_read,
    "pdf_read": _v_pdf_read,
    "notebook_edit": _v_notebook_edit,
    "tasks": _v_tasks,
    "spawn_agent": _v_spawn_agent,
}


_STRICT_VALIDATORS: dict[str, Any] = {
    "git": _v_git_strict,
    "notebook_edit": _v_notebook_edit_strict,
}


def validate_tool_call_args(
    tool_name: str, args: dict[str, Any], *, strict: bool = False
) -> list[str]:
    """Public entry point for per-tool argument validation.

    Returns a list of human-readable error strings; empty list means the
    args satisfy that tool's schema. Used by ``blueprints.py`` to reject
    malformed LLM output BEFORE executor + narrator spend is incurred.

    ``strict=True`` activates the blueprint-generation variants that reject
    param aliases / actions tolerated on legacy anchor samples but that the
    real Godspeed tools don't implement (``notebook_path``, ``git.action``
    in {add,branch,push,...}). The record-level validator used by
    ``assemble.py`` keeps ``strict=False`` so existing gold anchors stay
    valid through assembly.

    Unknown tool names return a single error; callers are expected to have
    verified ``tool_name in ALL_TOOLS`` first, but we fail closed just in case.
    """
    validator = _VALIDATORS.get(tool_name)
    if validator is None:
        return [f"unknown tool {tool_name!r}"]
    if not isinstance(args, dict):
        return [f"{tool_name}.arguments must be an object, got {type(args).__name__}"]
    if strict:
        strict_fn = _STRICT_VALIDATORS.get(tool_name)
        if strict_fn is not None:
            return strict_fn(args)
    return validator(args)


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def _validate_tools_field(tools: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(tools, list):
        return [f"tools must be a list, got {type(tools).__name__}"]
    if len(tools) != EXPECTED_TOOL_COUNT:
        errs.append(f"tools must have {EXPECTED_TOOL_COUNT} entries, got {len(tools)}")
    names: list[str] = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            errs.append(f"tools[{i}] not an object")
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else None
        if fn is None:
            errs.append(f"tools[{i}].function missing")
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            errs.append(f"tools[{i}].function.name not a string")
            continue
        names.append(name)
    if names:
        missing = _ALL_TOOLS_SET - set(names)
        extra = set(names) - _ALL_TOOLS_SET
        if missing:
            errs.append(f"tools missing canonical names: {sorted(missing)}")
        if extra:
            errs.append(f"tools have unknown names: {sorted(extra)}")
    return errs


def _validate_messages(messages: Any) -> tuple[list[str], Counter[str]]:
    """Validate the message sequence. Returns (errors, tool_usage_counter)."""
    tool_usage: Counter[str] = Counter()

    if not isinstance(messages, list) or not messages:
        return ["messages must be a non-empty list"], tool_usage

    errs: list[str] = []

    # Roles: system first (optional), then user, then assistant(/tool)+
    if messages[0].get("role") not in ("system", "user"):
        errs.append(f"first message role must be system or user, got {messages[0].get('role')!r}")

    pending_tool_call_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errs.append(f"msg[{i}] not an object")
            continue
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            errs.append(f"msg[{i}].role invalid: {role!r}")
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                errs.append(f"msg[{i}].tool_calls must be a list")
                tool_calls = []
            for j, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    errs.append(f"msg[{i}].tool_calls[{j}] not an object")
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function")
                if not isinstance(tc_id, str) or not tc_id:
                    errs.append(f"msg[{i}].tool_calls[{j}].id missing")
                if not isinstance(fn, dict):
                    errs.append(f"msg[{i}].tool_calls[{j}].function missing")
                    continue
                name = fn.get("name")
                args_raw = fn.get("arguments")
                if name not in _ALL_TOOLS_SET:
                    errs.append(f"msg[{i}].tool_calls[{j}].function.name {name!r} not in registry")
                    continue
                # OpenAI format uses a JSON-encoded string for arguments.
                if isinstance(args_raw, str):
                    try:
                        args_obj = json.loads(args_raw) if args_raw else {}
                    except json.JSONDecodeError as e:
                        errs.append(
                            f"msg[{i}].tool_calls[{j}].function.arguments not valid JSON: {e}"
                        )
                        continue
                elif isinstance(args_raw, dict):
                    args_obj = args_raw
                else:
                    errs.append(
                        f"msg[{i}].tool_calls[{j}].function.arguments must be JSON string or object"
                    )
                    continue
                tool_usage[name] += 1
                for sub_err in _VALIDATORS[name](args_obj):
                    errs.append(f"msg[{i}].tool_calls[{j}] {sub_err}")
                if isinstance(tc_id, str) and tc_id:
                    pending_tool_call_ids.add(tc_id)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if not isinstance(tc_id, str) or not tc_id:
                errs.append(f"msg[{i}].tool_call_id missing")
            elif tc_id not in pending_tool_call_ids:
                errs.append(f"msg[{i}].tool_call_id {tc_id!r} has no prior assistant tool_call")
            else:
                pending_tool_call_ids.discard(tc_id)

    if pending_tool_call_ids:
        errs.append(
            f"{len(pending_tool_call_ids)} unfulfilled tool_call ids: "
            f"{sorted(pending_tool_call_ids)[:3]}..."
        )

    return errs, tool_usage


def _infer_category(tool_usage: Counter[str]) -> str:
    total = sum(tool_usage.values())
    if total == 0:
        return "no_tool"
    if total == 1:
        return "single_tool"
    return "multi_turn"


def validate_record(record: dict[str, Any]) -> tuple[list[str], Counter[str], str]:
    """Return (errors, tool_usage, inferred_category) for one record."""
    errs: list[str] = []
    if not isinstance(record, dict):
        return [f"record not an object: {type(record).__name__}"], Counter(), "unknown"
    errs.extend(_validate_tools_field(record.get("tools")))
    msg_errs, tool_usage = _validate_messages(record.get("messages"))
    errs.extend(msg_errs)
    return errs, tool_usage, _infer_category(tool_usage)


# ---------------------------------------------------------------------------
# File-level validation
# ---------------------------------------------------------------------------


def validate_file(
    input_path: Path,
    *,
    min_coverage: int = DEFAULT_MIN_COVERAGE,
    fail_fast: bool = False,
) -> ValidationReport:
    """Run ``validate_record`` on every line and aggregate into a report."""
    report = ValidationReport()
    if not input_path.exists():
        report.errors.append(RecordError(-1, f"input does not exist: {input_path}"))
        return report

    with input_path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            report.total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                report.errors.append(RecordError(idx, f"invalid JSON: {e}"))
                if fail_fast:
                    return report
                continue

            errs, usage, category = validate_record(record)
            if errs:
                for err in errs:
                    report.errors.append(RecordError(idx, err))
                if fail_fast:
                    return report
            else:
                report.valid_records += 1
            report.tool_usage.update(usage)
            report.category_counts[category] += 1

    # Coverage floor
    for tool in ALL_TOOLS:
        if report.tool_usage[tool] < min_coverage:
            report.coverage_violations.append(f"{tool}: {report.tool_usage[tool]} < {min_coverage}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase A1 training JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/phase_a1/data/phase_a1_smoke.jsonl"),
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=DEFAULT_MIN_COVERAGE,
        help="Per-tool minimum count; 0 to disable.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-errors-shown",
        type=int,
        default=50,
        help="Limit printed errors to keep terminal readable.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    report = validate_file(args.input, min_coverage=args.min_coverage, fail_fast=args.fail_fast)

    logger.info(
        "validated %d records: %d ok, %d errors, %d coverage violations",
        report.total_records,
        report.valid_records,
        len(report.errors),
        len(report.coverage_violations),
    )
    logger.info("categories: %s", dict(report.category_counts))
    logger.info(
        "tool usage (sorted): %s",
        dict(sorted(report.tool_usage.items(), key=lambda kv: -kv[1])),
    )

    for e in report.errors[: args.max_errors_shown]:
        logger.error(str(e))
    if len(report.errors) > args.max_errors_shown:
        logger.error(
            "... and %d more errors suppressed", len(report.errors) - args.max_errors_shown
        )
    for v in report.coverage_violations:
        logger.error("coverage: %s", v)

    return 0 if report.ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
