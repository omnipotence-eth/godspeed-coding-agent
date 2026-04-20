"""Reconstruct intended patches from Godspeed session logs.

For every instance in a predictions.jsonl, find the matching session log
under ~/.godspeed/training and report what file_edit / file_write tool
calls the agent made and whether they succeeded. This surfaces cases
where the agent edited files successfully but the patch capture came up
empty (the iter1 stochastic bug).

Not a submission-path — this is a diagnostic + debugging tool.

Usage:
    python experiments/swebench_lite/reconstruct.py \\
        --predictions experiments/swebench_lite/predictions_iter1.jsonl \\
        --training-dir ~/.godspeed/training \\
        --out experiments/swebench_lite/intent_iter1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_predictions(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["instance_id"]] = row.get("model_patch", "")
    return out


def _load_sessions(training_dir: Path) -> list[tuple[Path, list[dict]]]:
    sessions = []
    for p in sorted(training_dir.glob("*.conversation.jsonl")):
        try:
            recs = [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError:
            continue
        sessions.append((p, recs))
    return sessions


def _instance_id_of(recs: list[dict]) -> str | None:
    """Heuristically extract the instance_id from the first user message."""
    for r in recs:
        if r.get("role") != "user":
            continue
        content = str(r.get("content", ""))
        # SWE-Bench prompts start with the problem statement; look for
        # repo-slug fragments in the first ~3000 chars.
        for marker in (
            "pvlib__pvlib-python-",
            "sqlfluff__sqlfluff-",
            "marshmallow-code__marshmallow-",
            "pydicom__pydicom-",
            "pylint-dev__astroid-",
            "pyvista__pyvista-",
            "django__django-",
            "sympy__sympy-",
        ):
            # Fragile — the prompt doesn't literally contain the
            # instance_id, but matching repo-slug substrings scopes to a
            # family. Use timestamp + content signature as tiebreaker.
            if marker.split("__")[0] in content[:3000].lower():
                return None  # fall through to heuristic match
        return None
    return None


def _edits_in_session(recs: list[dict]) -> list[dict]:
    """Extract every file_edit / file_write / diff_apply call + its result."""
    out = []
    pending: dict[str, dict] = {}
    for r in recs:
        role = r.get("role")
        if role == "assistant" and r.get("tool_calls"):
            for tc in r["tool_calls"]:
                fn = (tc.get("function") or {}).get("name")
                args_raw = (tc.get("function") or {}).get("arguments", "")
                if fn not in ("file_edit", "file_write", "diff_apply"):
                    continue
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {"_raw": str(args_raw)[:200]}
                call_id = tc.get("id") or f"nc_{len(pending)}"
                pending[call_id] = {
                    "tool": fn,
                    "file_path": args.get("file_path"),
                    "old_string_len": len(str(args.get("old_string", ""))),
                    "new_string_len": len(str(args.get("new_string", ""))),
                    "content_len": len(str(args.get("content", ""))),
                }
        elif role == "tool":
            call_id = r.get("tool_call_id")
            if call_id and call_id in pending:
                entry = pending.pop(call_id)
                content = str(r.get("content", ""))
                entry["result_preview"] = content[:200]
                entry["success"] = not any(
                    m in content.lower()[:200] for m in ("error", "failed", "not found")
                )
                out.append(entry)
    # Orphaned pending (no result seen) — probably mid-stream
    for entry in pending.values():
        entry["result_preview"] = "<no result recorded>"
        entry["success"] = False
        out.append(entry)
    return out


def _match_sessions_to_predictions(
    predictions: dict[str, str], sessions: list[tuple[Path, list[dict]]]
) -> dict[str, list[dict]]:
    """Approximate match: session maps to instance via problem-statement substring.

    Each SWE-Bench instance has a distinctive first ~200-char problem statement.
    We use that as a fingerprint.
    """
    # Build instance_id -> problem_statement fingerprint map from the dataset.
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    instance_fp: dict[str, str] = {}
    for row in ds:
        ps = row["problem_statement"]
        # First 150 non-whitespace chars
        fp = "".join(ps.split())[:150]
        instance_fp[row["instance_id"]] = fp

    # For each session, find the instance whose fingerprint appears in the first user msg
    by_iid: dict[str, list[dict]] = defaultdict(list)
    for path, recs in sessions:
        first_user = next(
            (str(r.get("content", "")) for r in recs if r.get("role") == "user"), ""
        )
        first_user_compact = "".join(first_user.split())[:600]
        for iid, fp in instance_fp.items():
            if fp and fp in first_user_compact:
                edits = _edits_in_session(recs)
                by_iid[iid].append(
                    {
                        "session_path": str(path),
                        "records": len(recs),
                        "edit_tool_calls": len(edits),
                        "successful_edits": sum(1 for e in edits if e.get("success")),
                        "edits": edits,
                    }
                )
                break
    return by_iid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions)
    sessions = _load_sessions(args.training_dir.expanduser())
    print(f"Loaded {len(predictions)} predictions, {len(sessions)} sessions", file=sys.stderr)

    matched = _match_sessions_to_predictions(predictions, sessions)
    print(f"Matched {len(matched)} instances to sessions", file=sys.stderr)

    summary: list[dict] = []
    for iid, patch in sorted(predictions.items()):
        patch_empty = not patch.strip()
        sess_list = matched.get(iid, [])
        # Pick the LAST matching session (most recent run of that instance)
        last = sess_list[-1] if sess_list else None
        summary.append(
            {
                "instance_id": iid,
                "patch_empty": patch_empty,
                "patch_lines": len(patch.splitlines()) if patch else 0,
                "matched_sessions": len(sess_list),
                "last_session": last["session_path"] if last else None,
                "edit_tool_calls": last["edit_tool_calls"] if last else 0,
                "successful_edits": last["successful_edits"] if last else 0,
                "edits_summary": [
                    {
                        "tool": e["tool"],
                        "file_path": e["file_path"],
                        "success": e.get("success"),
                    }
                    for e in (last["edits"] if last else [])
                ],
            }
        )

    with args.out.open("w", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row) + "\n")

    # Headline table
    print()
    print(f"{'instance_id':<45} {'patch':>6} {'edits':>6} {'succ':>5}  flag")
    print("-" * 80)
    for row in summary:
        flag = ""
        if row["patch_empty"] and row["successful_edits"] > 0:
            flag = "*** SUCCESS EDITS BUT EMPTY PATCH ***"
        elif row["patch_empty"]:
            flag = "(no edits attempted)"
        print(
            f"{row['instance_id']:<45} {row['patch_lines']:>6} "
            f"{row['edit_tool_calls']:>6} {row['successful_edits']:>5}  {flag}"
        )

    lost_edits = sum(
        1 for r in summary if r["patch_empty"] and r["successful_edits"] > 0
    )
    print()
    print(f"Empty patches despite successful edits: {lost_edits} / {len(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
