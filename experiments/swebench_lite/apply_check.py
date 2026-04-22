"""Per-instance `git apply --check` signal for candidate patches.

A test-free signal: for each candidate patch, we can ask "would this
patch apply cleanly at the instance's base_commit?" without actually
running tests. Patches that *do not apply cleanly* are almost certainly
wrong (bad hunks, wrong line numbers, edits to files that don't exist
at base_commit). Patches that *do* apply cleanly are merely *possible*
solutions — still no test access.

This module runs `git apply --check` inside a freshly-cloned repo
checked out to the instance's `base_commit`. The repo is cached under
``~/.godspeed/swebench_repos_cache/`` so the clone is a one-time cost
per unique (repo, base_commit) pair.

Usage (standalone)
------------------

    python experiments/swebench_lite/apply_check.py \\
        --pairs \\
            predictions_kimi.jsonl:e1_kimi \\
            predictions_gpt_oss.jsonl:gpt_oss \\
            ... \\
        --split dev \\
        --out experiments/swebench_lite/apply_check_results.jsonl

The output is JSONL with one row per (instance_id, label) pair:

    {"instance_id": "...", "label": "e1_kimi", "applies": true/false,
     "reason": "(stderr if failed)"}

Integration with the LLM judge
------------------------------

When used as an augmentation to ``llm_judge_selector.py``, a candidate
whose ``applies=false`` can be DISQUALIFIED (treated like an empty slot
from the judge's perspective). That narrows the selection pool to only
"possibly valid" patches without touching test knowledge.

Eligibility
-----------

``git apply --check`` inspects only the patch's interaction with the
pre-patch tree. It does not run tests, read `PASS_TO_PASS` / `FAIL_TO_PASS`
/ `hints_text`, or compare against the gold patch. It is a legitimate
test-free signal for best@k aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".godspeed" / "swebench_repos_cache"


@dataclass
class ApplyCheckResult:
    instance_id: str
    label: str
    applies: bool
    reason: str


def _repo_cache_key(repo: str, base_commit: str) -> str:
    """Stable filesystem-safe directory name for a (repo, base_commit) pair."""
    # Not a security context — just a filesystem-safe key for a cache dir.
    digest = hashlib.sha1(f"{repo}@{base_commit}".encode(), usedforsecurity=False).hexdigest()[:12]
    safe_repo = repo.replace("/", "__")
    return f"{safe_repo}__{digest}"


def _ensure_repo(repo: str, base_commit: str, cache_dir: Path) -> Path:
    """Clone `repo` at `base_commit` under cache_dir (idempotent). Returns worktree path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _repo_cache_key(repo, base_commit)
    if (target / ".git").exists():
        logger.debug("repo_cache hit: %s", target)
        return target
    logger.info("cloning %s @ %s -> %s", repo, base_commit, target)
    # Shallow clone + fetch specific commit; much faster than full clone on big repos.
    subprocess.run(
        ["git", "init", "--quiet", str(target)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "remote", "add", "origin", f"https://github.com/{repo}.git"],
        check=True,
        capture_output=True,
    )
    fetch = subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth=1", "--quiet", "origin", base_commit],
        capture_output=True,
        text=True,
    )
    if fetch.returncode != 0:
        # Shallow fetch of arbitrary commit isn't always supported — fall back to full fetch.
        logger.debug("shallow fetch failed, retrying full: %s", fetch.stderr[:200])
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--quiet", "origin"],
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--quiet", base_commit],
        check=True,
        capture_output=True,
    )
    return target


def check_patch(repo_path: Path, patch: str) -> tuple[bool, str]:
    """Run `git apply --check` on `patch` inside `repo_path`. Returns (applies_cleanly, reason).

    `patch` is passed on stdin. Any non-zero exit → does not apply.

    We disable Windows-side CRLF munging (``-c core.autocrlf=false
    -c core.safecrlf=false``) so line-ending conversion doesn't cause
    false-negative apply failures on repos that checked out with CRLF
    by default (e.g. sqlfluff). Patches produced by agents are LF-only;
    if git locally normalised the working tree to CRLF, `git apply`
    fails even though the logical change applies cleanly.
    """
    if not patch or not patch.strip():
        return False, "empty patch"
    try:
        # --ignore-whitespace handles LF-vs-CRLF mismatches between LF-only
        # patches (how agents emit them) and CRLF-normalised working trees
        # (how git checks out on Windows by default). --recount tolerates
        # slightly-off hunk line counts that some models produce. This keeps
        # the signal honest: only *real* apply failures (wrong file, wrong
        # context, missing lines) show up as applies=False.
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.safecrlf=false",
                "-C",
                str(repo_path),
                "apply",
                "--check",
                "--recount",
                "--ignore-whitespace",
                "-",
            ],
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "git apply --check timed out after 30s"
    if result.returncode == 0:
        return True, "ok"
    stderr_tail = (result.stderr or result.stdout or "")[-200:]
    return False, f"apply failed: {stderr_tail.strip()}"


def _load_predictions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["instance_id"]] = row.get("model_patch", "") or ""
    return out


def _load_dataset_repos(split: str) -> dict[str, tuple[str, str]]:
    """instance_id -> (repo, base_commit)."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return {row["instance_id"]: (row["repo"], row["base_commit"]) for row in ds}


def run_apply_checks(
    pairs: list[tuple[Path, str]],
    split: str,
    cache_dir: Path,
    instance_filter: set[str] | None = None,
) -> list[ApplyCheckResult]:
    """Check every (instance, label) candidate patch via `git apply --check`.

    Repos are cloned on demand into cache_dir. Patches that are empty are
    marked applies=False with reason='empty patch' (no git call made).
    """
    per_run = [(label, _load_predictions(p)) for p, label in pairs]
    instance_to_repo = _load_dataset_repos(split)

    all_ids: set[str] = set()
    for _, preds in per_run:
        all_ids.update(preds.keys())
    if instance_filter:
        all_ids &= instance_filter

    results: list[ApplyCheckResult] = []
    repo_paths: dict[str, Path] = {}

    for iid in sorted(all_ids):
        if iid not in instance_to_repo:
            logger.warning("skip instance not in dataset: %s", iid)
            continue
        repo, base_commit = instance_to_repo[iid]
        cache_key = _repo_cache_key(repo, base_commit)
        if cache_key not in repo_paths:
            try:
                repo_paths[cache_key] = _ensure_repo(repo, base_commit, cache_dir)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "clone/checkout failed for %s@%s: %s",
                    repo,
                    base_commit,
                    (exc.stderr or b"").decode(errors="replace")[-200:],
                )
                for label, preds in per_run:
                    if iid in preds:
                        results.append(
                            ApplyCheckResult(iid, label, False, f"clone/checkout failed: {exc}")
                        )
                continue
        repo_path = repo_paths[cache_key]
        for label, preds in per_run:
            patch = preds.get(iid, "")
            applies, reason = check_patch(repo_path, patch)
            results.append(ApplyCheckResult(iid, label, applies, reason))
            logger.info(
                "apply_check instance=%s label=%s applies=%s %s",
                iid,
                label,
                applies,
                reason[:60],
            )
    return results


def _parse_pairs(raw: list[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"--pairs entry must be 'path:label', got {entry!r}")
        path_s, label = entry.rsplit(":", 1)
        out.append((Path(path_s), label))
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--pairs", nargs="+", required=True, help="preds.jsonl:label entries")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory for cloned repos"
    )
    p.add_argument("--instances", nargs="*", default=None, help="Optional instance_id filter")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    pairs = _parse_pairs(args.pairs)
    instance_filter = set(args.instances) if args.instances else None
    results = run_apply_checks(pairs, args.split, args.cache_dir, instance_filter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "instance_id": r.instance_id,
                        "label": r.label,
                        "applies": r.applies,
                        "reason": r.reason[:200],
                    }
                )
                + "\n"
            )

    # Summary per label
    by_label: dict[str, dict[str, int]] = {}
    for r in results:
        stats = by_label.setdefault(r.label, {"total": 0, "applies": 0})
        stats["total"] += 1
        if r.applies:
            stats["applies"] += 1

    summary: dict[str, Any] = {
        "instances": len({r.instance_id for r in results}),
        "labels": by_label,
    }
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    return _main(_build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
