"""Tests for experiments/swebench_lite/apply_check.py.

Pure-function surface (parse, cache key, check_patch with tmp git repo) is
covered. Network-touching helpers (_ensure_repo, _load_dataset_repos) are
not tested here — covered by live-run smoke.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments" / "swebench_lite"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from apply_check import (  # noqa: E402
    _load_predictions,
    _parse_pairs,
    _repo_cache_key,
    check_patch,
)

# ---------------------------------------------------------------------------
# _repo_cache_key
# ---------------------------------------------------------------------------


def test_repo_cache_key_stable() -> None:
    k1 = _repo_cache_key("pvlib/pvlib-python", "abc123")
    k2 = _repo_cache_key("pvlib/pvlib-python", "abc123")
    assert k1 == k2


def test_repo_cache_key_depends_on_commit() -> None:
    assert _repo_cache_key("r/x", "aaa") != _repo_cache_key("r/x", "bbb")


def test_repo_cache_key_depends_on_repo() -> None:
    assert _repo_cache_key("r/x", "aaa") != _repo_cache_key("r/y", "aaa")


def test_repo_cache_key_safe_filename() -> None:
    k = _repo_cache_key("owner/repo", "abc")
    assert "/" not in k
    assert "\\" not in k


# ---------------------------------------------------------------------------
# _parse_pairs
# ---------------------------------------------------------------------------


def test_parse_pairs_happy_path() -> None:
    pairs = _parse_pairs(["a.jsonl:x", "b.jsonl:y"])
    assert pairs == [(Path("a.jsonl"), "x"), (Path("b.jsonl"), "y")]


def test_parse_pairs_rejects_missing_colon() -> None:
    with pytest.raises(ValueError, match="path:label"):
        _parse_pairs(["bad"])


# ---------------------------------------------------------------------------
# _load_predictions
# ---------------------------------------------------------------------------


def test_load_predictions_missing_patch_is_empty(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    p.write_text(
        json.dumps({"instance_id": "a"})
        + "\n"
        + json.dumps({"instance_id": "b", "model_patch": None})
        + "\n",
        encoding="utf-8",
    )
    preds = _load_predictions(p)
    assert preds == {"a": "", "b": ""}


# ---------------------------------------------------------------------------
# check_patch — uses a real tmp git repo
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Init a fresh git repo with one file committed. Returns the worktree path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], check=True, cwd=repo)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, cwd=repo)
    subprocess.run(["git", "config", "user.name", "Test"], check=True, cwd=repo)
    (repo / "hello.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], check=True, cwd=repo)
    subprocess.run(
        ["git", "commit", "-m", "init", "--quiet"],
        check=True,
        cwd=repo,
    )
    return repo


def test_check_patch_empty_patch(tmp_git_repo: Path) -> None:
    applies, reason = check_patch(tmp_git_repo, "")
    assert not applies
    assert "empty" in reason


def test_check_patch_whitespace_only(tmp_git_repo: Path) -> None:
    applies, _ = check_patch(tmp_git_repo, "   \n  \n")
    assert not applies


def test_check_patch_clean_apply(tmp_git_repo: Path) -> None:
    patch = (
        "diff --git a/hello.py b/hello.py\n"
        "--- a/hello.py\n"
        "+++ b/hello.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def hello():\n"
        "-    return 'world'\n"
        "+    return 'universe'\n"
    )
    applies, reason = check_patch(tmp_git_repo, patch)
    assert applies, f"should apply: {reason}"


def test_check_patch_malformed_hunk_rejected(tmp_git_repo: Path) -> None:
    patch = (
        "diff --git a/hello.py b/hello.py\n"
        "--- a/hello.py\n"
        "+++ b/hello.py\n"
        "@@ -99,1 +99,1 @@\n"  # line numbers don't exist
        "-something that isn't there\n"
        "+replacement\n"
    )
    applies, reason = check_patch(tmp_git_repo, patch)
    assert not applies
    assert "apply failed" in reason


def test_check_patch_nonexistent_file_rejected(tmp_git_repo: Path) -> None:
    patch = (
        "diff --git a/nonexistent.py b/nonexistent.py\n"
        "--- a/nonexistent.py\n"
        "+++ b/nonexistent.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-foo\n"
        "+bar\n"
    )
    applies, reason = check_patch(tmp_git_repo, patch)
    assert not applies
    assert "apply failed" in reason


def test_check_patch_gibberish_rejected(tmp_git_repo: Path) -> None:
    applies, reason = check_patch(tmp_git_repo, "this isn't a diff at all")
    assert not applies
    assert "apply failed" in reason


def test_check_patch_new_file_added(tmp_git_repo: Path) -> None:
    """Adding a brand-new file should apply cleanly if correctly formatted."""
    patch = (
        "diff --git a/newfile.py b/newfile.py\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        "+++ b/newfile.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def foo():\n"
        "+    return 1\n"
    )
    applies, reason = check_patch(tmp_git_repo, patch)
    assert applies, f"should apply: {reason}"
