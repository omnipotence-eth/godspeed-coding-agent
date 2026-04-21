"""Tests for experiments/swebench_lite/oracle_merge.py.

The helpers (_load_predictions, _load_resolved) are covered directly.
The main selection logic is exercised end-to-end via subprocess with
fake predictions + reports in tmp_path — that keeps v3.1.0's shipped
script exactly as-is (no refactor risk) while still giving us coverage.
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

ORACLE_MERGE = EXP_DIR / "oracle_merge.py"

from oracle_merge import _load_predictions, _load_resolved  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_report(path: Path, resolved_ids: list[str]) -> None:
    path.write_text(json.dumps({"resolved_ids": resolved_ids}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_load_predictions_returns_instance_id_to_patch(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    _write_jsonl(
        p,
        [
            {"instance_id": "a", "model_patch": "diff_a"},
            {"instance_id": "b", "model_patch": "diff_b"},
        ],
    )
    out = _load_predictions(p)
    assert out == {"a": "diff_a", "b": "diff_b"}


def test_load_predictions_coerces_missing_patch_to_empty(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    _write_jsonl(p, [{"instance_id": "a"}])  # no model_patch
    out = _load_predictions(p)
    assert out == {"a": ""}


def test_load_predictions_coerces_null_patch_to_empty(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    _write_jsonl(p, [{"instance_id": "a", "model_patch": None}])
    out = _load_predictions(p)
    assert out == {"a": ""}


def test_load_predictions_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    p.write_text(
        json.dumps({"instance_id": "a", "model_patch": "d"})
        + "\n\n   \n"
        + json.dumps({"instance_id": "b", "model_patch": "e"})
        + "\n",
        encoding="utf-8",
    )
    out = _load_predictions(p)
    assert out == {"a": "d", "b": "e"}


def test_load_resolved_returns_set(tmp_path: Path) -> None:
    p = tmp_path / "report.json"
    _write_report(p, ["i1", "i2", "i3"])
    assert _load_resolved(p) == {"i1", "i2", "i3"}


def test_load_resolved_missing_key_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "report.json"
    p.write_text("{}", encoding="utf-8")
    assert _load_resolved(p) == set()


# ---------------------------------------------------------------------------
# End-to-end: run oracle_merge.py via subprocess on fake inputs
# ---------------------------------------------------------------------------


def _run_oracle(
    tmp_path: Path, *, pairs: list[str], extra: list[str] = ()
) -> tuple[Path, Path, str]:
    """Invoke oracle_merge.py from tmp_path as cwd with relative paths.

    This matches the real invocation pattern (reproduce_v3_1.sh uses repo-relative
    paths). Passing absolute Windows paths via --pairs tokenizes wrong because
    rsplit(":", 1) hits the drive-letter colon; that path isn't a supported usage.
    """
    out_rel = "merged.jsonl"
    source_log_rel = "sources.jsonl"
    cmd = [
        sys.executable,
        str(ORACLE_MERGE),
        "--pairs",
        *pairs,
        "--out",
        out_rel,
        "--source-log",
        source_log_rel,
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(tmp_path))
    return tmp_path / out_rel, tmp_path / source_log_rel, result.stdout + result.stderr


def test_oracle_prefers_resolver_over_non_resolver(tmp_path: Path) -> None:
    # Run A resolves i1 only; Run B resolves nothing. Both have patches for both instances.
    _write_jsonl(
        tmp_path / "preds_a.jsonl",
        [
            {"instance_id": "i1", "model_patch": "a_patch_i1"},
            {"instance_id": "i2", "model_patch": "a_patch_i2_long" * 3},
        ],
    )
    _write_jsonl(
        tmp_path / "preds_b.jsonl",
        [
            {"instance_id": "i1", "model_patch": "b_patch_i1_long" * 3},
            {"instance_id": "i2", "model_patch": "short_i2"},
        ],
    )
    _write_report(tmp_path / "report_a.json", ["i1"])
    _write_report(tmp_path / "report_b.json", [])  # B resolves nothing

    out, sources, _ = _run_oracle(
        tmp_path,
        pairs=["preds_a.jsonl:report_a.json", "preds_b.jsonl:report_b.json"],
    )
    picks = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    srcs = [json.loads(line) for line in sources.read_text().splitlines() if line.strip()]

    # i1: Run A resolved → pick A's patch
    pick_i1 = next(p for p in picks if p["instance_id"] == "i1")
    src_i1 = next(s for s in srcs if s["instance_id"] == "i1")
    assert pick_i1["model_patch"] == "a_patch_i1"
    assert src_i1["strategy"] == "oracle_resolved"

    # i2: nobody resolved → shortest non-empty (Run B's "short_i2")
    pick_i2 = next(p for p in picks if p["instance_id"] == "i2")
    src_i2 = next(s for s in srcs if s["instance_id"] == "i2")
    assert pick_i2["model_patch"] == "short_i2"
    assert src_i2["strategy"] == "fallback_shortest_nonempty"


def test_oracle_prefers_shortest_among_resolvers(tmp_path: Path) -> None:
    """When multiple runs resolve, pick the shortest patch (most minimal)."""
    _write_jsonl(
        tmp_path / "preds_a.jsonl",
        [{"instance_id": "i1", "model_patch": "long_patch_verbose_fix" * 2}],
    )
    _write_jsonl(
        tmp_path / "preds_b.jsonl",
        [{"instance_id": "i1", "model_patch": "minimal_fix"}],
    )
    _write_report(tmp_path / "report_a.json", ["i1"])
    _write_report(tmp_path / "report_b.json", ["i1"])

    out, _sources, _ = _run_oracle(
        tmp_path,
        pairs=["preds_a.jsonl:report_a.json", "preds_b.jsonl:report_b.json"],
    )
    picks = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert picks[0]["model_patch"] == "minimal_fix"


def test_oracle_all_empty_emits_empty_patch(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "preds_a.jsonl", [{"instance_id": "i1", "model_patch": ""}])
    _write_jsonl(tmp_path / "preds_b.jsonl", [{"instance_id": "i1", "model_patch": ""}])
    _write_report(tmp_path / "report_a.json", [])
    _write_report(tmp_path / "report_b.json", [])

    out, sources, _ = _run_oracle(
        tmp_path,
        pairs=["preds_a.jsonl:report_a.json", "preds_b.jsonl:report_b.json"],
    )
    picks = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    srcs = [json.loads(line) for line in sources.read_text().splitlines() if line.strip()]
    assert picks[0]["model_patch"] == ""
    assert srcs[0]["strategy"] == "all_empty"


def test_oracle_single_pair_rejected(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "preds.jsonl", [{"instance_id": "i1", "model_patch": "d"}])
    _write_report(tmp_path / "report.json", ["i1"])
    result = subprocess.run(
        [
            sys.executable,
            str(ORACLE_MERGE),
            "--pairs",
            "preds.jsonl:report.json",
            "--out",
            "merged.jsonl",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "best-of-N" in result.stderr + result.stdout


def test_oracle_malformed_pair_rejected(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ORACLE_MERGE),
            "--pairs",
            "badinput",
            "badagain",
            "--out",
            "merged.jsonl",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "preds.jsonl:report.json" in result.stderr + result.stdout


def test_oracle_sets_uniform_model_name(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "preds_a.jsonl", [{"instance_id": "i1", "model_patch": "d"}])
    _write_jsonl(tmp_path / "preds_b.jsonl", [{"instance_id": "i1", "model_patch": "e"}])
    _write_report(tmp_path / "report_a.json", ["i1"])
    _write_report(tmp_path / "report_b.json", [])

    out, _sources, _ = _run_oracle(
        tmp_path,
        pairs=["preds_a.jsonl:report_a.json", "preds_b.jsonl:report_b.json"],
        extra=["--model-name", "oracle_test"],
    )
    picks = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all(p["model_name_or_path"] == "oracle_test" for p in picks)


def test_oracle_default_model_name_shows_count(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "preds_a.jsonl", [{"instance_id": "i1", "model_patch": "d"}])
    _write_jsonl(tmp_path / "preds_b.jsonl", [{"instance_id": "i1", "model_patch": "e"}])
    _write_jsonl(tmp_path / "preds_c.jsonl", [{"instance_id": "i1", "model_patch": "f"}])
    _write_report(tmp_path / "report_a.json", ["i1"])
    _write_report(tmp_path / "report_b.json", [])
    _write_report(tmp_path / "report_c.json", [])

    out, _sources, _ = _run_oracle(
        tmp_path,
        pairs=[
            "preds_a.jsonl:report_a.json",
            "preds_b.jsonl:report_b.json",
            "preds_c.jsonl:report_c.json",
        ],
    )
    picks = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert picks[0]["model_name_or_path"] == "oracle_best_of_3"


def test_oracle_missing_predictions_file_errors(tmp_path: Path) -> None:
    _write_report(tmp_path / "report.json", [])
    result = subprocess.run(
        [
            sys.executable,
            str(ORACLE_MERGE),
            "--pairs",
            "nope.jsonl:report.json",
            "nope.jsonl:report.json",
            "--out",
            "merged.jsonl",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "not found" in (result.stderr + result.stdout)


@pytest.mark.parametrize("n_runs", [2, 3, 5])
def test_oracle_scales_with_run_count(tmp_path: Path, n_runs: int) -> None:
    """Smoke test N-way merge."""
    pairs = []
    for i in range(n_runs):
        _write_jsonl(
            tmp_path / f"preds_{i}.jsonl",
            [{"instance_id": "i1", "model_patch": f"patch_{i}"}],
        )
        _write_report(
            tmp_path / f"report_{i}.json",
            ["i1"] if i == n_runs - 1 else [],
        )
        pairs.append(f"preds_{i}.jsonl:report_{i}.json")
    out, _srcs, _ = _run_oracle(tmp_path, pairs=pairs)
    picks = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert picks[0]["model_patch"] == f"patch_{n_runs - 1}"
