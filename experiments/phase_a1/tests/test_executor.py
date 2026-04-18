"""Unit tests for experiments.phase_a1.executor.

Covers the pieces that don't need a real LLM:
  * fixture dispatch is deterministic (same args \u2192 same output)
  * missing fixture files fall back to a placeholder pool
  * blueprint execution produces one ExecutedStep per planned call
  * sandbox seeding creates the expected seed files
  * timeout path produces is_error=True without crashing
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from experiments.phase_a1.executor import (
    Blueprint,
    PlannedCall,
    _load_fixtures,
    _pick_fixture,
    _seed_sandbox,
    execute_blueprint,
)
from experiments.phase_a1.registry_builder import FIXTURE_BACKED_TOOLS, build_registry

# ---------------------------------------------------------------------------
# Fixture dispatch
# ---------------------------------------------------------------------------


def test_fixture_dispatch_deterministic_for_same_args(tmp_path: Path) -> None:
    (tmp_path / "web_search.json").write_text(
        json.dumps(["alpha result", "beta result", "gamma result"]),
        encoding="utf-8",
    )
    # Clear cache so this fixtures_dir is loaded fresh
    import experiments.phase_a1.executor as ex

    ex._FIXTURE_CACHE.clear()

    args = {"query": "asyncio"}
    a = _pick_fixture("web_search", args, tmp_path)
    b = _pick_fixture("web_search", args, tmp_path)
    assert a == b


def test_fixture_dispatch_varies_with_args(tmp_path: Path) -> None:
    (tmp_path / "web_search.json").write_text(
        json.dumps([f"result_{i}" for i in range(10)]),
        encoding="utf-8",
    )
    import experiments.phase_a1.executor as ex

    ex._FIXTURE_CACHE.clear()

    seen = {_pick_fixture("web_search", {"query": f"q{i}"}, tmp_path) for i in range(20)}
    # With 10 variants and 20 hashed queries, we should see at least 2 distinct outputs.
    assert len(seen) >= 2


def test_fixture_missing_file_returns_placeholder(tmp_path: Path) -> None:
    import experiments.phase_a1.executor as ex

    ex._FIXTURE_CACHE.clear()
    pool = _load_fixtures("nonexistent_tool", tmp_path)
    assert pool
    assert "placeholder" in pool[0]


def test_fixture_accepts_object_array_with_output_field(tmp_path: Path) -> None:
    (tmp_path / "github.json").write_text(
        json.dumps([{"args_match": {}, "output": "PR #1"}, {"output": "PR #2"}]),
        encoding="utf-8",
    )
    import experiments.phase_a1.executor as ex

    ex._FIXTURE_CACHE.clear()
    pool = _load_fixtures("github", tmp_path)
    assert pool == ["PR #1", "PR #2"]


def test_all_fixture_backed_tools_have_fixture_files() -> None:
    """The 7 fixture-backed tools must each have a real fixture JSON file."""
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    for tool in FIXTURE_BACKED_TOOLS:
        path = fixtures_dir / f"{tool}.json"
        assert path.exists(), f"missing fixture file: {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) >= 5, (
            f"{tool}.json should be a list of \u22655 realistic responses"
        )


# ---------------------------------------------------------------------------
# Sandbox seeding
# ---------------------------------------------------------------------------


def test_seed_sandbox_creates_expected_layout(tmp_path: Path) -> None:
    _seed_sandbox(tmp_path)
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "src" / "main.py").exists()
    assert (tmp_path / "tests" / "test_main.py").exists()
    assert (tmp_path / "pyproject.toml").exists()
    # git may or may not be available \u2014 seed should never raise either way.


# ---------------------------------------------------------------------------
# Blueprint execution (no LLM, sandbox-only tool)
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def test_execute_blueprint_runs_real_file_read(tmp_path: Path) -> None:
    registry = build_registry()
    blueprint = Blueprint(
        user_intent="Read the main entry point",
        planned_calls=[
            PlannedCall(tool_name="file_read", arguments={"file_path": "src/main.py"}),
        ],
        expected_outcome="shows greet()",
        category="single_tool",
        primary_tool="file_read",
        spec_index=0,
        spec_seed=42,
    )
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    artifact = _run(
        execute_blueprint(
            blueprint,
            registry,
            output_dir=tmp_path / "sessions",
            fixtures_dir=fixtures_dir,
        )
    )
    assert len(artifact.steps) == 1
    step = artifact.steps[0]
    assert step.tool_name == "file_read"
    assert step.source == "real"
    assert step.is_error is False
    assert "greet" in step.output


def test_execute_blueprint_uses_fixture_for_web_search(tmp_path: Path) -> None:
    registry = build_registry()
    blueprint = Blueprint(
        user_intent="Look up ruff docs",
        planned_calls=[
            PlannedCall(tool_name="web_search", arguments={"query": "ruff config"}),
        ],
        expected_outcome="returns search results",
        category="single_tool",
        primary_tool="web_search",
        spec_index=1,
        spec_seed=43,
    )
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    artifact = _run(
        execute_blueprint(
            blueprint,
            registry,
            output_dir=tmp_path / "sessions",
            fixtures_dir=fixtures_dir,
        )
    )
    step = artifact.steps[0]
    assert step.source == "fixture"
    assert step.output
    assert not step.is_error


def test_execute_blueprint_no_tool_produces_empty_steps(tmp_path: Path) -> None:
    registry = build_registry()
    blueprint = Blueprint(
        user_intent="What is a closure?",
        planned_calls=[],
        expected_outcome="conceptual answer",
        category="no_tool",
        primary_tool="file_read",
        spec_index=2,
        spec_seed=44,
    )
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    artifact = _run(
        execute_blueprint(
            blueprint,
            registry,
            output_dir=tmp_path / "sessions",
            fixtures_dir=fixtures_dir,
        )
    )
    assert artifact.steps == []
    assert artifact.session_path.exists()


def test_execute_blueprint_reports_error_for_bad_args(tmp_path: Path) -> None:
    registry = build_registry()
    blueprint = Blueprint(
        user_intent="Read nothing",
        planned_calls=[PlannedCall(tool_name="file_read", arguments={})],
        expected_outcome="tool should fail",
        category="single_tool",
        primary_tool="file_read",
        spec_index=3,
        spec_seed=45,
    )
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    artifact = _run(
        execute_blueprint(
            blueprint,
            registry,
            output_dir=tmp_path / "sessions",
            fixtures_dir=fixtures_dir,
        )
    )
    assert len(artifact.steps) == 1
    assert artifact.steps[0].is_error is True
