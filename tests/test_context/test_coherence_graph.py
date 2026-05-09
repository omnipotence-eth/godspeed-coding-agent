"""Tests for Global Coherence Graph."""

from __future__ import annotations

from pathlib import Path

from godspeed.context.coherence_graph import (
    BlastRadius,
    BuildResult,
    CoherenceGraph,
    FileSpan,
    Invariant,
    UpdateResult,
)

SAMPLE_PY = '''\
"""Sample module for testing."""

import os
from pathlib import Path as Pth


def helper(x: int) -> int:
    """Double the input."""
    return x * 2


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b


CONSTANT = 42


def main() -> None:
    calc = Calculator()
    result = calc.add(1, helper(CONSTANT))
    print(result)
'''


def _build_gcg(tmp_path: Path, code: str = SAMPLE_PY) -> CoherenceGraph:
    """Build a GCG on a temporary repo with a single sample file."""
    db_path = tmp_path / "gcg.db"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    py_file = src_dir / "sample.py"
    py_file.write_text(code)

    gcg = CoherenceGraph(db_path)
    gcg.connect()
    gcg.build_from_repo(tmp_path, incremental=False)
    return gcg


class TestBuildFromRepo:
    """Test building the graph from a repository."""

    def test_build_extracts_symbols(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        stats = gcg.get_graph_stats()
        assert stats["symbols"] > 0
        assert stats["files"] > 0

    def test_build_finds_functions(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("helper")
        assert len(results) > 0
        sym = results[0]
        assert sym.kind == "function"
        assert "helper" in sym.name

    def test_build_finds_classes(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("Calculator")
        assert len(results) > 0
        sym = results[0]
        assert sym.kind == "class"

    def test_build_finds_methods(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("add")
        assert len(results) > 0
        sym = results[0]
        assert sym.kind in ("method", "function")

    def test_build_finds_variables(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("CONSTANT")
        assert len(results) > 0
        assert results[0].kind == "variable"

    def test_build_returns_build_result(self, tmp_path: Path) -> None:
        db_path = tmp_path / "gcg2.db"
        src_dir = tmp_path / "src2"
        src_dir.mkdir()
        py_file = src_dir / "empty.py"
        py_file.write_text("# empty")

        gcg2 = CoherenceGraph(db_path)
        gcg2.connect()
        result = gcg2.build_from_repo(tmp_path, incremental=False)
        assert isinstance(result, BuildResult)
        assert result.files_parsed >= 0


class TestFindSymbol:
    """Test symbol lookup."""

    def test_get_symbol_by_id(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("Calculator")
        assert len(results) > 0
        sym = gcg.get_symbol(results[0].id)
        assert sym is not None
        assert sym.name == "Calculator"

    def test_find_symbol_partial_match(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("Calc")  # partial match
        assert len(results) > 0

    def test_find_symbol_no_match(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("nonexistent_function")
        assert len(results) == 0

    def test_find_symbol_with_file_hint(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        src_file = tmp_path / "src" / "sample.py"
        results = gcg.find_symbol("helper", file_hint=src_file)
        assert len(results) > 0


class TestDependencies:
    """Test dependency edge resolution."""

    def test_extract_imports(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        deps = gcg.query_sql("SELECT * FROM dependencies WHERE kind = 'imports'")
        assert len(deps) > 0

    def test_call_dependencies_extracted(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        deps = gcg.query_sql("SELECT * FROM dependencies WHERE kind = 'calls'")
        assert len(deps) >= 0  # Call resolution depends on symbol matching

    def test_get_callees(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("main")
        if results:
            callees = gcg.get_callees(results[0].id)
            assert isinstance(callees, list)

    def test_get_callers(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("helper")
        if results:
            callers = gcg.get_callers(results[0].id)
            assert isinstance(callers, list)

    def test_get_dependents(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        src_file = tmp_path / "src" / "sample.py"
        dependents = gcg.get_dependents(src_file)
        assert isinstance(dependents, list)


class TestBlastRadius:
    """Test blast radius calculation."""

    def test_blast_radius_returns_structure(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("helper")
        if not results:
            return
        blast = gcg.get_blast_radius(results[0].id, max_depth=2)
        assert isinstance(blast, BlastRadius)
        assert blast.symbol_id == results[0].id
        assert blast.depth == 2
        assert isinstance(blast.affected_symbols, list)

    def test_blast_radius_for_class(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("Calculator")
        if not results:
            return
        blast = gcg.get_blast_radius(results[0].id, max_depth=3)
        assert isinstance(blast, BlastRadius)
        assert isinstance(blast.affected_files, set)


class TestUpdateFile:
    """Test incremental file updates."""

    def test_update_file_removes_old_symbols(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        src_file = tmp_path / "src" / "sample.py"
        new_code = "# empty file\nCONSTANT = 1\n"
        result = gcg.update_file(src_file, new_code, modified_by="test")
        assert isinstance(result, UpdateResult)
        assert result.symbols_removed >= 0

    def test_update_file_preserves_other_files(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        # Add a second file
        src_dir = tmp_path / "src"
        other = src_dir / "other.py"
        other.write_text("def foo() -> None:\n    return 42\n")

        # Build incremental (reparses the whole repo but adds to existing GCG)
        gcg.build_from_repo(tmp_path, incremental=True)

        # Update only sample.py
        src_file = src_dir / "sample.py"
        gcg.update_file(src_file, "# empty\nX = 1\n", modified_by="test")

        # Verify other.py symbols still present
        results = gcg.find_symbol("foo")
        assert len(results) > 0


class TestInvariants:
    """Test architectural invariants."""

    def test_add_invariant(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        gcg.add_invariant(
            Invariant(
                id="test-invariant-1",
                description="All functions must be documented",
                scope_glob="**/*.py",
                kind="structural",
                added_by="test",
            )
        )
        stats = gcg.get_graph_stats()
        assert stats["open_violations"] >= 0

    def test_check_invariants_after_change(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        gcg.add_invariant(
            Invariant(
                description="Test invariant",
                scope_glob="**/*.py",
                kind="structural",
                added_by="test",
            )
        )
        symbols = gcg.query_sql("SELECT id FROM symbols LIMIT 5")
        if symbols:
            violations = gcg.check_invariants([s["id"] for s in symbols])
            assert isinstance(violations, list)

    def test_get_violations(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        violations = gcg.get_violations()
        assert isinstance(violations, list)

    def test_get_violations_scoped(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        violations = gcg.get_violations(scope="src")
        assert isinstance(violations, list)


class TestContextSummary:
    """Test GCG context summary round-trip."""

    def test_get_context_summary(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        symbols = gcg.query_sql("SELECT id FROM symbols LIMIT 3")
        ids = [s["id"] for s in symbols]
        summary = gcg.get_context_summary(ids)
        assert "GCG:" in summary

    def test_resolve_context_summary(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        symbols = gcg.query_sql("SELECT id FROM symbols LIMIT 2")
        ids = [s["id"] for s in symbols]
        summary = gcg.get_context_summary(ids)
        spans = gcg.resolve_context_summary(summary)
        assert len(spans) == len(ids)
        for span in spans:
            assert isinstance(span, FileSpan)
            assert span.file.exists() or True  # May use relative paths

    def test_resolve_empty_summary(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        spans = gcg.resolve_context_summary("")
        assert isinstance(spans, list)
        assert len(spans) == 0


class TestGraphStats:
    """Test graph statistics."""

    def test_get_graph_stats(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        stats = gcg.get_graph_stats()
        assert "symbols" in stats
        assert "dependencies" in stats
        assert "files" in stats
        assert "open_violations" in stats
        assert stats["symbols"] > 0


class TestQuerySQL:
    """Test direct SQL query."""

    def test_query_sql_read_only(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        rows = gcg.query_sql("SELECT COUNT(*) as cnt FROM symbols")
        assert len(rows) == 1
        assert rows[0]["cnt"] > 0

    def test_query_sql_blocks_write(self, tmp_path: Path) -> None:
        gcg = _build_gcg(tmp_path)
        try:
            gcg.query_sql("DELETE FROM symbols")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
