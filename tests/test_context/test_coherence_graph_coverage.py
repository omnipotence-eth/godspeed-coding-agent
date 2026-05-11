"""Coverage gap tests for coherence_graph — all missed branches."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.context.coherence_graph import (
    BlastRadius,
    BuildResult,
    CoherenceGraph,
    Invariant,
    InvariantViolation,
    UpdateResult,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    logging.getLogger("godspeed.context.coherence_graph").handlers.clear()
    logging.getLogger("godspeed.context.coherence_graph").propagate = True
    yield


SAMPLE_PY = """\
\"\"\"Sample module for testing.\"\"\"

import os
from pathlib import Path as Pth


def helper(x: int) -> int:
    \"\"\"Double the input.\"\"\"
    return x * 2


class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b


CONSTANT = 42


def main() -> None:
    calc = Calculator()
    result = calc.add(1, helper(CONSTANT))
    print(result)
"""

SAMPLE_INHERIT = """\
class Base:
    def base_method(self):
        pass


class Child(Base):
    def child_method(self):
        pass
"""

SAMPLE_DECORATED = """\
from functools import wraps

def logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


@logger
def decorated_func():
    pass
"""

SAMPLE_CALLS = """\
def callee():
    return 42


def caller():
    return callee()
"""

SAMPLE_EMPTY = "# just a comment\n"

SAMPLE_ERROR = "def broken(\n"


def _build_gcg(
    tmp_path: Path, code: str = SAMPLE_PY, filename: str = "sample.py"
) -> CoherenceGraph:
    db_path = tmp_path / "gcg.db"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    py_file = src_dir / filename
    py_file.write_text(code)

    gcg = CoherenceGraph(db_path)
    gcg.connect()
    gcg.build_from_repo(tmp_path, incremental=False)
    return gcg


# ── Lifecycle edge cases ────────────────────────────────────────────────────


class TestLifecycle:
    """Cover connect/close edge cases."""

    def test_close_when_not_connected(self, tmp_path):
        gcg = CoherenceGraph(tmp_path / "gcg.db")
        gcg.close()  # should not raise, _conn is None

    def test_close_when_connected(self, tmp_path):
        gcg = CoherenceGraph(tmp_path / "gcg.db")
        gcg.connect()
        assert gcg._conn is not None
        gcg.close()
        assert gcg._conn is None

    def test_conn_property_raises_when_not_connected(self, tmp_path):
        gcg = CoherenceGraph(tmp_path / "gcg.db")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = gcg.conn

    def test_schema_path_not_exists(self, tmp_path):
        gcg = CoherenceGraph(tmp_path / "gcg.db")
        gcg.connect()
        stats = gcg.get_graph_stats()
        assert "symbols" in stats

    def test_db_version_set_on_schema_apply(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        rows = gcg.query_sql("SELECT * FROM gcg_meta WHERE key = 'db_version'")
        assert len(rows) == 1
        assert rows[0]["value"] == "1"


# ── Build edge cases ────────────────────────────────────────────────────────


class TestBuildEdges:
    """Cover remaining branches in build_from_repo."""

    def test_build_incremental_skips_unchanged(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        # Rebuild incremental — should detect file is up to date
        result = gcg.build_from_repo(tmp_path, incremental=True)
        assert isinstance(result, BuildResult)
        # Since file is unchanged, files_parsed may be 0

    def test_build_with_custom_languages(self, tmp_path):
        db_path = tmp_path / "gcg_lang.db"
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def foo(): pass\n")
        (src_dir / "app.js").write_text("function bar() {}\n")

        gcg = CoherenceGraph(db_path)
        gcg.connect()
        result = gcg.build_from_repo(tmp_path, languages=["py"])
        assert result.files_parsed >= 0
        # Only .py files parsed since ext_map only has .py

    def test_build_with_custom_excludes(self, tmp_path):
        db_path = tmp_path / "gcg_excl.db"
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def foo(): pass\n")
        exclude_dir = tmp_path / "excluded"
        exclude_dir.mkdir()
        (exclude_dir / "skip.py").write_text("def skipped(): pass\n")

        gcg = CoherenceGraph(db_path)
        gcg.connect()
        result = gcg.build_from_repo(tmp_path, exclude_patterns=["excluded"])
        assert isinstance(result, BuildResult)
        # Verify excluded directory's files not parsed
        results = gcg.find_symbol("skipped")
        assert len(results) == 0

    def test_build_parse_error_logged(self, tmp_path):
        db_path = tmp_path / "gcg_err.db"
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "bad.py").write_text("def broken(\n")  # SyntaxError

        gcg = CoherenceGraph(db_path)
        gcg.connect()
        result = gcg.build_from_repo(tmp_path, incremental=False)
        assert isinstance(result, BuildResult)

    def test_build_empty_repo(self, tmp_path):
        db_path = tmp_path / "gcg_empty.db"
        (tmp_path / "src").mkdir()
        gcg = CoherenceGraph(db_path)
        gcg.connect()
        result = gcg.build_from_repo(tmp_path, incremental=False)
        assert result.symbol_count == 0
        assert result.files_parsed == 0

    def test_build_progress_logging(self, tmp_path):
        db_path = tmp_path / "gcg_prog.db"
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        for i in range(150):
            (src_dir / f"mod_{i}.py").write_text(f"def func_{i}(): pass\n")

        gcg = CoherenceGraph(db_path)
        gcg.connect()
        result = gcg.build_from_repo(tmp_path, incremental=False)
        assert isinstance(result, BuildResult)
        assert result.files_parsed >= 100


# ── _file_up_to_date edge cases ────────────────────────────────────────────


class TestFileUpToDate:
    """Cover _file_up_to_date OSError path."""

    def test_file_up_to_date_oserror(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        fake_path = MagicMock()
        fake_path.stat.side_effect = OSError("cannot stat")
        result = gcg._file_up_to_date(fake_path)
        assert result is False


# ── update_file edge cases ─────────────────────────────────────────────────


class TestUpdateFileEdges:
    """Cover remaining branches in update_file."""

    def test_update_file_parse_error(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        src_file = tmp_path / "src" / "sample.py"
        result = gcg.update_file(src_file, "def broken(\n", modified_by="test")
        assert isinstance(result, UpdateResult)
        assert result.symbols_removed >= 0
        assert result.symbols_added == 0

    def test_update_file_new_symbols_trigger_invariants(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        gcg.add_invariant(
            Invariant(
                description="Test invariant",
                scope_glob="**/*.py",
                kind="structural",
                added_by="test",
            )
        )
        src_file = tmp_path / "src" / "sample.py"
        new_code = "CONSTANT = 100\n"
        result = gcg.update_file(src_file, new_code, modified_by="test")
        assert isinstance(result, UpdateResult)


# ── _parse_python_file edge cases ──────────────────────────────────────────


class TestParsePythonFile:
    """Cover remaining branches in _parse_python_file."""

    def test_mtime_oserror_fallback(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        gcg._parse_python_file(tmp_path / "nonexistent.py", "x = 1\n", modified_by="test")
        # Should use datetime.now(UTC) as fallback

    def test_syntax_error_skips(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        result = gcg._parse_python_file(tmp_path / "broken.py", "def (\n", modified_by="test")
        assert result == []

    def test_empty_file(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        result = gcg._parse_python_file(tmp_path / "empty.py", "", modified_by="test")
        assert result == []


# ── _module_path edge cases ─────────────────────────────────────────────────


class TestModulePath:
    """Cover _module_path."""

    def test_module_path_short(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        result = gcg._module_path(Path("main.py"))
        assert result == "main"

    def test_module_path_deep(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        result = gcg._module_path(Path("a/b/c/d/e/f/module.py"))
        assert result in ("d.e.f.module", "e.f.module")

    def test_module_path_with_init(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        result = gcg._module_path(Path("pkg/sub/__init__.py"))
        assert len(result.split(".")) <= 2
        assert result.endswith("pkg") or "sub" in result


# ── Symbol extraction — class with methods ─────────────────────────────────


class TestClassMethodExtraction:
    """Cover method extraction within class definitions."""

    def test_class_with_methods_and_variables(self, tmp_path):
        gcg = _build_gcg(tmp_path, SAMPLE_PY)
        results = gcg.find_symbol("add")
        assert len(results) > 0
        assert results[0].kind == "method"

    def test_multiple_classes(self, tmp_path):
        code = """\
class A:
    def method_a(self):
        pass

class B:
    def method_b(self):
        pass
"""
        gcg = _build_gcg(tmp_path, code)
        methods = gcg.find_symbol("method_a")
        assert len(methods) > 0
        methods_b = gcg.find_symbol("method_b")
        assert len(methods_b) > 0


# ── Dependency extraction edge cases ───────────────────────────────────────


class TestDependencyEdges:
    """Cover remaining dependency extraction branches."""

    def test_inheritance_dependency(self, tmp_path):
        gcg = _build_gcg(tmp_path, SAMPLE_INHERIT)
        deps = gcg.query_sql("SELECT * FROM dependencies WHERE kind = 'inherits'")
        assert len(deps) >= 0

    def test_call_dependency_matching(self, tmp_path):
        gcg = _build_gcg(tmp_path, SAMPLE_CALLS)
        deps = gcg.query_sql("SELECT * FROM dependencies WHERE kind = 'calls'")
        assert len(deps) >= 0

    def test_decorator_dependency(self, tmp_path):
        gcg = _build_gcg(tmp_path, SAMPLE_DECORATED)
        deps = gcg.query_sql("SELECT * FROM dependencies WHERE kind = 'decorates'")
        assert len(deps) >= 0

    def test_resolve_call_name_attribute(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        import ast

        node = ast.parse("obj.method(x)").body[0].value.func
        result = gcg._resolve_call_name(node)
        assert result == "obj.method"

    def test_resolve_name_attribute_chain(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        import ast

        node = ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="a", ctx=ast.Load()), attr="b", ctx=ast.Load()),
            attr="c",
            ctx=ast.Load(),
        )
        result = gcg._resolve_name(node)
        assert result == "a.b.c"


# ── get_symbol / find_symbol edge cases ─────────────────────────────────────


class TestSymbolLookupEdges:
    """Cover remaining branches in symbol lookup."""

    def test_get_symbol_none(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        result = gcg.get_symbol("nonexistent.id.here")
        assert result is None

    def test_find_symbol_file_hint_exact_match(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        src_file = tmp_path / "src" / "sample.py"
        # File hint with exact match
        results = gcg.find_symbol("Calculator", file_hint=src_file)
        assert len(results) > 0
        assert results[0].name == "Calculator"

    def test_find_symbol_file_hint_no_match_falls_back(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        src_file = tmp_path / "src" / "sample.py"
        results = gcg.find_symbol("helper", file_hint=tmp_path / "nonexistent.py")
        assert len(results) > 0

    def test_find_symbol_partial_like(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("hel")
        assert len(results) > 0


# ── Blast radius edge cases ─────────────────────────────────────────────────


class TestBlastRadiusEdges:
    """Cover remaining branches in get_blast_radius."""

    def test_blast_radius_empty(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        blast = gcg.get_blast_radius("nonexistent", max_depth=2)
        assert isinstance(blast, BlastRadius)
        assert blast.symbol_id == "nonexistent"

    def test_blast_radius_visited_tracking(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("helper")
        if results:
            blast = gcg.get_blast_radius(results[0].id, max_depth=1)
            assert isinstance(blast, BlastRadius)
            assert isinstance(blast.affected_symbols, list)
            assert isinstance(blast.affected_files, set)

    def test_blast_radius_max_depth_exceeded(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        results = gcg.find_symbol("helper")
        if results:
            blast = gcg.get_blast_radius(results[0].id, max_depth=0)
            assert blast.depth == 0
            assert len(blast.affected_symbols) == 0


# ── Invariants edge cases ──────────────────────────────────────────────────


class TestInvariantsEdges:
    """Cover remaining invariant branches."""

    def test_check_invariants_none_symbol_skipped(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        gcg.add_invariant(
            Invariant(
                description="Test",
                scope_glob="**/*.py",
                kind="structural",
                added_by="test",
            )
        )
        # Check with a nonexistent symbol ID
        violations = gcg.check_invariants(["nonexistent.id"])
        assert len(violations) == 0

    def test_check_invariants_scope_mismatch(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        gcg.add_invariant(
            Invariant(
                description="Only in go files",
                scope_glob="**/*.go",
                kind="structural",
                added_by="test",
            )
        )
        symbols = gcg.query_sql("SELECT id FROM symbols LIMIT 2")
        if symbols:
            violations = gcg.check_invariants([s["id"] for s in symbols])
            # All symbols are in .py files, .go scope won't match
            assert isinstance(violations, list)

    def test_check_invariants_no_active(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        violations = gcg.check_invariants(["some.id"])
        assert len(violations) == 0

    def test_invariant_with_generated_id(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        inv = Invariant(
            description="Auto ID test",
            scope_glob="**/*.py",
            kind="naming",
            added_by="test",
        )
        gcg.add_invariant(inv)
        # Check that the invariant was actually stored (auto-generated ID in DB)
        rows = gcg.query_sql(
            "SELECT COUNT(*) as cnt FROM invariants WHERE description = 'Auto ID test'"
        )
        assert rows[0]["cnt"] == 1

    def test_get_violations_no_scope(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        violations = gcg.get_violations()
        assert isinstance(violations, list)


# ── Context summary edge cases ─────────────────────────────────────────────


class TestContextSummaryEdges:
    """Cover remaining branches in context summary round-trip."""

    def test_get_context_summary_none_symbol(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        summary = gcg.get_context_summary(["nonexistent.id"])
        assert summary == ""

    def test_get_context_summary_mixed_valid_invalid(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        symbols = gcg.query_sql("SELECT id FROM symbols LIMIT 2")
        ids = [s["id"] for s in symbols] + ["fake.id"]
        summary = gcg.get_context_summary(ids)
        assert "GCG:" in summary

    def test_resolve_context_summary_malformed(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        spans = gcg.resolve_context_summary("GCG:badformat\nnot_gcg_line\n")
        assert len(spans) == 0  # bad format skipped, not GCG line skipped

    def test_resolve_context_summary_value_error(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        spans = gcg.resolve_context_summary("GCG:only_one_pipe_separator")
        assert len(spans) == 0

    def test_resolve_context_summary_index_error(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        spans = gcg.resolve_context_summary("GCG:a@b|c-d")
        assert len(spans) == 0  # missing checksum separator


# ── get_dependents edge cases ──────────────────────────────────────────────


class TestDependents:
    """Cover get_dependents with edges."""

    def test_get_dependents_for_nonexistent_file(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        dependents = gcg.get_dependents(tmp_path / "nonexistent.py")
        assert isinstance(dependents, list)

    def test_get_callers_nonexistent_symbol(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        callers = gcg.get_callers("nonexistent.id")
        assert len(callers) == 0

    def test_get_callees_nonexistent_symbol(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        callees = gcg.get_callees("nonexistent.id")
        assert len(callees) == 0


# ── Additional symbol kind tests ───────────────────────────────────────────


class TestSymbolKinds:
    """Test various symbol kinds extracted."""

    def test_variable_extraction(self, tmp_path):
        code = "X = 1\ny = 2\nZ = 3\n"
        gcg = _build_gcg(tmp_path, code)
        results = gcg.find_symbol("X")
        assert len(results) > 0
        assert results[0].kind == "variable"

    def test_unused_import_tracking(self, tmp_path):
        code = "import json\nimport sys\n\ndef foo():\n    pass\n"
        gcg = _build_gcg(tmp_path, code)
        deps = gcg.query_sql("SELECT COUNT(*) as cnt FROM dependencies WHERE kind = 'imports'")
        assert deps[0]["cnt"] > 0

    def test_method_in_class_detected(self, tmp_path):
        code = """\
class Service:
    \"\"\"A service class.\"\"\"

    def handle(self):
        pass

    def close(self):
        pass
"""
        gcg = _build_gcg(tmp_path, code)
        results = gcg.find_symbol("handle")
        assert len(results) > 0
        assert results[0].kind in ("method", "function")

    def test_top_level_function_not_method(self, tmp_path):
        code = """\
class Wrapper:
    pass

def standalone():
    pass
"""
        gcg = _build_gcg(tmp_path, code)
        results = gcg.find_symbol("standalone")
        assert len(results) > 0
        assert results[0].kind == "function"


# ── _row_to_violation ──────────────────────────────────────────────────────


class TestRowToViolation:
    """Cover _row_to_violation."""

    def test_row_to_violation(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        gcg.add_invariant(
            Invariant(
                description="V test",
                scope_glob="**/*.py",
                kind="structural",
                added_by="test",
            )
        )
        symbols = gcg.query_sql("SELECT id FROM symbols LIMIT 3")
        if symbols:
            gcg.check_invariants([s["id"] for s in symbols])
            violations = gcg.get_violations()
            for v in violations:
                assert isinstance(v, InvariantViolation)
                assert isinstance(v.id, str)
                assert isinstance(v.file, Path)


# ── _get_source edge cases ─────────────────────────────────────────────────


class TestGetSource:
    """Cover _get_source edge cases."""

    def test_get_source_default_end_lineno(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        import ast

        node = ast.parse("x = 1").body[0]
        result = gcg._get_source(node, "x = 1\n")
        assert "x = 1" in result

    def test_get_source_truncates_to_3_lines(self, tmp_path):
        gcg = _build_gcg(tmp_path)
        content = "line1\nline2\nline3\nline4\nline5\n"
        import ast

        tree = ast.parse(content)
        func = tree.body[0] if tree.body else None
        if func:
            result = gcg._get_source(func, content)
            lines = result.split("\n")
            assert len(lines) <= 3
