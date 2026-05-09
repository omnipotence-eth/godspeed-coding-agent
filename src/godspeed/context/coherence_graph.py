"""Global Coherence Graph — lossless persistent symbol-level dependency graph.

Stores: symbols (functions, classes, variables), their dependencies (calls,
imports, inherits, uses_type), and architectural invariants. Updated in
real-time on every file write. Queried by the retrieval subagent instead of
grep-searching raw files. Survives compaction (context summaries reference
GCG node IDs, not content).

Solves the paradox of supervision: global state lives here, not in the
context window.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SymbolKind = Literal["function", "class", "variable", "import", "method", "module"]
DependencyKind = Literal["calls", "imports", "inherits", "overrides", "uses_type", "decorates"]

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    id: str
    kind: SymbolKind
    file: Path
    start_line: int
    end_line: int
    name: str
    qualified_name: str
    signature: str
    docstring: str | None
    last_modified: datetime
    last_modified_by: str
    checksum: str


@dataclass
class Dependency:
    id: str
    from_symbol_id: str
    to_symbol_id: str
    kind: DependencyKind
    file: Path
    line: int


@dataclass
class Invariant:
    id: str = ""
    description: str = ""
    scope_glob: str = ""
    kind: Literal["structural", "behavioral", "naming", "coverage"] = "structural"
    added_by: str = ""
    added_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    active: bool = True


@dataclass
class InvariantViolation:
    id: str = ""
    invariant_id: str = ""
    symbol_id: str = ""
    file: Path = Path()
    line: int = 0
    description: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved: bool = False


@dataclass
class BuildResult:
    symbol_count: int = 0
    dependency_count: int = 0
    files_parsed: int = 0
    duration_ms: float = 0.0


@dataclass
class UpdateResult:
    symbols_added: int = 0
    symbols_removed: int = 0
    dependencies_added: int = 0
    dependencies_removed: int = 0
    violations: list[InvariantViolation] = field(default_factory=list)


@dataclass
class BlastRadius:
    symbol_id: str = ""
    affected_symbols: list[Symbol] = field(default_factory=list)
    affected_files: set[str] = field(default_factory=set)
    depth: int = 0


@dataclass
class FileSpan:
    file: Path
    start_line: int
    end_line: int
    symbol_id: str = ""


class CoherenceGraph:
    """SQLite-backed symbol graph.

    All methods are synchronous; wrap in asyncio.to_thread() when called
    from async agent loop.

    Args:
        db_path: Path to the SQLite database file.
    """

    DB_VERSION = 1

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._apply_schema()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _apply_schema(self) -> None:
        conn = self.conn
        schema_path = Path(__file__).parent / "coherence_graph_schema.sql"
        if schema_path.exists():
            schema = schema_path.read_text()
            conn.executescript(schema)
        conn.execute(
            "INSERT OR REPLACE INTO gcg_meta VALUES ('db_version', ?)",
            (str(self.DB_VERSION),),
        )
        conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("CoherenceGraph not connected — call connect() first")
        return self._conn

    # ── Build & update ────────────────────────────────────────────────────

    def build_from_repo(
        self,
        repo_root: Path,
        languages: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        incremental: bool = True,
    ) -> BuildResult:
        t0 = datetime.now(UTC)
        exclude = set(exclude_patterns or [])
        exclude.update({".git", "__pycache__", ".venv", "venv", "node_modules", ".tox"})
        languages = languages or ["py"]

        ext_map = {".py": "py"}
        target_exts = {
            ext for lang in languages for ext, ext_lang in ext_map.items() if ext_lang == lang
        }

        files_to_parse: list[Path] = []
        for ext in target_exts:
            for fpath in repo_root.rglob(f"*{ext}"):
                parts = set(fpath.parts)
                if parts & exclude:
                    continue
                if incremental and self._file_up_to_date(fpath):
                    continue
                files_to_parse.append(fpath)

        count = len(files_to_parse)
        for i, fpath in enumerate(files_to_parse):
            if i % 100 == 0 and count > 100:
                logger.debug("Parsing %d/%d: %s", i + 1, count, fpath)
            try:
                content = fpath.read_text(encoding="utf-8")
                self._parse_python_file(fpath, content, modified_by="initial_scan")
            except Exception as exc:
                logger.debug("Parse error %s: %s", fpath, exc)

        self.conn.commit()

        sym_count = self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        dep_count = self.conn.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
        duration_ms = (datetime.now(UTC) - t0).total_seconds() * 1000

        logger.info(
            "GCG build complete symbols=%d deps=%d files=%d duration_ms=%.0f",
            sym_count,
            dep_count,
            count,
            duration_ms,
        )
        return BuildResult(
            symbol_count=sym_count,
            dependency_count=dep_count,
            files_parsed=count,
            duration_ms=duration_ms,
        )

    def _file_up_to_date(self, file_path: Path) -> bool:
        try:
            mtime_str = datetime.fromtimestamp(file_path.stat().st_mtime, UTC).isoformat()
        except OSError:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM symbols WHERE file = ? AND last_modified = ? LIMIT 1",
            (str(file_path), mtime_str),
        ).fetchone()
        return row is not None

    def update_file(self, file_path: Path, new_content: str, modified_by: str) -> UpdateResult:
        old_symbols = [
            row[0]
            for row in self.conn.execute(
                "SELECT id FROM symbols WHERE file = ?", (str(file_path),)
            ).fetchall()
        ]

        # Remove old entries for this file
        self.conn.execute("DELETE FROM symbols WHERE file = ?", (str(file_path),))
        self.conn.execute("DELETE FROM dependencies WHERE file = ?", (str(file_path),))

        symbols_removed = len(old_symbols)
        try:
            symbols = self._parse_python_file(file_path, new_content, modified_by=modified_by)
        except Exception as exc:
            logger.warning("Parse error on update %s: %s", file_path, exc)
            self.conn.commit()
            return UpdateResult(symbols_removed=symbols_removed)

        self.conn.commit()

        violations = self.check_invariants([s.id for s in symbols])
        return UpdateResult(
            symbols_added=len(symbols),
            symbols_removed=symbols_removed,
            dependencies_added=len([s for s in symbols]),
            violations=violations,
        )

    # ── Python AST parsing ─────────────────────────────────────────────────

    def _parse_python_file(
        self, file_path: Path, content: str, modified_by: str = "initial_scan"
    ) -> list[Symbol]:
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, UTC)
        except OSError:
            mtime = datetime.now(UTC)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.debug("Syntax error in %s, skipping", file_path)
            return []

        module_name = self._module_path(file_path)
        symbols: list[Symbol] = []

        for node in ast.walk(tree):
            sym = self._extract_symbol(node, file_path, module_name, content, mtime, modified_by)
            if sym:
                symbols.append(sym)
                self._insert_symbol(sym)

        for node in ast.walk(tree):
            deps = self._extract_dependencies(node, file_path, symbols, module_name)
            for dep in deps:
                self._insert_dependency(dep)

        return symbols

    def _module_path(self, file_path: Path) -> str:
        parts = list(file_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        # Convert to dotted path, stripping leading directories if needed
        return ".".join(parts[-3:]) if len(parts) > 3 else ".".join(parts)

    def _extract_symbol(
        self,
        node: ast.AST,
        file_path: Path,
        module_name: str,
        content: str,
        mtime: datetime,
        modified_by: str,
    ) -> Symbol | None:
        if isinstance(node, ast.FunctionDef):
            kind: SymbolKind = "method" if self._is_method(node, content) else "function"
            name = node.name
            qual = f"{module_name}.{name}"
            sig = self._get_source(node, content)
            doc = ast.get_docstring(node)
            start = node.lineno
            end = node.end_lineno or start
            chk = hashlib.sha256(sig.encode()).hexdigest()
            return Symbol(
                id=f"{module_name}.{name}",
                kind=kind,
                file=file_path,
                start_line=start,
                end_line=end,
                name=name,
                qualified_name=qual,
                signature=sig,
                docstring=doc,
                last_modified=mtime,
                last_modified_by=modified_by,
                checksum=chk,
            )

        if isinstance(node, ast.ClassDef):
            name = node.name
            qual = f"{module_name}.{name}"
            doc = ast.get_docstring(node)
            start = node.lineno
            end = node.end_lineno or start
            sig = self._get_source(node, content)
            chk = hashlib.sha256(sig.encode()).hexdigest()
            sym = Symbol(
                id=qual,
                kind="class",
                file=file_path,
                start_line=start,
                end_line=end,
                name=name,
                qualified_name=qual,
                signature=sig,
                docstring=doc,
                last_modified=mtime,
                last_modified_by=modified_by,
                checksum=chk,
            )
            # Extract methods as child symbols
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    child_sym = self._extract_symbol(
                        child, file_path, qual, content, mtime, modified_by
                    )
                    if child_sym:
                        self._insert_symbol(child_sym)
            return sym

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    qual = f"{module_name}.{name}"
                    start = node.lineno
                    end = node.end_lineno or start
                    sig = self._get_source(node, content)
                    chk = hashlib.sha256(sig.encode()).hexdigest()
                    return Symbol(
                        id=qual,
                        kind="variable",
                        file=file_path,
                        start_line=start,
                        end_line=end,
                        name=name,
                        qualified_name=qual,
                        signature=sig,
                        docstring=None,
                        last_modified=mtime,
                        last_modified_by=modified_by,
                        checksum=chk,
                    )
        return None

    def _is_method(self, node: ast.FunctionDef, content: str) -> bool:
        for i in range(node.lineno - 2, 0, -1):
            line = content.split("\n")[i] if i < len(content.split("\n")) else ""
            if "class " in line and not line.strip().startswith("#"):
                # Verify indent: class is less indented than method
                cls_indent = len(line) - len(line.lstrip())
                fn_indent = node.col_offset
                if cls_indent < fn_indent:
                    return True
                return i < node.lineno - 1
        return False

    def _get_source(self, node: ast.stmt | ast.expr, content: str) -> str:
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        lines = content.split("\n")[start:end]
        return "\n".join(lines[:3]).strip()

    def _extract_dependencies(
        self,
        node: ast.AST,
        file_path: Path,
        symbols: list[Symbol],
        module_name: str,
    ) -> list[Dependency]:
        deps: list[Dependency] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                dep = Dependency(
                    id=str(uuid.uuid4()),
                    from_symbol_id=f"{module_name}._import",
                    to_symbol_id=f"{alias.name}.__init__",
                    kind="imports",
                    file=file_path,
                    line=node.lineno,
                )
                deps.append(dep)

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                target = f"{module}.{alias.name}"
                dep = Dependency(
                    id=str(uuid.uuid4()),
                    from_symbol_id=f"{module_name}._import",
                    to_symbol_id=target,
                    kind="imports",
                    file=file_path,
                    line=node.lineno,
                )
                deps.append(dep)

        if isinstance(node, ast.Call):
            func_name = self._resolve_call_name(node.func) if hasattr(node, "func") else ""
            if func_name:
                for sym in symbols:
                    if sym.name == func_name or sym.qualified_name.endswith(f".{func_name}"):
                        dep = Dependency(
                            id=str(uuid.uuid4()),
                            from_symbol_id=f"{module_name}.{func_name}",
                            to_symbol_id=sym.id,
                            kind="calls",
                            file=file_path,
                            line=node.lineno,
                        )
                        deps.append(dep)

        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = self._resolve_name(base)
                if base_name:
                    dep = Dependency(
                        id=str(uuid.uuid4()),
                        from_symbol_id=f"{module_name}.{node.name}",
                        to_symbol_id=base_name,
                        kind="inherits",
                        file=file_path,
                        line=node.lineno,
                    )
                    deps.append(dep)

        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                dec_name = self._resolve_name(decorator)
                if dec_name:
                    dep = Dependency(
                        id=str(uuid.uuid4()),
                        from_symbol_id=f"{module_name}.{node.name}",
                        to_symbol_id=dec_name,
                        kind="decorates",
                        file=file_path,
                        line=node.lineno,
                    )
                    deps.append(dep)

        return deps

    def _resolve_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, (ast.Name, ast.Attribute)):
                prefix = self._resolve_name(node.value)
                return f"{prefix}.{node.attr}"
            return node.attr
        return ""

    def _resolve_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._resolve_name(node.value)}.{node.attr}"
        return ""

    # ── Insert helpers ─────────────────────────────────────────────────────

    def _insert_symbol(self, sym: Symbol) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO symbols
               (id, kind, file, start_line, end_line, name,
                qualified_name, signature, docstring, last_modified,
                last_modified_by, checksum)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sym.id,
                sym.kind,
                str(sym.file),
                sym.start_line,
                sym.end_line,
                sym.name,
                sym.qualified_name,
                sym.signature,
                sym.docstring,
                sym.last_modified.isoformat(),
                sym.last_modified_by,
                sym.checksum,
            ),
        )

    def _insert_dependency(self, dep: Dependency) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO dependencies
               (id, from_symbol_id, to_symbol_id, kind, file, line)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (dep.id, dep.from_symbol_id, dep.to_symbol_id, dep.kind, str(dep.file), dep.line),
        )

    # ── Query methods ──────────────────────────────────────────────────────

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        row = self.conn.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_symbol(row)

    def find_symbol(self, name: str, file_hint: Path | None = None) -> list[Symbol]:
        if file_hint:
            rows = self.conn.execute(
                "SELECT * FROM symbols WHERE name = ? AND file = ? LIMIT 20",
                (name, str(file_hint)),
            ).fetchall()
            if rows:
                return [self._row_to_symbol(r) for r in rows]

        # Prioritize exact name matches first
        exact = self.conn.execute(
            "SELECT * FROM symbols WHERE name = ? LIMIT 20", (name,)
        ).fetchall()

        # Then partial matches via LIKE
        partial = self.conn.execute(
            "SELECT * FROM symbols WHERE (name LIKE ? OR qualified_name LIKE ?)"
            " AND name != ? LIMIT 20",
            (f"%{name}%", f"%{name}%", name),
        ).fetchall()

        rows = exact + partial
        return [self._row_to_symbol(r) for r in rows]

    def get_callers(self, symbol_id: str) -> list[tuple[Symbol, Dependency]]:
        rows = self.conn.execute(
            """SELECT s.*, d.id as dep_id, d.from_symbol_id, d.to_symbol_id,
                      d.kind as dep_kind, d.file as dep_file, d.line as dep_line
               FROM symbols s
               JOIN dependencies d ON s.id = d.from_symbol_id
               WHERE d.to_symbol_id = ? AND d.kind = 'calls'
               LIMIT 50""",
            (symbol_id,),
        ).fetchall()
        return [self._row_to_sym_dep(r) for r in rows]

    def get_callees(self, symbol_id: str) -> list[tuple[Symbol, Dependency]]:
        rows = self.conn.execute(
            """SELECT s.*, d.id as dep_id, d.from_symbol_id, d.to_symbol_id,
                      d.kind as dep_kind, d.file as dep_file, d.line as dep_line
               FROM symbols s
               JOIN dependencies d ON s.id = d.to_symbol_id
               WHERE d.from_symbol_id = ? AND d.kind = 'calls'
               LIMIT 50""",
            (symbol_id,),
        ).fetchall()
        return [self._row_to_sym_dep(r) for r in rows]

    def get_dependents(self, file_path: Path) -> list[Symbol]:
        rows = self.conn.execute(
            """SELECT DISTINCT s.* FROM symbols s
               JOIN dependencies d ON d.from_symbol_id = s.id
               WHERE d.to_symbol_id IN (
                   SELECT id FROM symbols WHERE file = ?
               ) AND s.file != ?
               LIMIT 50""",
            (str(file_path), str(file_path)),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def get_blast_radius(self, symbol_id: str, max_depth: int = 3) -> BlastRadius:
        affected: dict[str, Symbol] = {}
        affected_files: set[str] = set()
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(symbol_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)

            sym = self.get_symbol(current_id)
            if sym and current_id != symbol_id:
                affected[current_id] = sym
                affected_files.add(str(sym.file))

            # Descendants (callers of this symbol)
            for caller_sym, _ in self.get_callers(current_id):
                if caller_sym.id not in visited:
                    queue.append((caller_sym.id, depth + 1))

            # Ancestors (symbols called by this one)
            for callee_sym, _ in self.get_callees(current_id):
                if callee_sym.id not in visited:
                    queue.append((callee_sym.id, depth + 1))

        return BlastRadius(
            symbol_id=symbol_id,
            affected_symbols=list(affected.values()),
            affected_files=affected_files,
            depth=max_depth,
        )

    def query_sql(self, sql: str, params: tuple = ()) -> list[dict]:
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
        rows = self.conn.execute(sql, params).fetchall()
        cols = [desc[0] for desc in self.conn.execute(sql, params).description or []]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    # ── Invariants ─────────────────────────────────────────────────────────

    def add_invariant(self, invariant: Invariant) -> None:
        cols = "id, description, scope_glob, kind, added_by, added_at, active"
        vals = "?, ?, ?, ?, ?, ?, ?"
        self.conn.execute(
            f"INSERT INTO invariants ({cols}) VALUES ({vals})",  # noqa: S608  # nosec
            (
                invariant.id or str(uuid.uuid4()),
                invariant.description,
                invariant.scope_glob,
                invariant.kind,
                invariant.added_by,
                invariant.added_at.isoformat(),
                1 if invariant.active else 0,
            ),
        )
        self.conn.commit()

    def check_invariants(self, changed_symbols: list[str]) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        active = self.conn.execute("SELECT * FROM invariants WHERE active = 1").fetchall()
        for row in active:
            inv_id = row[0]
            inv = Invariant(
                id=inv_id,
                description=row[1],
                scope_glob=row[2],
                kind=row[3],
                added_by=row[4],
                added_at=datetime.fromisoformat(row[5]),
                active=bool(row[6]),
            )
            for sym_id in changed_symbols:
                sym = self.get_symbol(sym_id)
                if sym is None:
                    continue
                import fnmatch

                if not fnmatch.fnmatch(str(sym.file), inv.scope_glob):
                    continue
                v = InvariantViolation(
                    id=str(uuid.uuid4()),
                    invariant_id=inv_id,
                    symbol_id=sym_id,
                    file=sym.file,
                    line=sym.start_line,
                    description=f"Symbol {sym.name} violates: {inv.description}",
                    detected_at=datetime.now(UTC),
                )
                self.conn.execute(
                    """INSERT INTO invariant_violations
                       (id, invariant_id, symbol_id, file, line, description, detected_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        v.id,
                        v.invariant_id,
                        v.symbol_id,
                        str(v.file),
                        v.line,
                        v.description,
                        v.detected_at.isoformat(),
                    ),
                )
                violations.append(v)
        self.conn.commit()
        return violations

    def get_violations(self, scope: str | None = None) -> list[InvariantViolation]:
        if scope:
            rows = self.conn.execute(
                "SELECT * FROM invariant_violations WHERE resolved = 0 AND file LIKE ?",
                (f"%{scope}%",),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM invariant_violations WHERE resolved = 0"
            ).fetchall()
        return [self._row_to_violation(r) for r in rows]

    # ── Context integration ───────────────────────────────────────────────

    def get_context_summary(self, symbol_ids: list[str]) -> str:
        lines: list[str] = []
        for sid in symbol_ids:
            sym = self.get_symbol(sid)
            if sym is None:
                continue
            lines.append(
                f"GCG:{sym.id}@{sym.file}|{sym.start_line}-{sym.end_line}|{sym.checksum[:8]}"
            )
        return "\n".join(lines)

    def resolve_context_summary(self, summary: str) -> list[FileSpan]:
        spans: list[FileSpan] = []
        for line in summary.strip().split("\n"):
            line = line.strip()
            if not line.startswith("GCG:"):
                continue
            try:
                # Format: GCG:sym_id@file|start-end|checksum
                rest = line[4:]
                sym_ref, remaining = rest.split("|", 1)
                span_str, _checksum = remaining.rsplit("|", 1)
                sym_id, file_str = sym_ref.rsplit("@", 1)
                start_str, end_str = span_str.split("-", 1)
                spans.append(
                    FileSpan(
                        file=Path(file_str),
                        start_line=int(start_str),
                        end_line=int(end_str),
                        symbol_id=sym_id,
                    )
                )
            except (ValueError, IndexError):
                continue
        return spans

    def get_graph_stats(self) -> dict:
        sym_count = self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        dep_count = self.conn.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
        file_count = self.conn.execute("SELECT COUNT(DISTINCT file) FROM symbols").fetchone()[0]
        violation_count = self.conn.execute(
            "SELECT COUNT(*) FROM invariant_violations WHERE resolved = 0"
        ).fetchone()[0]
        return {
            "symbols": sym_count,
            "dependencies": dep_count,
            "files": file_count,
            "open_violations": violation_count,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _row_to_symbol(self, row: tuple) -> Symbol:
        return Symbol(
            id=row[0],
            kind=row[1],
            file=Path(row[2]),
            start_line=row[3],
            end_line=row[4],
            name=row[5],
            qualified_name=row[6],
            signature=row[7] or "",
            docstring=row[8],
            last_modified=datetime.fromisoformat(row[9]),
            last_modified_by=row[10],
            checksum=row[11],
        )

    def _row_to_sym_dep(self, row: tuple) -> tuple[Symbol, Dependency]:
        sym = Symbol(
            id=row[0],
            kind=row[1],
            file=Path(row[2]),
            start_line=row[3],
            end_line=row[4],
            name=row[5],
            qualified_name=row[6],
            signature=row[7] or "",
            docstring=row[8],
            last_modified=datetime.fromisoformat(row[9]),
            last_modified_by=row[10],
            checksum=row[11],
        )
        dep = Dependency(
            id=row[12],
            from_symbol_id=row[13],
            to_symbol_id=row[14],
            kind=row[15],
            file=Path(row[16]),
            line=row[17],
        )
        return (sym, dep)

    def _row_to_violation(self, row: tuple) -> InvariantViolation:
        return InvariantViolation(
            id=row[0],
            invariant_id=row[1],
            symbol_id=row[2],
            file=Path(row[3]),
            line=row[4] or 0,
            description=row[5],
            detected_at=datetime.fromisoformat(row[6]),
            resolved=bool(row[7]),
        )
