CREATE TABLE IF NOT EXISTS gcg_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    file TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    signature TEXT,
    docstring TEXT,
    last_modified TEXT NOT NULL,
    last_modified_by TEXT NOT NULL,
    checksum TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dependencies (
    id TEXT PRIMARY KEY,
    from_symbol_id TEXT NOT NULL,
    to_symbol_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS invariants (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    scope_glob TEXT NOT NULL,
    kind TEXT NOT NULL,
    added_by TEXT NOT NULL,
    added_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS invariant_violations (
    id TEXT PRIMARY KEY,
    invariant_id TEXT NOT NULL REFERENCES invariants(id),
    symbol_id TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    description TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_dependencies_from ON dependencies(from_symbol_id);
CREATE INDEX IF NOT EXISTS idx_dependencies_to ON dependencies(to_symbol_id);
CREATE INDEX IF NOT EXISTS idx_dependencies_file ON dependencies(file);
CREATE INDEX IF NOT EXISTS idx_violations_open ON invariant_violations(resolved)
    WHERE resolved = 0;
