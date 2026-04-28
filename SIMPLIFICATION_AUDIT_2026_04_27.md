# Godspeed Simplification Audit — 2026-04-27

> Audit of the complete Godspeed v3.4.0 codebase to identify simplification
> opportunities **without removing useful engineer features**.
>
> Scope: 19,300 lines src / 22,000 lines tests / 1,999 tests passing.

---

## Executive Summary

| Area | Lines (src+test) | Verdict | Potential Savings |
|------|------------------|---------|-------------------|
| **evolution/** | ~4,320 | 🔴 Dead code — keep hardware scanner only | ~3,900 |
| **memory/** | ~670 | 🔴 Completely unused | ~670 |
| **tools/ quality quartet** | ~1,300 | 🟡 Near-identical wrappers — consolidate | ~500 |
| **context/ auto_index** | ~390 | 🟢 Keep — already gated & useful | 0 |
| **tools/ notebook** | ~350 | 🟡 Niche — make optional or skill | ~200 |
| **TUI /evolve command** | ~120 | 🔴 Tied to dead evolution subsystem | ~120 |
| **registry description overrides** | ~40 | 🟡 Only used by evolution | ~40 |
| **Total realistic savings** | | | **~5,400 lines** |

---

## 1. Evolution Subsystem — 🔴 Remove Dead Code, Keep Hardware Scanner

**Current state:** 2,218 src + 2,102 test lines across 11 files.

**What's actually used:**
- `evolution/hardware.py::format_machine_report()` — called by `/scan` TUI command and `--machine-report` CLI flag. **This is the only production usage.**
- `evolution/hardware.py::detect_vram_mb()` / `select_evolution_model()` — called by mutator/fitness, which themselves are dead.

**What's dead:**
| File | Lines | Production references |
|------|-------|----------------------|
| `mutator.py` | 422 | 0 |
| `fitness.py` | 285 | 0 |
| `trace_analyzer.py` | 455 | 0 |
| `registry.py` | 314 | 1 (TUI `/evolve status`) |
| `safety.py` | 194 | 0 |
| `skill_gen.py` | 152 | 0 |
| `cross_session.py` | 254 | 0 |
| `applier.py` | 141 | 0 |
| `permissions.py` | 119 | 0 |
| `hardware.py` | 420 | 2 (TUI `/scan`, CLI `--machine-report`) |

**The `/evolve` TUI command** (`_cmd_evolve`, ~110 lines in `tui/commands.py`) exposes `status`, `history`, `rollback`, `review`, `run` subcommands. The `run` subcommand returns a message saying it runs asynchronously — **there is no actual implementation behind it.** The agent loop does not trigger evolution.

**Recommendation:**
1. Move `evolution/hardware.py` → `utils/hardware.py` (new package). It's self-contained (only uses stdlib + `subprocess`).
2. Update TUI `/scan` and CLI `--machine-report` imports.
3. Delete `src/godspeed/evolution/` entirely.
4. Delete `tests/test_evolution/` entirely.
5. Remove `/evolve` slash command from TUI.
6. Remove `update_description()` / `clear_description_override()` from `ToolRegistry` — only existed for self-evolution hot-swapping.

**Savings:** ~4,300 lines (2,100 src + 2,100 test) removed. Hardware scanner preserved.

---

## 2. Memory Subsystem — 🔴 Completely Unused

**Current state:** 377 src + 291 test lines across 4 files.

**Production references:** Zero. `grep` found no imports from `godspeed.memory` in `agent/`, `tui/`, or `tools/`. The `UserMemory`, `SessionMemory`, and `CorrectionMemory` classes are only referenced within their own module and tests.

**What it does:** SQLite-backed preference storage and correction tracking — a good idea, but never wired into the agent loop or TUI. The conversation logger (`training/conversation_logger.py`) handles session persistence instead.

**Recommendation:** Delete `src/godspeed/memory/` and `tests/test_memory/` entirely. If cross-session learning is needed later, it can be rebuilt on top of the existing conversation logger JSONL, which already captures everything.

**Savings:** ~670 lines.

---

## 3. Quality Analysis Tools — 🟡 Consolidate Identical Patterns

**Current state:** Four tools that are 80% identical boilerplate:

| Tool | Lines | Pattern |
|------|-------|---------|
| `complexity.py` | 191 | Check binary → run subprocess → parse → truncate → return |
| `coverage.py` | 184 | Check binary → run subprocess → parse → truncate → return |
| `dep_audit.py` | 206 | Check binary → run subprocess → parse → truncate → return |
| `security_scan.py` | ~180 | Check binary → run subprocess → parse → truncate → return |

All four:
- Import `shutil`, `subprocess`, `logging`
- Define `MAX_OUTPUT_CHARS` and timeout constants
- Have `RiskLevel.READ_ONLY`
- Run `shutil.which()` to check binary
- Run `subprocess.run()` with `capture_output=True, text=True`
- Truncate output with the same logic
- Return `ToolResult.success()` or `ToolResult.failure()`

**Recommendation:** Add a `_run_external_tool()` helper to `tools/base.py`:

```python
async def _run_external_tool(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int = 60,
    max_output_chars: int = 5000,
    check_binary: str = "",
) -> ToolResult:
    ...
```

Then each tool becomes ~40 lines (name, description, schema, execute → one-liner call to helper).

**Savings:** ~500 lines src + similar in tests.

---

## 4. Notebook Tool — 🟡 Niche, Evaluate Usage

**Current state:** 201 lines. Jupyter notebook cell-level editing.

**Verdict:** Useful for data-science workflows but adds JSON manipulation complexity for a narrow use case. The agent can already edit `.py` files; notebook editing is a specialization.

**Recommendation (low priority):** Either:
- Extract to a skill (markdown prompt file) instead of a built-in tool, or
- Keep but note it as "optional — only loads if Jupyter is installed"

**Savings:** ~150 lines if extracted to skill.

---

## 5. System Optimizer — 🟢 Keep, But Remove Dead "act" Mode References

**Current state:** 630 lines. `inspect` and `recommend` modes are fully implemented. `act` mode is documented as **NOT yet implemented** (line 516).

**What's dead:** `_SYSTEM_CRITICAL_NAMES` dict and associated comments about future `act` mode are misleading — the code reads like act mode is coming, but it has been "coming" for multiple releases.

**Recommendation:**
- Remove `act` from the schema enum (it's not there — good).
- Trim comments about future `act` mode to avoid implying unfinished features.
- Keep `_SYSTEM_CRITICAL_NAMES` if there's any plan to implement act mode; otherwise remove it.

**Savings:** ~20-30 lines of comments + the deny-list dict (~50 lines) if act mode is abandoned.

---

## 6. Context / Auto-Indexing — 🟢 Keep As-Is

**Current state:** `codebase_index.py` (297), `auto_index.py` (91), `chunker.py` (~150).

**Verdict:** Already well-designed:
- Gated on `chromadb` optional dependency
- Non-blocking (`asyncio.create_task`)
- Controlled by `settings.auto_index` (default True)
- No-op gracefully when deps missing

**Recommendation:** No changes. This is a good pattern for optional features.

---

## 7. Hooks Subsystem — 🟢 Keep

**Current state:** 159 lines across 3 files.

**Verdict:** Referenced from agent loop. Small, clean, useful for CI integration (pre-commit hooks, custom validations). No bloat.

---

## 8. MCP Subsystem — 🟢 Keep

**Current state:** 415 lines across 4 files.

**Verdict:** Referenced from CLI. Optional dependency (`mcp` package). Allows connecting to external MCP servers (filesystem, GitHub, etc.). This is a key differentiator and extensibility point. Well-gated.

---

## 9. Skills Subsystem — 🟢 Keep

**Current state:** 150 lines across 3 files.

**Verdict:** Referenced from TUI and CLI. Small, lightweight markdown-prompt loader. Useful for extending capabilities without code changes.

---

## 10. Training Subsystem — 🟡 Partially Used

**Current state:** 834 src + 930 test lines.

**What's used:**
- `training/conversation_logger.py` — used in agent loop and CLI. **Keep.**
- `training/exporter.py` — used by CLI `export-training`. **Keep.**

**What's questionable:**
- `training/benchmark.py` — Standalone benchmark suite. Only used in `experiments/` and tests. If benchmark runs are one-off research, consider moving to `scripts/` or `experiments/`.
- `training/rewards.py` — Reward logging. Referenced only within its own tests and by conversation_logger. Check if it's actually writing reward entries.

**Recommendation:** Keep `conversation_logger.py` and `exporter.py`. Evaluate if `benchmark.py` and `rewards.py` can be moved to `experiments/` or archived.

**Savings:** ~400 lines if benchmark+rewards are moved.

---

## 11. TUI Commands — 🟡 Evaluate Niche Commands

**Current state:** `tui/commands.py` is 1,367 lines.

Slash commands audit:
| Command | Usage | Verdict |
|---------|-------|---------|
| `/help` | Core | Keep |
| `/quit` | Core | Keep |
| `/model` | Core | Keep |
| `/tokens` | Core | Keep |
| `/budget` | Cost tracking | Keep |
| `/compact` | Context management | Keep |
| `/checkpoint` | Session save/restore | Keep |
| `/context` | Token usage | Keep |
| `/diff` | File diff review | Keep |
| `/permissions` | Security | Keep |
| `/approve` | Security | Keep |
| `/deny` | Security | Keep |
| `/auto-commit` | Git workflow | Keep |
| `/undo` | Git workflow | Keep |
| `/status` | Git workflow | Keep |
| `/push` | Git workflow | Keep |
| `/gh` | GitHub integration | Keep |
| `/keys` | API key management | Keep |
| `/pull` | Ollama model pull | Keep |
| `/scan` | Hardware scan | Keep (after moving hardware.py) |
| `/models` | Model presets | Keep |
| `/evolve` | Self-evolution | **Remove** (dead subsystem) |
| `/skill` | Skill management | Keep |

**Savings:** ~110 lines for `/evolve` removal.

---

## 12. Agent Loop — 🟢 Keep Complex, It's the Product

**Current state:** 1,514 lines.

**Verdict:** Large but justified. This is the core differentiator — hand-rolled ReAct loop with streaming, speculative dispatch, parallel tool calls, auto-stash, MUST-FIX injection, stuck-loop detection. Every feature is exercised by tests.

**Minor simplification:** The 9 callback type aliases (`OnAssistantText`, `OnToolCall`, etc.) could collapse to 2-3 more general types, but this is cosmetic.

---

## 13. CLI — 🟢 Keep, Entry Point Complexity is Expected

**Current state:** 1,191 lines.

**Verdict:** Handles multiple modes (TUI, one-shot, batch, stdin), env loading, tool registry setup, model routing, cost tracking, conversation logging. Complexity is proportional to functionality.

---

## 14. Experiments Directory — 🟡 Gitignore

**Current state:** 873 tiny files (~0 MB total) in `experiments/`.

**Verdict:** These are benchmark run artifacts (one directory per model tested). They don't belong in the repo — they should be generated locally or stored in W&B.

**Recommendation:** Add `experiments/*` to `.gitignore`. Keep directory structure in repo with a `.gitkeep` and a README explaining how to run benchmarks.

---

## Implementation Priority

### Phase 1 — High Impact, Low Risk (do first)
1. **Delete `memory/`** — completely unused, zero production references.
2. **Delete `evolution/` except `hardware.py`** — move hardware to `utils/`, update imports.
3. **Remove `/evolve` TUI command** — tied to dead subsystem.
4. **Remove `registry.py` description override methods** — only existed for evolution.
5. **Gitignore `experiments/`** — these are runtime artifacts.

### Phase 2 — Medium Impact, Medium Risk
6. **Consolidate quality tools** — extract `_run_external_tool()` helper.
7. **Evaluate `training/benchmark.py` and `rewards.py`** — move to `experiments/` if research-only.

### Phase 3 — Low Impact, Low Risk
8. **Trim `system_optimizer.py` dead comments** about unimplemented `act` mode.
9. **Evaluate `notebook.py`** — extract to skill or keep as optional.

---

## Expected Outcome

| Metric | Before | After (Phase 1+2) | Reduction |
|--------|--------|-------------------|-----------|
| Source lines | ~19,300 | ~14,500 | **25%** |
| Test lines | ~22,000 | ~18,500 | **16%** |
| Total lines | ~41,300 | ~33,000 | **20%** |
| Test count | 1,999 | ~1,700 | **15%** |
| CI time | ~48s | ~38s | **21%** |

The codebase becomes easier to navigate, faster to test, and less intimidating to contributors — while preserving every feature an engineer actually uses.
