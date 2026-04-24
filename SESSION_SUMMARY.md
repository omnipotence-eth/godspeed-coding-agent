# Godspeed — last session handoff

This file is **auto-loaded** after `MEMORY.md` in the same instruction merge. It is for **short-lived**, session-to-session continuity: what you were doing last time, open questions, and the next step so a fresh chat can resume without re-explaining the repo.

## At the end of a session, overwrite this with:

1. **Goal** — one sentence: what you were trying to achieve.
2. **Done** — bullets of completed work, commits, or PR links if useful.
3. **Next** — the very next action (command, file, or decision).
4. **Blockers** — optional: errors, missing API keys, or decisions waiting on a human.
5. **Context** — optional: branches, feature flags, or people to ping.

## Example skeleton

```text
**Goal:** Add streaming-safe Rich output in the TUI.

**Done:**
- Fixed `_on_assistant_chunk` to use `markup=False`.
- Pushed to `fix/rich-streaming`.

**Next:**
- Run `pytest tests/test_tui_*.py` and open PR.

**Blockers:** None.
```

## Relationship to `MEMORY.md`

| File | Purpose |
|------|--------|
| `MEMORY.md` | Lasts months — stable team/project facts. |
| `SESSION_SUMMARY.md` | Lasts until the next handoff — “where we left off.” |

When the next task is done, **trim or replace** this file so it does not bloat the context window. Keep `SESSION_SUMMARY.md` to roughly one screen of text if possible.
