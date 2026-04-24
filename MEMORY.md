# Godspeed — long-term memory

This file is **auto-loaded** into the agent’s system context (see `GODSPEED.md` merge order in `godspeed.context.project_instructions`). Use it for **stable** facts the agent should remember across sessions: architecture decisions, naming conventions, API keys and env layout (not the secrets themselves), repo quirks, and standing preferences.

## How to keep it useful

- **Be concise.** Bullet lists beat essays; remove items that stop being true.
- **Date major entries** when it matters (`2026-04: switched default model to …`).
- **Avoid secrets.** Never store passwords, tokens, or private keys; reference *where* to load them (e.g. `NVIDIA_NIM_API_KEY` in the shell).
- **Conflict resolution:** if this contradicts `GODSPEED.md` or `AGENTS.md`, those files are merged earlier — prefer editing those for authoritative project rules, and use this file for *your* durable notes.

## What to put here (examples)

- Default branch, PR checklist, or CI commands you always want followed.
- “We use Ruff not Black”, “migrations go in `alembic/versions/`”, etc.
- Per-developer or per-machine context that does not belong in the main README.

## What not to put here

- A full design doc — link to it or use `docs/`.
- Transient work-in-progress; use the chat or a scratch file instead, then distill outcomes here if they stay relevant.

---

*Optional:* Point other tools at this file so the same “memory” applies everywhere, or keep a short pointer in `GODSPEED.md` like: “See `MEMORY.md` for personal/project memory.”
