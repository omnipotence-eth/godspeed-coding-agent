# Godspeed Coding Agent — Documentation Audit
**Date:** 2026-04-27
**Scope:** All user-facing and contributor-facing documentation
**Criteria:** Accuracy, completeness, professionalism, consistency

---

## Executive Summary

**Grade: B+**

The documentation is extensive and generally well-written. The README is a strong portfolio piece, the architecture doc is thorough, and the permissions guide is excellent. However, **stale version references, inconsistent tool naming, a broken comparison table, and one clearly accidental file** drag the grade down. Most issues are quick fixes.

---

## Critical Issues (Fix Immediately)

### 1. `docs/fastapi-cors-configuration.md` — Accidental Inclusion

**Issue:** A generic FastAPI CORS tutorial (~230 lines) that has nothing to do with Godspeed. Appears to be an AI-session artifact that was committed by mistake.

**Impact:** Makes the repo look unprofessional. A reviewer will wonder why a coding agent ships a random FastAPI middleware guide.

**Fix:** Delete the file. Add `docs/*.md` to a pre-commit or PR checklist if this happens regularly.

---

### 2. Tool Name Inconsistency: `Bash`/`Shell` vs. `shell`

**Issue:** The actual tool name in code is `shell` (lowercase). The permission engine matches against `shell(command)`. However, documentation and examples use `Bash` and `Shell` (capitalized) interchangeably.

**Affected files:**
- `README.md` — settings.yaml example uses `Bash(git *)`, `Bash(ruff *)`
- `docs/permissions.md` — examples use `Bash(git *)`, `Shell(pytest *)`
- `settings.yaml.example` — uses `Bash(git *)`, `Bash(ruff *)`
- `src/godspeed/config.py` — default allow rules use `Bash(git *)`, `Bash(ruff *)`, `Bash(make *)`

**Impact:** Users copy-paste these examples into their `settings.yaml`. The rules silently fail to match because the tool name is `shell`, not `Bash`.

**Fix:** Standardize on `shell(...)` everywhere. Verify in `config.py`, `README.md`, `docs/permissions.md`, `settings.yaml.example`.

---

### 3. `SECURITY.md` — Supported Versions Table is Stale

**Issue:** Lists `2.3.x` and `2.2.x` as supported. Current version is `3.4.0`.

**Fix:** Update to `3.4.x` supported, `< 3.0` unsupported (or whatever the actual policy is).

---

### 4. `SECURITY.md` — Permission Evaluation Order is Wrong

**Issue:** Line 31 says "deny-first evaluation (deny > ask > allow)". The actual order is:

```
deny > dangerous > session > allow > ask > default (risk level)
```

**Fix:** Correct to match `docs/permissions.md` which gets this right.

---

## High-Priority Issues

### 5. `README.md` — Stale Personal Path in Public Doc

**Issue:** Lines 48–52 reference `C:\Users\ttimm\Desktop\godspeed_benchmark\production_audit.md`. This is a Windows user path that has no place in a public README.

**Fix:** Either commit `docs/production_audit.md` to the repo and link it, or remove the reference.

---

### 6. `README.md` — Duplicate Table Header

**Issue:** The "What's new in v3.4.0" section has a formatting error. Lines 37 and 40 both contain `| Area | What changed | Why |`, creating a broken table in the v3.3.0 subsection.

**Fix:** Remove the duplicate header on line 40.

---

### 7. `README.md` — Missing Slash Commands

**Issue:** The slash commands table (lines 266–289) does not include `/keys` or `/pull`, both of which were added in the latest commit.

**Fix:** Add both commands with descriptions.

---

### 8. `docs/quickstart_windows.md` — Incorrect Model Reference

**Issue:** Line 62 shows `model: anthropic/claude-opus-4-7`. This string does not match any known model identifier.

**Fix:** Use the correct 2026 model string, e.g. `claude-opus-4.6-20250929`.

---

### 9. `docs/quickstart_windows.md` — False Claim About Auto-Starting Ollama

**Issue:** Line 75 says "Godspeed will auto-start Ollama on first launch if it's installed." The code checks if Ollama is running (`OLLAMA_URL`) but does not auto-start the daemon.

**Fix:** Change to "Godspeed will detect a running Ollama instance automatically. Start Ollama first with `ollama serve`."

---

### 10. `docs/troubleshooting.md` — References Future Version

**Issue:** Line 50 says "Fix (v3.5.0+)" for `.env.local` auto-loading. Current version is 3.4.0, and this feature already shipped.

**Fix:** Change to "Fix (v3.4.0+)" or "Current versions".

---

### 11. `GODSPEED_ARCHITECTURE.md` — Very Stale Version Header

**Issue:** Line 1 reads "Godspeed Architecture (v2.3.0)". Current version is 3.4.0. Line 5 says "1,559+ tests passing"; current count is 1,999+.

**Fix:** Update version and test count. Consider removing the version from the title to avoid future staleness.

---

## Medium-Priority Issues

### 12. `README.md` — Install Command Package Name Mismatch

**Issue:** The install section says `pip install godspeed`, but `pyproject.toml` names the package `godspeed` (the script entry point), while the actual PyPI package may be `godspeed-coding-agent` (referenced in `docs/quickstart_windows.md`).

**Fix:** Verify the correct PyPI package name and standardize across all docs.

---

### 13. `README.md` — Product Comparison Table Removed

**Issue:** The README previously included a feature-comparison table against other coding agents.

**Fix:** Table removed. Project positioning is now strictly feature-driven with no competitive comparisons.

---

### 14. `README.md` — Dangerous Pattern Count is Stale

**Issue:** Line 69 says "71 dangerous patterns". After the security hardening commit, there are more (PowerShell, doas, pkexec, runas, base64 obfuscation).

**Fix:** Count current patterns and update. Or use "70+" to avoid future staleness.

---

### 15. `README.md` — Em-dash vs. Double-dash Inconsistency

**Issue:** Line 61 uses `--` (double dash) while the rest of the document uses `—` (em-dash). This is a typography inconsistency.

**Fix:** Standardize on em-dash throughout.

---

### 16. `CONTRIBUTING.md` — References Missing Makefile

**Issue:** Lines 30, 46–49 reference `make test`, `make lint`, `make type-check`, `make security`. No `Makefile` exists in the repo.

**Fix:** Either create a `Makefile` with these targets, or change the docs to use `uv run` commands:
```bash
uv run ruff check . --fix && uv run ruff format .
uv run ty check src/ || uv run mypy src/ --ignore-missing-imports
uv run pip-audit --ignore-vuln CVE-2026-28684
uv run bandit -r src/ -ll
uv run pytest --cov
```

---

### 17. Missing `CLAUDE.md` in Repo Root

**Issue:** The README and code both load `CLAUDE.md` for cross-agent compatibility, but the repo does not ship one. Other agents in the ecosystem typically ship this file.

**Fix:** Add a `CLAUDE.md` with build commands, test commands, architecture notes, and coding standards — consistent with what the project already documents.

---

### 18. Missing Documentation for New Features

**Issue:** The hardware scanner (`utils/hardware.py`), API key manager (`utils/api_keys.py`), `/keys`, and `/pull` commands have no user-facing documentation beyond the inline help in the TUI.

**Fix:** Add brief sections to README under "Configuration" or "Getting Started", or create `docs/api-keys.md` and `docs/hardware.md`.

---

## Low-Priority / Polish

### 19. `docs/permissions.md` — Minor Typos

- Line 121: "explaination" → "explanation" (if present — check full file)

### 20. `CHANGELOG.md` — Structure Check

**Issue:** The `[Unreleased]` section is now 50+ lines long. It covers multiple logical changes. This is fine, but verify that each bullet under `[Unreleased]` will eventually move to a version header.

**Fix:** No action needed now, but cut a release soon.

### 21. `docs/demo.md` — Version in Title

**Issue:** Line 26 references "Godspeed v3.3" in the recording title.

**Fix:** Update to current version when re-recording.

### 22. `GODSPEED_ARCHITECTURE.md` — Minor Inaccuracy

**Issue:** Line 16 referenced a specific open-source agent (74%+ SWE-bench) as a proven pattern. The 74% claim is from a specific paper/configuration and may be misleading without citation.

**Fix:** Softened to "following patterns from top-performing open-source coding agents" and removed specific product references.

---

## What Is Excellent

| File | Strength |
|------|----------|
| `README.md` | Strong hero section, clear architecture diagram, comprehensive feature list, honest benchmarks with null results published |
| `docs/permissions.md` | Best-in-class security documentation. Clear evaluation order, worked examples, runtime vs. config distinction |
| `GODSPEED.md.example` | Concise template with good guidance on what to include |
| `CONTRIBUTING.md` | Clear workflow, good code standards, excellent versioning policy with rationale |
| `docs/adding_a_driver.md` | Clean 3-step flow, good field reference table, practical worked example |
| `docs/troubleshooting.md` | Platform-specific fixes, clear symptom/cause/fix structure |
| `CHANGELOG.md` | Follows Keep a Changelog, honest about null results and limitations |

---

## Priority Fix List

| Priority | File | Fix |
|----------|------|-----|
| **P0** | `docs/fastapi-cors-configuration.md` | Delete (accidental inclusion) |
| **P0** | `README.md`, `docs/permissions.md`, `settings.yaml.example`, `config.py` | Standardize tool name to `shell(...)` instead of `Bash(...)`/`Shell(...)` |
| **P0** | `SECURITY.md` | Update supported versions to 3.4.x; fix permission evaluation order |
| **P1** | `README.md` | Remove personal Windows path; fix duplicate table header; add `/keys` and `/pull` |
| **P1** | `docs/quickstart_windows.md` | Fix model reference; correct Ollama auto-start claim |
| **P1** | `docs/troubleshooting.md` | Change "v3.5.0+" to "v3.4.0+" |
| **P1** | `GODSPEED_ARCHITECTURE.md` | Update version header and test count |
| **P2** | `CONTRIBUTING.md` | Fix Makefile references or create Makefile |
| **P2** | `README.md` | Remove product comparison table; update dangerous pattern count |
| **P2** | Root | Add `CLAUDE.md`; add docs for `/keys` and `/pull` |

---

*Audit completed 2026-04-27. 22 documentation files reviewed.*
