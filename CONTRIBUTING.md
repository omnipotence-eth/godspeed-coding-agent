# Contributing to Godspeed

Thank you for your interest in contributing to Godspeed! This guide will help you
get started.

## Development Setup

1. **Fork and clone** the repository:

   ```bash
   gh repo fork omnipotence-eth/godspeed-coding-agent --clone
   cd godspeed
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   uv sync --all-extras
   ```

3. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   ```

4. **Run tests** to verify your setup:

   ```bash
   make test
   ```

## Development Workflow

1. **Create a branch** from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the code standards below.

3. **Run the full check suite**:

   ```bash
   make lint        # ruff check + format
   make type-check  # ty or mypy
   make security    # pip-audit + bandit
   make test        # pytest with coverage
   ```

4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):

   ```bash
   git commit -m "feat: add new tool for X"
   ```

5. **Push and open a PR**:

   ```bash
   git push -u origin feat/your-feature-name
   gh pr create
   ```

## Code Standards

- **Type hints** on all public functions (`from __future__ import annotations`)
- **No `print()` in production** — use `logging.getLogger(__name__)`
- **Specific exceptions** — never bare `except:`
- **Ruff** for linting and formatting (line length 100)
- **Tests** for every new feature, especially security-related code

## Security Contributions

Security is Godspeed's core differentiator. If you're contributing to the security
module:

- Every dangerous command pattern needs a test
- Every secret detection pattern needs a test with real-world examples
- Permission engine changes need edge case tests
- Audit trail changes must maintain hash chain integrity

## Versioning Policy

Godspeed follows [Semantic Versioning](https://semver.org/) under a strict
interpretation, because the version number is a signal to users about stability
and to reviewers about project maturity. Inflated versions damage both.

| Bump | When | Evidence required in PR |
|------|------|-------------------------|
| **Patch** (`2.4.0 → 2.4.1`) | Bug fix, dependency refresh, documentation, internal refactor | Linked issue or CHANGELOG entry under *Fixed* |
| **Minor** (`2.4.0 → 2.5.0`) | New user-facing capability added **without** breaking existing usage | New CLI command / tool / configuration field documented in README or CHANGELOG under *Added* |
| **Major** (`2.x.y → 3.0.0`) | A breaking change to a documented public API or CLI surface | Breaking-change note in CHANGELOG under *Changed* with migration guidance, AND the PR description states the break explicitly |

A release with *no user-observable change* must not bump any digit. Lockfile
refreshes, CI tweaks, and internal hardening go out as patch releases at most,
and usually ride the next feature release.

### Why this matters

Version numbers are a conversation with your users. A project that ships `v2.x`
after 5 days implies stability guarantees it cannot keep, and a reviewer seeing
13 releases in a week will read that as version-number inflation, not momentum.
For comparison: the \`Development Status :: 3 - Alpha\` classifier in
\`pyproject.toml\` says "breaking changes are expected" — which is incompatible
with shipping major versions as fast as we can open PRs.

**Checklist before bumping the major digit**:
1. Is there a documented public API or CLI that this PR changes in a way users must adapt to?
2. Is the breaking change called out in the CHANGELOG under *Changed* with migration guidance?
3. Has the `Development Status` classifier graduated past *Alpha*? If no, reconsider.

If any answer is "no," bump minor or patch.

## Reporting Vulnerabilities

See [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.
