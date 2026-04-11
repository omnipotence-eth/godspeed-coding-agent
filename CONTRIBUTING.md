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

## Reporting Vulnerabilities

See [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.
