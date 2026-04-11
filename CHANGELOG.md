# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Route logs to stderr and scope verbose mode to `godspeed.*` namespace only — eliminates LiteLLM/httpx/markdown_it debug noise in TUI
- Add debug logging on tiktoken encoding fallback instead of silent `pass`
- Add docstrings to `PermissionEvaluator` and `AuditRecorder` protocol methods

### Added

- Audit trail retention cleanup — expired sessions are purged on startup based on `retention_days` setting
- Token counter model mappings for Claude, Gemini, DeepSeek, Ollama models

### Changed

- Expanded pyproject.toml classifiers and keywords for PyPI discoverability

## [0.1.0] - 2026-04-10

### Added

- Hand-rolled agent loop — model decides when to stop, no framework dependency
- 4-tier permission engine (deny > ask > allow) with deny-first evaluation, pattern matching, and 46 dangerous command patterns
- Hash-chained JSONL audit trail (SHA-256) with tamper detection and `godspeed audit verify`
- 4-layer secret protection: file deny rules, context cleaning, output filtering, audit redaction (27 regex patterns + Shannon entropy)
- LiteLLM integration for 200+ LLM providers (Claude, GPT, Gemini, Ollama, etc.) with fallback chains
- 7 built-in tools: file_read, file_write, file_edit (with fuzzy matching), shell, glob_search, grep_search, git
- Rich + prompt-toolkit TUI with syntax highlighting, diff rendering, and streaming
- Slash commands: /help, /model, /undo, /audit, /compact, /clear, /quit
- GODSPEED.md project instructions (walk-up-tree loading, like CLAUDE.md)
- Conversation compaction when approaching context limit
- Configuration cascade: global (~/.godspeed/settings.yaml) > project (.godspeed/settings.yaml) > CLI flags (deny rules are additive — project can't weaken global denies)
- CLI entry points: `godspeed` (TUI), `godspeed version`, `godspeed audit verify`
- 243 tests, 82% coverage
- Full repo-standards scaffolding: CI pipeline, pre-commit hooks, dependabot, issue/PR templates, CONTRIBUTING.md, SECURITY.md, LICENSE (MIT)
