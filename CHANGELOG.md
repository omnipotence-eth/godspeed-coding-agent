# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-11

### Added

- **Sub-agent coordinator**: `AgentCoordinator` spawns isolated sub-agents with separate conversations; depth limit 3, iteration limit 25; `SpawnAgentTool` (HIGH risk) enables multi-agent orchestration via `agent_loop()` reuse
- **`spawn_parallel()`**: run multiple sub-agents concurrently via `asyncio.gather()`
- **MCP client**: `MCPClient` connects to MCP servers via stdio transport; `MCPToolAdapter` maps MCP tool definitions to Godspeed Tool ABC (all HIGH risk); graceful when `mcp` package not installed
- **MCP server discovery**: `mcp_servers` config in `settings.yaml` auto-discovers and registers remote tools at startup
- **Model routing**: `ModelRouter` routes LLM calls by task type (plan/edit/chat) to different models; configurable via `routing` in settings
- **Human-in-the-loop pause/resume**: `asyncio.Event` shared between TUI and agent loop; `/pause` stops at next iteration, `/resume` continues, `/guidance <msg>` injects mid-conversation correction and resumes
- **Rich permission prompts**: contextual detail in permission dialogs -- file_edit shows mini-diff, file_write shows first 10 lines, shell shows syntax-highlighted command, file_read shows path
- 549 tests, ~90% coverage

### Changed

- `LLMClient.chat()` accepts optional `task_type` parameter for model routing
- `agent_loop()` accepts `pause_event` parameter for human-in-the-loop control
- `format_permission_prompt()` accepts optional `arguments` dict for contextual display

## [0.3.0] - 2026-04-11

### Added

- **Tree-sitter repo map tool**: `RepoMapTool` extracts symbol outlines (functions, classes, methods) from Python/JS/TS/Go files using tree-sitter; graceful degradation when tree-sitter not installed
- **Plan mode**: `/plan` command toggles read-only mode -- permission engine blocks all non-READ_ONLY tools; system prompt updated to instruct explore-only behavior
- **Git stash/stash_pop actions**: `GitTool` supports `stash` (with auto-message) and `stash_pop` operations
- **Auto-stash before risky operations**: agent loop tracks consecutive file edits; after 3+ consecutive writes, auto-stashes working state as a safety net
- **Model-aware compaction**: compaction prompts adapt to model context window size -- small models (<=32K) get aggressive summarization, frontier models (>100K) get detailed preservation
- **`MODEL_CONTEXT_WINDOWS` mapping**: prefix-matched context window sizes for Claude, GPT, Gemini, Ollama models with `get_model_context_window()` utility
- **`/checkpoint [name]` command**: save conversation state snapshots to `.godspeed/checkpoints/`; list checkpoints with metadata (tokens, messages, model, timestamp)
- **`/restore <name>` command**: restore a saved checkpoint, rebuilding full conversation state
- **Checkpoint management**: `save_checkpoint()`, `load_checkpoint()`, `list_checkpoints()`, `delete_checkpoint()` in `context/checkpoint.py`
- 485 tests, ~90% coverage

### Changed

- Compaction in agent loop now uses model-aware prompts via `get_compaction_prompt()` instead of hardcoded prompt
- Moved `tests/test_context.py` into `tests/test_context/` package for better organization

## [0.2.0] - 2026-04-11

### Added

- **Stuck-loop detection**: after 3 identical tool errors, injects a replan message forcing the model to try a different approach
- **Verification cascade**: `VerifyTool` runs `ruff check` on Python files; auto-verifies after every `file_edit`/`file_write` so the agent self-corrects lint errors
- **`/extend N` command**: override max iterations per agent turn (default: 50)
- **`/context` command**: show context window usage — tokens, percentage, message count with color-coded thresholds
- **Audit trail compression**: `compress_session()` rotates `.jsonl` → `.jsonl.gz`; `verify_chain()` transparently handles compressed logs
- **FileEdit confidence reporting**: output includes `[match=exact confidence=1.00]` or `[match=fuzzy confidence=0.87 line=42]` so the agent can gauge match quality
- **26 new dangerous command patterns**: iptables, mount/umount, fdisk, shutdown/reboot, docker rm -f, kubectl delete, env exfiltration, Windows destructive ops, supply-chain attacks
- **Ollama auto-start**: detects when Ollama is not running and starts `ollama serve` as a background process before first LLM call
- **Lazy LiteLLM import**: deferred import reduces cold startup from ~1.5s to ~300ms
- **Smart retry for connection errors**: skips retry+sleep when Ollama/server is down, returns actionable error immediately
- **Non-TTY crash guard**: graceful error message when launched from non-interactive shells
- `ToolRegistry.has_tool()` method
- `agent_loop()` accepts `max_iterations` parameter
- 411 tests, 90% coverage

### Fixed

- Route logs to stderr and scope verbose mode to `godspeed.*` namespace only — eliminates LiteLLM/httpx/markdown_it debug noise in TUI
- Add debug logging on tiktoken encoding fallback instead of silent `pass`
- Add docstrings to `PermissionEvaluator` and `AuditRecorder` protocol methods

### Changed

- `godspeed init` command — creates `~/.godspeed/` and default `settings.yaml`
- `godspeed models` command — shows popular model options with provider, cost, and API key info
- `settings.yaml.example` — full reference config with free/paid model examples and permission rules
- Audit trail retention cleanup — expired sessions are purged on startup based on `retention_days` setting; handles both `.jsonl` and `.jsonl.gz`
- Token counter model mappings for Claude, Gemini, DeepSeek, Ollama models
- Default model changed from paid `claude-sonnet-4-20250514` to free `ollama/qwen3:4b` — zero-cost out of the box
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
