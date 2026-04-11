<div align="center">

# Godspeed

**Security-first open-source coding agent.**

[![CI](https://github.com/omnipotence-eth/godspeed-coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/omnipotence-eth/godspeed-coding-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-orange?style=flat-square)](https://github.com/BerriAI/litellm)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

An AI coding agent that treats security as a first-class concern -- not an afterthought.

[Getting Started](#getting-started) | [Features](#features) | [Architecture](#architecture) | [Configuration](#configuration) | [Contributing](CONTRIBUTING.md)

</div>

---

## Why

Every open-source coding agent gives an LLM the ability to read files, write code, and run shell commands. None of them ship with a deny-first permission engine, a tamper-evident audit trail, or multi-layer secret protection out of the box. You are expected to bolt security on yourself, or trust the model not to `rm -rf /`.

Godspeed closes that gap. It pairs full coding capability (7 tools, 200+ LLM providers, conversation compaction) with a security model that fails closed by default. Every tool call passes through a 4-tier permission engine. Every action is recorded in a hash-chained audit log you can cryptographically verify. Secrets are caught at four separate layers before they ever reach the model or the log file.

If you want a coding agent you can actually point at a production codebase, this is it.

## Features

### Security

- **4-tier permission engine** -- deny-first evaluation with pattern matching, dangerous command detection (46 patterns), and fail-closed defaults. No tool call executes without explicit permission.
- **Hash-chained audit trail** -- SHA-256 JSONL log where each entry chains to the previous. Tamper-evident and verifiable with `godspeed audit verify`.
- **Secret protection** -- 4 layers of defense: file deny-listing, context cleaning, output filtering, and audit redaction. 27 regex patterns plus Shannon entropy analysis catch API keys, tokens, and credentials before they leak.

### Capability

- **200+ LLM providers** -- Claude, GPT, Gemini, Ollama, and everything else LiteLLM supports. Configure fallback chains so work never stops.
- **7 built-in tools** -- `file_read`, `file_write`, `file_edit` (with fuzzy matching), `shell`, `glob`, `grep`, and `git`. Everything a coding agent needs, nothing it doesn't.
- **GODSPEED.md project instructions** -- drop a `GODSPEED.md` in any project root to give the agent persistent context about your codebase, conventions, and constraints.
- **Conversation compaction** -- automatically summarizes context when approaching the token limit, so long sessions don't degrade.
- **Rich TUI** -- syntax highlighting, diff rendering, streaming output, and slash commands via Rich and prompt-toolkit.

## Architecture

```mermaid
flowchart LR
    User([User]) --> TUI["TUI\n(Rich + prompt-toolkit)"]
    TUI --> Loop["Agent Loop\n(loop.py)"]
    Loop --> LLM["LLM\n(LiteLLM)"]
    LLM -->|tool calls| Perm["Permission\nEngine"]
    Perm -->|allowed| Tools["Tools\n(7 built-in)"]
    Perm -->|denied| Deny[Deny + Log]
    Tools --> Audit["Audit Trail\n(SHA-256 JSONL)"]
    Deny --> Audit
    Audit --> Loop
    Loop -->|response| TUI

    style Perm fill:#e74c3c,color:#fff
    style Audit fill:#2ecc71,color:#fff
```

**How it works:**

The agent loop is hand-rolled (no framework) following the same pattern proven by top-performing coding agents. The LLM decides when to stop. On each turn, the LLM either responds with text (done) or requests tool calls. Every tool call passes through the **permission engine** before execution: deny rules are evaluated first and always win, then dangerous command detection (25+ regex patterns) blocks destructive operations, then session grants and allow rules, and finally the tool's risk level determines the default. If anything is ambiguous, it fails closed. After execution, the tool call, its result, and the permission decision are recorded in the **audit trail** -- a hash-chained JSONL file where each record includes the SHA-256 hash of the previous record. Secrets are redacted at four layers: file access deny rules, context cleaning before the LLM sees content, output filtering on LLM responses, and audit log redaction.

**Key modules:**

| Module | Path | Purpose |
|--------|------|---------|
| Agent loop | `src/godspeed/agent/` | Conversation management, LLM interaction, tool dispatch |
| Security | `src/godspeed/security/` | Permission engine, dangerous command detection, secret scanning |
| Audit | `src/godspeed/audit/` | Hash-chained event logging, redaction, verification |
| Tools | `src/godspeed/tools/` | 7 built-in tools with Pydantic schemas |
| LLM | `src/godspeed/llm/` | LiteLLM client wrapper, token counting |
| Context | `src/godspeed/context/` | Project instructions, conversation compaction |
| TUI | `src/godspeed/tui/` | Terminal UI, output rendering, slash commands |

## Getting Started

### Install

```bash
pip install godspeed
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add godspeed
```

### Run

```bash
# Set your LLM API key (example: Claude)
export ANTHROPIC_API_KEY="sk-..."

# Launch in any project directory
cd your-project/
godspeed
```

Or use a local model with [Ollama](https://ollama.com/) -- zero cost, full privacy:

```bash
ollama pull qwen3:4b
godspeed -m ollama/qwen3:4b
```

Godspeed auto-upgrades `ollama/` to `ollama_chat/` for tool-capable models (Qwen, Llama, Gemma, Mistral, etc.).

Godspeed reads `GODSPEED.md` from the project root for persistent instructions -- similar to how other agents use `CLAUDE.md`.

### First session

```
$ godspeed
godspeed> Explain the authentication flow in this codebase
```

The agent will read your code, answer questions, write files, and run commands -- all gated by the permission engine.

### Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model [name]` | Show or switch the active model |
| `/clear` | Clear conversation history |
| `/undo` | Undo last git commit (`git reset --soft HEAD~1`) |
| `/audit` | Show audit trail stats and verify chain integrity |
| `/permissions` | Show current permission rules and session grants |
| `/quit` | Exit Godspeed |

## Configuration

### Project-level: `GODSPEED.md`

Drop a `GODSPEED.md` in your project root. The agent loads it as system context on every session. Use it for coding standards, architecture notes, or constraints. See [`GODSPEED.md.example`](GODSPEED.md.example) for a template.

### Global: `~/.godspeed/settings.yaml`

```yaml
model: claude-sonnet-4-20250514
fallback_models:
  - gpt-4o
  - gemini-2.0-flash

permissions:
  deny:
    - "FileRead(.env)"
    - "FileRead(*.pem)"
    - "FileRead(.ssh/*)"
  allow:
    - "Bash(git *)"
    - "Bash(ruff *)"
    - "Bash(pytest *)"
  ask:
    - "Bash(*)"

audit:
  enabled: true
  retention_days: 30
```

Permission rules use glob-style matching against `ToolName(argument)` strings. Deny rules are additive across config levels -- a project config cannot weaken global denies.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude access |
| `OPENAI_API_KEY` | GPT access |
| `GEMINI_API_KEY` | Gemini access |
| `GODSPEED_MODEL` | Override default model |

## Development

```bash
# Clone and install
git clone https://github.com/omnipotence-eth/godspeed-coding-agent.git
cd godspeed
uv sync --all-extras

# Lint and format
ruff check . --fix && ruff format .

# Run tests
pytest --cov

# Verify audit trail integrity
godspeed audit verify
```

## License

[MIT](LICENSE)

---

<div align="center">

Built by [Tremayne Timms](https://github.com/omnipotence-eth)

</div>
