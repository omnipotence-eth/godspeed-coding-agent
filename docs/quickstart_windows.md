# Godspeed on Windows — quickstart

Godspeed ships on Windows, macOS, and Linux. This guide is the
Windows-specific path from zero to "agent completed its first
task" in under 2 minutes. For other platforms see the main
`README.md`.

---

## Prerequisites

- **Windows 10 / 11**
- **Python 3.11+** (3.13 recommended; Godspeed's CI matrix covers
  3.11 / 3.12 / 3.13)
- **Git** — for git-bash (Godspeed's shell tool uses it for shell
  command execution on Windows) and for cloning constituent repos
  when running benchmarks

Optional but recommended:
- **Ollama** for a free local model fallback (~3 GB VRAM for qwen3:4b)
- **WSL2 + Docker Desktop** if you want to run SWE-bench benchmarks
  locally (POSIX-only harness)

---

## 1. Install

The lowest-friction path uses Miniconda for isolation:

```powershell
# 1. Install Miniconda (one-time, if not already):
# https://docs.conda.io/en/latest/miniconda.html

# 2. Create + activate an env
conda create -n godspeed python=3.13 -y
conda activate godspeed

# 3. Install Godspeed
pip install godspeed-coding-agent
```

Alternatives:
- `pipx install godspeed-coding-agent` — isolated tool-install, no conda
- `uv pip install godspeed-coding-agent` — if you use `uv`

---

## 2. First-time config

```powershell
godspeed init
```

Creates `~/.godspeed/settings.yaml` with a default model pointing at a
free-tier NVIDIA NIM driver (requires `NVIDIA_NIM_API_KEY`) and a
fallback chain to local Ollama.

Open `~/.godspeed/settings.yaml` and either:
1. **Add an API key** for a paid provider (recommended for sustained
   use — NIM free tier is 40 RPM shared across all users):
   ```yaml
   model: anthropic/claude-opus-4-7
   ```
   Set the env var:
   ```powershell
   setx ANTHROPIC_API_KEY "sk-ant-..."    # persistent; open a new shell after
   ```

2. **Or use Ollama locally** (free, no cloud):
   ```powershell
   # Install Ollama from https://ollama.com then:
   ollama serve
   ollama pull qwen3:4b
   ```
   Godspeed will auto-start Ollama on first launch if it's installed.

---

## 3. Environment variables for Windows

Godspeed's TUI renders non-ASCII characters (arrows, em-dashes).
Set UTF-8 encoding to prevent `UnicodeEncodeError` crashes on
legacy `cp1252` consoles:

```powershell
# Persistent across sessions (PowerShell):
[Environment]::SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8', 'User')

# Or for one-off sessions (cmd.exe):
set PYTHONIOENCODING=utf-8
```

> Godspeed v3.3.0+ self-wraps stdout/stderr to UTF-8 at startup, so
> this is only needed on older installs. Still a good global to set.

Restart your terminal after `setx` / `[Environment]::Set...`.

---

## 4. Run your first task

```powershell
godspeed run "Create hello.py with a function greet(name) that returns 'Hello, <name>!'"
```

Expected: ~20-30 seconds, `hello.py` appears in the current directory,
exit code 0.

For interactive mode:

```powershell
godspeed
```

Launches the TUI. Type a task at the prompt, watch the agent work
live, Ctrl+C to cancel mid-turn (v3.3.0+).

---

## 5. WSL for SWE-bench benchmarks

If you want to run the SWE-bench local evaluation harness (for
reproducing benchmark numbers or generating leaderboard-submission
logs), the harness is POSIX-only and must run inside WSL:

```powershell
# Install WSL Ubuntu (one-time)
wsl --install -d Ubuntu

# Inside Ubuntu:
sudo apt update && sudo apt install -y python3-pip docker.io
pip install swebench
```

Docker Desktop must be running with WSL integration enabled
(Settings → Resources → WSL Integration → enable Ubuntu).

Once set up, Godspeed's runner script handles the path translation:

```powershell
# From a Windows terminal, inside the godspeed repo:
bash experiments/swebench_lite/leaderboard_submission/run_local_swebench_eval.sh --dry-run
```

The `--dry-run` flag validates prereqs without actually running the
eval. Remove it for a real run (~1-2 hours + 20-50 GB of Docker
images).

---

## Common Windows-specific issues

See `docs/troubleshooting.md`. The top three for new Windows users:

1. **`UnicodeEncodeError`** — set `PYTHONIOENCODING=utf-8`
2. **Ctrl+C doesn't cancel mid-turn** — only in TUI; on the
   `ProactorEventLoop` (default), cancel works via the standard
   KeyboardInterrupt path which requires a cancel point to be hit
3. **`psutil not available`** — reinstall from a fresh pip env

---

## Next steps

- Read `README.md` → Features for the full tool surface
- Read `GODSPEED_ARCHITECTURE.md` for the agent loop, tool system,
  permission engine, and audit trail design
- `godspeed models` — shows preconfigured model options + env-var
  requirements for each
- `godspeed audit verify` — confirms your session's audit trail is
  unbroken

All glory to God. Happy shipping.
