# Godspeed — troubleshooting guide

Common issues and their fixes, in order from most-hit to least-hit.

---

## Windows: `UnicodeEncodeError` on exit

**Symptom:** Agent finishes the task, then the CLI crashes just as it's about
to print the final summary:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in
position 126: character maps to <undefined>
```

**Cause:** the default Windows console encoding is `cp1252`, which can't
encode arrows (`→`), em-dashes (`—`), or smart quotes. The agent already
finished its real work — it's just the final write that crashes.

**Fix (v3.3.0+):** the CLI now force-wraps stdout/stderr in UTF-8 at
startup. If you're on an older install, set the environment variable:

```bash
export PYTHONIOENCODING=utf-8            # bash / zsh / git-bash
set PYTHONIOENCODING=utf-8               # cmd.exe
$env:PYTHONIOENCODING="utf-8"            # PowerShell
```

Persist it by adding to `~/.bashrc` / `~/.zshrc` / PowerShell profile.

---

## NVIDIA NIM free-tier: 40 RPM rate limit hits during long runs

**Symptom:** `litellm.Timeout: APITimeoutError - Request timed out` after
~40 calls in a minute, often during SWE-Bench evaluation or heavy refactors.

**Cause:** the NIM free tier caps concurrent usage at 40 requests per
minute *shared across all users*. Other users' traffic can push you
over even if your own rate is low.

**Mitigations:**

1. **Configure a fallback chain** in `~/.godspeed/settings.yaml`:
   ```yaml
   model: nvidia_nim/moonshotai/kimi-k2.5
   fallback_models:
     - ollama/qwen3:4b   # local fallback, always available
   ```
   When NIM returns 429, Godspeed's built-in retry / fallback loop
   drops to the next model automatically.

2. **Enable instance-cooldown** for benchmark runs:
   ```bash
   python experiments/swebench_lite/run.py --instance-cooldown 90
   ```
   Inserts a 90 s sleep between instances so the RPM window resets.

3. **Use paid direct API** for sustained load:
   ```yaml
   model: moonshot/kimi-k2.5     # requires MOONSHOT_API_KEY
   ```

---

## WSL + Docker for SWE-Bench harness

**Symptom:** `python -m swebench.harness.run_evaluation` on Windows raises:
```
ModuleNotFoundError: No module named 'resource'
```

**Cause:** the SWE-bench harness uses Python's `resource` module, which
is POSIX-only.

**Fix:** run the harness inside WSL Ubuntu. Godspeed's runner script
handles this automatically:

```bash
bash experiments/swebench_lite/leaderboard_submission/run_local_swebench_eval.sh --dry-run
bash experiments/swebench_lite/leaderboard_submission/run_local_swebench_eval.sh --max-workers 4
```

The script converts Windows paths to `/mnt/c/...`, invokes the harness
via `wsl -d Ubuntu`, and copies per-instance artifacts back into the
leaderboard submission dir.

**Known bug (upstream):** swebench 4.1.0's pvlib testbed image ships
NumPy 2.0+ which breaks pvlib's `np.Inf` usage. Our audit caught this;
see `experiments/swebench_lite/leaderboard_submission/README.md`
"⚠ Environmental caveat" for the full analysis. Don't use local
harness numbers as the leaderboard submission until upstream fixes;
use sb-cli cloud-graded reports instead.

---

## Audit trail: verify session integrity

Every Godspeed session records a hash-chained audit log at
`~/.godspeed/audit/<session_id>.audit.jsonl`. Each record links to the
previous via `prev_hash`, forming a cryptographically tamper-evident
chain.

```bash
# Verify one session
godspeed audit verify 9e320a20-fbde-4004-b9cc-4fbc94601a05

# Verify every session in the audit dir
godspeed audit verify
```

Expected output: `VALID -- Chain verified: N records`.

If `verify` reports a break, the log has been tampered with between
the listed record and the next — or a file-system corruption event
occurred. The records up to the break are still valid history; the
remainder should be discarded.

---

## Ollama first run: model not installed

**Symptom:**
```
connection refused: http://localhost:11434
```
or
```
model 'qwen3:4b' not found
```

**Fix:**

```bash
# Install ollama (https://ollama.com) then:
ollama serve                 # in a separate terminal
ollama pull qwen3:4b         # ~2.5 GB download, one-time
```

Then point Godspeed at it:

```yaml
# ~/.godspeed/settings.yaml
model: ollama/qwen3:4b
```

Godspeed's CLI `init` command will auto-start Ollama on session
launch if it's installed but not running. The 15-second startup
timeout is controlled by `OLLAMA_STARTUP_TIMEOUT` in `src/godspeed/cli.py`.

---

## `file_edit` rejected by post-edit syntax gate

**Symptom (v3.3.0+):** an edit that looked like it should work returns:
```
Post-edit syntax check failed — the edit would leave the file
unparseable. Line N: <reason>.
```

**Cause:** the fuzzy matcher found your edit location, but the
`new_string` dropped indentation or introduced a syntax error. The
gate is protecting you from silently writing a broken .py / .pyi /
.json file.

**Fix:** re-read the file first (so you have exact whitespace context),
then include enough surrounding lines in `old_string` that the
replacement `new_string` will preserve proper indentation.

---

## Diff reviewer: accept/reject the proposed edit

**Symptom (v3.3.0+):** the TUI prompts
```
Review proposed edit
  file_edit  src/main.py
  +2 -1 lines
  @@ ... @@
  Apply? (y)es · (n)o
```

**Usage:**
- **y / yes / ⏎ (empty)** — apply the change
- **n / no** — reject; file stays unchanged, agent gets a "rejected"
  ToolResult and will typically retry with a different approach
- **a / always** — accept *and* suppress future review prompts for
  the rest of the session (useful when you trust a refactor)

The reviewer is TUI-only. Headless mode (`godspeed run`) never asks
— it auto-accepts so CI stays deterministic.

---

## Mid-turn cancel (v3.3.0+): Ctrl+C to interrupt

**Symptom:** agent is heading in the wrong direction and you want
to redirect without waiting for the turn to finish.

**Fix:**
- **First Ctrl+C** — cancels the current turn cleanly. The agent
  unwinds at the next checkpoint (between streaming chunks). You
  see: `Agent cancelled. Send another prompt or /quit.`
- **Second Ctrl+C within 1 s** — hard interrupt
  (KeyboardInterrupt). Use if the first press didn't take effect
  (rare; happens if the agent is mid-non-cancellable IO).

Windows note: on the `ProactorEventLoop`, the first-press handler
isn't installable via `asyncio.add_signal_handler`. Ctrl+C still
works via the standard KeyboardInterrupt path.

---

## psutil not installed on a custom conda env

**Symptom:**
```
psutil not available; cannot force-kill process tree for pid=NNN
```

**Cause:** `psutil` is declared as a core dep, but a custom conda env
may not have installed it.

**Fix:**
```bash
conda run -n <env> pip install 'psutil>=5.9,<8.0'
# or:
pip install godspeed-coding-agent --force-reinstall
```

Without psutil, Godspeed's shell tool can't kill grandchildren when
a timeout fires — a long-running subprocess may orphan.

---

## Getting help / reporting a bug

- Issues: https://github.com/omnipotence-eth/godspeed-coding-agent/issues
- Architecture reference: `GODSPEED_ARCHITECTURE.md`
- Full-flow walkthrough: `README.md` → Getting Started
