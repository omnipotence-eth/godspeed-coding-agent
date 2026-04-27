# Godspeed — permissions guide

Godspeed ships with a **4-tier deny-first permission engine** wrapping every
tool call. The goal is that an agent with a compromised model, prompt
injection, or a bad tool parse cannot silently exfiltrate secrets, rewrite
`.env`, or `rm -rf` your home directory. Everything dangerous is blocked
by default and stays blocked until you say otherwise.

---

## The four tiers

Evaluated in this exact order for every tool call. First match wins.

| # | Tier | When it fires |
|---|------|---------------|
| 1 | **DENY** | Any `deny:` rule matches the tool call — blocked, no prompt |
| 2 | **Dangerous command detection** | Shell command matches a built-in dangerous pattern (e.g. `rm -rf /`) — blocked |
| 3 | **SESSION grant** | You previously approved this pattern this session — allowed |
| 4 | **ALLOW** | Any `allow:` rule matches — allowed, no prompt |
| 5 | **ASK** | Any `ask:` rule matches — prompts the user |
| 6 | **Risk-level default** | Falls back to the tool's risk level (READ_ONLY → allow, LOW → ask, HIGH → ask, DESTRUCTIVE → deny) |

Key invariant: **deny rules always win**. A project-level `allow:` entry
cannot weaken a global `deny:` entry.

---

## Where rules live

| Scope | File | Precedence |
|-------|------|------------|
| Global | `~/.godspeed/settings.yaml` | Loaded first |
| Project | `<repo>/.godspeed/settings.yaml` | Loaded second — can only **add** denies, never remove them |

Rules are YAML under the `permissions:` key:

```yaml
permissions:
  deny:
    - "FileRead(.env)"
    - "FileRead(.ssh/*)"
    - "FileWrite(.env*)"
  allow:
    - "shell(git *)"
    - "shell(pytest *)"
    - "shell(make *)"
  ask:
    - "shell(*)"
```

Pattern format is `Tool(glob)` — fnmatch globbing on the argument.
Examples: `Shell(git *)`, `FileRead(*.pem)`, `FileWrite(src/*.py)`.

---

## Writing rules at runtime: the `/remember` command

Manually editing YAML is friction. Use `/remember` inside the TUI to
persist a rule without leaving the session:

```text
/remember approve Shell(pytest *)        # ALLOW, global
/remember deny FileWrite(*.env*)         # DENY,  global
/remember ask Shell(rm *)                # ASK,   global
/remember approve Shell(make) --project  # ALLOW, this repo only
```

Each command:

1. Validates pattern syntax (`Tool(argument)` form required).
2. Writes to `~/.godspeed/settings.yaml` (or `.godspeed/settings.yaml`
   with `--project`).
3. Adds the rule to the **live** permission engine so it takes effect
   on the next tool call — no restart needed.

Duplicates are silently skipped — re-running the same `/remember` is
idempotent.

Accepted action words: `approve` (alias for `allow`), `allow`, `deny`,
`ask`. `approve` reads more naturally in conversation.

---

## Three worked examples

### 1. Approve a specific test command once vs. forever

During a session you see the ASK prompt:

```
[?] shell(pytest tests/test_foo.py -xvs)
    Allow (a)lways · this-session (s) · just-this-time (y) · deny (n)
```

- Pressing `s` grants a session-scoped allow that expires in 1 hour.
- Pressing `a` calls `/remember approve` for you and writes the rule
  to `~/.godspeed/settings.yaml` — persists across restarts.

You could also pre-authorize ahead of time:

```text
/remember approve Shell(pytest *)
```

### 2. Scope a rule to one repo only

When testing an experimental tool that should only run in *this* repo:

```text
/remember approve Shell(./experimental_script.sh) --project
```

Goes to `<repo>/.godspeed/settings.yaml`. Other repos on this machine
are unaffected.

### 3. Why you cannot override a global deny at project level

By design. Project-level configs can only **add** denies, never
remove them — the merge in `config._merge_configs` takes the union
of deny lists.

If you wrote:

```yaml
# global: ~/.godspeed/settings.yaml
permissions:
  deny:
    - "FileRead(.env)"

# project: ./.godspeed/settings.yaml
permissions:
  allow:
    - "FileRead(.env)"   # ← will NOT override the global deny
```

the deny still wins. This is intentional: a repo you `git clone` can't
ship a `.godspeed/settings.yaml` that removes the global `.env` block.

To actually lift a global deny, edit `~/.godspeed/settings.yaml`
directly — it's your file, under your control.

---

## Inspecting the current state

```text
/permissions
```

Shows a table of every DENY / ALLOW / ASK rule and every active
SESSION grant. Useful when debugging why a tool got blocked.

---

## Dangerous command detection (tier 2)

Separate from the rule list. Godspeed's shell tool ships built-in
detection for a small set of always-dangerous patterns:

- `rm -rf /`, `rm -rf ~`, `rm -rf $HOME`
- `dd if=... of=/dev/*`
- `mkfs`, `fdisk /dev/*`
- Fork bombs (`:() { :|:& };:`)
- `chmod -R 777 /`

Even a matching ALLOW rule or SESSION grant **cannot** bypass this
tier. If you really need to run one of these patterns, do it outside
Godspeed.

---

## Audit trail

Every permission decision — allow, deny, or ask — is recorded in the
session's hash-chained audit log at
`~/.godspeed/audit/<session_id>.audit.jsonl`. Verify with:

```bash
godspeed audit verify <session_id>
```

See `docs/troubleshooting.md` → "Audit trail: verify session integrity"
for the full workflow.
