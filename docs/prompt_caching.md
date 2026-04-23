# Godspeed — prompt caching per provider

Prompt caching lets the model skip re-billing input tokens that
haven't changed between turns — the system prompt, tool schemas,
older conversation history. Godspeed enables it by default
(`prompt_caching: true` in `settings.yaml`) and applies the
provider-specific mechanism automatically.

Heaviest real-session costs come from the replayed context (89k-122k
tokens/turn in our benchmarks). With caching active, only the newest
user input + the model's new response are re-billed at full rate.

---

## Per-provider behavior

| Provider | Mechanism | Godspeed does | Expected savings |
|----------|-----------|---------------|------------------|
| **Anthropic** (direct, Bedrock, Vertex) | Explicit `cache_control` on content blocks | Marks system prompt + last stable turn (2 breakpoints) | ~90% on cached input after 1st call |
| **OpenAI** (gpt-4o / gpt-4.1 / o1 / o3 / o4) | Automatic prefix-hash caching | No-op marker (harmless) | ~50% on cached input |
| **DeepSeek** | Automatic | No-op marker (harmless) | ~75% on cached input |
| **Gemini** | Separate `cachedContent` API | **Skipped** (different shape) | N/A via LiteLLM |
| **Ollama** (local) | Process-local KV cache | Skipped | Already local — irrelevant |
| **NVIDIA NIM** | Depends on upstream provider | Skipped by default | Varies |
| **Groq** | Not supported | Skipped | N/A |
| **Mistral** | Not supported | Skipped | N/A |

See `src/godspeed/llm/client.py` for the authoritative allow/deny lists
(`_CACHING_ALLOWLIST` / `_CACHING_DENYLIST`).

---

## What gets cached

Two breakpoints for Anthropic-family models:

1. **End of system prompt.** Contains tool descriptions, project
   instructions (GODSPEED.md), auto-indexed repo context. Stable for
   the whole session. First cache hit lands on the second LLM call
   in a session.

2. **End of last stable conversation turn.** The latest `tool`-role
   message (tool result) or `assistant` message is marked — caches
   the entire history except the caller's newest user input. Every
   subsequent turn only re-bills the last user message and the new
   assistant response.

Anthropic allows up to 4 cache breakpoints; Godspeed uses 2 because
adding more rarely raises the hit rate meaningfully and doubles the
complexity.

---

## Telemetry

LiteLLM surfaces Anthropic's cache metrics on the `usage` block:

- `cache_read_input_tokens` — tokens served from cache (priced at 10%)
- `cache_creation_input_tokens` — tokens written to cache (priced at 125%)

Godspeed accumulates both on `LLMClient.total_cache_read_tokens` and
`LLMClient.total_cache_creation_tokens`. The `/stats` slash command
displays them when non-zero:

```
session   claude-opus-4-7   turns=7   tokens=42k in / 1.2k out
cache    read=38k  (90.5% of input)   created=4k
cost     $0.08 (saved ~$0.45 via cache)
```

Providers that don't surface cache metrics leave both counters at 0 —
not "zero hits," just "unknown."

---

## Disabling

Set `prompt_caching: false` in `~/.godspeed/settings.yaml` if a
specific provider rejects or errors on the `cache_control` marker.
The default allow/deny lists should cover all mainstream providers
correctly, but new/rare providers may need this escape hatch.

```yaml
# ~/.godspeed/settings.yaml
prompt_caching: false
```

Or via environment variable for a single run:

```bash
GODSPEED_PROMPT_CACHING=false godspeed
```

---

## What Godspeed does NOT cache

- **Tool schemas block.** Anthropic's API accepts `cache_control` on
  the tools array, but LiteLLM translates this inconsistently across
  provider backends. Safer to let the tools block ride in the
  system-prompt cache (which includes the tool-description text
  Godspeed generates into the system prompt anyway).
- **Gemini context caching.** Gemini uses a separate `cachedContent`
  endpoint with an explicit create/list/delete lifecycle — not
  appropriate for per-turn agent loops where conversation history
  grows every call.
- **Per-file read results.** When `file_read` returns, that content
  lives in the next tool-result message — it becomes part of the
  "last stable turn" cache breakpoint on the call AFTER it's injected.
  No special handling needed.
