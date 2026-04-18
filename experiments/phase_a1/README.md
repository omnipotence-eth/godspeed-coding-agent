# Phase A1 — Synthetic Tool-Calling Data Pipeline

Generate ~6,200 tool-calling training samples covering all 21 Godspeed tools,
free-tier friendly, pluggable into the ml-lab SFT pipeline.

## Layout

```
experiments/phase_a1/
├── providers.py         # Cerebras → Z.ai → Groq → Ollama async router
├── specs.py             # stratified (tool, category, seed) generation
├── blueprints.py        # Stage A — LLM blueprint generation
├── executor.py          # Stage B — real tool execution on tmp sandbox
├── fixtures/            # canned outputs for non-sandbox tools
├── narrator.py          # Stage C — LLM assistant-narration around real I/O
├── emit.py              # Stage D — ConversationLogger → OpenAI JSONL
├── judge.py             # Stage E — GLM-Flash 4-dim filter
├── swesmith_distill.py  # diversity-sample existing SWE-Smith corpus
├── augment.py           # param-shuffle + intent paraphrase
├── orchestrate.py       # top-level runner (resumable, checkpointed)
├── validate.py          # final schema + tool-name + arg-shape check
├── anchor_claude.py     # 50-sample Sonnet-4.6 anchor set (held-out)
├── data/                # intermediate + final JSONL outputs
└── tests/               # pytest unit tests
```

## Targets

| Command | What it does |
|---------|--------------|
| `make a1-smoke` | 20-sample Ollama-only pipeline smoke test |
| `make a1-anchor` | Generate 50 Claude anchor samples (≤$2) |
| `make a1-run` | Full 6,200-sample background run (5 days, resumable) |
| `make a1-validate` | Schema + coverage stats on final JSONL |

## Output format

JSONL at `data/phase_a1_final.jsonl`. Each line:
```json
{
  "messages": [{"role": "system", ...}, {"role": "user", ...}, ...],
  "tools": [... 21-tool OpenAI function-calling schemas ...]
}
```

Consumable by the ml-lab training config's `format: messages_raw`. Tool
rendering is handled by TRL's chat template at train time.

## Gates to Phase A2

- ≥200 samples per tool
- Category mix within 2% of 70/15/10/5 (single / multi / no-tool / error-recovery)
- <1% `validate.py` failures
- Judge median ≥4.5, p5 ≥4.0
- 95th-percentile sample ≤4096 tokens
