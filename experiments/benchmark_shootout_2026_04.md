# Godspeed Benchmark Shootout — 2026-04

Comparison across local Ollama and NVIDIA NIM free-tier models. Each run used the 20-task suite in `benchmarks/tasks.jsonl` with the polished fixtures in `benchmarks/fixtures/`.

| Model | Overall | Pass (J>=0.6) | Easy | Medium | Hard | Mech | Waste | tok/s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nvidia_nim/qwen/qwen3.5-397b-a17b` | 0.608 | 11/20 | 0.840 | 0.831 | 0.189 | 7/13 | 0.054 | 10.3 | 556.3 |
| `nvidia_nim/moonshotai/kimi-k2.5` | 0.548 | 9/20 | 0.840 | 0.727 | 0.135 | 6/13 | 0.043 | 13.5 | 519.8 |
| `nvidia_nim/mistralai/devstral-2-123b-instruct-2512` | 0.446 | 5/20 | 0.450 | 0.473 | 0.413 | 2/13 | 0.043 | 23.2 | 290.9 |
| `nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct` | 0.333 | 5/20 | 0.870 | 0.138 | 0.174 | 2/13 | 0.025 | 3.9 | 575.8 |
| `ollama/qwen3-coder:latest` | 0.107 | 1/20 | 0.150 | 0.125 | 0.057 | 1/13 | 0.041 | 10.7 | 770.5 |

**Winner (highest overall):** `nvidia_nim/qwen/qwen3.5-397b-a17b` — 0.608 overall, 11/20 tasks pass Jaccard>=0.6.

**Columns:** *Mech* = tasks where the `verify.py` hook confirmed mechanical success. *Waste* = mean `waste_penalty` (higher is worse, capped at 0.3). *tok/s* = mean output tokens per second.
