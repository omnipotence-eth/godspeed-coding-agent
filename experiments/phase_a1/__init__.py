"""Godspeed Phase A1 — synthetic tool-calling training data pipeline.

Multi-provider free-tier synthesis: Cerebras GLM-4.6 → Z.ai GLM-4.5-Flash →
Groq Llama-3.3-70B → local Ollama cascade. Output: 6.2K samples in OpenAI
format ({messages, tools}) consumable by the ml-lab training pipeline's
``messages_raw`` reader.
"""

from __future__ import annotations
