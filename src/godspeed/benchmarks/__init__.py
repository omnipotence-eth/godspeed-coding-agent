"""Godspeed benchmark infrastructure.

Provides:
- Base runner class for all benchmarks
- SWE-bench runner (Verified + Lite)
- API key rotation integration
- Cost tracking and per-instance metrics
"""

from __future__ import annotations

from godspeed.benchmarks.nim_key_rotation import NIMKeyManager

__all__ = ["NIMKeyManager"]
