"""Lightweight structured metrics for the agent loop.

Emits JSONL lines that can be scraped by external collectors. No external
dependencies — everything is stdlib.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTOGRAM_BUCKETS = 100


@dataclass(slots=True)
class _Histogram:
    """Rolling histogram with fixed-size buckets."""

    buckets: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_HISTOGRAM_BUCKETS))

    def observe(self, value: float) -> None:
        self.buckets.append(value)

    @property
    def count(self) -> int:
        return len(self.buckets)

    @property
    def p50(self) -> float | None:
        if not self.buckets:
            return None
        return sorted(self.buckets)[len(self.buckets) // 2]

    @property
    def p99(self) -> float | None:
        if not self.buckets:
            return None
        s = sorted(self.buckets)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def mean(self) -> float | None:
        if not self.buckets:
            return None
        return sum(self.buckets) / len(self.buckets)


@dataclass
class LoopMetrics:
    """Session-level metrics accumulator with histograms and counters.

    Designed to be cheap to update on every loop iteration — all operations
    are O(1) except percentile reads (which sort at most 100 elements).
    """

    # Counters
    iterations: int = 0
    tool_calls_total: int = 0
    tool_errors_total: int = 0
    tool_denials_total: int = 0
    speculative_hits: int = 0
    speculative_misses: int = 0
    must_fix_injections: int = 0
    compactions: int = 0

    # Timing histograms (milliseconds)
    _loop_duration_ms: _Histogram = field(default_factory=_Histogram)
    _tool_latency_ms: _Histogram = field(default_factory=_Histogram)
    _llm_latency_ms: _Histogram = field(default_factory=_Histogram)

    # Token tracking
    _token_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=60))
    _last_token_time: float = field(default_factory=time.monotonic)

    # State
    start_time: float = field(default_factory=time.monotonic)

    def record_iteration(self, duration_sec: float) -> None:
        self.iterations += 1
        self._loop_duration_ms.observe(duration_sec * 1000)

    def record_tool_call(self, name: str, duration_sec: float, is_error: bool = False) -> None:
        self.tool_calls_total += 1
        self._tool_latency_ms.observe(duration_sec * 1000)
        if is_error:
            self.tool_errors_total += 1

    def record_tool_denial(self) -> None:
        self.tool_denials_total += 1

    def record_speculative_hit(self) -> None:
        self.speculative_hits += 1

    def record_speculative_miss(self) -> None:
        self.speculative_misses += 1

    def record_llm_call(self, duration_sec: float) -> None:
        self._llm_latency_ms.observe(duration_sec * 1000)

    def record_token_count(self, tokens: int) -> None:
        now = time.monotonic()
        self._token_samples.append((now, tokens))
        self._last_token_time = now

    def record_compaction(self) -> None:
        self.compactions += 1

    def record_must_fix(self) -> None:
        self.must_fix_injections += 1

    @property
    def token_velocity_per_min(self) -> float | None:
        """Tokens per minute based on the last 60 samples."""
        if len(self._token_samples) < 2:
            return None
        first_t, first_n = self._token_samples[0]
        last_t, last_n = self._token_samples[-1]
        delta_t = last_t - first_t
        delta_n = last_n - first_n
        if delta_t <= 0:
            return None
        return (delta_n / delta_t) * 60

    @property
    def speculative_hit_rate(self) -> float:
        total = self.speculative_hits + self.speculative_misses
        if total == 0:
            return 0.0
        return self.speculative_hits / total

    @property
    def duration_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON emission."""
        return {
            "iterations": self.iterations,
            "tool_calls_total": self.tool_calls_total,
            "tool_errors_total": self.tool_errors_total,
            "tool_denials_total": self.tool_denials_total,
            "speculative_hits": self.speculative_hits,
            "speculative_misses": self.speculative_misses,
            "speculative_hit_rate": round(self.speculative_hit_rate, 3),
            "must_fix_injections": self.must_fix_injections,
            "compactions": self.compactions,
            "duration_seconds": round(self.duration_seconds, 2),
            "token_velocity_per_min": (
                round(v, 1) if (v := self.token_velocity_per_min) is not None else None
            ),
            "loop_duration_ms": {
                "count": self._loop_duration_ms.count,
                "p50": round(p, 2) if (p := self._loop_duration_ms.p50) is not None else None,
                "p99": round(p, 2) if (p := self._loop_duration_ms.p99) is not None else None,
                "mean": round(m, 2) if (m := self._loop_duration_ms.mean) is not None else None,
            },
            "tool_latency_ms": {
                "count": self._tool_latency_ms.count,
                "p50": round(p, 2) if (p := self._tool_latency_ms.p50) is not None else None,
                "p99": round(p, 2) if (p := self._tool_latency_ms.p99) is not None else None,
                "mean": round(m, 2) if (m := self._tool_latency_ms.mean) is not None else None,
            },
            "llm_latency_ms": {
                "count": self._llm_latency_ms.count,
                "p50": round(p, 2) if (p := self._llm_latency_ms.p50) is not None else None,
                "p99": round(p, 2) if (p := self._llm_latency_ms.p99) is not None else None,
                "mean": round(m, 2) if (m := self._llm_latency_ms.mean) is not None else None,
            },
        }


class MetricsSink:
    """JSONL metrics sink that writes to a file or stdout.

    Lightweight — no threading, no buffering beyond Python's file IO.
    Each call to emit() writes a single JSON line.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._file: Any | None = None

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        line = json.dumps({"ts": time.time(), "event": event_type, **payload}, default=str)
        if self._path is not None:
            try:
                if self._file is None:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
                self._file.write(line + "\n")
            except OSError as exc:
                logger.debug("Metrics emit failed: %s", exc)
        else:
            logger.debug("metrics %s", line)

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    def __enter__(self) -> MetricsSink:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
