"""NVIDIA NIM API key rotation with rate-limit aware dispatch.

Free-tier NIM keys get ~30 RPM each. This module:
1. Manages a pool of API keys, tracking RPM usage per key.
2. Selects the least-constrained key for each request (round-robin + RPM awareness).
3. Handles 429 responses by rotating to the next key with backoff.
4. Integrates transparently with the existing LLMClient via environment variable swap.

Usage:
    from godspeed.benchmarks.nim_key_rotation import NIMKeyManager

    manager = NIMKeyManager.from_env()  # reads NVIDIA_NIM_API_KEYS=key1,key2,key3
    async with manager.key_context() as api_key:
        response = await llm_client.chat(...)

    # Or for CLI: set the active key before spawning subprocesses
    manager.rotate_on_429(response_status, headers)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RPM_LIMIT_DEFAULT = 30
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_BACKOFF_MAX = 60.0
DEFAULT_JITTER = 0.25


@dataclass
class _KeySlot:
    key: str
    requests_this_minute: int = 0
    window_start: float = field(default_factory=time.monotonic)
    cooldown_until: float = 0.0
    consecutive_429s: int = 0

    @property
    def rpm_available(self) -> int:
        now = time.monotonic()
        if now - self.window_start >= 60.0:
            return RPM_LIMIT_DEFAULT  # fresh window
        return max(0, RPM_LIMIT_DEFAULT - self.requests_this_minute)

    @property
    def is_cooling_down(self) -> bool:
        return time.monotonic() < self.cooldown_until

    def record_request(self) -> None:
        now = time.monotonic()
        if now - self.window_start >= 60.0:
            self.window_start = now
            self.requests_this_minute = 0
        self.requests_this_minute += 1


class NIMKeyManager:
    """Manages a pool of NVIDIA NIM API keys with rate-limit aware dispatching.

    Keys are consumed round-robin, preferring keys with available RPM budget.
    Keys that receive a 429 are temporarily cooled down with exponential backoff.
    """

    def __init__(
        self,
        keys: list[str],
        rpm_limit: int = RPM_LIMIT_DEFAULT,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        jitter: float = DEFAULT_JITTER,
    ):
        if not keys:
            raise ValueError("At least one API key is required")
        self._slots = {k: _KeySlot(key=k) for k in keys}
        self._order = deque(keys)
        self._rpm_limit = rpm_limit
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._jitter = jitter
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls,
        env_var: str = "NVIDIA_NIM_API_KEYS",
        fallback_env: str = "NVIDIA_NIM_API_KEY",
    ) -> NIMKeyManager:
        """Create from comma-separated environment variable or single fallback key.

        Examples:
            NVIDIA_NIM_API_KEYS=nvapi-key1,nvapi-key2,nvapi-key3
            NVIDIA_NIM_API_KEY=nvapi-single-key  (single-key fallback)
        """
        raw = os.environ.get(env_var, "")
        if raw:
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        else:
            single = os.environ.get(fallback_env, "")
            keys = [single.strip()] if single.strip() else []
        if not keys:
            raise ValueError(
                f"No API keys found. Set {env_var} (comma-separated) "
                f"or {fallback_env} (single key)."
            )
        return cls(keys=keys)

    @classmethod
    def from_config_file(cls, path: str | Path) -> NIMKeyManager:
        """Load keys from a text file, one key per line."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"API key file not found: {p}")
        keys = [
            line.strip()
            for line in p.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return cls(keys=keys)

    @property
    def key_count(self) -> int:
        return len(self._slots)

    async def get_key(self) -> str:
        """Return the best available API key based on RPM budget and cooldown state.

        Prefers keys with:
        1. Not in cooldown
        2. Highest remaining RPM budget
        """
        async with self._lock:
            return self._select_best_key()

    def _select_best_key(self) -> str:
        best_key = None
        best_rpm = -1

        for key in self._order:
            slot = self._slots[key]
            if slot.is_cooling_down:
                continue
            rpm = slot.rpm_available
            if rpm > best_rpm:
                best_rpm = rpm
                best_key = key

        if best_key is None:
            # All keys cooling down — pick the one with earliest cooldown end
            best_key = min(self._order, key=lambda k: self._slots[k].cooldown_until)
            cooldown_remaining = self._slots[best_key].cooldown_until - time.monotonic()
            logger.warning(
                "All %d keys in cooldown. Using %s (%.1fs remaining)",
                self.key_count,
                best_key[-8:],
                cooldown_remaining,
            )

        # Rotate the selected key to back of queue for distribution
        self._order.remove(best_key)
        self._order.append(best_key)

        self._slots[best_key].record_request()
        return best_key

    async def report_429(self, key: str) -> float:
        """Report a 429 response for the given key.

        Returns the backoff duration in seconds. The caller should sleep this
        long before retrying with a different key.
        """
        async with self._lock:
            slot = self._slots[key]
            slot.consecutive_429s += 1
            cooldown = min(
                self._backoff_base * (2 ** (slot.consecutive_429s - 1)),
                self._backoff_max,
            )
            cooldown *= 1.0 + random.uniform(-self._jitter, self._jitter)  # noqa: S311
            slot.cooldown_until = time.monotonic() + cooldown
            logger.info(
                "Key %s rate-limited (429 #%d). Cooldown: %.1fs",
                key[-8:],
                slot.consecutive_429s,
                cooldown,
            )
            return cooldown

    async def report_success(self, key: str) -> None:
        """Report a successful request — resets consecutive 429 counter for this key."""
        async with self._lock:
            slot = self._slots[key]
            if slot.consecutive_429s > 0:
                slot.consecutive_429s = 0
                slot.cooldown_until = 0.0

    def stats(self) -> dict[str, int]:
        """Return per-key RPM usage stats for monitoring."""
        return {key[-8:]: self._slots[key].requests_this_minute for key in sorted(self._slots)}

    @asynccontextmanager
    async def key_context(self) -> AsyncIterator[str]:
        """Async context manager that sets NVIDIA_NIM_API_KEY for the duration.

        Usage:
            async with manager.key_context() as api_key:
                # NVIDIA_NIM_API_KEY is set to api_key for the duration
                await llm_client.chat(...)
        """
        key = await self.get_key()
        old_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
        os.environ["NVIDIA_NIM_API_KEY"] = key
        try:
            yield key
        finally:
            if old_key:
                os.environ["NVIDIA_NIM_API_KEY"] = old_key
            else:
                os.environ.pop("NVIDIA_NIM_API_KEY", None)
