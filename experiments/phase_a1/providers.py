"""Multi-provider async LLM router with daily-quota tracking and cascade fallback.

Providers:
  - ``cerebras``  → Cerebras free tier (GLM-4.6 / Qwen3-235B)     1M tok/day, 30 RPM
  - ``zai``       → Z.ai free tier (GLM-4.5-Flash / GLM-4.7-Flash) ~1M tok/day
  - ``groq``      → Groq free tier (Llama-3.3-70B / Qwen QwQ)     1K req/day, 30 RPM
  - ``ollama``    → Local Ollama (Qwen2.5-Coder-32B-Q4)           unlimited
  - ``anthropic`` → Claude Sonnet 4.6 (anchor only)               paid

Model tiers (used by pipeline stages):
  - ``primary``   → blueprint + narrator generation (best quality)
  - ``secondary`` → same, different provider for resilience
  - ``judge``     → GLM-Flash scoring
  - ``overflow``  → spillover when primary/secondary rate-limited
  - ``anchor``    → Claude Sonnet (50-sample held-out set)

Quota state persists to SQLite at ``data/provider_quota.db``; counters reset
at each provider's local UTC midnight. Cascade order is tier-specific.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_env_local(dotenv_path: Path | None = None) -> None:
    """Populate ``os.environ`` from a sibling ``.env.local`` file if present.

    Only sets keys that are not already in the environment — real env vars
    always win. Simple ``KEY=VALUE`` parser; no quoting or variable expansion.
    The file is ``.env.local`` beside this module by default (gitignored via
    the repo's existing ``.env.*`` rule).
    """
    path = dotenv_path or (Path(__file__).parent / ".env.local")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_local()

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Static config for one provider."""

    name: str
    model: str
    daily_token_cap: int | None = None
    daily_request_cap: int | None = None
    rpm_cap: int | None = None
    supports_tool_schema: bool = True

    def is_local(self) -> bool:
        return self.name == "ollama"


# Models verified against the live free-tier keys supplied (Apr 2026).
# Cerebras access: qwen-3-235b-a22b-instruct-2507, llama3.1-8b.
# Z.ai access:    glm-4.7-flash, glm-4.5-flash.
# Paid-only on current keys: Cerebras gpt-oss-120b / zai-glm-4.7;
#                            Z.ai glm-4.5-air / glm-4.6.
CEREBRAS_QWEN = ProviderConfig(
    name="cerebras",
    model="qwen-3-235b-a22b-instruct-2507",  # top-tier open model for this key
    daily_token_cap=1_000_000,
    rpm_cap=30,
)
CEREBRAS_LLAMA_SMALL = ProviderConfig(
    name="cerebras",
    model="llama3.1-8b",  # small fast fallback on same provider
    daily_token_cap=1_000_000,
    rpm_cap=30,
)
ZAI_GLM_PRIMARY = ProviderConfig(
    name="zai",
    model="glm-4.7-flash",  # BFCL-champion tool calling (flash tier is free)
    daily_token_cap=1_000_000,
    rpm_cap=30,
)
ZAI_GLM_JUDGE = ProviderConfig(
    name="zai",
    model="glm-4.5-flash",  # lighter, consistent rubric application
    daily_token_cap=1_000_000,
    rpm_cap=30,
)
GROQ_OVERFLOW = ProviderConfig(
    name="groq",
    model="llama-3.3-70b-versatile",
    daily_request_cap=1_000,
    rpm_cap=30,
)
OLLAMA_LOCAL = ProviderConfig(
    # Smallest-viable default for 16GB VRAM. ~2GB download, ~3GB active.
    # Upgrade to ``qwen2.5-coder:7b`` (~4.4GB) if judge reject-rate is high on
    # Ollama-produced samples during the full run.
    name="ollama",
    model="qwen2.5-coder:3b",
    supports_tool_schema=False,
)
ANTHROPIC_ANCHOR = ProviderConfig(
    name="anthropic",
    model="claude-sonnet-4-6",
)


# Three independent quota pools: Cerebras (1M tok/day), Z.ai (~1M tok/day),
# Groq (1K req/day). Primary = Qwen-235B; secondary = GLM-4.7-Flash — opposing
# model families for data diversity. Judge is a different model than the
# generators on every sample to limit self-collusion.
TIER_CASCADES: dict[str, list[ProviderConfig]] = {
    "primary": [CEREBRAS_QWEN, ZAI_GLM_PRIMARY, GROQ_OVERFLOW, OLLAMA_LOCAL],
    "secondary": [ZAI_GLM_PRIMARY, CEREBRAS_QWEN, GROQ_OVERFLOW, OLLAMA_LOCAL],
    "judge": [ZAI_GLM_JUDGE, CEREBRAS_LLAMA_SMALL, GROQ_OVERFLOW],
    "overflow": [GROQ_OVERFLOW, OLLAMA_LOCAL, CEREBRAS_QWEN, ZAI_GLM_PRIMARY],
    "ollama_only": [OLLAMA_LOCAL],
    "anchor": [ANTHROPIC_ANCHOR],
}


# ---------------------------------------------------------------------------
# Quota tracker
# ---------------------------------------------------------------------------


class QuotaTracker:
    """SQLite-backed per-provider daily quota state. UTC midnight reset."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS usage (
        provider TEXT NOT NULL,
        utc_date TEXT NOT NULL,
        tokens_used INTEGER NOT NULL DEFAULT 0,
        requests_used INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (provider, utc_date)
    )
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(self._SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def record(self, provider: str, tokens: int, requests: int = 1) -> None:
        today = self._today()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO usage (provider, utc_date, tokens_used, requests_used) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(provider, utc_date) DO UPDATE SET "
                "  tokens_used = tokens_used + ?, "
                "  requests_used = requests_used + ?",
                (provider, today, tokens, requests, tokens, requests),
            )
            conn.commit()

    def get_usage(self, provider: str) -> tuple[int, int]:
        """Return (tokens_used, requests_used) for provider today."""
        today = self._today()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tokens_used, requests_used FROM usage WHERE provider=? AND utc_date=?",
                (provider, today),
            ).fetchone()
            return (row[0], row[1]) if row else (0, 0)

    def has_headroom(self, cfg: ProviderConfig, approx_input_tokens: int = 5000) -> bool:
        """Check if provider has enough quota for one more call."""
        if cfg.is_local():
            return True
        tokens_used, requests_used = self.get_usage(cfg.name)
        if (
            cfg.daily_token_cap is not None
            and tokens_used + approx_input_tokens > cfg.daily_token_cap
        ):
            return False
        if cfg.daily_request_cap is not None and requests_used + 1 > cfg.daily_request_cap:
            return False
        return True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Base for provider errors."""


class AllProvidersExhausted(ProviderError):
    """All providers in the cascade failed or are over quota."""


class ProviderAuthError(ProviderError):
    """API key missing or invalid."""


# ---------------------------------------------------------------------------
# LLM response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalized response across providers."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------


class _ProviderBackend:
    """Thin async wrapper around a provider SDK. Subclasses implement `_call`."""

    cfg: ProviderConfig

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    async def call(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> LLMResponse:
        start = time.perf_counter()
        text, input_tokens, output_tokens, meta = await self._call(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        latency = time.perf_counter() - start
        return LLMResponse(
            text=text,
            provider=self.cfg.name,
            model=self.cfg.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            metadata=meta,
        )

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int, dict[str, Any]]:
        raise NotImplementedError


class _CerebrasBackend(_ProviderBackend):
    """Cerebras Cloud SDK — OpenAI-compatible."""

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        from cerebras.cloud.sdk import AsyncCerebras

        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ProviderAuthError("CEREBRAS_API_KEY not set")
        self._client = AsyncCerebras(api_key=api_key)

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return (
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            {"finish_reason": getattr(choice, "finish_reason", None)},
        )


class _ZAIBackend(_ProviderBackend):
    """Z.ai SDK (GLM). OpenAI-compatible surface via AsyncOpenAI against Z.ai endpoint."""

    _BASE_URL = "https://api.z.ai/api/paas/v4/"

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        from openai import AsyncOpenAI

        api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("ZAI_API_KEY not set")
        self._client = AsyncOpenAI(api_key=api_key, base_url=self._BASE_URL)

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # GLM "flash" tier is a reasoning model by default. With thinking
            # enabled, max_tokens is split between hidden reasoning and final
            # answer — tight budgets produce an empty ``content``. Disable it
            # so ``max_tokens`` goes entirely to the answer.
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        # Fallback: if content still empty (e.g. thinking param ignored by a
        # future model variant), surface reasoning_content instead.
        text = choice.message.content or getattr(choice.message, "reasoning_content", "") or ""
        usage = resp.usage
        return (
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            {"finish_reason": getattr(choice, "finish_reason", None)},
        )


class _GroqBackend(_ProviderBackend):
    """Groq SDK — OpenAI-compatible."""

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        from groq import AsyncGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ProviderAuthError("GROQ_API_KEY not set")
        self._client = AsyncGroq(api_key=api_key)

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return (
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            {"finish_reason": getattr(choice, "finish_reason", None)},
        )


class _OllamaBackend(_ProviderBackend):
    """Local Ollama via OpenAI-compatible /v1 endpoint."""

    _BASE_URL = "http://localhost:11434/v1/"

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key="ollama", base_url=self._BASE_URL)

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return (
            text,
            input_tokens,
            output_tokens,
            {"finish_reason": getattr(choice, "finish_reason", None)},
        )


class _AnthropicBackend(_ProviderBackend):
    """Claude Sonnet-4.6 (anchor set only)."""

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        from anthropic import AsyncAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY not set")
        self._client = AsyncAnthropic(api_key=api_key)

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int, dict[str, Any]]:
        resp = await self._client.messages.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        if json_mode:
            # Strip common fencing defensively; Anthropic has no native JSON mode.
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
                if stripped.endswith("```"):
                    stripped = stripped[:-3]
                text = stripped.strip()
        usage = resp.usage
        return (
            text,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            {"stop_reason": getattr(resp, "stop_reason", None)},
        )


_BACKEND_CLASSES: dict[str, type[_ProviderBackend]] = {
    "cerebras": _CerebrasBackend,
    "zai": _ZAIBackend,
    "groq": _GroqBackend,
    "ollama": _OllamaBackend,
    "anthropic": _AnthropicBackend,
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ProviderRouter:
    """Tier-based async router with quota-aware cascade."""

    def __init__(
        self,
        quota_db: Path,
        *,
        max_retries: int = 3,
        tier_overrides: dict[str, list[ProviderConfig]] | None = None,
    ) -> None:
        self._tracker = QuotaTracker(quota_db)
        self._max_retries = max_retries
        self._cascades = dict(TIER_CASCADES)
        if tier_overrides:
            self._cascades.update(tier_overrides)
        self._backend_cache: dict[tuple[str, str], _ProviderBackend] = {}

    def _backend(self, cfg: ProviderConfig) -> _ProviderBackend:
        key = (cfg.name, cfg.model)
        if key not in self._backend_cache:
            backend_cls = _BACKEND_CLASSES[cfg.name]
            self._backend_cache[key] = backend_cls(cfg)
        return self._backend_cache[key]

    async def complete(
        self,
        *,
        tier: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Call the first available provider in the tier cascade."""
        cascade = self._cascades.get(tier)
        if not cascade:
            raise ValueError(f"Unknown tier: {tier!r}. Choices: {sorted(self._cascades)}")

        approx_input = max(1000, len(system) // 4 + len(user) // 4)
        last_error: Exception | None = None

        for cfg in cascade:
            if not self._tracker.has_headroom(cfg, approx_input_tokens=approx_input):
                logger.info("skip provider=%s reason=quota_exhausted", cfg.name)
                continue

            try:
                backend = self._backend(cfg)
            except ProviderAuthError as e:
                logger.info("skip provider=%s reason=no_auth (%s)", cfg.name, e)
                continue

            for attempt in range(self._max_retries):
                try:
                    resp = await backend.call(
                        system=system,
                        user=user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_mode=json_mode,
                    )
                    self._tracker.record(
                        cfg.name,
                        tokens=resp.input_tokens + resp.output_tokens,
                    )
                    return resp
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    transient = any(
                        s in msg
                        for s in ("rate", "429", "timeout", "temporarily", "overload", "502", "503")
                    )
                    if not transient or attempt == self._max_retries - 1:
                        logger.warning(
                            "provider=%s attempt=%d/%d failed: %s",
                            cfg.name,
                            attempt + 1,
                            self._max_retries,
                            e,
                        )
                        break
                    wait = 2**attempt + random.uniform(0, 1)
                    logger.info(
                        "provider=%s transient, backoff %.1fs (attempt %d/%d): %s",
                        cfg.name,
                        wait,
                        attempt + 1,
                        self._max_retries,
                        e,
                    )
                    await asyncio.sleep(wait)

            # Move to next cascade step
            continue

        raise AllProvidersExhausted(
            f"tier={tier!r} — all providers failed or quota-exhausted. Last error: {last_error}"
        )

    def usage_snapshot(self) -> dict[str, dict[str, int]]:
        """Return {provider: {tokens, requests}} for today."""
        snapshot: dict[str, dict[str, int]] = {}
        for name in _BACKEND_CLASSES:
            t, r = self._tracker.get_usage(name)
            snapshot[name] = {"tokens": t, "requests": r}
        return snapshot


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def default_router(data_dir: Path | None = None) -> ProviderRouter:
    """Build a router with quota DB under the pipeline data dir."""
    base = data_dir or Path(__file__).parent / "data"
    return ProviderRouter(quota_db=base / "provider_quota.db")


async def _cli_test() -> int:
    """Ad-hoc test: send one request to each available provider."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    router = default_router()
    for tier in ["ollama_only"]:
        try:
            resp = await router.complete(
                tier=tier,
                system="You are a concise assistant. Reply with only valid JSON.",
                user='Respond with exactly: {"ok": true}',
                max_tokens=32,
                temperature=0.0,
                json_mode=True,
            )
            logger.info("tier=%s provider=%s model=%s", tier, resp.provider, resp.model)
            logger.info("  text=%r", resp.text)
            logger.info(
                "  tokens in=%d out=%d latency=%.2fs",
                resp.input_tokens,
                resp.output_tokens,
                resp.latency_s,
            )
        except AllProvidersExhausted as e:
            logger.warning("tier=%s exhausted: %s", tier, e)
    logger.info("usage: %s", json.dumps(router.usage_snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(_cli_test()))
