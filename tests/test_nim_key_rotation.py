"""Tests for NVIDIA NIM API key rotation manager."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.benchmarks.nim_key_rotation import NIMKeyManager, _KeySlot


class TestKeySlot:
    def test_rpm_available_fresh_window(self):
        slot = _KeySlot(key="test-key")
        slot.requests_this_minute = 25
        slot.window_start = time.monotonic() - 61
        assert slot.rpm_available == 30

    def test_rpm_available_partial_used(self):
        slot = _KeySlot(key="test-key")
        slot.requests_this_minute = 10
        slot.window_start = time.monotonic()
        assert slot.rpm_available == 20

    def test_rpm_available_exhausted(self):
        slot = _KeySlot(key="test-key")
        slot.requests_this_minute = 30
        slot.window_start = time.monotonic()
        assert slot.rpm_available == 0

    def test_record_request_resets_window(self):
        slot = _KeySlot(key="test-key")
        slot.requests_this_minute = 29
        slot.window_start = time.monotonic() - 61
        slot.record_request()
        assert slot.requests_this_minute == 1
        assert slot.window_start <= time.monotonic()

    def test_cooldown_tracking(self):
        slot = _KeySlot(key="test-key")
        assert not slot.is_cooling_down
        slot.cooldown_until = time.monotonic() + 10
        assert slot.is_cooling_down


class TestNIMKeyManager:
    def test_requires_at_least_one_key(self):
        with pytest.raises(ValueError):
            NIMKeyManager(keys=[])

    def test_key_count(self):
        mgr = NIMKeyManager(keys=["key1", "key2", "key3"])
        assert mgr.key_count == 3

    async def test_get_key_round_robin(self):
        mgr = NIMKeyManager(keys=["k1", "k2", "k3"])
        keys = [await mgr.get_key() for _ in range(6)]
        expected = ["k1", "k2", "k3", "k1", "k2", "k3"]
        assert keys == expected

    async def test_report_429_sets_cooldown(self):
        mgr = NIMKeyManager(keys=["k1", "k2"])
        await mgr.get_key()  # use k1
        key = await mgr.get_key()  # use k2
        cooldown = await mgr.report_429(key)
        assert cooldown > 0
        assert mgr._slots[key].is_cooling_down

    async def test_report_success_resets_429_counter(self):
        mgr = NIMKeyManager(keys=["k1", "k2"])
        key = await mgr.get_key()
        await mgr.report_429(key)
        assert mgr._slots[key].consecutive_429s == 1
        await mgr.report_success(key)
        assert mgr._slots[key].consecutive_429s == 0
        assert not mgr._slots[key].is_cooling_down

    async def test_skips_cooling_down_keys(self):
        mgr = NIMKeyManager(keys=["k1", "k2"])
        k1 = await mgr.get_key()
        await mgr.report_429(k1)
        # k1 is now cooling down, should get k2
        next_key = await mgr.get_key()
        assert next_key == "k2"

    async def test_stats_tracks_rpm(self):
        mgr = NIMKeyManager(keys=["key_a", "key_b"])
        for _ in range(5):
            await mgr.get_key()
        stats = mgr.stats()
        assert sum(stats.values()) == 5

    async def test_key_context_sets_env(self):
        mgr = NIMKeyManager(keys=["test-key-123"])
        assert (
            "NVIDIA_NIM_API_KEY" not in os.environ
            or os.environ["NVIDIA_NIM_API_KEY"] != "test-key-123"
        )
        async with mgr.key_context() as key:
            assert key == "test-key-123"
            assert os.environ["NVIDIA_NIM_API_KEY"] == "test-key-123"
        # Should restore previous state
        assert os.environ.get("NVIDIA_NIM_API_KEY", "") != "test-key-123"

    def test_from_env_single_key(self):
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "single-key"}):
            mgr = NIMKeyManager.from_env()
            assert mgr.key_count == 1
            assert mgr._slots["single-key"] is not None

    def test_from_env_multiple_keys(self):
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEYS": "k1, k2 , k3"}):
            mgr = NIMKeyManager.from_env()
            assert mgr.key_count == 3

    def test_from_env_raises_when_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API keys found"):
                NIMKeyManager.from_env()

    def test_from_config_file(self, tmp_path: Path):
        cfg = tmp_path / "keys.txt"
        cfg.write_text("  key-alpha  \n\n# comment\n  key-beta\n")
        mgr = NIMKeyManager.from_config_file(cfg)
        assert mgr.key_count == 2
