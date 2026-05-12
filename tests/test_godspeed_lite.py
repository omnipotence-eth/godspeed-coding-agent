"""Tests for Godspeed Lite agent."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.lite.agent import (
    MODES,
    EMPTY_PATCH_RETRY,
    LITE_SYSTEM_PROMPT,
    BUDGET_PROMPT,
    _extract_command,
    _load_agents_md,
    LiteMode,
    GodspeedLite,
)


class TestLiteMode:
    def test_smart_max_steps(self):
        assert MODES["smart"].max_steps == 40

    def test_rush_is_faster(self):
        assert MODES["rush"].max_steps < MODES["smart"].max_steps

    def test_deep_is_most_thorough(self):
        assert MODES["deep"].max_steps > MODES["smart"].max_steps

    def test_custom_model_overrides_default(self):
        cfg = LiteMode(name="test", model="custom/model")
        assert cfg.model == "custom/model"


class TestCommandExtraction:
    def test_extracts_backtick_command(self):
        content = "Let me check files.\n```bash\nls -la\n```"
        assert _extract_command(content) == "ls -la"

    def test_extracts_no_language_tag(self):
        content = "Running:\n```\ngrep -rn test .\n```"
        assert _extract_command(content) == "grep -rn test ."

    def test_extracts_first_match_only(self):
        content = "```\nls\n```\n```\npwd\n```"
        assert _extract_command(content) == "ls"

    def test_fallback_to_known_prefix(self):
        content = "git status"
        assert _extract_command(content) == "git status"

    def test_returns_none_for_plain_text(self):
        content = "I think the fix is in foo.py. Let me look at it."
        assert _extract_command(content) is None

    def test_handles_empty_content(self):
        assert _extract_command("") is None


class TestAgentsMdLoading:
    def test_finds_agents_md_in_cwd(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("Run tests: pytest")
        result = _load_agents_md(tmp_path)
        assert "pytest" in result

    def test_finds_claude_md_fallback(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("Build: make")
        result = _load_agents_md(tmp_path)
        assert "Build: make" in result

    def test_returns_empty_when_none_found(self, tmp_path: Path):
        result = _load_agents_md(tmp_path)
        assert result == ""


class TestGodspeedLite:
    def test_init_defaults(self):
        agent = GodspeedLite()
        assert agent._cfg.name == "smart"
        assert agent._cfg.max_steps == 40

    def test_accepts_custom_mode(self):
        agent = GodspeedLite(mode="rush")
        assert agent._cfg.max_steps == 15

    def test_model_override(self):
        agent = GodspeedLite(model="openai/gpt-oh")
        assert agent._cfg.model == "openai/gpt-oh"

    def test_pick_model_single(self):
        agent = GodspeedLite()
        model = agent._pick_model()
        assert model == agent._cfg.model

    def test_pick_model_roulette(self):
        agent = GodspeedLite(roulette_models=["m1", "m2"])
        models = {agent._pick_model() for _ in range(20)}
        assert len(models) >= 2  # roulette actually switches

    @pytest.mark.asyncio
    async def test_run_produces_patch_on_submit(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=MagicMock(
                content="SUBMIT_PATCH",
                usage={"cost_usd": 0.001},
            )
        )
        with patch("godspeed.lite.agent.LLMClient", return_value=mock_client):
            agent = GodspeedLite()
            agent._run_bash = MagicMock(return_value="ok")  # type: ignore[method-assign]
            agent._capture_diff = MagicMock(return_value="diff --git a/out.txt")  # type: ignore[method-assign]
            result = await agent.run("fix it")
            assert "diff" in result

    @pytest.mark.asyncio
    async def test_empty_patch_retries(self):
        call_count = 0

        def make_response(content: str, cost: float = 0.0):
            r = MagicMock()
            r.content = content
            r.usage = {"cost_usd": cost}
            return r

        async def side_effect(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_response("SUBMIT_PATCH")
            return make_response("Let me try again.\n```bash\necho ok\n```\nSUBMIT_PATCH")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=side_effect)

        with patch("godspeed.lite.agent.LLMClient", return_value=mock_client):
            agent = GodspeedLite(max_steps=3)
            agent._run_bash = MagicMock(return_value="ok")  # type: ignore[method-assign]
            agent._capture_diff = MagicMock(side_effect=["", "diff fix"])  # type: ignore[method-assign]
            result = await agent.run("fix it")
            assert "diff" in result
            assert call_count >= 2

    def test_run_bash_executes(self):
        agent = GodspeedLite()
        output = agent._run_bash("echo hello")
        assert "hello" in output

    def test_run_bash_truncates_timeout(self):
        agent = GodspeedLite(step_timeout=1)
        if sys.platform == "win32":
            output = agent._run_bash("ping -n 10 127.0.0.1 > nul")
        else:
            output = agent._run_bash("sleep 10")
        assert "timed out" in output.lower() or "timeout" in output.lower()

    def test_capture_diff_no_repo(self):
        agent = GodspeedLite()
        diff = agent._capture_diff()
        # May return empty or error — should be a string
        assert isinstance(diff, str)

    def test_detect_test_framework_python(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
        agent = GodspeedLite(workdir=tmp_path)
        result = agent._detect_test_framework()
        assert "pytest" in result

    def test_detect_test_framework_js(self, tmp_path: Path):
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        agent = GodspeedLite(workdir=tmp_path)
        result = agent._detect_test_framework()
        assert "npm test" in result

    def test_cost_and_steps_tracking(self):
        agent = GodspeedLite()
        assert agent.cost_usd == 0.0
        assert agent.steps_taken == 0

    def test_system_prompt_constant(self):
        assert len(LITE_SYSTEM_PROMPT) > 200
        assert "bash" in LITE_SYSTEM_PROMPT.lower()
        assert "SUBMIT_PATCH" in LITE_SYSTEM_PROMPT

    def test_budget_prompt_constant(self):
        assert "SUBMIT_PATCH" in BUDGET_PROMPT

    def test_empty_patch_retry_constant(self):
        assert "empty" in EMPTY_PATCH_RETRY.lower()
