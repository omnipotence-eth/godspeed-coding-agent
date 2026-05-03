"""Tests for godspeed.tools.ollama_manager."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import RiskLevel, ToolResult
from godspeed.tools.ollama_manager import (
    OllamaModelInfo,
    OllamaTool,
    _run_ollama,
    delete_model,
    ensure_model_pulled,
    is_model_installed,
    list_models,
    pull_model,
    show_model,
)

# ─── OllamaModelInfo ────────────────────────────────────────


class TestOllamaModelInfo:
    def test_size_gb_from_bytes(self):
        info = OllamaModelInfo(name="test", size_bytes=1_073_741_824)  # 1 GB
        assert abs(info.size_gb - 1.0) < 0.01

    def test_size_gb_zero(self):
        info = OllamaModelInfo(name="test")
        assert info.size_gb == 0.0

    def test_name(self):
        info = OllamaModelInfo(name="llama3:8b", size_bytes=4_000_000_000)
        assert info.name == "llama3:8b"


# ─── _run_ollama ───────────────────────────────────────────


class TestRunOllama:
    def test_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
            rc, _stdout, _stderr = _run_ollama(["list"])
            assert rc == 0
            assert _stdout == "ok\n"
            mock_run.assert_called_once()

    def test_not_installed(self):
        with patch("shutil.which", return_value=None):
            from godspeed.tools.ollama_manager import _ollama_installed

            assert _ollama_installed() is False

    def test_timeout(self):
        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("ollama", 1)

        with patch("subprocess.run", side_effect=_raise_timeout):
            rc, _stdout, stderr = _run_ollama(["pull", "test"], timeout=1)
            assert rc == -2
            assert "timed out" in stderr

    def test_file_not_found(self):
        def _raise_filenotfound(*args, **kwargs):
            raise FileNotFoundError("ollama not found")

        with patch("subprocess.run", side_effect=_raise_filenotfound):
            rc, _stdout, stderr = _run_ollama(["list"])
            assert rc == -1
            assert "not installed" in stderr


# ─── list_models ────────────────────────────────────────────


class TestListModels:
    def test_returns_list_when_ollama_missing(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            result = list_models()
            assert result == []

    def test_parses_model_list(self):
        stdout = "llama3:8b  4.2 GB  2 hours ago\nqwen2.5-coder:14b  8.1 GB  1 day ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 2
                assert models[0].name == "llama3:8b"
                assert abs(models[0].size_gb - 4.2) < 0.5
                assert models[1].name == "qwen2.5-coder:14b"

    def test_parses_size_mb(self):
        stdout = "tiny:1b  512 MB  1 hour ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert abs(models[0].size_gb - 0.5) < 0.1

    def test_handles_failure(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(1, "", "error")):
                result = list_models()
                assert result == []

    def test_skips_empty_lines(self):
        stdout = "\n  \nllama3:8b  4 GB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1


# ─── pull_model ─────────────────────────────────────────────


class TestPullModel:
    def test_success(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "subprocess.run", return_value=MagicMock(returncode=0, stdout="done", stderr="")
            ):
                assert pull_model("llama3:8b") is True

    def test_failure(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "subprocess.run",
                return_value=MagicMock(returncode=1, stdout="", stderr="pull failed"),
            ):
                assert pull_model("bad-model") is False

    def test_not_installed(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            assert pull_model("llama3:8b") is False

    def test_with_progress_callback(self):
        callback = MagicMock()
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "subprocess.run", return_value=MagicMock(returncode=0, stdout="done", stderr="")
            ):
                result = pull_model("llama3:8b", on_progress=callback)
                assert result is True

    def test_timeout(self):
        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("ollama", 600)

        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("subprocess.run", side_effect=_raise_timeout):
                assert pull_model("llama3:8b") is False


# ─── delete_model ───────────────────────────────────────────


class TestDeleteModel:
    def test_success(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "godspeed.tools.ollama_manager._run_ollama", return_value=(0, "deleted", "")
            ):
                success, msg = delete_model("llama3:8b")
                assert success is True
                assert "deleted" in msg

    def test_failure(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "godspeed.tools.ollama_manager._run_ollama", return_value=(1, "", "not found")
            ):
                success, msg = delete_model("bad-model")
                assert success is False
                assert "not found" in msg

    def test_not_installed(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            success, msg = delete_model("llama3:8b")
            assert success is False
            assert "not installed" in msg


# ─── show_model ─────────────────────────────────────────────


class TestShowModel:
    def test_success(self):
        stdout = "architecture    llama3\nparameters      8B\ncontext length  8192\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("llama3:8b")
                assert info is not None
                assert info["name"] == "llama3:8b"
                assert info["architecture"] == "llama3"

    def test_not_found(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "godspeed.tools.ollama_manager._run_ollama", return_value=(1, "", "not found")
            ):
                assert show_model("bad-model") is None

    def test_not_installed(self):
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            assert show_model("llama3:8b") is None

    def test_skips_indented_lines(self):
        stdout = "architecture    llama3\n  some indented line\nparameters      8B\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("llama3:8b")
                assert "some indented line" not in info


# ─── is_model_installed ────────────────────────────────────


class TestIsModelInstalled:
    def test_found(self):
        with patch("godspeed.tools.ollama_manager.list_models") as mock_list:
            mock_list.return_value = [
                OllamaModelInfo(name="llama3:8b"),
                OllamaModelInfo(name="qwen2.5:14b"),
            ]
            assert is_model_installed("llama3:8b") is True
            assert is_model_installed("nonexistent") is False

    def test_empty_list(self):
        with patch("godspeed.tools.ollama_manager.list_models", return_value=[]):
            assert is_model_installed("llama3:8b") is False


# ─── ensure_model_pulled ───────────────────────────────────


class TestEnsureModelPulled:
    def test_already_installed(self):
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=True):
            assert ensure_model_pulled("llama3:8b") is True

    def test_needs_pull_success(self):
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=False):
            with patch("godspeed.tools.ollama_manager.pull_model", return_value=True):
                assert ensure_model_pulled("llama3:8b") is True

    def test_needs_pull_failure(self):
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=False):
            with patch("godspeed.tools.ollama_manager.pull_model", return_value=False):
                assert ensure_model_pulled("llama3:8b") is False

    def test_with_progress_callback(self):
        callback = MagicMock()
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=False):
            with patch("godspeed.tools.ollama_manager.pull_model", return_value=True):
                result = ensure_model_pulled("llama3:8b", on_progress=callback)
                assert result is True


# ─── OllamaTool (agent tool wrapper) ─────────────────────


class TestOllamaTool:
    def test_tool_metadata(self):
        tool = OllamaTool()
        assert tool.name == "ollama"
        assert tool.risk_level == RiskLevel.LOW
        assert "list" in tool.description.lower()

    def test_get_schema(self):
        tool = OllamaTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert "model" in schema["properties"]
        assert "action" in schema["required"]

    @pytest.mark.asyncio
    async def test_execute_list(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.list_models", return_value=[]):
            result = await tool.execute({"action": "list"}, MagicMock())
            assert isinstance(result, ToolResult)
            assert "No local models" in result.output

    @pytest.mark.asyncio
    async def test_execute_list_with_models(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.list_models") as mock_list:
            mock_list.return_value = [
                OllamaModelInfo(name="llama3:8b", size_bytes=4_000_000_000),
            ]
            result = await tool.execute({"action": "list"}, MagicMock())
            assert isinstance(result, ToolResult)
            assert "llama3:8b" in result.output

    @pytest.mark.asyncio
    async def test_execute_pull_no_model(self):
        tool = OllamaTool()
        result = await tool.execute({"action": "pull"}, MagicMock())
        assert result.is_error is True
        assert "required" in result.error.lower() or "model" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_pull_success(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.pull_model", return_value=True):
            result = await tool.execute({"action": "pull", "model": "llama3:8b"}, MagicMock())
            assert result.is_error is False
            assert "Successfully" in result.output

    @pytest.mark.asyncio
    async def test_execute_pull_failure(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.pull_model", return_value=False):
            result = await tool.execute({"action": "pull", "model": "bad"}, MagicMock())
            assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_show_no_model(self):
        tool = OllamaTool()
        result = await tool.execute({"action": "show"}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_show_success(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.show_model") as mock_show:
            mock_show.return_value = {"name": "llama3:8b", "architecture": "llama3"}
            result = await tool.execute({"action": "show", "model": "llama3:8b"}, MagicMock())
            assert result.is_error is False
            assert "llama3:8b" in result.output

    @pytest.mark.asyncio
    async def test_execute_show_not_found(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.show_model", return_value=None):
            result = await tool.execute({"action": "show", "model": "bad"}, MagicMock())
            assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_delete_no_model(self):
        tool = OllamaTool()
        result = await tool.execute({"action": "delete"}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_delete_success(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.delete_model", return_value=(True, "Deleted")):
            result = await tool.execute({"action": "delete", "model": "llama3:8b"}, MagicMock())
            assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_delete_failure(self):
        tool = OllamaTool()
        with patch("godspeed.tools.ollama_manager.delete_model", return_value=(False, "Error")):
            result = await tool.execute({"action": "delete", "model": "bad"}, MagicMock())
            assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        tool = OllamaTool()
        result = await tool.execute({"action": "unknown"}, MagicMock())
        assert result.is_error is True
        assert "Unknown action" in result.error
