"""Tests for godspeed.tools.ollama_manager."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.tools.base import RiskLevel, ToolResult
from godspeed.tools.ollama_manager import (
    OllamaModelInfo,
    OllamaTool,
    _ollama_installed,
    _run_ollama,
    delete_model,
    ensure_model_pulled,
    is_model_installed,
    list_models,
    pull_model,
    pull_model_async,
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


# ─── ListModels additional edge cases ─────────────────────────


class TestListModelsAdditional:
    """Cover missed branches in list_models parsing."""

    def test_skips_empty_line_parts(self):
        """Line 74: line with only whitespace parts yields empty parts list."""
        stdout = "\n  \nllama3:8b  4 GB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1

    def test_value_error_in_size_parsing_first_format(self):
        """Lines 93-94: ValueError in first size parsing format (e.g. '4.X' GB)."""
        stdout = "bad:1b  4.abc GB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert models[0].name == "bad:1b"
                assert models[0].size_bytes == 0

    def test_size_with_attached_suffix_gb(self):
        """Line 99: size directly with GB suffix (e.g. '4GB')."""
        stdout = "model:7b  4.5GB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert abs(models[0].size_gb - 4.5) < 0.1

    def test_size_with_attached_suffix_mb(self):
        """Line 100: size directly with MB suffix (e.g. '512MB')."""
        stdout = "tiny:1b  512MB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert abs(models[0].size_gb - 0.5) < 0.1

    def test_value_error_in_attached_suffix(self):
        """ValueError when attached GB/MB suffix has non-numeric prefix."""
        stdout = "bad:1b  abcGB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert models[0].size_bytes == 0

    def test_ago_parsing_with_remaining_parts(self):
        """Lines 104-105: 'ago' found and remaining parts appended as modified_at."""
        stdout = "model:7b  4 GB  2 days ago  something_after\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert "ago something_after" in models[0].modified_at

    def test_ago_parsing_at_end_of_line(self):
        """Ago parsing when 'ago' is at end of line — modified_at stays empty."""
        stdout = "model:7b  4 GB  ago\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                models = list_models()
                assert len(models) == 1
                assert models[0].modified_at == ""


# ─── PullModel with progress callback, not_installed, timeout ─────────


class TestPullModelAdditional:
    """Cover missed branches in pull_model with on_progress."""

    def test_not_installed_with_progress(self):
        """Line 120: on_progress called when ollama not installed."""
        callback = MagicMock()
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            result = pull_model("llama3:8b", on_progress=callback)
            assert result is False
            callback.assert_called_once_with("Ollama is not installed")

    def test_pull_failure_with_progress(self):
        """Line 134: on_progress called with truncated stderr on pull failure."""
        callback = MagicMock()
        long_stderr = "x" * 300
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "subprocess.run",
                return_value=MagicMock(returncode=1, stdout="", stderr=long_stderr),
            ):
                result = pull_model("bad", on_progress=callback)
                assert result is False
                callback.assert_called()
                arg = callback.call_args[0][0]
                assert "Pull failed" in arg
                assert len(arg) <= 215  # "Pull failed: " (14) + up to 200 chars

    def test_filenotfound_with_progress(self):
        """Line 137-139: FileNotFoundError calls on_progress."""
        callback = MagicMock()
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("subprocess.run", side_effect=FileNotFoundError("ollama")):
                result = pull_model("llama3:8b", on_progress=callback)
                assert result is False
                callback.assert_called_once_with("Ollama is not installed")

    def test_timeout_with_progress(self):
        """Line 142: TimeoutExpired calls on_progress."""
        callback = MagicMock()

        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("ollama", 600)

        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("subprocess.run", side_effect=_raise_timeout):
                result = pull_model("llama3:8b", on_progress=callback)
                assert result is False
                callback.assert_called_once_with("Pull timed out after 10min")

    def test_pull_success_silent_no_progress(self):
        """Pull success with no callback doesn't error."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ok", stderr=""),
            ):
                assert pull_model("llama3:8b") is True

    def test_file_not_found_no_progress(self):
        """FileNotFoundError without progress callback."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("subprocess.run", side_effect=FileNotFoundError("ollama")):
                assert pull_model("llama3:8b") is False

    def test_timeout_no_progress(self):
        """Timeout without progress callback."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("o", 600)):
                assert pull_model("llama3:8b") is False


# ─── pull_model_async ─────────────────────────────────────────────


class TestPullModelAsync:
    """Lines 154-202: full pull_model_async coverage."""

    @pytest.mark.asyncio
    async def test_not_installed(self):
        """ollama not installed."""
        callback = MagicMock()
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            result = await pull_model_async("llama3:8b", on_progress=callback)
            assert result is False
            callback.assert_called_once_with("Ollama is not installed")

    @pytest.mark.asyncio
    async def test_not_installed_no_callback(self):
        """ollama not installed, no callback."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=False):
            result = await pull_model_async("llama3:8b")
            assert result is False

    @pytest.mark.asyncio
    async def test_successful_pull(self):
        """Streaming pull completes successfully."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b'{"status":"downloading","completed":50,"total":100,"digest":"abc123def456"}\n',
                    b'{"status":"done"}\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                callback = MagicMock()
                result = await pull_model_async("llama3:8b", on_progress=callback)
                assert result is True
                assert callback.called

    @pytest.mark.asyncio
    async def test_successful_pull_no_progress_data(self):
        """Streaming pull with JSON that has no status."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b'{"not_status": "irrelevant"}\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                callback = MagicMock()
                result = await pull_model_async("llama3:8b", on_progress=callback)
                assert result is True
                # callback should not be called for no-status lines
                callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_pull_with_progress_percentage(self):
        """Progress with completed/total yields percentage."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b'{"status":"pulling","completed":25,"total":100,"digest":"sha256:abc123def456789012345678"}\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                callback = MagicMock()
                result = await pull_model_async("llama3:8b", on_progress=callback)
                assert result is True
                callback.assert_called()
                call_arg = callback.call_args[0][0]
                assert "25%" in call_arg

    @pytest.mark.asyncio
    async def test_pull_with_zero_total(self):
        """Progress with total=0 shouldn't divide by zero."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b'{"status":"downloading","completed":0,"total":0,"digest":"abc123"}\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                callback = MagicMock()
                result = await pull_model_async("llama3:8b", on_progress=callback)
                assert result is True
                callback.assert_called()
                call_arg = callback.call_args[0][0]
                assert "0%" in call_arg

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        """JSON decode error is caught and logged — doesn't crash."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b"not json at all\n",
                    b'{"status":"done"}\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await pull_model_async("llama3:8b")
                assert result is True

    @pytest.mark.asyncio
    async def test_unicode_decode_error(self):
        """Unicode decode error in progress is caught."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b'\x80\x81invalid utf8\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await pull_model_async("llama3:8b")
                assert result is True

    @pytest.mark.asyncio
    async def test_pull_failure_returncode(self):
        """Non-zero returncode returns False."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await pull_model_async("llama3:8b")
                assert result is False

    @pytest.mark.asyncio
    async def test_file_not_found_error(self):
        """FileNotFoundError during async pull."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError("no ollama"),
            ):
                callback = MagicMock()
                result = await pull_model_async("llama3:8b", on_progress=callback)
                assert result is False
                callback.assert_called_once_with("Ollama is not installed")

    @pytest.mark.asyncio
    async def test_null_stdout(self):
        """stdout is None despite PIPE (safety check)."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.stdout = None  # safety branch
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await pull_model_async("llama3:8b")
                assert result is False

    @pytest.mark.asyncio
    async def test_finally_kills_running_process(self):
        """Finally block kills process if still running."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = None  # still running
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=Exception("boom")  # cause early exit
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                with pytest.raises(Exception):
                    await pull_model_async("llama3:8b")
                mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_only_no_percent(self):
        """Status message without completed/total."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                side_effect=[
                    b'{"status":"verifying sha256 digest"}\n',
                    b"",
                ]
            )
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                callback = MagicMock()
                result = await pull_model_async("llama3:8b", on_progress=callback)
                assert result is True
                callback.assert_called_once_with("verifying sha256 digest")


# ─── ShowModel additional edge cases ───────────────────────────


class TestShowModelAdditional:
    """Cover missed branches in show_model."""

    def test_skip_blank_lines(self):
        """Line 230: blank lines in output are skipped."""
        stdout = "architecture    llama3\n\nparameters      8B\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("llama3:8b")
                assert info is not None
                assert "architecture" in info
                assert "parameters" in info
                # Blank line didn't cause errors
                assert "" not in info

    def test_skip_tab_indented_lines(self):
        """Line 235: tab-indented continuation lines are skipped."""
        stdout = "architecture    llama3\n\tindented continuation\nparameters      8B\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("llama3:8b")
                assert "indented continuation" not in info

    def test_skip_space_indented_lines(self):
        """Line 235: space-indented continuation lines are skipped."""
        stdout = "architecture    llama3\n  space indented\nparameters      8B\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("llama3:8b")
                assert "space indented" not in info

    def test_line_partition_value_error(self):
        """Line 237-238: line.partition doesn't raise ValueError."""
        stdout = "singleword\nparameters      8B\n"
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("test")
                assert info is not None
                assert "parameters" in info
                # singleword becomes key with empty value
                assert "singleword" in info

    def test_multiline_value_continuation(self):
        """Multi-line values where continuation is indented get skipped."""
        stdout = (
            "architecture    llama3\n"
            "  Additional architecture details\n"
            "parameters      8B\n"
        )
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, stdout, "")):
                info = show_model("llama3:8b")
                assert "Additional" not in info

    def test_empty_show_output(self):
        """Empty stdout should still produce a dict with name."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch("godspeed.tools.ollama_manager._run_ollama", return_value=(0, "", "")):
                info = show_model("test")
                assert info == {"name": "test"}


# ─── Additional _run_ollama tests ────────────────────────────────


class TestRunOllamaAdditional:
    def test_returns_stderr_on_error(self):
        """Non-zero returncode returns stderr."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
            rc, stdout, stderr = _run_ollama(["rm", "bad"])
            assert rc == 1
            assert stderr == "error msg"

    def test_custom_timeout_passed(self):
        """Timeout parameter is forwarded to subprocess.run."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_ollama(["pull", "model"], timeout=999)
            assert mock_run.call_args[1]["timeout"] == 999


# ─── DeleteModel additional ─────────────────────────────────────


class TestDeleteModelAdditional:
    def test_returns_stdout_when_stderr_empty_on_success(self):
        """Success with empty stderr returns stdout."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "godspeed.tools.ollama_manager._run_ollama",
                return_value=(0, "model removed", ""),
            ):
                ok, msg = delete_model("test")
                assert ok
                assert "removed" in msg

    def test_returns_stderr_on_failure(self):
        """Failure prefers stderr."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "godspeed.tools.ollama_manager._run_ollama",
                return_value=(1, "", "model not found"),
            ):
                ok, msg = delete_model("test")
                assert not ok
                assert "not found" in msg

    def test_returns_stdout_on_failure_when_stderr_empty(self):
        """Failure with empty stderr falls back to stdout."""
        with patch("godspeed.tools.ollama_manager._ollama_installed", return_value=True):
            with patch(
                "godspeed.tools.ollama_manager._run_ollama",
                return_value=(1, "something went wrong", ""),
            ):
                ok, msg = delete_model("test")
                assert not ok
                assert "wrong" in msg


# ─── _ollama_installed ──────────────────────────────────────────


class TestOllamaInstalled:
    def test_installed_when_found(self):
        with patch("shutil.which", return_value="/usr/local/bin/ollama"):
            assert _ollama_installed() is True

    def test_not_installed_when_none(self):
        with patch("shutil.which", return_value=None):
            assert _ollama_installed() is False
