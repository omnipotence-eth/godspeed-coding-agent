"""Tests for llama.cpp server management tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.llamacpp_manager import (
    LlamaCppTool,
    configure_litellm_env,
    get_server_status,
    is_server_running,
    start_server,
    stop_server,
)


class TestFindServerBinary:
    """Test binary discovery."""

    @patch.object(Path, "exists", return_value=True)
    def test_default_path_found(self, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_server_binary

        result = _find_server_binary()
        assert result is not None

    @patch.object(Path, "exists", return_value=False)
    @patch("shutil.which", return_value="/usr/bin/llama-server")
    def test_path_found(self, mock_which: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_server_binary

        result = _find_server_binary()
        assert result == Path("/usr/bin/llama-server")

    @patch.object(Path, "exists", return_value=False)
    @patch("shutil.which", return_value=None)
    def test_not_found(self, mock_which: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_server_binary

        result = _find_server_binary()
        assert result is None


class TestFindModel:
    """Test model file discovery."""

    @patch.object(Path, "exists", return_value=True)
    def test_default_model_found(self, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_model

        result = _find_model()
        assert result is not None

    @patch.object(Path, "exists", side_effect=[False, True])
    @patch("pathlib.Path.glob", return_value=[Path("/models/other.gguf")])
    def test_any_gguf_found(self, mock_glob: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_model

        result = _find_model()
        assert result == Path("/models/other.gguf")

    @patch.object(Path, "exists", return_value=False)
    @patch("pathlib.Path.glob", return_value=[])
    def test_no_model_found(self, mock_glob: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_model

        result = _find_model()
        assert result is None


class TestIsServerRunning:
    """Test server health check."""

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_server_running(self, mock_request: MagicMock, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__.return_value = MagicMock()
        assert is_server_running("http://127.0.0.1:8080") is True

    @patch("urllib.request.urlopen", side_effect=Exception("Connection refused"))
    @patch("urllib.request.Request")
    def test_server_not_running(self, mock_request: MagicMock, mock_urlopen: MagicMock) -> None:
        assert is_server_running("http://127.0.0.1:8080") is False


class TestStartServer:
    """Test server startup logic."""

    def test_already_running(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True):
            result = start_server()
        assert result is None

    def test_binary_not_found(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch("godspeed.tools.llamacpp_manager._find_server_binary", return_value=None):
                result = start_server()
        assert result is None

    def test_model_not_found(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch("godspeed.tools.llamacpp_manager._find_model", return_value=None):
                    result = start_server()
        assert result is None

    def test_start_success(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(timeout=1)
        assert result is mock_proc

    def test_start_os_error(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", side_effect=OSError("Permission denied")):
                        result = start_server()
        assert result is None

    def test_server_exits_early(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.communicate.return_value = ("stdout", "stderr")
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(timeout=1)
        assert result is None


class TestStopServer:
    """Test server shutdown."""

    def test_none_process(self) -> None:
        assert stop_server(None) is True

    def test_already_terminated(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        assert stop_server(mock_proc) is True

    def test_graceful_stop(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        assert stop_server(mock_proc) is True
        mock_proc.terminate.assert_called_once()

    def test_force_kill(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
        assert stop_server(mock_proc) is True
        mock_proc.kill.assert_called_once()

    def test_error_handling(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = Exception("boom")
        assert stop_server(mock_proc) is False


class TestGetServerStatus:
    """Test status querying."""

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False)
    def test_not_running(self, mock_is_running: MagicMock) -> None:
        status = get_server_status()
        assert status["running"] is False
        assert status["model"] is None

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_running_with_model(self, mock_request, mock_urlopen, mock_is_running) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "test-model"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        status = get_server_status()
        assert status["running"] is True
        assert status["model"] == "test-model"

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_running_list_format(self, mock_request, mock_urlopen, mock_is_running) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'[{"id": "list-model"}]'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        status = get_server_status()
        assert status["model"] == "list-model"


class TestConfigureLitellmEnv:
    """Test environment configuration."""

    @patch.dict("os.environ", {}, clear=True)
    def test_sets_defaults(self) -> None:
        configure_litellm_env()
        import os

        assert os.environ.get("LLAMACPP_API_BASE") == "http://127.0.0.1:8080/v1"
        assert os.environ.get("OPENAI_API_BASE") == "http://127.0.0.1:8080/v1"
        assert os.environ.get("OPENAI_API_KEY") == "none"

    @patch.dict("os.environ", {"LLAMACPP_API_BASE": "existing"}, clear=True)
    def test_preserves_existing(self) -> None:
        configure_litellm_env()
        import os

        assert os.environ.get("LLAMACPP_API_BASE") == "existing"


class TestLlamaCppTool:
    """Test the LlamaCpp tool integration."""

    @pytest.mark.asyncio
    async def test_status_not_running(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            result = await tool.execute({"action": "status"}, tool_context)
            assert "NOT RUNNING" in result.output

    @pytest.mark.asyncio
    async def test_status_running(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch(
            "godspeed.tools.llamacpp_manager.get_server_status",
            return_value={
                "running": True,
                "url": "http://127.0.0.1:8080",
                "model": "test-model",
                "version": "1.0",
            },
        ):
            result = await tool.execute({"action": "status"}, tool_context)
            assert "RUNNING" in result.output
            assert "test-model" in result.output

    @pytest.mark.asyncio
    async def test_start_success(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.start_server", return_value=MagicMock()):
            result = await tool.execute({"action": "start"}, tool_context)
            assert "started successfully" in result.output

    @pytest.mark.asyncio
    async def test_start_already_running(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.start_server", return_value=None):
            with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True):
                result = await tool.execute({"action": "start"}, tool_context)
                assert "already running" in result.output

    @pytest.mark.asyncio
    async def test_start_failure(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.start_server", return_value=None):
            with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
                result = await tool.execute({"action": "start"}, tool_context)
                assert "Failed to start" in result.error

    @pytest.mark.asyncio
    async def test_stop(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        result = await tool.execute({"action": "stop"}, tool_context)
        assert "stop requested" in result.output

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        result = await tool.execute({"action": "fly"}, tool_context)
        assert result.is_error is True
        assert "Unknown action" in result.error

    def test_schema(self) -> None:
        tool = LlamaCppTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["status", "start", "stop"]

    def test_name(self) -> None:
        tool = LlamaCppTool()
        assert tool.name == "llamacpp"

    def test_risk_level(self) -> None:
        tool = LlamaCppTool()
        from godspeed.tools.base import RiskLevel

        assert tool.risk_level == RiskLevel.LOW
