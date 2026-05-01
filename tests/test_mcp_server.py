from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from godspeed.audit.trail import AuditTrail
from godspeed.mcp_server.server import GodspeedMCPServer
from godspeed.security.permissions import ALLOW, DENY, PermissionDecision
from godspeed.tools.base import ToolResult


def _settings_for(tmp_path: Path, *, audit_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        project_dir=tmp_path,
        global_dir=tmp_path,
        permissions=SimpleNamespace(deny=[], allow=[], ask=[]),
        audit=SimpleNamespace(enabled=audit_enabled, retention_days=30),
    )


def _make_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> GodspeedMCPServer:
    monkeypatch.setattr(
        "godspeed.mcp_server.server._load_settings_with_optional_config",
        lambda _path: _settings_for(tmp_path),
    )
    return GodspeedMCPServer()


@pytest.mark.asyncio
async def test_file_read_allowed_logs_allow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _make_server(monkeypatch, tmp_path)
    server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
    server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("file-content"))
    server.audit_trail = Mock()
    server.audit_trail.arecord = AsyncMock()

    result = await server.handle_tool_call(name="file_read", arguments={"file_path": "README.md"})

    assert result.isError is False
    assert result.content[0].text == "file-content"
    assert server.audit_trail.arecord.await_count == 1
    audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
    assert audit_detail["denied"] is False


@pytest.mark.asyncio
async def test_file_read_denied_returns_error_and_logs_deny(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _make_server(monkeypatch, tmp_path)
    server.permission_engine.evaluate = Mock(return_value=PermissionDecision(DENY, "blocked path"))
    server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("should-not-run"))
    server.audit_trail = Mock()
    server.audit_trail.arecord = AsyncMock()

    result = await server.handle_tool_call(name="file_read", arguments={"file_path": ".env"})

    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert payload["tool"] == "file_read"
    assert payload["reason"] == "blocked path"
    assert server.registry.dispatch.await_count == 0
    audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
    assert audit_detail["denied"] is True


@pytest.mark.asyncio
async def test_shell_dangerous_blocked_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _make_server(monkeypatch, tmp_path)
    server.permission_engine.evaluate = Mock(
        return_value=PermissionDecision(DENY, "Dangerous command detected")
    )
    server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("should-not-run"))
    server.audit_trail = Mock()
    server.audit_trail.arecord = AsyncMock()

    result = await server.handle_tool_call(name="shell", arguments={"command": "rm -rf /"})

    assert result.isError is True
    assert server.registry.dispatch.await_count == 0
    audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
    assert "Dangerous command" in audit_detail["reason"]


@pytest.mark.asyncio
async def test_file_read_output_secret_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _make_server(monkeypatch, tmp_path)
    server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
    server.registry.dispatch = AsyncMock(
        return_value=ToolResult.ok("token=sk-ant-api03-fakeabcdefghijklmnopqrstu")
    )
    server.audit_trail = Mock()
    server.audit_trail.arecord = AsyncMock()

    result = await server.handle_tool_call(name="file_read", arguments={"file_path": "secret.txt"})

    assert result.isError is False
    assert "sk-ant" not in result.content[0].text
    assert "[REDACTED]" in result.content[0].text


@pytest.mark.asyncio
async def test_unknown_caller_attribution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server = _make_server(monkeypatch, tmp_path)
    server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
    server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("ok"))
    server.audit_trail = Mock()
    server.audit_trail.arecord = AsyncMock()

    await server.handle_tool_call(name="file_read", arguments={"file_path": "README.md"})

    audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
    assert audit_detail["caller"] == "mcp_client:unknown"


def test_audit_verify_mixed_cli_and_mcp_session(tmp_path: Path) -> None:
    trail = AuditTrail(log_dir=tmp_path / "audit", session_id="mixed-session")
    trail.record(event_type="session_start", detail={"mode": "cli"}, outcome="success")
    trail.record(
        event_type="tool_call",
        detail={"mode": "mcp", "caller": "mcp_client:unknown", "tool": "file_read"},
        outcome="success",
    )
    is_valid, _message = trail.verify_chain()
    assert is_valid


def test_all_registered_tools_are_listed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server = _make_server(monkeypatch, tmp_path)
    listed = server.list_mcp_tools()
    registered = server.registry.list_tools()
    assert len(listed) == len(registered)
    assert len(listed) > 0


def test_server_starts_cleanly_with_default_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _make_server(monkeypatch, tmp_path)
    assert server.server is not None
    assert server.permission_engine is not None
    assert server.tool_context is not None


@pytest.mark.asyncio
async def test_sigint_style_shutdown_keeps_audit_chain_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _make_server(monkeypatch, tmp_path)
    assert server.audit_trail is not None
    await server._audit(
        outcome="success",
        detail={"caller": "mcp_client:unknown", "tool": "file_read", "argument": "README.md"},
    )
    await server.shutdown()
    is_valid, _message = server.audit_trail.verify_chain()
    assert is_valid
