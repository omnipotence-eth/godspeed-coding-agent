"""Tests for secret redaction in audit events."""

from __future__ import annotations

from godspeed.audit.redactor import _redact_recursive, redact_audit_detail
from godspeed.security.secrets import REDACTED

# ---------------------------------------------------------------------------
# Test: redact single values
# ---------------------------------------------------------------------------


class TestRedactStrings:
    def test_openai_key_redacted(self) -> None:
        result = redact_audit_detail({"text": "sk-proj-abcdefghijklmnopqrstuvwxyz123456"})
        assert result["text"] == REDACTED

    def test_anthropic_key_redacted(self) -> None:
        result = redact_audit_detail({"text": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"})
        assert result["text"] == REDACTED

    def test_aws_access_key_redacted(self) -> None:
        result = redact_audit_detail({"text": "AKIAIOSFODNN7EXAMPLE"})
        assert result["text"] == REDACTED

    def test_github_pat_redacted(self) -> None:
        result = redact_audit_detail({"text": "ghp_1234567890abcdefghijklmnopqrstuvwxyz"})
        assert result["text"] == REDACTED

    def test_gitlab_pat_redacted(self) -> None:
        result = redact_audit_detail({"text": "glpat-abcdefghijklmnopqrstuvwxyz"})
        assert result["text"] == REDACTED

    def test_sendgrid_key_redacted(self) -> None:
        result = redact_audit_detail(
            {"text": "SG.abcdefghijklmnopqrstuv.abcdefghijklmnopqrstuvwxyz"}
        )
        assert result["text"] == REDACTED

    def test_private_key_redacted(self) -> None:
        result = redact_audit_detail({"text": "-----BEGIN RSA PRIVATE KEY-----"})
        assert result["text"] == REDACTED

    def test_jwt_token_redacted(self) -> None:
        result = redact_audit_detail(
            {"text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"}
        )
        assert result["text"] == REDACTED

    def test_bearer_token_redacted(self) -> None:
        result = redact_audit_detail(
            {"text": "Bearer abcdefghijklmnopqrstuvwxyz12345678901234567890"}
        )
        assert result["text"] == REDACTED

    def test_database_connection_string_redacted(self) -> None:
        result = redact_audit_detail({"text": "postgres://user:password123@localhost:5432/db"})
        assert result["text"] == REDACTED

    def test_password_assignment_redacted(self) -> None:
        result = redact_audit_detail({"text": 'password: "mysupersecretpassword"'})
        assert result["text"] == REDACTED

    def test_api_key_assignment_redacted(self) -> None:
        result = redact_audit_detail({"text": 'api_key: "abcdefghijklmnopqrstuvwxyz"'})
        assert result["text"] == REDACTED


# ---------------------------------------------------------------------------
# Test: non-secret content preserved
# ---------------------------------------------------------------------------


class TestNonSecretPreserved:
    def test_normal_text_unchanged(self) -> None:
        result = redact_audit_detail({"text": "This is a normal message"})
        assert result["text"] == "This is a normal message"

    def test_empty_string_unchanged(self) -> None:
        result = redact_audit_detail({"text": ""})
        assert result["text"] == ""

    def test_numeric_values_unchanged(self) -> None:
        result = redact_audit_detail({"count": 42, "ratio": 0.5})
        assert result["count"] == 42
        assert result["ratio"] == 0.5

    def test_boolean_values_unchanged(self) -> None:
        result = redact_audit_detail({"flag": True, "enabled": False})
        assert result["flag"] is True
        assert result["enabled"] is False

    def test_none_preserved(self) -> None:
        result = redact_audit_detail({"key": None})
        assert result["key"] is None

    def test_file_paths_preserved(self) -> None:
        result = redact_audit_detail({"path": "/home/user/src/main.py"})
        assert result["path"] == "/home/user/src/main.py"

    def test_tool_names_preserved(self) -> None:
        result = redact_audit_detail({"tool": "file_read", "args": {"path": "test.py"}})
        assert result["tool"] == "file_read"
        assert result["args"]["path"] == "test.py"


# ---------------------------------------------------------------------------
# Test: nested object traversal
# ---------------------------------------------------------------------------


class TestNestedTraversal:
    def test_nested_dict(self) -> None:
        detail = {
            "session": {
                "model": "claude-3",
                "config": {"api_key": "sk-ant-api03-secretkey1234567890abcdef"},
            }
        }
        result = redact_audit_detail(detail)
        assert result["session"]["model"] == "claude-3"
        assert result["session"]["config"]["api_key"] == REDACTED

    def test_list_of_strings(self) -> None:
        detail = {
            "files": ["src/main.py", "src/config.py"],
            "tokens": ["ghp_1234567890abcdefghijklmnopqrstuvwxyz", "normal_token"],
        }
        result = redact_audit_detail(detail)
        assert result["files"] == ["src/main.py", "src/config.py"]
        assert result["tokens"][0] == REDACTED
        assert result["tokens"][1] == "normal_token"

    def test_list_of_dicts(self) -> None:
        detail = {
            "events": [
                {"type": "auth", "token": "glpat-abcdefghijklmnopqrstuvwxyz"},
                {"type": "read", "path": "test.py"},
            ]
        }
        result = redact_audit_detail(detail)
        assert result["events"][0]["token"] == REDACTED
        assert result["events"][1]["path"] == "test.py"

    def test_deeply_nested(self) -> None:
        detail = {
            "level1": {"level2": {"level3": {"secret": "-----BEGIN OPENSSH PRIVATE KEY-----"}}}
        }
        result = redact_audit_detail(detail)
        assert result["level1"]["level2"]["level3"]["secret"] == REDACTED

    def test_mixed_content_in_nested(self) -> None:
        detail = {
            "user": "admin",
            "secrets": ["AKIAIOSFODNN7EXAMPLE", "normal_value", "sk-proj-another1234567890abcdef"],
            "normal_dict": {"key": "value"},
        }
        result = redact_audit_detail(detail)
        assert result["user"] == "admin"
        assert result["secrets"][0] == REDACTED
        assert result["secrets"][1] == "normal_value"
        assert result["secrets"][2] == REDACTED
        assert result["normal_dict"]["key"] == "value"


# ---------------------------------------------------------------------------
# Test: large payload handling
# ---------------------------------------------------------------------------


class TestLargePayload:
    def test_large_dict(self) -> None:
        detail = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = redact_audit_detail(detail)
        assert len(result) == 100
        for key in detail:
            assert result[key] == detail[key]

    def test_many_nested(self) -> None:
        detail = {"items": [{"name": f"item_{i}", "token": f"xoxb-secret{i}"} for i in range(50)]}
        result = redact_audit_detail(detail)
        assert len(result["items"]) == 50
        # Each token is not long enough to be detected, so they may remain but let's just verify size
        for item in result["items"]:
            assert "name" in item

    def test_multiple_secrets_in_one_string(self) -> None:
        detail = {
            "environment": (
                "OPENAI_API_KEY=sk-proj-abcdef123456; ANTHROPIC=sk-ant-api03-zyxwvut987654"
            )
        }
        result = redact_audit_detail(detail)
        assert REDACTED in result["environment"]

    def test_huggingface_token_redacted(self) -> None:
        result = redact_audit_detail({"token": "hf_abcdefghijklmnopqrstuvwxyz123456"})
        assert result["token"] == REDACTED

    def test_google_api_key_redacted(self) -> None:
        result = redact_audit_detail({"key": "AIzaSyD8HxABCnAmkLbDEFGHIJKLMNOPQRSTUVW"})
        assert result["key"] == REDACTED

    def test_stripe_key_redacted(self) -> None:
        result = redact_audit_detail({"key": "sk_test_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"})
        assert result["key"] == REDACTED


# ---------------------------------------------------------------------------
# Test: _redact_recursive directly
# ---------------------------------------------------------------------------


class TestRedactRecursive:
    def test_non_string_non_collection(self) -> None:
        assert _redact_recursive(42) == 42
        assert _redact_recursive(3.14) == 3.14
        assert _redact_recursive(True) is True
        assert _redact_recursive(None) is None

    def test_empty_dict(self) -> None:
        assert _redact_recursive({}) == {}

    def test_empty_list(self) -> None:
        assert _redact_recursive([]) == []
