"""Tests for secret detection and redaction.

Real-world patterns tested — every secret type must have a detection test
and a redaction test.
"""

from __future__ import annotations

from godspeed.security.secrets import (
    REDACTED,
    _shannon_entropy,
    detect_secrets,
    redact_secrets,
)


class TestAPIKeyDetection:
    """Test detection of known API key formats."""

    def test_openai_key(self) -> None:
        text = "api_key = 'sk-proj-1234567890abcdefghijklmnopqrstuv'"
        findings = detect_secrets(text)
        assert any(f["type"] == "openai_api_key" for f in findings)

    def test_anthropic_key(self) -> None:
        text = "key = 'sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234'"
        findings = detect_secrets(text)
        assert any(f["type"] == "anthropic_api_key" for f in findings)

    def test_aws_access_key(self) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        findings = detect_secrets(text)
        assert any(f["type"] == "aws_access_key" for f in findings)

    def test_github_pat(self) -> None:
        text = "token = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij'"
        findings = detect_secrets(text)
        assert any(f["type"] == "github_pat" for f in findings)

    def test_gitlab_pat(self) -> None:
        text = "GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx1234"
        findings = detect_secrets(text)
        assert any(f["type"] == "gitlab_pat" for f in findings)

    def test_slack_bot_token(self) -> None:
        text = "SLACK_TOKEN=xoxb-1234567890-abcdefghij"
        findings = detect_secrets(text)
        assert any(f["type"] == "slack_bot_token" for f in findings)

    def test_sendgrid_key(self) -> None:
        text = "SG.abcdefghijklmnopqrstuv.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop"
        findings = detect_secrets(text)
        assert any(f["type"] == "sendgrid_api_key" for f in findings)


class TestPrivateKeyDetection:
    """Test detection of private key headers."""

    def test_rsa_private_key(self) -> None:
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK..."
        findings = detect_secrets(text)
        assert any(f["type"] == "private_key" for f in findings)

    def test_ec_private_key(self) -> None:
        text = "-----BEGIN EC PRIVATE KEY-----\nMHQCAQ..."
        findings = detect_secrets(text)
        assert any(f["type"] == "private_key" for f in findings)

    def test_openssh_private_key(self) -> None:
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC..."
        findings = detect_secrets(text)
        assert any(f["type"] == "openssh_private_key" for f in findings)


class TestGenericPatterns:
    """Test generic secret assignment patterns."""

    def test_password_assignment(self) -> None:
        text = "password = 'super_secret_password_123'"
        findings = detect_secrets(text)
        assert any(f["type"] == "password_assignment" for f in findings)

    def test_api_key_assignment(self) -> None:
        text = "api_key = 'my-secret-api-key-value-here'"
        findings = detect_secrets(text)
        assert any(f["type"] == "api_key_assignment" for f in findings)

    def test_bearer_token(self) -> None:
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test.signature"
        findings = detect_secrets(text)
        assert any(f["type"] == "bearer_token" for f in findings)

    def test_database_connection_string(self) -> None:
        text = "DATABASE_URL=postgres://user:password@host:5432/db"
        findings = detect_secrets(text)
        assert any(f["type"] == "database_connection_string" for f in findings)


class TestEntropyDetection:
    """Test high-entropy string detection."""

    def test_high_entropy_string(self) -> None:
        # Random-looking string with high entropy
        text = "secret = aB3xK9mQ2nR7pL5vW8jF4hG6dS1cY0tU"
        findings = detect_secrets(text)
        # Should be detected by either pattern or entropy
        assert len(findings) > 0

    def test_low_entropy_not_flagged(self) -> None:
        # Repetitive string with low entropy
        text = "value = aaaaaaaaaaaaaaaaaaaaaaaaa"
        findings = detect_secrets(text)
        entropy_findings = [f for f in findings if f["type"] == "high_entropy_string"]
        assert len(entropy_findings) == 0


class TestRedaction:
    """Test secret redaction in text."""

    def test_redact_api_key(self) -> None:
        text = "My key is sk-proj-1234567890abcdefghijklmnopqrstuv"
        redacted = redact_secrets(text)
        assert "sk-proj" not in redacted
        assert REDACTED in redacted

    def test_redact_multiple_secrets(self) -> None:
        text = "AWS_KEY=AKIAIOSFODNN7EXAMPLE\ntoken=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        redacted = redact_secrets(text)
        assert "AKIA" not in redacted
        assert "ghp_" not in redacted
        assert redacted.count(REDACTED) == 2

    def test_no_secrets_unchanged(self) -> None:
        text = "This is a normal piece of code with no secrets."
        assert redact_secrets(text) == text

    def test_preserves_surrounding_text(self) -> None:
        text = "before AKIAIOSFODNN7EXAMPLE after"
        redacted = redact_secrets(text)
        assert redacted.startswith("before ")
        assert redacted.endswith(" after")


class TestShannonEntropy:
    """Test entropy calculation."""

    def test_empty_string(self) -> None:
        assert _shannon_entropy("") == 0.0

    def test_single_char(self) -> None:
        assert _shannon_entropy("a") == 0.0

    def test_repeated_char(self) -> None:
        assert _shannon_entropy("aaaaaaa") == 0.0

    def test_high_entropy(self) -> None:
        # All unique characters = high entropy
        entropy = _shannon_entropy("abcdefghijklmnop")
        assert entropy > 3.5

    def test_binary_entropy(self) -> None:
        # Two equally frequent chars = 1 bit
        entropy = _shannon_entropy("abababab")
        assert abs(entropy - 1.0) < 0.01
