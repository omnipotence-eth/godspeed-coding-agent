"""Secret detection and redaction — regex + entropy analysis.

4-layer secret protection:
1. File access deny rules (handled by permission engine)
2. Context cleaning before LLM sees content
3. Output filtering on LLM responses
4. Audit log redaction (handled by audit.redactor)
"""

from __future__ import annotations

import math
import re

# Compiled secret detection patterns: (regex, type_name)
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # API keys with known prefixes (more specific patterns first)
    (re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}"), "anthropic_api_key"),
    (re.compile(r"sk-[a-zA-Z0-9\-]{20,}"), "openai_api_key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "github_pat"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "github_oauth"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{22,}"), "github_fine_grained"),
    (re.compile(r"glpat-[a-zA-Z0-9\-]{20,}"), "gitlab_pat"),
    (re.compile(r"xoxb-[0-9]{10,}-[a-zA-Z0-9]+"), "slack_bot_token"),
    (re.compile(r"xoxp-[0-9]{10,}-[a-zA-Z0-9]+"), "slack_user_token"),
    (re.compile(r"SG\.[a-zA-Z0-9_\-.]{20,}\.[a-zA-Z0-9_\-.]{20,}"), "sendgrid_api_key"),
    (re.compile(r"sq0[a-z]{3}-[a-zA-Z0-9\-_]{22,}"), "square_api_key"),
    # Private keys
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"), "private_key"),
    (re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"), "openssh_private_key"),
    # Generic patterns
    (
        re.compile(r"""(?:password|passwd|pwd)\s*[:=]\s*['"][^'"]{8,}['"]""", re.IGNORECASE),
        "password_assignment",
    ),
    (
        re.compile(r"""(?:api_key|apikey|api-key)\s*[:=]\s*['"][^'"]{10,}['"]""", re.IGNORECASE),
        "api_key_assignment",
    ),
    (
        re.compile(r"""(?:secret|token)\s*[:=]\s*['"][^'"]{10,}['"]""", re.IGNORECASE),
        "secret_assignment",
    ),
    # Bearer tokens
    (re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", re.IGNORECASE), "bearer_token"),
    # Connection strings
    (
        re.compile(r"(?:postgres|mysql|mongodb)://[^\s]+:[^\s]+@[^\s]+"),
        "database_connection_string",
    ),
    # Hugging Face
    (re.compile(r"hf_[a-zA-Z0-9]{20,}"), "huggingface_token"),
    # JWT tokens
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "jwt_token"),
    # Azure
    (re.compile(r"DefaultEndpointsProtocol=https;[^\s]{20,}"), "azure_connection_string"),
    # AWS secret key (generic assignment)
    (
        re.compile(
            r"""aws_secret_access_key\s*[:=]\s*['"]?[A-Za-z0-9/+=]{30,}['"]?""",
            re.IGNORECASE,
        ),
        "aws_secret_key",
    ),
    # Google Cloud
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "google_api_key"),
    # Stripe
    (re.compile(r"(?:sk|pk)_(?:test|live)_[a-zA-Z0-9]{20,}"), "stripe_key"),
    # PKCS8 private key
    (re.compile(r"-----BEGIN (?:ENCRYPTED )?PRIVATE KEY-----"), "pkcs8_private_key"),
    # Discord
    (re.compile(r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27,}"), "discord_bot_token"),
    # Unquoted password assignments (common in configs)
    (
        re.compile(r"""(?:password|passwd|pwd)\s*[:=]\s*\S{8,}""", re.IGNORECASE),
        "password_unquoted",
    ),
]

# Minimum entropy threshold for high-entropy string detection
ENTROPY_THRESHOLD = 4.5
ENTROPY_MIN_LENGTH = 20

# Redaction placeholder
REDACTED = "[REDACTED]"


def detect_secrets(text: str) -> list[dict[str, str]]:
    """Detect potential secrets in text using regex patterns and entropy.

    Returns:
        List of dicts with 'type', 'match', 'start', 'end' keys.
    """
    findings: list[dict[str, str]] = []
    seen_spans: set[tuple[int, int]] = set()

    # Pattern-based detection
    for pattern, secret_type in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in seen_spans:
                seen_spans.add(span)
                findings.append(
                    {
                        "type": secret_type,
                        "match": match.group(),
                        "start": str(match.start()),
                        "end": str(match.end()),
                    }
                )

    # Entropy-based detection for unmatched high-entropy strings
    for match in re.finditer(r"[a-zA-Z0-9+/\-_]{20,}", text):
        span = (match.start(), match.end())
        if span in seen_spans:
            continue
        token = match.group()
        if len(token) >= ENTROPY_MIN_LENGTH and _shannon_entropy(token) >= ENTROPY_THRESHOLD:
            seen_spans.add(span)
            findings.append(
                {
                    "type": "high_entropy_string",
                    "match": token,
                    "start": str(match.start()),
                    "end": str(match.end()),
                }
            )

    return findings


def redact_secrets(text: str) -> str:
    """Redact all detected secrets in text, replacing with [REDACTED]."""
    findings = detect_secrets(text)
    if not findings:
        return text

    # Sort by position descending so replacements don't shift indices
    findings.sort(key=lambda f: int(f["start"]), reverse=True)

    result = text
    for finding in findings:
        start = int(finding["start"])
        end = int(finding["end"])
        result = result[:start] + REDACTED + result[end:]

    return result


def _shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string in bits per character."""
    if not data:
        return 0.0

    freq: dict[str, int] = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1

    length = len(data)
    entropy = 0.0
    for count in freq.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy
