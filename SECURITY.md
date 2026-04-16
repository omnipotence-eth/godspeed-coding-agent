# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.3.x   | :white_check_mark: |
| 2.2.x   | :white_check_mark: |
| < 2.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Godspeed, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email **Ttimmsinternational@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

You will receive an acknowledgment within 48 hours and a detailed response within 7
days indicating next steps.

## Security Design

Godspeed is built with a security-first architecture:

- **4-tier permission engine**: deny-first evaluation (deny > ask > allow)
- **Dangerous command detection**: 71 regex patterns for destructive operations
- **Secret protection**: 4-layer defense (access control, context cleaning, output
  filtering, audit redaction) with 27 regex patterns + Shannon entropy analysis
- **Hash-chained audit trail**: tamper-evident JSONL logs with SHA-256 chain.
  Audit writes fail closed — any I/O error raises `AuditWriteError` and the
  chain state does not advance, so a successful retry chains cleanly from the
  last persisted record.
- **Fail-closed defaults**: permission timeouts result in denial

## Scope

The following are in scope for security reports:

- Permission bypass (tool executes without proper authorization)
- Audit trail tampering (hash chain can be modified without detection)
- Secret leakage (secrets appear in LLM context, output, or logs)
- Dangerous command bypass (destructive commands execute without detection)
- Prompt injection via GODSPEED.md or tool outputs
