# Threat Model

## 1. What Godspeed protects against

- Prompt injection in repository content that tries to trick the agent into exfiltrating secrets or running unsafe tool calls.
- Runaway or malicious tool-use loops that attempt destructive shell operations (for example recursive deletes, pipe-to-shell payloads, or force-push style history rewrites).
- Secret leakage from local files and tool outputs into model context and audit logs via regex and entropy-based redaction.
- Audit log tampering after the fact via hash-chained SHA-256 records and explicit chain verification.
- Dangerous behavior hidden behind wrappers (for example `bash -c`, command substitution, variable-expanded flags, or chained commands) when patterns match those forms.

## 2. What Godspeed explicitly does NOT protect against

- A compromised model provider that exfiltrates prompt data before local tool permissions are enforced.
- A user intentionally configuring permissive allow/ask rules that defeat deny-first safety posture.
- Side channels in model output (including steganographic leakage in otherwise normal responses).
- Physical machine compromise or local privilege escalation outside the Godspeed process.
- Vulnerabilities in upstream dependencies, Python runtime, OS shell, or kernel.

## 3. Assumptions the security model makes

- `settings.yaml` (global/project) is trusted and has not been tampered with by an attacker.
- The human approving prompts in interactive mode is trustworthy and not socially engineered.
- Audit log storage has appropriate filesystem protections to prevent unauthorized edits/deletes.
- TLS to remote model providers is valid and not intercepted by a hostile network actor.

## 4. Known limitations of the current implementation

- Dangerous-command blocking is regex-based; semantic equivalents can still evade detection through novel obfuscation not represented in current patterns.
- Keyword prefiltering is used before full regex scanning; if both keyword heuristics and regex patterns miss an obfuscated payload, it can fall through.
- Some shell obfuscation classes remain hard to robustly block without full shell parsing (for example complex nested expansions, here-doc generated payloads, and multi-stage decoding pipelines).
- Unicode lookalike command names are not comprehensively normalized; homoglyph-based spoofing remains a gap.
- Permission rules rely on formatted tool-call strings and glob matching, not AST-level argument semantics.
- Secret scanning focuses on known token formats and entropy thresholds; transformed/encrypted/chunked secrets can bypass detection.
- Cross-line and whitespace-obfuscated secret detection is improved for common key prefixes but still pattern-bound and format-specific.
- Entropy thresholds involve tradeoffs: lowering threshold catches more low-entropy secrets but can increase false positives on random-like non-secrets.
- Redaction is textual replacement and does not classify severity; it prevents raw secret exposure but does not provide DLP policy tiers.
- MCP caller attribution is carried in `action_detail` metadata, not a dedicated top-level audit schema field.
- Audit chain integrity detects post-write tampering but does not prevent deletion of the entire log directory by a sufficiently privileged local attacker.
