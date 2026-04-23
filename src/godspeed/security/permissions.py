"""4-tier permission engine — the core security differentiator.

Evaluation order: deny > dangerous > session > allow > ask > default (risk level).
Deny rules always win. Dangerous command detection runs before session grants
so that user-approved patterns cannot bypass destructive command blocking.
Fail-closed: any ambiguity results in denial.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time

from godspeed.security.dangerous import detect_dangerous_command
from godspeed.security.rules import RuleAction, parse_rules
from godspeed.tools.base import RiskLevel, ToolCall

logger = logging.getLogger(__name__)


class PermissionDecision:
    """Result of a permission evaluation."""

    def __init__(self, action: str, reason: str = "") -> None:
        self.action = action
        self.reason = reason

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.action == other
        if isinstance(other, PermissionDecision):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"PermissionDecision({self.action!r}, {self.reason!r})"


ALLOW = "allow"
DENY = "deny"
ASK = "ask"


class PermissionEngine:
    """4-tier permission engine with deny-first evaluation.

    Tiers (by tool risk level):
    - READ_ONLY: auto-allowed, no prompt
    - LOW: ask once, then session-scoped allow
    - HIGH: ask every time (unless pattern-matched to allow)
    - DESTRUCTIVE: blocked by default, requires explicit allow rule

    Rule evaluation:
    1. Check deny rules — if any match, DENY
    2. Check dangerous command patterns — if detected, DENY
    3. Check session grants — if granted, ALLOW
    4. Check allow rules — if any match, ALLOW
    5. Check ask rules — if any match, ASK
    6. Fall back to risk-level default
    """

    def __init__(
        self,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        ask_patterns: list[str] | None = None,
        tool_risk_levels: dict[str, RiskLevel] | None = None,
        mode: str = "normal",
    ) -> None:
        self._deny_rules = parse_rules(deny_patterns or [], RuleAction.DENY)
        self._allow_rules = parse_rules(allow_patterns or [], RuleAction.ALLOW)
        self._ask_rules = parse_rules(ask_patterns or [], RuleAction.ASK)
        self._tool_risk_levels = tool_risk_levels or {}
        self._session_grants: dict[str, float] = {}
        self._grant_ttl: float = 3600.0  # 1 hour default
        self._grants_lock = threading.Lock()
        self.plan_mode: bool = False
        # Permission mode — governs prompt frequency and hard-floor enforcement.
        #
        #   "strict" — treat ASK as DENY. Maximum caution.
        #   "normal" — default. Risk-level defaults + session grants.
        #   "auto"   — productivity mode: auto-allow READ_ONLY and LOW,
        #              still prompt on HIGH, deny DESTRUCTIVE by default.
        #              Deny rules + dangerous-command regex still apply.
        #   "yolo"   — auto-approve everything after the hard floor
        #              (deny rules + dangerous-command detection).
        #   "unsafe" — skip the entire engine including hard floor.
        #              Claude-Code-equivalent of
        #              --dangerously-skip-permissions. Only the Dockerized /
        #              disposable-VM use case justifies this.
        self.mode: str = mode

    def evaluate(self, tool_call: ToolCall) -> PermissionDecision:
        """Evaluate a tool call against all rules.

        Returns a PermissionDecision with action and reason.
        """
        # UNSAFE mode — skip the entire engine, including the hard floor.
        # This is the Claude-Code-equivalent of --dangerously-skip-permissions.
        # Only use in disposable sandboxes (Docker, ephemeral VM). Recorded
        # in the audit trail via the reason string.
        if self.mode == "unsafe":
            return PermissionDecision(ALLOW, "unsafe mode — all guards skipped")

        # Plan mode: block everything except READ_ONLY tools
        if self.plan_mode:
            risk = self._tool_risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
            if risk != RiskLevel.READ_ONLY:
                return PermissionDecision(DENY, "Plan mode active — read-only tools only")

        formatted = tool_call.format_for_permission()

        # 1. Deny rules first — always win (except in unsafe mode above)
        for rule in self._deny_rules:
            if rule.matches(formatted):
                return PermissionDecision(DENY, f"Matched deny rule: {rule.pattern}")

        # 2. Dangerous command detection (for shell commands) — BEFORE session grants
        #    so that a session grant like "Bash(npm *)" cannot bypass dangerous detection
        if tool_call.tool_name.lower() in ("bash", "shell"):
            command = ""
            if isinstance(tool_call.arguments, dict):
                # Prefer the 'command' key — do NOT use first string value,
                # which could be a benign 'description' field
                command = tool_call.arguments.get("command", "")
                if not isinstance(command, str):
                    command = ""
            if command:
                dangers = detect_dangerous_command(command)
                if dangers:
                    return PermissionDecision(
                        DENY,
                        f"Dangerous command detected: {', '.join(dangers)}",
                    )

        # 3. YOLO mode — auto-approve after the hard floor (deny + dangerous)
        #    has cleared. Skips session/allow/ask/risk-default entirely.
        if self.mode == "yolo":
            return PermissionDecision(ALLOW, "yolo mode (hard floor enforced)")

        # 3b. AUTO mode — productivity tier. Auto-approve READ_ONLY and LOW
        #     risk tools; fall through to normal evaluation for HIGH /
        #     DESTRUCTIVE so the user is still prompted on writes and shell.
        if self.mode == "auto":
            risk = self._tool_risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
            if risk in (RiskLevel.READ_ONLY, RiskLevel.LOW):
                return PermissionDecision(ALLOW, f"auto mode (risk={risk.name.lower()})")
            # fall through to session grants / allow rules / ask / default

        # 4. Session grants (user already approved this pattern)
        if self._check_session_grant(formatted):
            return PermissionDecision(ALLOW, "Session grant (time-limited)")

        # 5. Allow rules
        for rule in self._allow_rules:
            if rule.matches(formatted):
                return PermissionDecision(ALLOW, f"Matched allow rule: {rule.pattern}")

        # 6. Ask rules
        for rule in self._ask_rules:
            if rule.matches(formatted):
                if self.mode == "strict":
                    return PermissionDecision(
                        DENY, f"strict mode — ask rule denied: {rule.pattern}"
                    )
                return PermissionDecision(ASK, f"Matched ask rule: {rule.pattern}")

        # 7. Default based on risk level
        risk = self._tool_risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
        decision = self._default_for_risk(risk)
        if self.mode == "strict" and decision == ASK:
            return PermissionDecision(DENY, f"strict mode — default ASK denied ({decision.reason})")
        return decision

    def add_rule(self, pattern: str, action: str) -> None:
        """Add a pattern to the in-memory rule list at runtime.

        Used by the ``/remember`` slash command so a persisted rule
        takes effect immediately in the current session, not just on
        next restart.

        Args:
            pattern: ``Tool(glob)`` style pattern.
            action: ``"allow" | "deny" | "ask"``.
        """
        action_lc = action.lower()
        if action_lc == "allow":
            self._allow_rules.extend(parse_rules([pattern], RuleAction.ALLOW))
        elif action_lc == "deny":
            self._deny_rules.extend(parse_rules([pattern], RuleAction.DENY))
        elif action_lc == "ask":
            self._ask_rules.extend(parse_rules([pattern], RuleAction.ASK))
        else:
            msg = f"action must be 'allow' | 'deny' | 'ask', got {action!r}"
            raise ValueError(msg)
        logger.info("Runtime rule added action=%s pattern=%s", action_lc, pattern)

    def grant_session_permission(self, pattern: str) -> None:
        """Grant a session-scoped permission for a pattern.

        Called when the user approves an ASK prompt. Thread-safe.
        """
        with self._grants_lock:
            self._session_grants[pattern] = time.monotonic()
        logger.info("Session permission granted pattern=%s ttl=%ds", pattern, int(self._grant_ttl))

    def revoke_session_permission(self, pattern: str) -> None:
        """Revoke a single session-scoped permission. Thread-safe."""
        with self._grants_lock:
            self._session_grants.pop(pattern, None)

    def revoke_session_permissions(self) -> None:
        """Revoke all session-scoped permissions. Thread-safe."""
        with self._grants_lock:
            self._session_grants.clear()

    @property
    def deny_rules(self) -> list:
        """Read-only access to deny rules."""
        return list(self._deny_rules)

    @property
    def allow_rules(self) -> list:
        """Read-only access to allow rules."""
        return list(self._allow_rules)

    @property
    def ask_rules(self) -> list:
        """Read-only access to ask rules."""
        return list(self._ask_rules)

    @property
    def session_grants(self) -> dict[str, float]:
        """Read-only copy of active session grants. Thread-safe."""
        with self._grants_lock:
            return dict(self._session_grants)

    def _check_session_grant(self, tool_call_str: str) -> bool:
        """Check session grants, removing expired ones. Thread-safe."""
        now = time.monotonic()
        with self._grants_lock:
            expired = [p for p, t in self._session_grants.items() if now - t > self._grant_ttl]
            for p in expired:
                del self._session_grants[p]
                logger.info("Session grant expired pattern=%s", p)

            # Snapshot grants under lock, then match outside
            grants = list(self._session_grants.keys())

        return any(fnmatch.fnmatch(tool_call_str, pattern) for pattern in grants)

    @staticmethod
    def _default_for_risk(risk: RiskLevel) -> PermissionDecision:
        """Get the default permission decision for a risk level."""
        if risk == RiskLevel.READ_ONLY:
            return PermissionDecision(ALLOW, "read-only tool")
        if risk == RiskLevel.LOW:
            return PermissionDecision(ASK, "low-risk write tool")
        if risk == RiskLevel.DESTRUCTIVE:
            return PermissionDecision(DENY, "destructive tool blocked by default")
        return PermissionDecision(ASK, "high-risk tool")
