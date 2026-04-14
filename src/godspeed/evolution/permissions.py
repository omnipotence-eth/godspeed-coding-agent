"""Permission pattern learning — analyze denials/grants to suggest optimizations.

Mines audit trails for permission patterns and recommends allowlist changes.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter

from godspeed.evolution.trace_analyzer import SessionTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PermissionSuggestion:
    """A suggested permission configuration change."""

    tool_name: str
    action: str  # "add_to_allowlist" | "pre_approve" | "review_needed"
    denial_count: int
    grant_count: int
    rationale: str


# ---------------------------------------------------------------------------
# Permission Advisor
# ---------------------------------------------------------------------------


class PermissionAdvisor:
    """Analyze permission patterns to suggest allowlist optimizations."""

    def __init__(
        self,
        denial_threshold: int = 5,
        grant_threshold: int = 5,
    ) -> None:
        self._denial_threshold = denial_threshold
        self._grant_threshold = grant_threshold

    def analyze_denials(self, sessions: list[SessionTrace]) -> list[PermissionSuggestion]:
        """Find tools denied >threshold times — suggest adding to allowlist."""
        denials: Counter[str] = Counter()
        denial_reasons: dict[str, list[str]] = {}

        for session in sessions:
            for tool_name, reason in session.permission_denials:
                denials[tool_name] += 1
                if tool_name not in denial_reasons:
                    denial_reasons[tool_name] = []
                if len(denial_reasons[tool_name]) < 3:
                    denial_reasons[tool_name].append(reason)

        suggestions: list[PermissionSuggestion] = []
        for tool_name, count in denials.most_common():
            if count >= self._denial_threshold:
                reasons = denial_reasons.get(tool_name, [])
                rationale = (
                    f"Denied {count} times across sessions. "
                    f"Reasons: {', '.join(sorted(set(reasons))[:3])}"
                )
                suggestions.append(
                    PermissionSuggestion(
                        tool_name=tool_name,
                        action="add_to_allowlist",
                        denial_count=count,
                        grant_count=0,
                        rationale=rationale,
                    )
                )

        return suggestions

    def analyze_approvals(self, sessions: list[SessionTrace]) -> list[PermissionSuggestion]:
        """Find tools always approved — suggest pre-approving."""
        grants: Counter[str] = Counter()
        denials: Counter[str] = Counter()

        for session in sessions:
            for tool_name in session.permission_grants:
                grants[tool_name] += 1
            for tool_name, _ in session.permission_denials:
                denials[tool_name] += 1

        suggestions: list[PermissionSuggestion] = []
        for tool_name, count in grants.most_common():
            if count >= self._grant_threshold and denials.get(tool_name, 0) == 0:
                suggestions.append(
                    PermissionSuggestion(
                        tool_name=tool_name,
                        action="pre_approve",
                        denial_count=0,
                        grant_count=count,
                        rationale=f"Approved {count} times, never denied. Safe to pre-approve.",
                    )
                )

        return suggestions

    def generate_permission_config(self, suggestions: list[PermissionSuggestion]) -> dict:
        """Produce a config snippet for godspeed.yaml."""
        allow: list[str] = []
        for s in suggestions:
            if s.action in ("add_to_allowlist", "pre_approve"):
                allow.append(s.tool_name)

        return {"permissions": {"allow": sorted(set(allow))}}

    def get_all_suggestions(self, sessions: list[SessionTrace]) -> list[PermissionSuggestion]:
        """Get combined denial + approval suggestions."""
        return self.analyze_denials(sessions) + self.analyze_approvals(sessions)
