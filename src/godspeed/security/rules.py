"""Permission rule types for the 4-tier permission engine."""

from __future__ import annotations

import fnmatch
import logging
from enum import StrEnum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RuleAction(StrEnum):
    """What happens when a rule matches."""

    DENY = "deny"
    ALLOW = "allow"
    ASK = "ask"


class PermissionRule(BaseModel):
    """A permission rule matching tool calls by pattern.

    Pattern format: 'ToolName(argument_pattern)'
    Examples:
        - 'Bash(git *)' — matches any git command
        - 'FileRead(.env)' — matches reading .env
        - 'Bash(*)' — matches any bash command
        - 'FileRead(*.pem)' — matches reading any .pem file
    """

    pattern: str
    action: RuleAction

    def matches(self, tool_call_str: str) -> bool:
        """Check if this rule matches a formatted tool call string.

        Uses fnmatch for glob-style matching.
        """
        return fnmatch.fnmatch(tool_call_str, self.pattern)


def parse_rules(patterns: list[str], action: RuleAction) -> list[PermissionRule]:
    """Parse a list of pattern strings into PermissionRule objects."""
    return [PermissionRule(pattern=p, action=action) for p in patterns]
