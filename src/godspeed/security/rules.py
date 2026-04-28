"""Permission rule types for the 4-tier permission engine."""

from __future__ import annotations

import fnmatch
import logging
import re
from enum import StrEnum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _compile_glob(pattern: str) -> re.Pattern:
    """Compile a fnmatch glob pattern to a regex once."""
    return re.compile(fnmatch.translate(pattern))


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

    The glob pattern is compiled to a regex at construction time for
    ~3-5x faster matching per evaluation.
    """

    pattern: str
    action: RuleAction

    def model_post_init(self, _context: object) -> None:
        self._compiled: re.Pattern = _compile_glob(self.pattern)

    def matches(self, tool_call_str: str) -> bool:
        """Check if this rule matches a formatted tool call string (compiled regex)."""
        return bool(self._compiled.match(tool_call_str))


def parse_rules(patterns: list[str], action: RuleAction) -> list[PermissionRule]:
    """Parse a list of pattern strings into PermissionRule objects."""
    return [PermissionRule(pattern=p, action=action) for p in patterns]
