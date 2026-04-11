"""Secret redaction for audit logs — ensures no secrets leak into the trail."""

from __future__ import annotations

from typing import Any

from godspeed.security.secrets import redact_secrets


def redact_audit_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets from an audit event's action_detail.

    Recursively walks the detail dict and redacts any string values
    that contain detected secrets.
    """
    return _redact_recursive(detail)


def _redact_recursive(obj: Any) -> Any:
    """Recursively redact secrets in nested data structures."""
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        return {k: _redact_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_recursive(item) for item in obj]
    return obj
