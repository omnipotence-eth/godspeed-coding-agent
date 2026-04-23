"""Shared aiohttp session for HTTP tools.

Provides a singleton session with connection pooling for reduced latency
on consecutive HTTP calls. Session is created on first use and closed
when the event loop closes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy session — created on first use
_session: Any = None
_loop: Any = None


def _get_event_loop():
    """Get the current event loop."""
    import asyncio

    return asyncio.get_event_loop()


async def _create_session() -> Any:
    """Create a new aiohttp session with connection pooling."""
    try:
        import aiohttp
    except ImportError:
        logger.warning("aiohttp not installed; falling back to urllib")
        return None

    # Connection pooling configuration
    connector = aiohttp.TCPConnector(
        limit=100,  # Max total connections
        limit_per_host=10,  # Max connections per host
        ttl_dns_cache=300,  # DNS cache TTL (seconds)
        use_dns_cache=True,
    )

    session = aiohttp.ClientSession(connector=connector)
    return session


async def get_session() -> Any | None:
    """Get or create the shared aiohttp session.

    Returns None if aiohttp is not installed.
    """
    global _session, _loop

    current_loop = _get_event_loop()
    if _loop is not current_loop:
        # Loop changed; old session is invalid
        if _session is not None:
            await _session.close()
        _session = None
        _loop = current_loop

    if _session is None:
        _session = await _create_session()

    return _session


async def close_session() -> None:
    """Close the shared session."""
    global _session, _loop
    if _session is not None:
        await _session.close()
        _session = None
        _loop = None
    logger.debug("HTTP session closed")
