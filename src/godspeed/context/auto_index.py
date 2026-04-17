"""Auto-indexing helper — kick off a non-blocking codebase index on session start.

The ChromaDB-backed index in ``codebase_index.py`` was previously opt-in:
the user had to call ``/reindex`` before the agent could use semantic
search. This helper makes it automatic and non-blocking — the session
starts immediately and the index builds in the background.

Design notes:
- Gated on ``settings.auto_index`` (default True)
- No-op if the ``[index]`` extra isn't installed (chromadb missing)
- No-op if the index is already fresh (``needs_reindex()`` returns False)
- Uses ``asyncio.create_task`` so the main loop is not blocked
- Logs completion/failure; never raises out of the scheduling path
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from godspeed.context.codebase_index import CodebaseIndex

logger = logging.getLogger(__name__)


def maybe_start_auto_index(
    project_dir: Path,
    auto_index_enabled: bool,
) -> asyncio.Task[int] | None:
    """Schedule a background index build if needed.

    Returns the asyncio.Task so the caller can await or cancel it during
    shutdown if desired. Returns None when:

    - ``auto_index_enabled`` is False
    - chromadb isn't installed (``[index]`` extra missing)
    - The index is already fresh

    Never raises — a scheduling failure is logged and the function
    returns None so the session continues normally.
    """
    if not auto_index_enabled:
        logger.debug("Auto-index disabled by settings")
        return None

    try:
        from godspeed.context.codebase_index import CodebaseIndex
    except ImportError:
        logger.debug("codebase_index module unavailable; skipping auto-index")
        return None

    try:
        index = CodebaseIndex(project_dir=project_dir)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Auto-index init failed: %s", exc)
        return None

    if not index.is_available:
        logger.info("chromadb not installed; auto-index skipped (install `godspeed[index]`)")
        return None

    if not index.needs_reindex():
        logger.debug("Codebase index is fresh; skipping rebuild")
        return None

    logger.info("Scheduling background codebase re-index project_dir=%s", project_dir)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        logger.warning("No running event loop; cannot schedule auto-index")
        return None

    return loop.create_task(_run_auto_index(index))


async def _run_auto_index(index: CodebaseIndex) -> int:
    """Coroutine wrapper: builds the index and logs the outcome.

    Swallows exceptions — an indexing failure must not crash the agent
    session. Returns the chunk count on success, 0 on failure.
    """
    try:
        count = await index.build_index_async()
        logger.info("Auto-index built chunks=%d", count)
        return count
    except Exception as exc:
        logger.warning("Auto-index build failed: %s", exc, exc_info=True)
        return 0
