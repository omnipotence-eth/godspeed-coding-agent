"""Persistent codebase index using ChromaDB for semantic code search."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from godspeed.context.chunker import chunk_file

logger = logging.getLogger(__name__)

# Default exclusions for indexing
DEFAULT_EXCLUDES = [
    "node_modules",
    ".venv",
    "__pycache__",
    ".git",
    ".godspeed",
    "*.pyc",
    "*.pyo",
    ".egg-info",
    "dist",
    "build",
]

# File extensions to index
INDEXABLE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
}


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search result from the codebase index."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float


def _is_chromadb_available() -> bool:
    """Check if chromadb is installed."""
    try:
        import chromadb  # noqa: F401

        return True
    except ImportError:
        return False


class CodebaseIndex:
    """Persistent codebase index backed by ChromaDB.

    Provides semantic search over source code. Gracefully degrades
    if chromadb is not installed.

    Args:
        project_dir: Root directory of the project.
        db_path: Path for ChromaDB persistent storage.
            Defaults to ``project_dir/.godspeed/index/chroma``.
    """

    def __init__(
        self,
        project_dir: Path,
        db_path: Path | None = None,
    ) -> None:
        self._project_dir = project_dir
        self._db_path = db_path or (project_dir / ".godspeed" / "index" / "chroma")
        self._collection: Any | None = None
        self._client: Any | None = None
        self._building = False
        self._index_time: float | None = None

    @property
    def is_available(self) -> bool:
        """Whether chromadb is installed and usable."""
        return _is_chromadb_available()

    @property
    def is_building(self) -> bool:
        """Whether the index is currently being built."""
        return self._building

    def _ensure_collection(self) -> Any:
        """Get or create the ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        import chromadb

        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))
        self._collection = self._client.get_or_create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def build_index(self, exclude: list[str] | None = None) -> int:
        """Build the index by scanning and chunking all source files.

        Args:
            exclude: Additional directory/file patterns to exclude.

        Returns:
            Number of chunks indexed.
        """
        if not self.is_available:
            logger.warning("ChromaDB not available. Install with: pip install godspeed[index]")
            return 0

        self._building = True
        excludes = set(DEFAULT_EXCLUDES + (exclude or []))

        try:
            collection = self._ensure_collection()

            # Clear existing data
            existing = collection.count()
            if existing > 0:
                collection.delete(where={"indexed": True})

            # Scan files
            all_chunks = []
            for path in self._iter_files(excludes):
                chunks = chunk_file(path)
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.info("No files to index in %s", self._project_dir)
                return 0

            # Batch add to ChromaDB
            batch_size = 100
            total = 0
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                collection.add(
                    ids=[f"chunk_{i + j}" for j in range(len(batch))],
                    documents=[c.content for c in batch],
                    metadatas=[
                        {
                            "file_path": c.file_path,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                            "indexed": True,
                        }
                        for c in batch
                    ],
                )
                total += len(batch)

            self._index_time = time.time()
            logger.info(
                "Indexed %d chunks from %s",
                total,
                self._project_dir,
            )
            return total

        finally:
            self._building = False

    async def build_index_async(self, exclude: list[str] | None = None) -> int:
        """Build index in a background thread."""
        return await asyncio.get_event_loop().run_in_executor(None, self.build_index, exclude)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the codebase index.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.

        Returns:
            Ranked list of SearchResult objects.
        """
        if not self.is_available:
            return []

        if self._building:
            return []

        try:
            collection = self._ensure_collection()
            if collection.count() == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
            )

            search_results = []
            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
                dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, dists, strict=False):
                    score = max(0.0, 1.0 - dist)
                    search_results.append(
                        SearchResult(
                            file_path=meta.get("file_path", ""),
                            start_line=meta.get("start_line", 0),
                            end_line=meta.get("end_line", 0),
                            content=doc,
                            score=round(score, 3),
                        )
                    )

            return search_results

        except Exception as exc:
            logger.warning("Search failed: %s", exc)
            return []

    def needs_reindex(self) -> bool:
        """Check if the index is stale (newer files exist than index time)."""
        if self._index_time is None:
            # Check if DB exists with data
            if not self._db_path.exists():
                return True
            if self.is_available:
                try:
                    collection = self._ensure_collection()
                    return collection.count() == 0
                except Exception:
                    return True
            return True

        # Check if any source file is newer than index
        for path in self._iter_files(set(DEFAULT_EXCLUDES)):
            try:
                if path.stat().st_mtime > self._index_time:
                    return True
            except OSError:
                continue

        return False

    def _iter_files(self, excludes: set[str]) -> list[Path]:
        """Iterate indexable files, skipping excluded directories eagerly."""
        files = []
        dirs_to_visit = [self._project_dir]
        while dirs_to_visit:
            current = dirs_to_visit.pop()
            try:
                for entry in current.iterdir():
                    if entry.is_dir():
                        if entry.name not in excludes:
                            dirs_to_visit.append(entry)
                    elif entry.is_file() and entry.suffix in INDEXABLE_EXTENSIONS:
                        files.append(entry)
            except (PermissionError, OSError):
                continue
        return sorted(files)
