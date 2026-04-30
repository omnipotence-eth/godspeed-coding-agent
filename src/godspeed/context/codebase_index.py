"""Persistent codebase index using ChromaDB for semantic code search."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from godspeed.context.chunker import chunk_file

logger = logging.getLogger(__name__)

# Default exclusions for indexing — frozenset for O(1) lookups and efficient union ops
DEFAULT_EXCLUDES: frozenset[str] = frozenset(
    {
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
    }
)

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


_chromadb_available: bool | None = None
_chromadb_lock = threading.Lock()


def _is_chromadb_available() -> bool:
    """Check if chromadb is installed. Result is cached after first call (thread-safe)."""
    global _chromadb_available
    if _chromadb_available is None:
        with _chromadb_lock:
            if _chromadb_available is None:
                try:
                    import chromadb  # noqa: F401

                    _chromadb_available = True
                except ImportError:
                    _chromadb_available = False
    return _chromadb_available


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

    def close(self) -> None:
        """Release ChromaDB client and collection references.

        ChromaDB's PersistentClient holds open file handles; dropping
        references allows the underlying resources to be reclaimed.
        """
        self._collection = None
        self._client = None

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

        Streams chunks to ChromaDB in fixed-size batches to avoid memory
        exhaustion on large codebases. Previously accumulated all chunks
        in memory before writing; now writes incrementally.

        Args:
            exclude: Additional directory/file patterns to exclude.

        Returns:
            Number of chunks indexed.
        """
        if not self.is_available:
            logger.warning("ChromaDB not available. Install with: pip install godspeed[index]")
            return 0

        self._building = True
        excludes = DEFAULT_EXCLUDES | set(exclude or [])

        try:
            collection = self._ensure_collection()

            # Clear existing data
            existing = collection.count()
            if existing > 0:
                collection.delete(where={"indexed": True})

            # Stream chunks to ChromaDB in batches — avoid accumulating all
            # chunks in memory for large codebases.
            batch_size = 100
            batch: list[Any] = []
            total = 0
            chunk_counter = 0

            for path in self._iter_files(excludes):
                for c in chunk_file(path):
                    batch.append(c)
                    chunk_counter += 1
                    if len(batch) >= batch_size:
                        self._add_batch(collection, batch, chunk_counter - len(batch))
                        total += len(batch)
                        batch = []

            # Flush remaining chunks
            if batch:
                self._add_batch(collection, batch, chunk_counter - len(batch))
                total += len(batch)

            if total == 0:
                logger.info("No files to index in %s", self._project_dir)
                return 0

            self._index_time = time.time()
            logger.info(
                "Indexed %d chunks from %s",
                total,
                self._project_dir,
            )
            return total

        finally:
            self._building = False

    @staticmethod
    def _add_batch(collection: Any, batch: list[Any], base_id: int) -> None:
        """Add a batch of chunks to the ChromaDB collection."""
        collection.add(
            ids=[f"chunk_{base_id + j}" for j in range(len(batch))],
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
            doc_count = collection.count()
            if doc_count == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, doc_count),
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

    def _iter_files(self, excludes: set[str] | frozenset[str]) -> list[Path]:
        """Iterate indexable files, skipping excluded directories eagerly.

        Returns a sorted list for deterministic ordering during indexing.
        Uses iterative DFS to avoid recursion limits on deep directory trees.
        """
        files: list[Path] = []
        dirs_to_visit: list[Path] = [self._project_dir]
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
        files.sort()
        return files
