"""Tests for RetrievalSubagent."""

from __future__ import annotations

from pathlib import Path

from godspeed.agent.retrieval_subagent import (
    FileSpan,
    RetrievalResult,
    RetrievalSubagent,
)
from godspeed.context.coherence_graph import CoherenceGraph

SAMPLE_CODE = '''\
"""Sample module."""

import os


def _private() -> None:
    pass


class Processor:
    """Handles data processing."""

    def run(self, data: list[int]) -> list[int]:
        return [x * 2 for x in data]


CONFIG = {"debug": True}


def main() -> None:
    proc = Processor()
    proc.run([1, 2, 3])
'''


async def _build_retrieval(tmp_path: Path) -> RetrievalSubagent:
    db_path = tmp_path / "gcg.db"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(SAMPLE_CODE)

    gcg = CoherenceGraph(db_path)
    gcg.connect()
    gcg.build_from_repo(tmp_path, incremental=False)

    return RetrievalSubagent(gcg=gcg, repo_root=tmp_path, model="cheap")


class TestFileSpan:
    """Test FileSpan dataclass."""

    def test_creation(self) -> None:
        span = FileSpan(
            file=Path("src/main.py"),
            start_line=10,
            end_line=25,
            symbol_id="src.main.my_func",
            relevance_score=0.95,
            match_reason="gcg_direct",
        )
        assert span.file == Path("src/main.py")
        assert span.relevance_score == 0.95


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_defaults(self) -> None:
        r = RetrievalResult()
        assert r.spans == []
        assert r.gcg_hits == 0
        assert r.cache_hit is False


class TestGCGLookup:
    """Test GCG-first lookup."""

    async def test_find_symbol(self, tmp_path: Path) -> None:
        sub = await _build_retrieval(tmp_path)
        result = await sub.retrieve("Processor")
        assert isinstance(result, RetrievalResult)
        assert result.gcg_hits > 0
        assert any("gcg_direct" in s.match_reason for s in result.spans)

    async def test_find_function(self, tmp_path: Path) -> None:
        sub = await _build_retrieval(tmp_path)
        result = await sub.retrieve("main")
        assert isinstance(result, RetrievalResult)
        assert result.gcg_hits > 0

    async def test_miss_falls_back(self, tmp_path: Path) -> None:
        sub = await _build_retrieval(tmp_path)
        result = await sub.retrieve("nonexistent_symbol_xyz")
        assert isinstance(result, RetrievalResult)
        # GCG won't find it, but result should still be valid
        assert result.gcg_hits == 0


class TestSpanDeduplication:
    """Test span deduplication."""

    def test_removes_duplicates(self, tmp_path: Path) -> None:
        sub = RetrievalSubagent(
            gcg=None,  # type: ignore[arg-type]
            repo_root=tmp_path,
        )
        spans = [
            FileSpan(file=Path("a.py"), start_line=1, end_line=5),
            FileSpan(file=Path("a.py"), start_line=1, end_line=5),  # dup
            FileSpan(file=Path("b.py"), start_line=10, end_line=15),
        ]
        result = sub._rank_and_deduplicate(spans, max_spans=10)
        assert len(result) == 2

    def test_caps_at_max_spans(self, tmp_path: Path) -> None:
        sub = RetrievalSubagent(
            gcg=None,  # type: ignore[arg-type]
            repo_root=tmp_path,
        )
        spans = [
            FileSpan(
                file=Path(f"f{i}.py"),
                start_line=i,
                end_line=i + 1,
                relevance_score=float(10 - i),
            )
            for i in range(10)
        ]
        result = sub._rank_and_deduplicate(spans, max_spans=3)
        assert len(result) == 3
        # Highest score should be first
        assert result[0].relevance_score == 10.0


class TestCache:
    """Test span cache."""

    async def test_cache_hit(self, tmp_path: Path) -> None:
        sub = await _build_retrieval(tmp_path)
        _result1 = await sub.retrieve("Processor")
        result2 = await sub.retrieve("Processor")
        assert result2.cache_hit is True


class TestFormatSpans:
    """Test format_spans_for_agent."""

    def test_empty_spans(self, tmp_path: Path) -> None:
        sub = RetrievalSubagent(
            gcg=None,  # type: ignore[arg-type]
            repo_root=tmp_path,
        )
        output = sub.format_spans_for_agent([])
        assert "No relevant" in output

    def test_formats_spans(self, tmp_path: Path) -> None:
        sub = RetrievalSubagent(
            gcg=None,  # type: ignore[arg-type]
            repo_root=tmp_path,
        )
        spans = [
            FileSpan(
                file=Path("src/auth.py"),
                start_line=145,
                end_line=167,
                symbol_id="AuthManager.sign_token",
                match_reason="gcg_direct",
            ),
        ]
        output = sub.format_spans_for_agent(spans)
        assert "auth.py" in output
        assert "gcg_direct" in output
