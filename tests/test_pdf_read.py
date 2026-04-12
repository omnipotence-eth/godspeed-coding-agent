"""Tests for PDF read tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.pdf_read import MAX_PAGES_PER_REQUEST, PdfReadTool, parse_page_range


@pytest.fixture()
def tool() -> PdfReadTool:
    return PdfReadTool()


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


def _make_mock_doc(page_texts: list[str]) -> MagicMock:
    """Create a mock pymupdf document with given page texts."""
    doc = MagicMock()
    doc.__len__ = lambda self: len(page_texts)

    pages = []
    for text in page_texts:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)

    doc.__getitem__ = lambda self, idx: pages[idx]
    return doc


class TestParsePageRange:
    """Unit tests for parse_page_range."""

    def test_single_page(self) -> None:
        indices, error = parse_page_range("3", 10)
        assert error is None
        assert indices == [2]

    def test_page_range(self) -> None:
        indices, error = parse_page_range("1-5", 10)
        assert error is None
        assert indices == [0, 1, 2, 3, 4]

    def test_page_range_clamped_to_total(self) -> None:
        indices, error = parse_page_range("8-15", 10)
        assert error is None
        assert indices == [7, 8, 9]

    def test_single_page_out_of_range(self) -> None:
        _indices, error = parse_page_range("11", 10)
        assert error is not None
        assert "exceeds" in error

    def test_start_page_out_of_range(self) -> None:
        _indices, error = parse_page_range("15-20", 10)
        assert error is not None
        assert "exceeds" in error

    def test_end_less_than_start(self) -> None:
        _indices, error = parse_page_range("5-3", 10)
        assert error is not None
        assert "end" in error.lower() or "Invalid" in error

    def test_invalid_format(self) -> None:
        _indices, error = parse_page_range("abc", 10)
        assert error is not None
        assert "Invalid" in error

    def test_invalid_range_format(self) -> None:
        _indices, error = parse_page_range("a-b", 10)
        assert error is not None
        assert "Invalid" in error

    def test_empty_string(self) -> None:
        _indices, error = parse_page_range("", 10)
        assert error is not None

    def test_page_zero(self) -> None:
        _indices, error = parse_page_range("0", 10)
        assert error is not None
        assert ">= 1" in error

    def test_negative_page(self) -> None:
        _indices, error = parse_page_range("-1", 10)
        assert error is not None

    def test_max_pages_exceeded(self) -> None:
        _indices, error = parse_page_range("1-25", 100)
        assert error is not None
        assert str(MAX_PAGES_PER_REQUEST) in error


class TestPdfReadToolProperties:
    """Test tool metadata."""

    def test_name(self, tool: PdfReadTool) -> None:
        assert tool.name == "pdf_read"

    def test_risk_level(self, tool: PdfReadTool) -> None:
        from godspeed.tools.base import RiskLevel

        assert tool.risk_level == RiskLevel.READ_ONLY

    def test_schema_has_required_file_path(self, tool: PdfReadTool) -> None:
        schema = tool.get_schema()
        assert "file_path" in schema["properties"]
        assert "pages" in schema["properties"]
        assert schema["required"] == ["file_path"]


class TestPdfReadToolExecute:
    """Test tool execution with mocked pymupdf."""

    @pytest.mark.asyncio()
    async def test_pymupdf_not_installed(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        with patch.dict("sys.modules", {"pymupdf": None}):
            result = await tool.execute({"file_path": "test.pdf"}, ctx)

        assert result.is_error
        assert "pymupdf" in result.error
        assert "pip install pymupdf" in result.error

    @pytest.mark.asyncio()
    async def test_file_not_found(self, tool: PdfReadTool, ctx: ToolContext) -> None:
        mock_pymupdf = MagicMock()
        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "nonexistent.pdf"}, ctx)

        assert result.is_error
        assert "File not found" in result.error

    @pytest.mark.asyncio()
    async def test_path_traversal_blocked(self, tool: PdfReadTool, ctx: ToolContext) -> None:
        mock_pymupdf = MagicMock()
        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "../../etc/passwd"}, ctx)

        assert result.is_error
        assert "Access denied" in result.error

    @pytest.mark.asyncio()
    async def test_empty_file_path(self, tool: PdfReadTool, ctx: ToolContext) -> None:
        mock_pymupdf = MagicMock()
        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": ""}, ctx)

        assert result.is_error
        assert "non-empty string" in result.error

    @pytest.mark.asyncio()
    async def test_valid_pdf_read_all_pages(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        mock_doc = _make_mock_doc(["Page one text.", "Page two text.", "Page three text."])
        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "test.pdf"}, ctx)

        assert not result.is_error
        assert "[Page 1]:" in result.output
        assert "Page one text." in result.output
        assert "[Page 2]:" in result.output
        assert "Page two text." in result.output
        assert "[Page 3]:" in result.output
        assert "Page three text." in result.output

    @pytest.mark.asyncio()
    async def test_single_page_read(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        mock_doc = _make_mock_doc(["First.", "Second.", "Third."])
        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "test.pdf", "pages": "2"}, ctx)

        assert not result.is_error
        assert "[Page 2]:" in result.output
        assert "Second." in result.output
        assert "[Page 1]:" not in result.output
        assert "[Page 3]:" not in result.output

    @pytest.mark.asyncio()
    async def test_page_range_read(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        texts = [f"Content of page {i + 1}." for i in range(10)]
        mock_doc = _make_mock_doc(texts)
        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "test.pdf", "pages": "1-5"}, ctx)

        assert not result.is_error
        for i in range(1, 6):
            assert f"[Page {i}]:" in result.output
            assert f"Content of page {i}." in result.output
        assert "[Page 6]:" not in result.output

    @pytest.mark.asyncio()
    async def test_max_pages_limit_enforced(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        texts = [f"Page {i}" for i in range(50)]
        mock_doc = _make_mock_doc(texts)
        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "test.pdf", "pages": "1-25"}, ctx)

        assert result.is_error
        assert str(MAX_PAGES_PER_REQUEST) in result.error

    @pytest.mark.asyncio()
    async def test_default_caps_at_max_pages(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        """When no pages specified and doc has >20 pages, only first 20 are returned."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        texts = [f"Page {i + 1}" for i in range(30)]
        mock_doc = _make_mock_doc(texts)
        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "test.pdf"}, ctx)

        assert not result.is_error
        assert "[Page 20]:" in result.output
        assert "[Page 21]:" not in result.output
        assert "more pages" in result.output

    @pytest.mark.asyncio()
    async def test_non_pdf_file_rejected(
        self, tool: PdfReadTool, ctx: ToolContext, tmp_path: Path
    ) -> None:
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("not a pdf")

        mock_pymupdf = MagicMock()
        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "notes.txt"}, ctx)

        assert result.is_error
        assert "Not a PDF file" in result.error

    @pytest.mark.asyncio()
    async def test_windows_absolute_path_blocked(self, tool: PdfReadTool, ctx: ToolContext) -> None:
        mock_pymupdf = MagicMock()
        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            result = await tool.execute({"file_path": "C:\\Windows\\system.pdf"}, ctx)

        assert result.is_error
        assert "Access denied" in result.error
