"""PDF read tool — extract text from PDF files with page range support."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

MAX_PAGES_PER_REQUEST = 20


def parse_page_range(pages_str: str, total_pages: int) -> tuple[list[int], str | None]:
    """Parse a page range string into 0-based page indices.

    Args:
        pages_str: Human-readable page range, e.g. "1-5", "3", "10-20".
        total_pages: Total number of pages in the document.

    Returns:
        Tuple of (list of 0-based page indices, error message or None).
    """
    pages_str = pages_str.strip()
    if not pages_str:
        return [], "pages must be a non-empty string"

    if "-" in pages_str:
        parts = pages_str.split("-", maxsplit=1)
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            return [], f"Invalid page range: '{pages_str}'. Expected format: '1-5' or '3'"
        if start < 1:
            return [], f"Page numbers must be >= 1, got {start}"
        if end < start:
            return [], f"Invalid range: end ({end}) < start ({start})"
        if start > total_pages:
            return [], f"Start page {start} exceeds document length ({total_pages} pages)"
        # Clamp end to total_pages
        end = min(end, total_pages)
        indices = list(range(start - 1, end))
    else:
        try:
            page_num = int(pages_str)
        except ValueError:
            return [], f"Invalid page number: '{pages_str}'. Expected format: '1-5' or '3'"
        if page_num < 1:
            return [], f"Page numbers must be >= 1, got {page_num}"
        if page_num > total_pages:
            return [], f"Page {page_num} exceeds document length ({total_pages} pages)"
        indices = [page_num - 1]

    if len(indices) > MAX_PAGES_PER_REQUEST:
        return [], (
            f"Requested {len(indices)} pages, max is {MAX_PAGES_PER_REQUEST} per request. "
            f"Use a smaller range."
        )

    return indices, None


class PdfReadTool(Tool):
    """Read PDF files and extract text content with page range support.

    Requires pymupdf (aka fitz). Install with: pip install pymupdf
    """

    @property
    def name(self) -> str:
        return "pdf_read"

    @property
    def description(self) -> str:
        return (
            "Read a PDF file and extract text content. "
            "Supports page ranges like '1-5', '3', or '10-20'. "
            "Maximum 20 pages per request."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to read (relative to project root)",
                },
                "pages": {
                    "type": "string",
                    "description": (
                        "Page range to read, e.g. '1-5', '3', '10-20'. "
                        "1-based. Max 20 pages per request. "
                        "Omit to read from the beginning."
                    ),
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            import pymupdf
        except ImportError:
            return ToolResult.failure(
                "PDF support requires pymupdf. Install with: pip install pymupdf"
            )

        file_path_str = arguments.get("file_path", "")
        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")

        try:
            file_path = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not file_path.exists():
            return ToolResult.failure(f"File not found: {file_path}")

        if not file_path.is_file():
            return ToolResult.failure(f"Not a file: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            return ToolResult.failure(
                f"Not a PDF file: {file_path.name}. Only .pdf files are supported."
            )

        try:
            doc = pymupdf.open(str(file_path))
        except Exception as exc:
            logger.error("Failed to open PDF %s", file_path, exc_info=True)
            return ToolResult.failure(f"Failed to open PDF: {exc}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return ToolResult.failure("PDF has no pages")

        pages_str = arguments.get("pages")
        if pages_str is not None:
            if not isinstance(pages_str, str):
                pages_str = str(pages_str)
            page_indices, error = parse_page_range(pages_str, total_pages)
            if error:
                doc.close()
                return ToolResult.failure(error)
        else:
            # Default: read up to MAX_PAGES_PER_REQUEST from the beginning
            page_count = min(total_pages, MAX_PAGES_PER_REQUEST)
            page_indices = list(range(page_count))

        blocks: list[str] = []
        for idx in page_indices:
            page = doc[idx]
            text = page.get_text()
            blocks.append(f"[Page {idx + 1}]:\n{text}")

        doc.close()

        output = "\n".join(blocks)
        if total_pages > len(page_indices) and pages_str is None:
            output += (
                f"\n\n... ({total_pages - len(page_indices)} more pages. "
                f"Use 'pages' parameter to read specific ranges.)"
            )

        return ToolResult.success(output)
