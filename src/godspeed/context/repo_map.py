"""Repo map — tree-sitter-based symbol extraction for codebase overview.

Parses source files to extract top-level symbols (classes, functions, methods)
and produces a compressed outline for the LLM's context window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.tools.excludes import is_excluded

logger = logging.getLogger(__name__)

# File extension → tree-sitter language name
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
}

# Tree-sitter node types that represent symbols we want to extract
SYMBOL_NODE_TYPES = frozenset(
    {
        "function_definition",
        "class_definition",
        "decorated_definition",
        # JS/TS
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
        # Go
        "method_declaration",
        "type_declaration",
    }
)


class Symbol:
    """A parsed symbol from a source file."""

    __slots__ = ("children", "kind", "line", "name")

    def __init__(self, name: str, kind: str, line: int) -> None:
        self.name = name
        self.kind = kind
        self.line = line
        self.children: list[Symbol] = []

    def format(self, indent: int = 0) -> str:
        """Format as compressed outline string."""
        prefix = "  " * indent
        parts = [f"{prefix}{self.name}(L{self.line})"]
        for child in self.children:
            parts.append(child.format(indent + 1))
        return "\n".join(parts)


class RepoMapper:
    """Extract symbol outlines from source files using tree-sitter.

    Gracefully degrades when tree-sitter is not installed — returns
    an empty map instead of crashing.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}
        self._available = self._check_availability()

    @staticmethod
    def _check_availability() -> bool:
        """Check if tree-sitter is installed."""
        try:
            from tree_sitter_language_pack import get_parser  # noqa: F401

            return True
        except ImportError:
            logger.info("tree-sitter not available — repo map disabled")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def _get_parser(self, language: str) -> Any:
        """Get or create a parser for the given language."""
        if language not in self._parsers:
            from tree_sitter_language_pack import get_parser

            self._parsers[language] = get_parser(language)
        return self._parsers[language]

    def parse_file(self, file_path: Path) -> list[Symbol]:
        """Parse a single file and return top-level symbols."""
        if not self._available:
            return []

        suffix = file_path.suffix.lower()
        language = LANGUAGE_MAP.get(suffix)
        if language is None:
            return []

        try:
            source = file_path.read_bytes()
        except (OSError, PermissionError):
            return []

        if not source.strip():
            return []

        try:
            parser = self._get_parser(language)
            tree = parser.parse(source)
        except Exception as exc:
            logger.warning("tree-sitter parse failed file=%s error=%s", file_path, exc)
            return []

        return self._extract_symbols(tree.root_node, language)

    def _extract_symbols(self, node: Any, language: str) -> list[Symbol]:
        """Walk tree-sitter AST and extract symbols."""
        symbols: list[Symbol] = []

        for child in node.children:
            symbol = self._node_to_symbol(child, language)
            if symbol is not None:
                # Extract methods from class body
                if symbol.kind == "class":
                    self._extract_class_members(child, symbol, language)
                symbols.append(symbol)

        return symbols

    def _node_to_symbol(self, node: Any, language: str) -> Symbol | None:
        """Convert a tree-sitter node to a Symbol, or None if not a symbol."""
        node_type = node.type

        # Handle decorated definitions (Python)
        if node_type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return self._node_to_symbol(child, language)
            return None

        # Handle export statements (JS/TS)
        if node_type == "export_statement":
            for child in node.children:
                sym = self._node_to_symbol(child, language)
                if sym is not None:
                    return sym
            return None

        if node_type in ("function_definition", "function_declaration"):
            name = self._get_name(node)
            if name:
                return Symbol(name=name, kind="function", line=node.start_point[0] + 1)

        if node_type in ("class_definition", "class_declaration"):
            name = self._get_name(node)
            if name:
                return Symbol(name=name, kind="class", line=node.start_point[0] + 1)

        # Go method declarations
        if node_type == "method_declaration":
            name = self._get_name(node)
            if name:
                return Symbol(name=name, kind="method", line=node.start_point[0] + 1)

        # Go type declarations
        if node_type == "type_declaration":
            name = self._get_name(node)
            if name:
                return Symbol(name=name, kind="type", line=node.start_point[0] + 1)

        return None

    def _extract_class_members(self, class_node: Any, symbol: Symbol, language: str) -> None:
        """Extract methods/properties from a class body."""
        for child in class_node.children:
            # Python: class body is a 'block' node
            if child.type in ("block", "class_body"):
                for member in child.children:
                    member_sym = self._member_to_symbol(member, language)
                    if member_sym is not None:
                        symbol.children.append(member_sym)

    def _member_to_symbol(self, node: Any, language: str) -> Symbol | None:
        """Convert a class member node to a Symbol."""
        if node.type == "decorated_definition":
            for child in node.children:
                sym = self._member_to_symbol(child, language)
                if sym is not None:
                    return sym
            return None

        if node.type in ("function_definition", "method_definition"):
            name = self._get_name(node)
            if name:
                return Symbol(name=name, kind="method", line=node.start_point[0] + 1)

        return None

    @staticmethod
    def _get_name(node: Any) -> str | None:
        """Extract the name identifier from a definition node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()
            # Go type_spec inside type_declaration
            if child.type == "type_spec":
                for subchild in child.children:
                    if subchild.type == "type_identifier":
                        return subchild.text.decode()
        return None

    def map_directory(
        self,
        directory: Path,
        max_depth: int = 5,
        pattern: str = "",
    ) -> str:
        """Generate a repo map for a directory.

        Args:
            directory: Root directory to scan.
            max_depth: Maximum directory depth to traverse.
            pattern: Optional glob pattern to filter files (e.g. "*.py").

        Returns:
            Formatted string outline of the codebase symbols.
        """
        if not self._available:
            return "tree-sitter not available. Install with: pip install godspeed[context]"

        if not directory.is_dir():
            return f"Not a directory: {directory}"

        lines: list[str] = []
        files = sorted(directory.rglob(pattern or "*"))

        for file_path in files:
            if not file_path.is_file():
                continue
            if is_excluded(file_path.relative_to(directory)):
                continue
            if file_path.suffix.lower() not in LANGUAGE_MAP:
                continue

            # Check depth
            rel = file_path.relative_to(directory)
            if len(rel.parts) - 1 > max_depth:
                continue

            symbols = self.parse_file(file_path)
            if not symbols:
                continue

            rel_str = str(rel).replace("\\", "/")
            lines.append(f"{rel_str}: {', '.join(s.format() for s in symbols)}")

        if not lines:
            return "No symbols found in directory."

        return "\n".join(lines)
