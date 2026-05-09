"""Tests for WikiBridge — wiki page to SKILL.md generation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from godspeed.skills.wiki_bridge import WikiBridge


# ── Helpers ────────────────────────────────────────────────────────────


def _strip(text: str) -> str:
    """Remove leading/trailing whitespace from each line in a block of text."""
    return "\n".join(line.strip() for line in text.strip().splitlines())


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def wiki_dir() -> Path:
    """Create a temp wiki directory with sample pages.

    Filenames are carefully chosen so that slugified topics match them.
    """
    d = Path(tempfile.mkdtemp())
    (d / "nvfp4-benchmarks.md").write_text(_strip("""
        ---
        title: NVFP4 Benchmarks
        short: NVFP4 performance results
        tags: [nvfp4, blackwell, benchmark, performance]
        confidence: high
        ---

        # NVFP4 Benchmarks

        ## Results
        - PP: 993 tok/s at 50 GPU layers
        - TG: 1.06 tok/s at 50 GPU layers
    """))
    (d / "qwen3-architecture.md").write_text(_strip("""
        ---
        title: Qwen3 Architecture
        short: Architecture overview of Qwen3
        tags: [qwen3, architecture, transformer]
        ---

        # Qwen3 Architecture

        Qwen3 uses a standard transformer decoder architecture.
    """))
    (d / "untagged-page.md").write_text(_strip("""
        ---
        title: Untagged
        short: No tags here
        tags: [untagged]
        ---

        Just some content.
    """))
    (d / "no-frontmatter.md").write_text(_strip("""
        Just a plain page with no frontmatter.
    """))
    return d


@pytest.fixture
def bridge(wiki_dir: Path) -> WikiBridge:
    return WikiBridge(wiki_dir=str(wiki_dir))


# ── Constructor ───────────────────────────────────────────────────────


class TestConstructor:
    def test_default_wiki_dir_uses_home(self) -> None:
        bridge = WikiBridge()
        assert bridge._wiki_dir is not None
        assert isinstance(bridge._wiki_dir, Path)

    def test_uses_provided_wiki_dir(self) -> None:
        bridge = WikiBridge(wiki_dir=str(Path("/tmp", "test-wiki")))
        assert bridge._wiki_dir == Path("/tmp", "test-wiki")

    def test_output_dir_is_godspeed_skills(self) -> None:
        bridge = WikiBridge(wiki_dir="/tmp")
        assert bridge._output_dir == Path.home() / ".godspeed" / "skills"


# ── Slugify ───────────────────────────────────────────────────────────


class TestSlugify:
    def test_lowercases(self) -> None:
        assert WikiBridge._slugify("Hello World") == "hello-world"

    def test_replaces_spaces(self) -> None:
        assert WikiBridge._slugify("foo bar baz") == "foo-bar-baz"

    def test_replaces_underscores(self) -> None:
        assert WikiBridge._slugify("foo_bar") == "foo-bar"

    def test_strips_special_chars(self) -> None:
        assert WikiBridge._slugify("hello!@#world") == "helloworld"

    def test_collapses_multiple_hyphens(self) -> None:
        assert WikiBridge._slugify("foo---bar") == "foo-bar"

    def test_strips_trailing_hyphens(self) -> None:
        assert WikiBridge._slugify("-hello-") == "hello"

    def test_empty_string(self) -> None:
        assert WikiBridge._slugify("") == ""

    def test_handles_single_word(self) -> None:
        assert WikiBridge._slugify("hello") == "hello"

    def test_handles_mixed_case(self) -> None:
        assert WikiBridge._slugify("NVFP4 Benchmarks") == "nvfp4-benchmarks"


# ── Frontmatter extraction ────────────────────────────────────────────


class TestExtractFrontmatter:
    def test_extracts_yaml(self) -> None:
        text = "---\ntitle: Hello\n---\nbody"
        result = WikiBridge._extract_frontmatter(text)
        assert result == {"title": "Hello"}

    def test_returns_empty_for_no_frontmatter(self) -> None:
        text = "Just content\nno frontmatter"
        result = WikiBridge._extract_frontmatter(text)
        assert result == {}

    def test_returns_empty_for_malformed_yaml(self) -> None:
        text = "---\n: invalid yaml\n---\nbody"
        result = WikiBridge._extract_frontmatter(text)
        assert result == {}

    def test_returns_empty_for_non_dict_yaml(self) -> None:
        text = "---\n- list\n- item\n---\nbody"
        result = WikiBridge._extract_frontmatter(text)
        assert result == {}

    def test_handles_empty_frontmatter(self) -> None:
        text = "------\nbody"
        result = WikiBridge._extract_frontmatter(text)
        assert result == {}

    def test_extracts_multi_field_yaml(self) -> None:
        text = "---\ntitle: Test\ntags: [a, b]\nconfidence: high\n---\nbody"
        result = WikiBridge._extract_frontmatter(text)
        assert result["title"] == "Test"
        assert result["tags"] == ["a", "b"]
        assert result["confidence"] == "high"


# ── Strip frontmatter ────────────────────────────────────────────────


class TestStripFrontmatter:
    def test_strips_frontmatter_block(self) -> None:
        text = "---\ntitle: Hello\n---\nbody content"
        result = WikiBridge._strip_frontmatter(text)
        assert result == "body content"

    def test_returns_original_if_no_frontmatter(self) -> None:
        text = "just content"
        result = WikiBridge._strip_frontmatter(text)
        assert result == text

    def test_returns_original_if_unclosed(self) -> None:
        text = "---\ntitle: unclosed"
        result = WikiBridge._strip_frontmatter(text)
        assert result == text

    def test_handles_empty_frontmatter_delims(self) -> None:
        text = "------\nbody"
        result = WikiBridge._strip_frontmatter(text)
        assert result == "body"


# ── Clean body ────────────────────────────────────────────────────────


class TestCleanBody:
    def test_removes_horizontal_rules(self) -> None:
        body = "content\n---\nmore"
        result = WikiBridge._clean_body(body)
        assert result == "content\nmore"

    def test_preserves_content_without_rules(self) -> None:
        body = "# Hello\n\nSome text"
        result = WikiBridge._clean_body(body)
        assert result == "# Hello\n\nSome text"

    def test_handles_empty_body(self) -> None:
        assert WikiBridge._clean_body("") == ""

    def test_removes_multiple_rules(self) -> None:
        body = "a\n---\nb\n---\nc"
        result = WikiBridge._clean_body(body)
        assert result == "a\nb\nc"


# ── Find page ─────────────────────────────────────────────────────────


class TestFindPage:
    def test_finds_by_exact_slug(self, bridge: WikiBridge) -> None:
        page = bridge._find_page("nvfp4-benchmarks")
        assert page is not None
        assert page.name == "nvfp4-benchmarks.md"

    def test_finds_by_partial_match(self, bridge: WikiBridge) -> None:
        page = bridge._find_page("nvfp4")
        assert page is not None
        assert "nvfp4" in page.name

    def test_returns_none_for_no_match(self, bridge: WikiBridge) -> None:
        page = bridge._find_page("nonexistent-topic-xyz")
        assert page is None

    def test_finds_by_slug_directly(self, bridge: WikiBridge) -> None:
        page = bridge._find_page("qwen3-architecture")
        assert page is not None
        assert page.name == "qwen3-architecture.md"

    def test_uses_slug_for_topic(self, bridge: WikiBridge) -> None:
        page = bridge._find_page("NVFP4 Benchmarks")
        assert page is not None
        assert page.name == "nvfp4-benchmarks.md"

    def test_finds_no_frontmatter_page(self, bridge: WikiBridge) -> None:
        page = bridge._find_page("no-frontmatter")
        assert page is not None


# ── Generate skill ────────────────────────────────────────────────────


class TestGenerateSkill:
    def test_generates_skill_from_topic(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("NVFP4 Benchmarks")
        assert result is not None
        skill_md = result / "SKILL.md"
        assert skill_md.is_file()
        content = skill_md.read_text()
        assert "name: nvfp4-benchmarks" in content
        assert "description: NVFP4 performance results" in content
        assert "trigger: nvfp4-benchmarks" in content
        assert "NVFP4 Benchmarks" in content
        assert "confidence: high" in content

    def test_generates_skill_with_custom_name(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("NVFP4 Benchmarks", output_name="nvfp4-bench")
        assert result is not None
        skill_md = result / "SKILL.md"
        content = skill_md.read_text()
        assert "name: nvfp4-bench" in content
        assert "trigger: nvfp4-bench" in content

    def test_returns_none_for_unknown_topic(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("does-not-exist")
        assert result is None

    def test_creates_references_source(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("NVFP4 Benchmarks")
        assert result is not None
        ref_file = result / "references" / "source.md"
        assert ref_file.is_file()
        assert "NVFP4 Benchmarks" in ref_file.read_text()

    def test_includes_metadata_in_frontmatter(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("Qwen3 Architecture")
        assert result is not None
        content = (result / "SKILL.md").read_text()
        assert "source: llm-wiki" in content
        assert "topic: Qwen3 Architecture" in content

    def test_output_dir_is_created(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("untagged-page")
        assert result is not None
        assert result.is_dir()

    def test_no_frontmatter_page_uses_title_fallback(self, bridge: WikiBridge) -> None:
        result = bridge.generate_skill("no-frontmatter")
        assert result is not None
        content = (result / "SKILL.md").read_text()
        assert "no-frontmatter" in content


# ── Generate all ──────────────────────────────────────────────────────


class TestGenerateAll:
    def test_generates_all_pages(self, bridge: WikiBridge) -> None:
        results = bridge.generate_all()
        assert len(results) >= 3

    def test_filters_by_tag(self, bridge: WikiBridge) -> None:
        results = bridge.generate_all(tag_filter="qwen3")
        assert len(results) >= 1
        for r in results:
            assert "qwen3" in r.name

    def test_filter_excludes_non_matching(self, bridge: WikiBridge) -> None:
        results = bridge.generate_all(tag_filter="nonexistent-tag-xyz")
        assert len(results) == 0

    def test_empty_wiki_dir_returns_empty(self) -> None:
        empty = Path(tempfile.mkdtemp())
        bridge = WikiBridge(wiki_dir=str(empty))
        results = bridge.generate_all()
        assert results == []

    def test_generate_all_with_tag_on_matching_page(self, bridge: WikiBridge) -> None:
        results = bridge.generate_all(tag_filter="untagged")
        assert len(results) >= 1

    def test_generate_all_logs_warning_for_missing_dir(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge = WikiBridge(wiki_dir=str(Path(tempfile.mkdtemp(), "does-not-exist")))
        results = bridge.generate_all()
        assert results == []


# ── Default wiki dir ─────────────────────────────────────────────────


class TestDefaultWiki:
    def test_uses_existing_wiki_dir(self) -> None:
        bridge = WikiBridge()
        assert bridge._wiki_dir is not None

    def test_fallback_when_no_wiki_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_home = Path(tempfile.mkdtemp())
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        bridge = WikiBridge()
        assert bridge._wiki_dir.name == "wiki"
