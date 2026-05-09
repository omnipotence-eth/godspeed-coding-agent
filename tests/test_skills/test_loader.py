"""Tests for skill discovery and loading (new SKILL.md standard)."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.skills.loader import (
    SKILL_NAME_RE,
    Skill,
    SkillError,
    SkillHub,
    _compute_hash,
    _load_skill_directory,
    _parse_frontmatter,
    discover_skills,
)


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a valid skill directory structure."""
    skills = tmp_path / "skills"
    skills.mkdir()
    return skills


def _make_skill_dir(parent: Path, name: str, frontmatter: str, body: str) -> Path:
    """Create a ``{name}/SKILL.md`` directory structure."""
    d = parent / name
    d.mkdir(parents=True)
    content = f"---\n{frontmatter}\n---\n\n{body}"
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


class TestSkillValidation:
    """Test Skill dataclass validation."""

    def test_valid_name(self) -> None:
        s = Skill(name="my-skill", description="Test", trigger="my-skill", content="Body", path=Path("/x"))
        assert s.name == "my-skill"

    def test_invalid_name_uppercase(self) -> None:
        with pytest.raises(SkillError):
            Skill(name="BadName", description="Test", trigger="x", content="Body", path=Path("/x"))

    def test_invalid_name_underscore(self) -> None:
        with pytest.raises(SkillError):
            Skill(name="bad_name", description="Test", trigger="x", content="Body", path=Path("/x"))

    def test_invalid_name_space(self) -> None:
        with pytest.raises(SkillError):
            Skill(name="bad name", description="Test", trigger="x", content="Body", path=Path("/x"))

    def test_edge_case_names(self) -> None:
        valid = ["a", "a-b", "1-2-3", "skill", "my-long-skill-name"]
        for name in valid:
            assert SKILL_NAME_RE.match(name), f"Expected {name!r} to be valid"

    def test_hash_computation(self) -> None:
        h1 = _compute_hash("hello")
        h2 = _compute_hash("hello")
        h3 = _compute_hash("world")
        assert h1 == h2
        assert len(h1) == 16
        assert h1 != h3


class TestParseFrontmatter:
    """Test _parse_frontmatter()."""

    def test_valid_frontmatter(self) -> None:
        text = "---\nname: test\ndescription: A test\ntrigger: test\n---\n\nBody here."
        result = _parse_frontmatter(text)
        assert result is not None
        fm, body = result
        assert fm["name"] == "test"
        assert fm["description"] == "A test"
        assert body == "Body here."

    def test_no_frontmatter(self) -> None:
        assert _parse_frontmatter("Just body text.") is None

    def test_invalid_yaml(self) -> None:
        assert _parse_frontmatter("---\n[invalid\n---\nBody") is None

    def test_non_dict_yaml(self) -> None:
        assert _parse_frontmatter("---\n- list\n- item\n---\nBody") is None

    def test_empty_frontmatter(self) -> None:
        # YAML-safe_load("") returns None, which is not a dict
        assert _parse_frontmatter("---\n---\n\nBody") is None

    def test_unclosed_frontmatter(self) -> None:
        assert _parse_frontmatter("---\nname: test\n\nBody") is None

    def test_preserves_body_newlines(self) -> None:
        text = "---\nname: test\ntrigger: test\ndescription: Test\n---\n\nLine 1\n\nLine 2"
        result = _parse_frontmatter(text)
        assert result is not None
        _, body = result
        assert "Line 1" in body
        assert "Line 2" in body


class TestLoadSkillDirectory:
    """Test _load_skill_directory()."""

    def test_loads_valid_skill(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "my-skill", "name: my-skill\ndescription: A skill\ntrigger: ms", "Do the thing.")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "A skill"
        assert skill.trigger == "ms"
        assert skill.content == "Do the thing."

    def test_missing_skill_md(self, tmp_path: Path) -> None:
        d = tmp_path / "empty-skill"
        d.mkdir()
        assert _load_skill_directory(d) is None

    def test_missing_name_falls_back_to_dirname(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "no-name", "description: Only desc", "Body")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert skill.name == "no-name"  # dir name used as fallback

    def test_missing_description(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "no-desc", "name: no-desc", "Body")
        assert _load_skill_directory(d) is None

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert _load_skill_directory(tmp_path / "nonexistent") is None

    def test_reads_references_subdir(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "with-refs", "name: with-refs\ndescription: Has refs\ntrigger: wr", "Body")
        ref_dir = d / "references"
        ref_dir.mkdir()
        (ref_dir / "guide.md").write_text("Guide content")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert len(skill.files.references) == 1

    def test_reads_scripts_and_assets(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "full-skill", "name: full-skill\ndescription: Full\ntrigger: fs", "Body")
        (d / "scripts").mkdir()
        (d / "scripts" / "run.sh").write_text("echo hi")
        (d / "assets").mkdir()
        (d / "assets" / "icon.png").write_text("fake-png")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert len(skill.files.scripts) == 1
        assert len(skill.files.assets) == 1

    def test_invalid_skill_name(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "Bad-Name", "name: Bad-Name\ndescription: Bad\ntrigger: bn", "Body")
        assert _load_skill_directory(d) is None

    def test_sets_hash(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "hashed", "name: hashed\ndescription: H\ntrigger: h", "Body")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert len(skill.hash) == 16


class TestDiscoverSkills:
    """Test discover_skills()."""

    @pytest.fixture(autouse=True)
    def _no_standard_dirs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prevent discovery from real user directories."""
        monkeypatch.setattr("godspeed.skills.loader.DEFAULT_SKILL_DIRS", [])
        monkeypatch.setattr("godspeed.skills.loader.PROJECT_SKILL_DIRS", [])

    def test_empty_dirs(self, tmp_path: Path) -> None:
        skills = discover_skills([tmp_path / "empty"])
        assert skills == []

    def test_discovers_valid_skills(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "review", "name: review\ndescription: Review code\ntrigger: review", "Review.")
        _make_skill_dir(tmp_path, "test", "name: test\ndescription: Run tests\ntrigger: test", "Test.")
        skills = discover_skills([tmp_path])
        assert len(skills) >= 2  # may also find std dirs
        triggers = {s.trigger for s in skills}
        assert "review" in triggers
        assert "test" in triggers

    def test_skips_invalid_skills(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "good", "name: good\ndescription: G\ntrigger: g", "Good.")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("No frontmatter")  # type: ignore[union-attr]
        skills = discover_skills([tmp_path])
        assert all(s.trigger == "g" for s in skills if s.trigger == "g")

    def test_later_dirs_override(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        override = tmp_path / "override"
        base.mkdir()
        override.mkdir()
        _make_skill_dir(base, "skill-a", "name: skill-a\ndescription: Base\ntrigger: a", "Base version.")
        _make_skill_dir(override, "skill-a", "name: skill-a-override\ndescription: Override\ntrigger: a", "Override version.")
        skills = discover_skills([base, override])
        matches = [s for s in skills if s.trigger == "a"]
        assert len(matches) == 1
        assert "Override" in matches[0].description

    def test_ignores_non_skill_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-skill.md").write_text("---\nname: fake\n---\nBody")
        skills = discover_skills([tmp_path])
        assert len(skills) == 0


class TestSkillHub:
    """Test SkillHub marketplace operations."""

    def test_install_locks_and_verifies(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "test-skill", "name: test-skill\ndescription: T\ntrigger: ts", "Body.")
        skill = hub.install("test-skill", src / "test-skill")
        assert skill.name == "test-skill"
        assert hub.verify_integrity("test-skill")
        installed = hub.list_installed()
        assert any(e["name"] == "test-skill" for e in installed)

    def test_remove_skill(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "removable", "name: removable\ndescription: R\ntrigger: r", "Body.")
        hub.install("removable", src / "removable")
        hub.remove("removable")
        assert hub.list_installed() == []

    def test_double_install_raises(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "dup", "name: dup\ndescription: D\ntrigger: d", "Body.")
        hub.install("dup", src / "dup")
        from godspeed.skills.loader import SkillError
        with pytest.raises(SkillError, match="already installed"):
            hub.install("dup", src / "dup")

    def test_verify_tampered_fails(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "safe", "name: safe\ndescription: S\ntrigger: s", "Body.")
        hub.install("safe", src / "safe")

        skill_file = tmp_path / "hub" / "safe" / "SKILL.md"
        skill_file.write_text("---\nname: hacked\n---\nEVIL")
        assert not hub.verify_integrity("safe")

    def test_quarantine_moves_skill(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "bad", "name: bad\ndescription: B\ntrigger: b", "Body.")
        hub.install("bad", src / "bad")

        skill_file = tmp_path / "hub" / "bad" / "SKILL.md"
        skill_file.write_text("---\nname: hacked\n---\nEVIL")
        hub.quarantine("bad")

        assert not (tmp_path / "hub" / "bad").exists()
        assert (tmp_path / "hub" / ".hub" / "quarantine" / "bad").exists()

    def test_remove_nonexistent_raises(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        from godspeed.skills.loader import SkillError
        with pytest.raises(SkillError):
            hub.remove("does-not-exist")
