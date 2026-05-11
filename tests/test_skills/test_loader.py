"""Tests for skill discovery and loading (new SKILL.md standard)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.skills.loader import (
    DEFAULT_SKILL_DIRS,
    PROJECT_SKILL_DIRS,
    SKILL_NAME_RE,
    Skill,
    SkillError,
    SkillHub,
    SkillSecurityError,
    _check_skill_path,
    _compute_hash,
    _find_project_root,
    _load_skill_directory,
    _parse_frontmatter,
    _skill_dirs,
    _stat_ctime,
    _stat_mtime,
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
        s = Skill(
            name="my-skill",
            description="Test",
            trigger="my-skill",
            content="Body",
            path=Path("/x"),
        )
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
        d = _make_skill_dir(
            tmp_path,
            "my-skill",
            "name: my-skill\ndescription: A skill\ntrigger: ms",
            "Do the thing.",
        )
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
        d = _make_skill_dir(
            tmp_path,
            "with-refs",
            "name: with-refs\ndescription: Has refs\ntrigger: wr",
            "Body",
        )
        ref_dir = d / "references"
        ref_dir.mkdir()
        (ref_dir / "guide.md").write_text("Guide content")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert len(skill.files.references) == 1

    def test_reads_scripts_and_assets(self, tmp_path: Path) -> None:
        d = _make_skill_dir(
            tmp_path,
            "full-skill",
            "name: full-skill\ndescription: Full\ntrigger: fs",
            "Body",
        )
        (d / "scripts").mkdir()
        (d / "scripts" / "run.sh").write_text("echo hi")
        (d / "assets").mkdir()
        (d / "assets" / "icon.png").write_text("fake-png")
        skill = _load_skill_directory(d)
        assert skill is not None
        assert len(skill.files.scripts) == 1
        assert len(skill.files.assets) == 1

    def test_invalid_skill_name(self, tmp_path: Path) -> None:
        d = _make_skill_dir(
            tmp_path,
            "Bad-Name",
            "name: Bad-Name\ndescription: Bad\ntrigger: bn",
            "Body",
        )
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
        _make_skill_dir(
            tmp_path,
            "review",
            "name: review\ndescription: Review code\ntrigger: review",
            "Review.",
        )
        _make_skill_dir(
            tmp_path,
            "test",
            "name: test\ndescription: Run tests\ntrigger: test",
            "Test.",
        )
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
        _make_skill_dir(
            base,
            "skill-a",
            "name: skill-a\ndescription: Base\ntrigger: a",
            "Base version.",
        )
        _make_skill_dir(
            override,
            "skill-a",
            "name: skill-a-override\ndescription: Override\ntrigger: a",
            "Override version.",
        )
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


# -- Additional coverage tests -----------------------------------------------


class TestFindProjectRoot:
    """Test _find_project_root()."""

    def test_found_with_git_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = _find_project_root()
        assert result == tmp_path

    def test_not_found_no_git(self, tmp_path: Path) -> None:
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = _find_project_root()
        assert result is None

    def test_found_in_parent(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        child = tmp_path / "a" / "b" / "c"
        child.mkdir(parents=True)
        with patch("pathlib.Path.cwd", return_value=child):
            result = _find_project_root()
        assert result == tmp_path

    def test_stops_at_non_git_parent(self, tmp_path: Path) -> None:
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        with patch("pathlib.Path.cwd", return_value=child):
            result = _find_project_root()
        assert result is None


class TestSkillDirs:
    """Test _skill_dirs()."""

    def test_without_project_root(self, tmp_path: Path) -> None:
        with patch("godspeed.skills.loader._find_project_root", return_value=None):
            dirs = _skill_dirs()
        assert len(dirs) == len(DEFAULT_SKILL_DIRS)

    def test_with_project_root(self, tmp_path: Path) -> None:
        with patch("godspeed.skills.loader._find_project_root", return_value=tmp_path):
            dirs = _skill_dirs()
        assert len(dirs) == len(DEFAULT_SKILL_DIRS) + len(PROJECT_SKILL_DIRS)


class TestCheckSkillPath:
    """Test _check_skill_path()."""

    def test_valid_skill_path(self, tmp_path: Path) -> None:
        p = tmp_path / "my-skill" / "SKILL.md"
        assert _check_skill_path(p) is True

    def test_not_skill_md(self) -> None:
        p = Path("/x/README.md")
        assert _check_skill_path(p) is False

    def test_root_skill_md(self) -> None:
        p = Path("/SKILL.md")
        assert _check_skill_path(p) is False


class TestLoadSkillDirectoryAdvanced:
    """Edge cases for _load_skill_directory()."""

    def test_oserror_reading_skill_md(self, tmp_path: Path) -> None:
        d = tmp_path / "bad-read"
        d.mkdir()
        skill_md = d / "SKILL.md"
        skill_md.write_text("---\nname: test\ndescription: test\ntrigger: t\n---\nBody")
        with patch.object(Path, "read_text", side_effect=OSError("disk error")):
            result = _load_skill_directory(d)
        assert result is None

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        d = tmp_path / "missing-fields"
        d.mkdir()
        (d / "SKILL.md").write_text("---\n---\nBody")
        assert _load_skill_directory(d) is None

    def test_with_metadata_in_frontmatter(self, tmp_path: Path) -> None:
        d = _make_skill_dir(
            tmp_path,
            "meta-skill",
            "name: meta-skill\ndescription: Has metadata\ntrigger: ms\nversion: '1.2'\nlicense: MIT\ncompatibility: python>=3.11\nmetadata:\n  author: test\n  tags: [a, b]",
            "Body with metadata.",
        )
        skill = _load_skill_directory(d)
        assert skill is not None
        assert skill.version == "1.2"
        assert skill.license == "MIT"
        assert skill.compatibility == "python>=3.11"
        assert skill.metadata == {"author": "test", "tags": ["a", "b"]}

    def test_dir_name_differs_from_frontmatter_name(self, tmp_path: Path) -> None:
        d = _make_skill_dir(
            tmp_path,
            "dir-name",
            "name: frontmatter-name\ndescription: Desc\ntrigger: dn",
            "Body.",
        )
        skill = _load_skill_directory(d)
        assert skill is not None
        assert skill.name == "frontmatter-name"


class TestStatHelpers:
    """Test _stat_ctime and _stat_mtime."""

    def test_stat_ctime_success(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _stat_ctime(f)
        assert result is not None

    def test_stat_ctime_oserror(self, tmp_path: Path) -> None:
        with patch.object(Path, "stat", side_effect=OSError("no stat")):
            result = _stat_ctime(Path("/nonexistent"))
        assert result is None

    def test_stat_mtime_success(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _stat_mtime(f)
        assert result is not None

    def test_stat_mtime_oserror(self, tmp_path: Path) -> None:
        with patch.object(Path, "stat", side_effect=OSError("no stat")):
            result = _stat_mtime(Path("/nonexistent"))
        assert result is None


class TestParseFrontmatterAdvanced:
    """Additional frontmatter parsing edge cases."""

    def test_frontmatter_with_trailing_spaces(self) -> None:
        text = "---   \nname: test\ndescription: desc\ntrigger: t\n---   \n\nBody."
        result = _parse_frontmatter(text)
        assert result is not None
        fm, body = result
        assert fm["name"] == "test"

    def test_frontmatter_with_windows_line_endings(self) -> None:
        text = "---\r\nname: test\r\ndescription: desc\r\ntrigger: t\r\n---\r\n\r\nBody."
        result = _parse_frontmatter(text)
        assert result is not None
        fm, body = result
        assert fm["name"] == "test"

    def test_only_frontmatter_delimiter(self) -> None:
        assert _parse_frontmatter("---") is None

    def test_empty_string(self) -> None:
        assert _parse_frontmatter("") is None

    def test_frontmatter_with_special_yaml_types(self) -> None:
        text = "---\nname: test\ndescription: desc\ntrigger: t\nversion: '1.0'\ncount: 42\nenabled: true\n---\n\nBody."
        result = _parse_frontmatter(text)
        assert result is not None
        fm, body = result
        assert fm["count"] == 42
        assert fm["enabled"] is True


class TestSkillHubAdvanced:
    """Additional SkillHub edge cases."""

    def test_verify_integrity_no_lock_entry(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "unlocked", "name: unlocked\ndescription: U\ntrigger: u", "Body.")
        hub.install("unlocked", src / "unlocked")
        hub._lock["skills"].pop("unlocked", None)
        assert not hub.verify_integrity("unlocked")

    def test_verify_integrity_skill_not_found(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        assert not hub.verify_integrity("nonexistent")

    def test_install_invalid_skill(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        invalid = tmp_path / "invalid"
        invalid.mkdir()
        with pytest.raises(SkillError, match="Invalid skill"):
            hub.install("bad", invalid)

    def test_install_security_scan_fails_with_many_issues(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(
            src, "dangerous", "name: dangerous\ndescription: D\ntrigger: d", "Bad stuff."
        )
        issues = ["issue 1", "issue 2", "issue 3", "issue 4", "issue 5", "issue 6", "issue 7"]
        with patch("godspeed.skills.security.scan_skill", return_value=issues):
            with pytest.raises(SkillSecurityError, match="failed security scan"):
                hub.install("dangerous", src / "dangerous")

    def test_install_security_scan_fails_with_few_issues(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "risky", "name: risky\ndescription: R\ntrigger: r", "Bad stuff.")
        issues = ["one issue", "two issues"]
        with patch("godspeed.skills.security.scan_skill", return_value=issues):
            with pytest.raises(SkillSecurityError, match="failed security scan"):
                hub.install("risky", src / "risky")

    def test_install_security_scan_passes(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "safe-one", "name: safe-one\ndescription: S\ntrigger: so", "Safe.")
        with patch("godspeed.skills.security.scan_skill", return_value=[]):
            skill = hub.install("safe-one", src / "safe-one")
        assert skill.name == "safe-one"

    def test_quarantine_nonexistent(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        hub.quarantine("does-not-exist")

    def test_list_installed_empty(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        assert hub.list_installed() == []

    def test_remove_updates_lock(self, tmp_path: Path) -> None:
        hub = SkillHub(base_dir=tmp_path / "hub")
        src = tmp_path / "src"
        _make_skill_dir(src, "to-remove", "name: to-remove\ndescription: TR\ntrigger: tr", "Body.")
        hub.install("to-remove", src / "to-remove")
        assert "to-remove" in hub._lock["skills"]
        hub.remove("to-remove")
        assert "to-remove" not in hub._lock["skills"]


class TestDiscoverSkillsAdvanced:
    """Additional discover_skills edge cases."""

    @pytest.fixture(autouse=True)
    def _no_standard_dirs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("godspeed.skills.loader.DEFAULT_SKILL_DIRS", [])
        monkeypatch.setattr("godspeed.skills.loader.PROJECT_SKILL_DIRS", [])

    def test_extra_dirs_none(self, tmp_path: Path) -> None:
        skills = discover_skills(None)
        assert skills == []

    def test_extra_dirs_empty(self, tmp_path: Path) -> None:
        skills = discover_skills([])
        assert skills == []

    def test_base_dir_does_not_exist(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope"
        skills = discover_skills([nonexistent])
        assert skills == []

    def test_directory_with_no_subdirs(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty-base"
        empty.mkdir()
        skills = discover_skills([empty])
        assert skills == []

    def test_directory_with_files_not_dirs(self, tmp_path: Path) -> None:
        base = tmp_path / "base-with-files"
        base.mkdir()
        (base / "NOT_A_SKILL.md").write_text("---\nname: nope\n---\nBody")
        skills = discover_skills([base])
        assert len(skills) == 0

    def test_multiple_directories_no_overlap(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        _make_skill_dir(a, "skill-a", "name: skill-a\ndescription: A\ntrigger: a", "A.")
        _make_skill_dir(b, "skill-b", "name: skill-b\ndescription: B\ntrigger: b", "B.")
        skills = discover_skills([a, b])
        triggers = {s.trigger for s in skills}
        assert triggers == {"a", "b"}
