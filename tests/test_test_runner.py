"""Tests for the test runner tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.test_runner import TestRunnerTool, detect_framework


@pytest.fixture
def runner() -> TestRunnerTool:
    return TestRunnerTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


class TestDetectFramework:
    """Test framework auto-detection."""

    def test_detects_pytest_from_conftest(self, tmp_path: Path) -> None:
        (tmp_path / "conftest.py").write_text("", encoding="utf-8")
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_pytest_from_tests_conftest(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "conftest.py").write_text("", encoding="utf-8")
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_pytest_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]',
            encoding="utf-8",
        )
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_jest(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"scripts": {"test": "jest"}}',
            encoding="utf-8",
        )
        assert detect_framework(tmp_path) == "jest"

    def test_detects_vitest(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"scripts": {"test": "vitest run"}}',
            encoding="utf-8",
        )
        assert detect_framework(tmp_path) == "vitest"

    def test_detects_go(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module example.com/foo", encoding="utf-8")
        assert detect_framework(tmp_path) == "go"

    def test_detects_cargo(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "foo"', encoding="utf-8")
        assert detect_framework(tmp_path) == "cargo"

    def test_no_framework(self, tmp_path: Path) -> None:
        assert detect_framework(tmp_path) == ""


class TestTestRunnerTool:
    """Test the test runner tool."""

    def test_name(self, runner: TestRunnerTool) -> None:
        assert runner.name == "test_runner"

    def test_schema(self, runner: TestRunnerTool) -> None:
        schema = runner.get_schema()
        assert "target" in schema["properties"]
        assert "framework" in schema["properties"]

    @pytest.mark.asyncio
    async def test_no_framework_detected(self, runner: TestRunnerTool, ctx: ToolContext) -> None:
        result = await runner.execute({}, ctx)
        assert not result.is_error
        assert "No test framework detected" in result.output

    @pytest.mark.asyncio
    async def test_unknown_framework(self, runner: TestRunnerTool, ctx: ToolContext) -> None:
        result = await runner.execute({"framework": "foobar"}, ctx)
        assert result.is_error
        assert "Unknown test framework" in result.error
