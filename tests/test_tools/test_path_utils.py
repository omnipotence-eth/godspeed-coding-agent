"""Tests for shared path resolution utilities."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.path_utils import resolve_tool_path


class TestResolveToolPath:
    def test_relative_path_within_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            (cwd / "subdir").mkdir()
            result = resolve_tool_path("subdir", cwd)
            assert result == cwd / "subdir"

    def test_absolute_path_within_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            abs_path = cwd / "file.txt"
            abs_path.touch()
            result = resolve_tool_path(str(abs_path), cwd)
            assert result == abs_path

    def test_windows_drive_letter_path_on_non_windows_raises(self) -> None:
        with patch("os.name", "posix"):
            mock_cwd = MagicMock(spec=Path)
            mock_cwd.resolve.return_value = mock_cwd
            mock_cwd.__str__.return_value = "/home/user/project"
            with pytest.raises(ValueError, match=r"Access denied.*Windows absolute path"):
                resolve_tool_path(r"C:\Users\test\file.txt", mock_cwd)

    @pytest.mark.skipif(sys.platform != "win32", reason="WindowsPath only on Windows")
    def test_windows_drive_letter_path_on_windows_passes(self) -> None:
        with (
            patch("os.name", "nt"),
            tempfile.TemporaryDirectory() as tmp,
        ):
            cwd = Path(tmp).resolve()
            safe_path = cwd / "safe_file.txt"
            safe_path.touch()
            result = resolve_tool_path(str(safe_path), cwd)
            assert result == safe_path

    def test_path_outside_cwd_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            outside = Path(tmp).parent / "outside_file.txt"
            try:
                outside.touch()
                with pytest.raises(ValueError, match=r"Access denied.*outside the project"):
                    resolve_tool_path(str(outside), cwd)
            finally:
                if outside.exists():
                    outside.unlink()

    def test_symlink_traversal_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            sub = cwd / "sub"
            sub.mkdir()
            outside_real = Path(tmp).parent / "real_outside.txt"
            try:
                outside_real.touch()
                evil_link = sub / "evil_link.txt"
                evil_link.touch()
                evil_link_str = os.path.normpath(str(evil_link.resolve()))
                outside_str = os.path.normpath(str(outside_real.resolve()))

                call_count = [0]

                def fake_realpath(p: str, **kwargs: object) -> str:  # noqa: ARG001
                    call_count[0] += 1
                    norm = os.path.normpath(p)
                    if norm == evil_link_str and call_count[0] > 1:
                        return outside_str
                    return norm

                with patch("os.path.realpath", side_effect=fake_realpath):
                    with pytest.raises(ValueError, match=r"resolves via symlinks to outside"):
                        resolve_tool_path(str(evil_link), cwd)
            finally:
                if outside_real.exists():
                    outside_real.unlink()

    def test_oserror_during_realpath_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            sub = cwd / "sub"
            sub.mkdir()
            file_in = sub / "ok.txt"
            file_in.touch()
            file_in_resolved = file_in.resolve()
            file_in_str = os.path.normpath(str(file_in_resolved))

            call_count = [0]

            def fake_realpath(p: str, **kwargs: object) -> str:  # noqa: ARG001
                call_count[0] += 1
                norm = os.path.normpath(p)
                if norm == file_in_str and call_count[0] > 1:
                    raise OSError("Permission denied")
                return norm

            with patch("os.path.realpath", side_effect=fake_realpath):
                result = resolve_tool_path(str(file_in), cwd)
            assert result.resolve() == file_in_resolved

    def test_expanduser_tilde(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            with patch("pathlib.Path.expanduser", return_value=cwd / "expanded"):
                result = resolve_tool_path("~/file.txt", cwd)
                assert result.resolve() == (cwd / "expanded").resolve()
