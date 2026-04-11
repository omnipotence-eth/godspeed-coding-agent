"""Dangerous command detection — pattern matching for destructive operations.

Inspired by Hermes Agent's Tirith security scanner. Detects commands that
could cause irreversible damage: recursive deletes, disk operations,
pipe-to-shell, SQL injection, etc.
"""

from __future__ import annotations

import re

# Compiled patterns for dangerous commands
DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Filesystem destruction
    (re.compile(r"rm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)*[/~]"), "recursive delete from root/home"),
    (re.compile(r"rm\s+-[a-zA-Z]*[rf]"), "recursive/force delete"),
    (re.compile(r"chmod\s+777"), "world-writable permissions"),
    (re.compile(r"chmod\s+-R\s+777"), "recursive world-writable permissions"),
    (re.compile(r"chown\s+-R\s+"), "recursive ownership change"),
    # Disk operations
    (re.compile(r"mkfs\."), "filesystem format"),
    (re.compile(r"dd\s+if="), "raw disk write"),
    (re.compile(r">\s*/dev/sd"), "direct disk overwrite"),
    # Pipe-to-shell (supply chain attack vector)
    (re.compile(r"curl\s.*\|\s*(ba)?sh"), "pipe curl to shell"),
    (re.compile(r"wget\s.*\|\s*(ba)?sh"), "pipe wget to shell"),
    (re.compile(r"curl\s.*\|\s*python"), "pipe curl to python"),
    # SQL injection / destructive SQL
    (re.compile(r"DROP\s+(TABLE|DATABASE|INDEX|VIEW)", re.IGNORECASE), "SQL DROP"),
    (re.compile(r"DELETE\s+FROM\s+\w+\s*;", re.IGNORECASE), "unfiltered SQL DELETE"),
    (re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE), "SQL TRUNCATE"),
    # Git destructive operations
    (re.compile(r"git\s+push\s+.*--force"), "git force push"),
    (re.compile(r"git\s+reset\s+--hard"), "git hard reset"),
    (re.compile(r"git\s+clean\s+-[a-zA-Z]*f"), "git clean force"),
    # System operations
    (re.compile(r"kill\s+-9\s+"), "force kill process"),
    (re.compile(r"pkill\s+-9\s+"), "force kill by name"),
    (re.compile(r"systemctl\s+(stop|disable|mask)\s+"), "stop/disable service"),
    # Code execution patterns (injection vectors)
    (re.compile(r"eval\s*\("), "eval execution"),
    (re.compile(r"exec\s*\("), "exec execution"),
    # Fork bomb
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"), "fork bomb"),
    # Environment destruction
    (re.compile(r"unset\s+(PATH|HOME|USER)", re.IGNORECASE), "unset critical env var"),
    # Git short flag
    (re.compile(r"git\s+push\s+.*-f\b"), "git force push (short flag)"),
    # Privilege escalation
    (re.compile(r"sudo\s+"), "sudo command"),
    (re.compile(r"su\s+-"), "switch user"),
    # Reverse shell / network
    (re.compile(r"nc\s+.*-[le]"), "netcat listener/exec"),
    (re.compile(r"ncat\s+"), "ncat network tool"),
    # Supply chain
    (re.compile(r"npm\s+publish"), "npm publish"),
    (re.compile(r"pip\s+install\s+--force-reinstall"), "pip force reinstall"),
    (re.compile(r"twine\s+upload"), "PyPI upload"),
    # Pipe to more interpreters
    (re.compile(r"curl\s.*\|\s*(?:perl|ruby|node)"), "pipe download to interpreter"),
    (re.compile(r"wget\s.*\|\s*(?:perl|ruby|node)"), "pipe download to interpreter"),
    # Persistence
    (re.compile(r"crontab\s+-[er]"), "crontab modification"),
    # Environment manipulation
    (re.compile(r"history\s+-c"), "clear shell history"),
    # Container escape
    (re.compile(r"docker\s+run\s+.*--privileged"), "privileged container"),
    (re.compile(r"nsenter\s+"), "namespace enter"),
    # Command execution via interpreters
    (re.compile(r"python[23]?\s+-c\s+"), "python command execution"),
    (re.compile(r"perl\s+-e\s+"), "perl command execution"),
    (re.compile(r"ruby\s+-e\s+"), "ruby command execution"),
    (re.compile(r"node\s+-e\s+"), "node command execution"),
    # find/xargs destructive
    (re.compile(r"find\s+.*-exec\s+"), "find with exec"),
    (re.compile(r"find\s+.*-delete"), "find with delete"),
    (re.compile(r"xargs\s+.*rm"), "xargs with rm"),
    # awk/perl system execution
    (re.compile(r"awk\s+.*system\s*\("), "awk system() call"),
    # Network/firewall manipulation
    (re.compile(r"iptables\s+"), "firewall rule modification"),
    (re.compile(r"nft\s+"), "nftables rule modification"),
    # Mount/unmount (filesystem manipulation)
    (re.compile(r"\bmount\s+"), "filesystem mount"),
    (re.compile(r"\bumount\s+"), "filesystem unmount"),
    # Disk partitioning
    (re.compile(r"fdisk\s+"), "disk partitioning"),
    (re.compile(r"parted\s+"), "disk partitioning"),
    # System shutdown/reboot
    (re.compile(r"\bshutdown\b"), "system shutdown"),
    (re.compile(r"\breboot\b"), "system reboot"),
    # Download-to-overwrite
    (re.compile(r"wget\s+.*-O\s*/"), "wget overwrite to root path"),
    (re.compile(r"curl\s+.*-o\s*/"), "curl download to root path"),
    # Environment exfiltration via pipe
    (re.compile(r"\benv\b.*\|\s*curl"), "environment exfiltration via curl"),
    (re.compile(r"\benv\b.*\|\s*nc"), "environment exfiltration via netcat"),
    (re.compile(r"cat\s+/etc/passwd.*\|"), "password file exfiltration"),
    # Pipe to interpreter (broader)
    (re.compile(r"echo\s.*\|\s*(?:python|perl|ruby|node|sh|bash)"), "echo pipe to interpreter"),
    # Container destruction
    (re.compile(r"docker\s+rm\s+-f"), "force remove container"),
    (re.compile(r"docker\s+system\s+prune"), "docker system prune"),
    # Kubernetes destructive
    (re.compile(r"kubectl\s+delete\s+"), "kubernetes resource deletion"),
    # SSH key overwrite
    (re.compile(r"ssh-keygen\s+.*-f\s+"), "SSH key generation/overwrite"),
    # GPG key deletion
    (re.compile(r"gpg\s+--delete-key"), "GPG key deletion"),
    # Windows-specific destructive commands
    (re.compile(r"\bdel\s+/[sS]"), "Windows recursive delete"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE), "Windows disk format"),
    (re.compile(r"reg\s+delete", re.IGNORECASE), "Windows registry deletion"),
    (re.compile(r"powershell\s+.*-[eE]nc"), "PowerShell encoded command execution"),
    # Supply chain — local install with arbitrary setup.py
    (re.compile(r"pip\s+install\s+.*--no-verify"), "pip install without verification"),
    (re.compile(r"npm\s+install\s+.*--ignore-scripts\s*$"), "npm install ignoring scripts"),
]


def detect_dangerous_command(command: str) -> list[str]:
    """Check a shell command against dangerous patterns.

    Args:
        command: The shell command to check.

    Returns:
        List of danger descriptions. Empty list means safe.
    """
    dangers = []
    for pattern, description in DANGEROUS_PATTERNS:
        if pattern.search(command):
            dangers.append(description)
    return dangers


def is_dangerous(command: str) -> bool:
    """Quick check: is this command dangerous?"""
    return len(detect_dangerous_command(command)) > 0
