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
    (re.compile(r"rm\s+(?:--recursive|--force|--dir)"), "recursive/force delete (long flags)"),
    (re.compile(r"rm\s+-r\s+-f\b"), "recursive force delete (separate flags)"),
    (re.compile(r"\brmdir\s+/[sS]"), "Windows recursive directory removal"),
    (re.compile(r"\bshred\b"), "secure file deletion"),
    (re.compile(r"\bwipe\b"), "secure file wipe"),
    (re.compile(r"\bsrm\b"), "secure file removal"),
    (re.compile(r">\s*/etc/"), "overwrite critical config file"),
    (re.compile(r"chmod\s+(?:777|0777|a\+rwx|ugo\+rwx|a=rwx)"), "world-writable permissions"),
    (
        re.compile(r"chmod\s+-R\s+(?:777|0777|a\+rwx|ugo\+rwx)"),
        "recursive world-writable permissions",
    ),
    (re.compile(r"chmod\s+-R\s+[ao]\+w\b"), "recursive write for all/others"),
    (re.compile(r"chown\s+-R\s+"), "recursive ownership change"),
    # Disk operations
    (re.compile(r"mkfs\."), "filesystem format"),
    (re.compile(r"dd\s+.*of=/dev/"), "raw disk write (of= device)"),
    (re.compile(r"dd\s+if=/dev/zero\s+of="), "zero-fill disk"),
    (
        re.compile(r">\s*/dev/(?:sd|nvme|hd|vd|xd|xvd|dm-|mapper|loop|mmcblk|ram)"),
        "direct disk overwrite",
    ),
    (re.compile(r"\bcat\s+/dev/zero\s*>\s*/dev/"), "write zeros to device"),
    (re.compile(r"\btee\s+/etc/"), "tee to critical config file"),
    (re.compile(r"\bmv\s+\S+\s+/dev/null"), "data destruction via mv to null"),
    # Pipe-to-shell (supply chain attack vector)
    (re.compile(r"curl\s.*\|\s*(?:ba)?sh"), "pipe curl to shell"),
    (re.compile(r"wget\s.*\|\s*(?:ba)?sh"), "pipe wget to shell"),
    (re.compile(r"curl\s.*\|\s*python"), "pipe curl to python"),
    (re.compile(r"curl\s.*\|\s*sudo\s+(?:ba)?sh"), "pipe curl to sudo shell"),
    (re.compile(r"curl\s.*\s*>\s*\S+\.sh\s*&&\s*(?:ba)?sh"), "download and execute script"),
    (re.compile(r"source\s+<\s*\(\s*curl"), "process substitution pipe to shell"),
    (re.compile(r"\.\s+<\s*\(\s*curl"), "dot process substitution pipe"),
    # SQL injection / destructive SQL
    (re.compile(r"DROP\s+(TABLE|DATABASE|INDEX|VIEW)", re.IGNORECASE), "SQL DROP"),
    (re.compile(r"DELETE\s+FROM\s+\w+\s*;", re.IGNORECASE), "unfiltered SQL DELETE"),
    (re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE), "SQL TRUNCATE"),
    # Git destructive operations
    (re.compile(r"git\s+push\s+.*--force"), "git force push"),
    (re.compile(r"git\s+push\s+.*-f\b"), "git force push (short flag)"),
    (re.compile(r"git\s+push\s+.*--force-with-lease"), "git force push with lease"),
    (re.compile(r"git\s+push\s+.*--delete"), "git delete remote branch"),
    (re.compile(r"git\s+reset\s+--hard"), "git hard reset"),
    (re.compile(r"git\s+clean\s+-[a-zA-Z]*f"), "git clean force"),
    (re.compile(r"git\s+branch\s+-D\b"), "git force branch delete"),
    (re.compile(r"git\s+tag\s+-d\b"), "git delete tag"),
    (re.compile(r"git\s+config\s+remote\.\S+\.url"), "git config remote URL change"),
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
    # Privilege escalation (narrowed to dangerous subcommands only)
    (
        re.compile(r"sudo\s+(?:rm|dd|mkfs|fdisk|parted|mount|umount|shutdown|reboot)\b"),
        "sudo dangerous command",
    ),
    (re.compile(r"su\s+-"), "switch user"),
    (re.compile(r"sudo\s+-i\b"), "sudo interactive root shell"),
    # Reverse shell / network
    (re.compile(r"nc\s+.*-[le]"), "netcat listener/exec"),
    (re.compile(r"ncat\s+"), "ncat network tool"),
    (re.compile(r"\bsocat\b"), "socat network tool"),
    (re.compile(r"/dev/tcp/"), "bash reverse shell via /dev/tcp"),
    # Encoded payload execution
    (re.compile(r"base64\s+.*\|\s*(?:ba)?sh"), "base64 decode to shell"),
    (re.compile(r"openssl\s+enc\s+.*\|\s*(?:ba)?sh"), "decrypt to shell"),
    (re.compile(r"xxd\s+-r\b.*\|\s*(?:ba)?sh"), "hex decode to shell"),
    # Supply chain
    (re.compile(r"npm\s+publish"), "npm publish"),
    (re.compile(r"(?:npm|pnpm|yarn)\s+dlx\b"), "package runner (arbitrary code)"),
    (re.compile(r"\bnpx\b"), "npx package runner"),
    (re.compile(r"pip\s+install\s+--force-reinstall"), "pip force reinstall"),
    (re.compile(r"twine\s+upload"), "PyPI upload"),
    (re.compile(r"npm\s+unpublish"), "npm unpublish"),
    (re.compile(r"cargo\s+yank"), "cargo yank"),
    (re.compile(r"gem\s+yank"), "gem yank"),
    # Pipe to more interpreters
    (re.compile(r"curl\s.*\|\s*(?:perl|ruby|node)"), "pipe download to interpreter"),
    (re.compile(r"wget\s.*\|\s*(?:perl|ruby|node)"), "pipe download to interpreter"),
    # Persistence
    (re.compile(r"crontab\s+-[er]"), "crontab modification"),
    (re.compile(r"schtasks\s+/create"), "Windows scheduled task creation"),
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
    # awk/system execution
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
    (
        re.compile(
            r"cat\s+/(?:etc/(?:passwd|shadow|gshadow|sudoers|group)"
            r"|home/\S+/.ssh/id_)"
        ),
        "sensitive file exfiltration",
    ),
    (re.compile(r"\bprintenv\b.*\|\s*curl"), "environment exfiltration via curl"),
    # Pipe to interpreter (broader)
    (re.compile(r"echo\s.*\|\s*(?:python|perl|ruby|node|sh|bash)"), "echo pipe to interpreter"),
    # Container destruction
    (re.compile(r"docker\s+rm\s+-f"), "force remove container"),
    (re.compile(r"docker\s+system\s+prune"), "docker system prune"),
    (re.compile(r"docker\s+push\b"), "docker push image"),
    # Kubernetes destructive
    (re.compile(r"kubectl\s+delete\s+"), "kubernetes resource deletion"),
    (re.compile(r"kubectl\s+delete\s+ns\b"), "kubernetes namespace deletion"),
    (re.compile(r"kubectl\s+delete\s+pvc\b"), "kubernetes PVC deletion"),
    (re.compile(r"helm\s+uninstall\b"), "helm uninstall"),
    # Cloud resource destruction
    (
        re.compile(r"aws\s+(?:s3\s+rm|ec2\s+terminate|rds\s+delete|lambda\s+delete)"),
        "AWS resource destruction",
    ),
    (
        re.compile(r"gcloud\s+(?:compute\s+instances\s+delete|sql\s+instances\s+delete)"),
        "GCP resource destruction",
    ),
    (
        re.compile(r"az\s+(?:group\s+delete|vm\s+delete|sql\s+server\s+delete)"),
        "Azure resource destruction",
    ),
    (re.compile(r"terraform\s+destroy\b"), "terraform infrastructure destruction"),
    # SSH key overwrite
    (re.compile(r"ssh-keygen\s+.*-[tf]\s+"), "SSH key generation/overwrite"),
    # GPG key deletion
    (re.compile(r"gpg\s+--delete-key"), "GPG key deletion"),
    # Windows-specific destructive commands
    (re.compile(r"\bdel\s+/[sS]"), "Windows recursive delete"),
    (re.compile(r"\brmdir\s+/[sS]"), "Windows recursive dir removal"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE), "Windows disk format"),
    (re.compile(r"reg\s+delete", re.IGNORECASE), "Windows registry deletion"),
    (re.compile(r"reg\s+add", re.IGNORECASE), "Windows registry addition"),
    (re.compile(r"reg\s+import", re.IGNORECASE), "Windows registry import"),
    (re.compile(r"powershell\s+.*-[eE]nc"), "PowerShell encoded command execution"),
    (re.compile(r"powershell\s+.*-(?:NoP|NonI|W\s+Hidden)"), "PowerShell evasion flags"),
    (re.compile(r"\bwmic\s+process\s+call\s+create"), "WMIC process creation"),
    (re.compile(r"\bmshta\b"), "mshta HTML application host"),
    (re.compile(r"\brundll32\b"), "rundll32 execution"),
    (re.compile(r"\bregsvr32\b"), "regsvr32 execution"),
    (re.compile(r"\bcertutil\s+-urlcache\b"), "certutil download"),
    (re.compile(r"\bbitsadmin\s+/transfer\b"), "bitsadmin download"),
    (re.compile(r"\btakeown\s+/[fF]"), "takeown ownership change"),
    (re.compile(r"\bicacls\b"), "icacls ACL manipulation"),
    (re.compile(r"\bnet\s+(?:user|localgroup)"), "Windows user/group manipulation"),
    (re.compile(r"\bsc\s+(?:stop|delete|config)"), "Windows service manipulation"),
    # Supply chain — local install with arbitrary setup.py
    (re.compile(r"pip\s+install\s+.*--no-verify"), "pip install without verification"),
    # Container runtime alternative
    (re.compile(r"podman\s+run\s+.*--privileged"), "privileged podman container"),
    # sed/perl in-place file editing (destructive)
    (re.compile(r"(?:perl|sed)\s+-i\b"), "in-place file editing"),
    (re.compile(r"chattr\s+[+-]i\b"), "immutable file attribute change"),
    (re.compile(r"cat\s+/dev/null\s*>\s*"), "file truncation via /dev/null"),
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
