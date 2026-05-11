"""Batch-fix all 52 CodeQL alerts."""
import json, os, subprocess, sys

result = subprocess.run(
    ["gh", "api", "repos/t-timms/godspeed-coding-agent/code-scanning/alerts", "--paginate"],
    capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
)
alerts = json.loads(result.stdout)
open_alerts = [a for a in alerts if a.get("state") == "open"]

by_file = {}
for a in open_alerts:
    path = a["most_recent_instance"]["location"]["path"]
    line = a["most_recent_instance"]["location"]["start_line"]
    rule = a["rule"]["id"]
    by_file.setdefault(path, {})[line] = rule

for path, issues in sorted(by_file.items()):
    if not os.path.exists(path):
        print(f"SKIP: {path}")
        continue
    print(f"\n{path}: {len(issues)} issues")
    for line, rule in sorted(issues.items()):
        print(f"  L{line}: {rule}")
