"""Batch-fix CodeQL alerts in test files."""
import json, os, sys

with open("alerts.json", encoding="utf-8-sig") as f:
    alerts = [a for a in json.load(f) if a.get("state") == "open"]

# Group by file
by_file = {}
for a in alerts:
    loc = a["most_recent_instance"]["location"]
    path = loc["path"]
    line = loc["start_line"]
    rule = a["rule"]["id"]
    by_file.setdefault(path, set()).add((line, rule))

for path, issues in sorted(by_file.items()):
    if not os.path.exists(path):
        print(f"SKIP (no file): {path}")
        continue
    rules = {r for _, r in issues}
    print(f"\n{path}: {len(issues)} alerts - {rules}")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, rule in sorted(issues):
        if 1 <= line_no <= len(lines):
            content = lines[line_no - 1].rstrip()
            print(f"  L{line_no}: [{rule}] {content[:80]}")
