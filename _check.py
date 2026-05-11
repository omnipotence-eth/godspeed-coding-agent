import json, sys, subprocess
r = subprocess.run(["gh","api","repos/t-timms/godspeed-coding-agent/code-scanning/alerts","--paginate"], capture_output=True, text=True)
data = json.loads(r.stdout)
open_alerts = [a for a in data if a.get("state") == "open"]
print(f"Remaining: {len(open_alerts)}")
by_file = {}
for a in open_alerts:
    loc = a["most_recent_instance"]["location"]
    rule = a["rule"]["id"]
    k = loc["path"]
    by_file.setdefault(k, []).append((loc["start_line"], rule))
for path in sorted(by_file):
    issues = by_file[path]
    print(f"\n{path} ({len(issues)})")
    for line, rule in sorted(issues):
        print(f"  L{line}: {rule}")
