import json

with open("alerts.json", encoding="utf-8") as f:
    data = json.load(f)

seen = set()
for a in data:
    if a.get("state") != "open":
        continue
    loc = a["most_recent_instance"]["location"]
    path = loc["path"]
    line = loc["start_line"]
    rule = a["rule"]["id"]
    key = (path, rule)
    if key not in seen:
        seen.add(key)
        print(f"{rule:40s} {path}:{line}")

print(f"\nTotal unique alerts: {len(seen)}")
