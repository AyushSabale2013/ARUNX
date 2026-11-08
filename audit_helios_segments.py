import re
from pathlib import Path
from collections import defaultdict

text = Path("hel1os_2026Nov08T145547498.py").read_text()

pattern = (
    r'HLS_(2026\d{4})_(\d{6})_(\d+)sec_lev1_(V\d+)\.zip'
)

records = defaultdict(list)

for date, start, duration, version in re.findall(pattern, text):
    records[date].append({
        "start": start,
        "duration": int(duration),
        "version": version
    })

print("Dates with questionable segment counts:\n")

for date in sorted(records):
    files = sorted(records[date], key=lambda x: x["start"])

    if len(files) != 2:
        print(f"\n{date}: {len(files)} files")
        for f in files:
            print(
                f"  Start: {f['start']} | "
                f"Duration: {f['duration']} sec | "
                f"Version: {f['version']}"
            )

print("\nAudit complete.")
