import re
from pathlib import Path
from datetime import date, timedelta
from collections import Counter

script = Path("hel1os_2026Nov08T145547498.py")

if not script.exists():
    print("HEL1OS PRADAN script not found.")
    raise SystemExit(1)

content = script.read_text()

dates = re.findall(r"HLS_(2026\d{4})_", content)

start = date(2026, 1, 1)
end = date(2026, 10, 31)

dates = [d for d in dates if "20260101" <= d <= "20261031"]
counts = Counter(dates)

expected = set()
current = start
while current <= end:
    expected.add(current.strftime("%Y%m%d"))
    current += timedelta(days=1)

missing = sorted(expected - set(counts))
incorrect = {d: n for d, n in counts.items() if n != 2}

print("Total ZIP paths Jan-Oct:", len(dates))
print("Unique dates:", len(counts))
print("Expected ZIP files:", len(expected) * 2)
print("Missing dates:", len(missing))
print("Dates not having exactly 2 ZIPs:", len(incorrect))

print("\nMissing dates:")
for d in missing:
    print(f"{d[6:8]}-{d[4:6]}-{d[:4]}")

print("\nDates with unexpected file counts:")
for d, n in sorted(incorrect.items()):
    print(f"{d}: {n} ZIP files")
