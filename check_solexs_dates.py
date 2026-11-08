import re
from pathlib import Path
from datetime import date, timedelta, datetime
from collections import Counter

script = Path("solexs_2026Nov08T144629101.py")

if not script.exists():
    print("PRADAN script not found.")
    raise SystemExit(1)

content = script.read_text()

all_dates = re.findall(r"AL1_SLX_L1_(2026\d{4})_v1\.0\.zip", content)

start = date(2026, 1, 1)
end = date(2026, 10, 31)

dates = [d for d in all_dates if "20260101" <= d <= "20261031"]
available = set(dates)

expected = set()
current = start

while current <= end:
    expected.add(current.strftime("%Y%m%d"))
    current += timedelta(days=1)

missing = sorted(expected - available)
duplicates = {d: n for d, n in Counter(dates).items() if n > 1}

print("Total ZIP paths in script:", len(all_dates))
print("ZIP paths from Jan-Oct:", len(dates))
print("Unique dates from Jan-Oct:", len(available))
print("Missing dates:", len(missing))
print("Duplicate dates:", duplicates)

print("\nMissing dates:")
for d in missing:
    print(datetime.strptime(d, "%Y%m%d").strftime("%d %b %Y"))
available = set(dates)

expected = set()
current = start
while current <= end:
    expected.add(current.strftime("%Y%m%d"))
    current += timedelta(days=1)

missing = sorted(expected - available)
duplicates = {d: n for d, n in Counter(dates).items() if n > 1}

print("Total ZIP paths in script:", len(all_dates))
print("ZIP paths from Jan-Oct:", len(dates))
print("Unique dates from Jan-Oct:", len(available))
print("Missing dates:", len(missing))
print("Duplicate dates:", duplicates)

print("\nMissing dates:")
for d in missing:
    print(datetime.strptime(d, "%Y%m%d").strftime("%d %b %Y"))
