import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

text = Path("hel1os_2026Nov08T145547498.py").read_text()

pattern = r'HLS_(2026\d{4})_(\d{6})_(\d+)sec_lev1_(V\d+)\.zip'

daily_intervals = defaultdict(list)

for date, start, duration, version in re.findall(pattern, text):
    day = datetime.strptime(date, "%Y%m%d")
    start_time = datetime.strptime(start, "%H%M%S").time()

    start_dt = datetime.combine(day.date(), start_time)
    end_dt = start_dt + timedelta(seconds=int(duration))

    # Split the observation across all calendar days it overlaps
    current = start_dt

    while current < end_dt:
        next_day = datetime.combine(
            current.date() + timedelta(days=1),
            datetime.min.time()
        )

        segment_end = min(end_dt, next_day)
        daily_intervals[current.strftime("%Y%m%d")].append(
            (current, segment_end)
        )
        current = segment_end

print("DATE       COVERAGE  MISSING")

full_days = 0
partial_days = 0
low_coverage_days = []

for day in sorted(daily_intervals):
    date = datetime.strptime(day, "%Y%m%d")
    day_start = date
    day_end = date + timedelta(days=1)

    intervals = sorted(daily_intervals[day])
    merged = []

    for start, end in intervals:
        start = max(start, day_start)
        end = min(end, day_end)

        if start >= end:
            continue

        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    covered = sum(
        (end - start).total_seconds()
        for start, end in merged
    )

    coverage = covered / 3600
    missing = 24 - coverage

    if coverage >= 23.5:
        full_days += 1
    else:
        partial_days += 1
        low_coverage_days.append(day)

    print(f"{day}  {coverage:6.2f}h  {missing:6.2f}h")

print("\n===== SUMMARY =====")
print("Dates with any HEL1OS observation:", len(daily_intervals))
print("Days with >= 23.5 hours coverage:", full_days)
print("Days with < 23.5 hours coverage:", partial_days)
print("Incomplete dates:")
for day in low_coverage_days:
    print(day)
