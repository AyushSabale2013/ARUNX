import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# ---------- HEL1OS coverage ----------
text = Path("hel1os_2026Nov08T145547498.py").read_text()
pattern = r'HLS_(2026\d{4})_(\d{6})_(\d+)sec_lev1_(V\d+)\.zip'

daily = defaultdict(list)

for date, start, duration, version in re.findall(pattern, text):
    day = datetime.strptime(date, "%Y%m%d")
    start_dt = datetime.combine(
        day.date(),
        datetime.strptime(start, "%H%M%S").time()
    )
    end_dt = start_dt + timedelta(seconds=int(duration))

    current = start_dt
    while current < end_dt:
        next_day = datetime.combine(
            current.date() + timedelta(days=1),
            datetime.min.time()
        )
        segment_end = min(end_dt, next_day)
        daily[current.strftime("%Y%m%d")].append(
            (current, segment_end)
        )
        current = segment_end

coverage = {}

for date, intervals in daily.items():
    day = datetime.strptime(date, "%Y%m%d")
    day_end = day + timedelta(days=1)

    merged = []
    for start, end in sorted(intervals):
        start = max(start, day)
        end = min(end, day_end)

        if start >= end:
            continue

        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    seconds = sum(
        (end - start).total_seconds()
        for start, end in merged
    )
    coverage[date] = seconds / 3600

# ---------- SoLEXS availability ----------
sol_text = Path("solexs_2026Nov08T144629101.py").read_text()
sol_dates = set(
    re.findall(r'AL1_SLX_L1_(2026\d{4})_v1\.0\.zip', sol_text)
)

# ---------- GOES events ----------
goes = pd.read_csv("output/goes_events_parsed.csv")
goes["date"] = pd.to_datetime(goes["date"]).dt.strftime("%Y%m%d")
goes["class_type"] = goes["class_type"].astype(str).str.upper()

goes = goes[goes["class_type"].isin(["M", "X"])]

m_counts = goes[goes["class_type"] == "M"].groupby("date").size()
x_counts = goes[goes["class_type"] == "X"].groupby("date").size()

# ---------- Daily report ----------
rows = []

for day in pd.date_range("2026-01-01", "2026-10-31"):
    date = day.strftime("%Y%m%d")
    hours = coverage.get(date, 0.0)
    sol_available = date in sol_dates

    rows.append({
        "date": day.strftime("%Y-%m-%d"),
        "hel1os_hours": round(hours, 2),
        "solexs_available": int(sol_available),
        "M_flares": int(m_counts.get(date, 0)),
        "X_flares": int(x_counts.get(date, 0)),
        "MX_flares": int(m_counts.get(date, 0) + x_counts.get(date, 0)),
        "hel1os_pass": int(hours >= 23.5),
        "both_pass": int(hours >= 23.5 and sol_available)
    })

report = pd.DataFrame(rows)
report.to_csv("output/final_date_report.csv", index=False)

eligible = report[report["both_pass"] == 1]

print("\n===== FINAL COVERAGE SUMMARY =====")
print("Calendar days:", len(report))
print("HEL1OS >= 23.5 hours:", int(report["hel1os_pass"].sum()))
print("SoLEXS available:", int(report["solexs_available"].sum()))
print("Both instruments pass:", len(eligible))
print("M flares on eligible dates:", int(eligible["M_flares"].sum()))
print("X flares on eligible dates:", int(eligible["X_flares"].sum()))
print("Total M/X events on eligible dates:", int(eligible["MX_flares"].sum()))

print("\n===== MONTHLY ELIGIBLE DATA =====")
monthly = eligible.groupby(
    eligible["date"].str[:7]
).agg(
    eligible_days=("date", "count"),
    M_flares=("M_flares", "sum"),
    X_flares=("X_flares", "sum"),
    MX_flares=("MX_flares", "sum")
)
print(monthly.to_string())

print("\n===== M/X EVENTS ON EXCLUDED DATES =====")
excluded = report[
    (report["both_pass"] == 0) & (report["MX_flares"] > 0)
]
print(excluded.to_string(index=False) if not excluded.empty else "None")

print("\nSaved: output/final_date_report.csv")
