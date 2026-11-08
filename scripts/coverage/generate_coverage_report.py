import re
import csv
from pathlib import Path
from collections import Counter
import pandas as pd

HEL_FILE = Path("hel1os_2026Nov08T145547498.py")
SOL_FILE = Path("solexs_2026Nov08T144629101.py")
GOES_FILE = Path("output/goes_events_parsed.csv")

# Extract HEL1OS archive filenames
hel_text = HEL_FILE.read_text()
hel_matches = re.findall(
    r'HLS_(2026\d{4})_(\d{6})_(\d+)sec_lev1_(V\d+)\.zip',
    hel_text
)
hel_records = [
    {"date": date, "start": start, "duration": int(duration), "version": version}
    for date, start, duration, version in hel_matches
]

# Extract SoLEXS archive dates
sol_text = SOL_FILE.read_text()
sol_dates = re.findall(r'AL1_SLX_L1_(2026\d{4})_v1\.0\.zip', sol_text)

hel_counts = Counter(r["date"] for r in hel_records)
sol_counts = Counter(sol_dates)

# Load GOES catalogue
goes = pd.read_csv(GOES_FILE)
goes["date"] = pd.to_datetime(goes["date"]).dt.strftime("%Y%m%d")
goes["class_type"] = goes["class_type"].astype(str).str.upper()

mx = goes[goes["class_type"].isin(["M", "X"])]
mx_counts = mx.groupby("date").size().to_dict()
m_counts = mx[mx["class_type"] == "M"].groupby("date").size().to_dict()
x_counts = mx[mx["class_type"] == "X"].groupby("date").size().to_dict()

# Build daily report for Jan-Oct 2026
dates = pd.date_range("2026-01-01", "2026-10-31")
rows = []

for dt in dates:
    d = dt.strftime("%Y%m%d")
    h = hel_counts[d]
    s = sol_counts[d]
    rows.append({
        "date": dt.strftime("%Y-%m-%d"),
        "hel1os_zip_count": h,
        "solexs_zip_count": s,
        "both_have_files": int(h > 0 and s > 0),
        "goes_m_count": m_counts.get(d, 0),
        "goes_x_count": x_counts.get(d, 0),
        "goes_mx_count": mx_counts.get(d, 0),
    })

report = pd.DataFrame(rows)
report.to_csv("output/common_coverage_report.csv", index=False)

# Summary
common = report[report["both_have_files"] == 1]
print("\n===== ARCHIVE COVERAGE SUMMARY =====")
print("Calendar dates:", len(report))
print("HEL1OS dates with files:", (report["hel1os_zip_count"] > 0).sum())
print("SoLEXS dates with files:", (report["solexs_zip_count"] > 0).sum())
print("Dates with files from BOTH:", len(common))
print("Dates missing from either instrument:", len(report) - len(common))

print("\n===== GOES M/X EVENTS =====")
print("Total M/X events in catalogue:", len(mx))
print("M-class events:", len(mx[mx["class_type"] == "M"]))
print("X-class events:", len(mx[mx["class_type"] == "X"]))
print("M/X events on common archive dates:",
      int(common["goes_mx_count"].sum()))

print("\n===== MONTHLY COMMON-DATE M/X COUNTS =====")
monthly = common.groupby(common["date"].str[:7]).agg(
    common_dates=("date", "count"),
    M_flares=("goes_m_count", "sum"),
    X_flares=("goes_x_count", "sum"),
    MX_flares=("goes_mx_count", "sum"),
)
print(monthly.to_string())

print("\n===== DATES WITH M/X EVENTS BUT NO COMMON ARCHIVE FILES =====")
missing_events = report[
    (report["goes_mx_count"] > 0) & (report["both_have_files"] == 0)
]
if missing_events.empty:
    print("None")
else:
    print(missing_events[
        ["date", "hel1os_zip_count", "solexs_zip_count", "goes_mx_count"]
    ].to_string(index=False))

print("\nSaved: output/common_coverage_report.csv")
