
from pathlib import Path
from collections import Counter
import csv
import re

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DOWNLOAD_DIR = (
    BASE_DIR
    / "downloaders"
    / "pradan1.issdc.gov.in"
    / "al1"
    / "protected"
    / "downloadData"
)

OUTPUT_DIR = BASE_DIR / "output"
REPORT_FILE = OUTPUT_DIR / "archive_inventory_report.csv"

# Find downloaded archives
helios_archives = sorted(
    (DOWNLOAD_DIR / "hel1os" / "level1" / "2026").rglob("HLS_*.zip")
)

solexs_archives = sorted(
    (DOWNLOAD_DIR / "solexs" / "level1" / "2026").rglob("AL1_SLX_L1_*.zip")
)

archives = [
    ("HEL1OS", path) for path in helios_archives
] + [
    ("SoLEXS", path) for path in solexs_archives
]

# Inventory
rows = []

for instrument, path in archives:
    if instrument == "HEL1OS":
        match = re.search(r"HLS_(\d{8})_", path.name)
    else:
        match = re.search(r"AL1_SLX_L1_(\d{8})_", path.name)

    date = match.group(1) if match else ""

    rows.append({
        "instrument": instrument,
        "date": date,
        "filename": path.name,
        "relative_path": str(path.relative_to(BASE_DIR)),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 ** 2), 2),
    })

# Save CSV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with REPORT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "instrument",
        "date",
        "filename",
        "relative_path",
        "size_bytes",
        "size_mb",
    ])
    writer.writeheader()
    writer.writerows(rows)

# Summary
counts = Counter(row["instrument"] for row in rows)

print("=" * 55)
print("ARUN-X ARCHIVE INVENTORY")
print("=" * 55)

print("HEL1OS archives:", counts["HEL1OS"])
print("SoLEXS archives:", counts["SoLEXS"])
print("Total archives:", len(rows))

print(
    "Total size (GB):",
    round(sum(row["size_bytes"] for row in rows) / (1024 ** 3), 2)
)

print("Report saved to:", REPORT_FILE)
print("=" * 55)
