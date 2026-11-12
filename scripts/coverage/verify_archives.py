
from pathlib import Path
import zipfile
import csv
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[2]
INVENTORY = BASE_DIR / "output" / "archive_inventory_report.csv"
REPORT = BASE_DIR / "output" / "archive_integrity_report.csv"

results = []

with INVENTORY.open("r", newline="", encoding="utf-8") as file:
    archives = list(csv.DictReader(file))

print("=" * 60)
print("ARUN-X ZIP ARCHIVE INTEGRITY VERIFICATION")
print("=" * 60)
print("Total archives to verify:", len(archives))
print()

for index, archive in enumerate(archives, start=1):
    path = BASE_DIR / archive["relative_path"]

    status = "OK"
    error = ""

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_file = zf.testzip()

            if bad_file is not None:
                status = "CORRUPT"
                error = f"CRC failure: {bad_file}"

    except zipfile.BadZipFile as exc:
        status = "CORRUPT"
        error = str(exc)

    except Exception as exc:
        status = "ERROR"
        error = str(exc)

    results.append({
        "instrument": archive["instrument"],
        "date": archive["date"],
        "filename": archive["filename"],
        "size_bytes": archive["size_bytes"],
        "status": status,
        "error": error,
    })

    print(
        f"[{index}/{len(archives)}] "
        f"{archive['filename']} : {status}"
    )

with REPORT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "instrument",
            "date",
            "filename",
            "size_bytes",
            "status",
            "error",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

counts = Counter(row["status"] for row in results)

print()
print("=" * 60)
print("INTEGRITY VERIFICATION SUMMARY")
print("=" * 60)
print("Total:", len(results))
print("Healthy:", counts["OK"])
print("Corrupt:", counts["CORRUPT"])
print("Errors:", counts["ERROR"])
print("Report:", REPORT)
print("=" * 60)
