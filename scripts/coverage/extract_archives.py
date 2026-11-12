
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv
import re
import shutil

BASE = Path(__file__).resolve().parents[2]

SOURCE = BASE / "downloaders" / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData"
HEL1OS_SOURCE = SOURCE / "hel1os" / "level1" / "2026"
SOLEXS_SOURCE = SOURCE / "solexs" / "level1" / "2026"

HEL1OS_OUTPUT = BASE / "raw_hel1os"
SOLEXS_OUTPUT = BASE / "raw_solexs"
REPORT_DIR = BASE / "output"

START_DATE = "20260101"
END_DATE = "20260731"

# Keep True for the next verification run.
DRY_RUN = False

def get_archive_date(filename):
    match = re.search(r"2026\d{4}", filename)
    return match.group(0) if match else None

def required_file(instrument, filename):
    name = filename.lower()
    base = name.split("/")[-1]

    if instrument == "HEL1OS":
        return (
            base.startswith("lightcurve_") and base.endswith(".fits")
        ) or (
            "/aux/" in "/" + name
            and base.startswith("gti")
            and base.endswith(".fits")
        )

    if instrument == "SoLEXS":
        return (
            "/sdd2/" in "/" + name
            and base.endswith((".lc.gz", ".gti.gz"))
        )

    return False

def safe_target(destination, relative_path):
    root = destination.resolve()
    target = (destination / relative_path).resolve()

    if not target.is_relative_to(root):
        raise ValueError(f"Unsafe output path: {relative_path}")

    return target

def save_report(rows, filename):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "instrument", "date", "archive", "member",
        "output_path", "size_bytes", "status", "error"
    ]

    with (REPORT_DIR / filename).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        ("HEL1OS", HEL1OS_SOURCE, HEL1OS_OUTPUT),
        ("SoLEXS", SOLEXS_SOURCE, SOLEXS_OUTPUT),
    ]

    # Estimate required extraction space before doing anything.
    planned_bytes = 0
    planned_files = 0

    for instrument, source, _ in sources:
        for archive in source.rglob("*.zip"):
            date = get_archive_date(archive.name)
            if not date or not START_DATE <= date <= END_DATE:
                continue

            with ZipFile(archive) as z:
                for info in z.infolist():
                    if not info.is_dir() and required_file(instrument, info.filename):
                        planned_bytes += info.file_size
                        planned_files += 1

    free_bytes = shutil.disk_usage(BASE.drive or BASE).free

    print(f"Planned files: {planned_files}")
    print(f"Required space: {planned_bytes / 1024**3:.2f} GB")
    print(f"Available space: {free_bytes / 1024**3:.2f} GB")

    if not DRY_RUN and free_bytes < planned_bytes + 2 * 1024**3:
        raise RuntimeError("Insufficient disk space. Extraction stopped.")

    report_rows = []
    total_archives = total_files = total_bytes = errors = 0

    for instrument, source, destination in sources:
        print(f"\nProcessing {instrument}")

        archives = sorted(source.rglob("*.zip"))
        selected = []

        for archive in archives:
            date = get_archive_date(archive.name)
            if date and START_DATE <= date <= END_DATE:
                selected.append((archive, date))

        print("Selected archives:", len(selected))

        for number, (archive, date) in enumerate(selected, 1):
            total_archives += 1

            with ZipFile(archive) as z:
                members = [
                    info for info in z.infolist()
                    if not info.is_dir()
                    and required_file(instrument, info.filename)
                ]

                for info in members:
                    try:
                        member_path = PurePosixPath(info.filename)

                        if member_path.is_absolute() or ".." in member_path.parts:
                            raise ValueError("Unsafe ZIP member path")

                        if instrument == "HEL1OS":
                            relative = Path(*member_path.parts)
                        else:
                            relative = (
                                Path("2026") / date[4:6] / date[6:8]
                                / Path(*member_path.parts)
                            )

                        target = safe_target(destination, relative)

                        row = {
                            "instrument": instrument,
                            "date": date,
                            "archive": archive.name,
                            "member": info.filename,
                            "output_path": str(target.relative_to(BASE)),
                            "size_bytes": info.file_size,
                            "status": "",
                            "error": "",
                        }

                        total_files += 1
                        total_bytes += info.file_size

                        if DRY_RUN:
                            row["status"] = "planned"

                        elif target.exists():
                            if target.stat().st_size != info.file_size:
                                raise RuntimeError("Existing file size mismatch")

                            with target.open("rb") as existing:
                                import zlib
                                crc = 0
                                while chunk := existing.read(1024 * 1024):
                                    crc = zlib.crc32(chunk, crc)

                            if crc & 0xFFFFFFFF != info.CRC:
                                raise RuntimeError("Existing file CRC mismatch")

                            row["status"] = "already_exists"

                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_name(target.name + ".part")

                            try:
                                with z.open(info) as src, temporary.open("wb") as dst:
                                    shutil.copyfileobj(src, dst)

                                if temporary.stat().st_size != info.file_size:
                                    raise RuntimeError("Extracted size mismatch")

                                with temporary.open("rb") as f:
                                    import zlib
                                    crc = 0
                                    while chunk := f.read(1024 * 1024):
                                        crc = zlib.crc32(chunk, crc)

                                if crc & 0xFFFFFFFF != info.CRC:
                                    raise RuntimeError("Extracted CRC mismatch")

                                temporary.replace(target)
                                row["status"] = "extracted"

                            finally:
                                temporary.unlink(missing_ok=True)

                        report_rows.append(row)

                    except Exception as exc:
                        errors += 1
                        report_rows.append({
                            "instrument": instrument,
                            "date": date,
                            "archive": archive.name,
                            "member": info.filename,
                            "output_path": "",
                            "size_bytes": info.file_size,
                            "status": "error",
                            "error": str(exc),
                        })

            if number % 25 == 0:
                print(f"Processed {number}/{len(selected)} archives")

            report_name = (
                "selective_extraction_plan.csv" if DRY_RUN
                else "selective_extraction_report.csv"
            )
            save_report(report_rows, report_name)

    print("\n========== SUMMARY ==========")
    print("Mode:", "DRY RUN" if DRY_RUN else "EXTRACTION")
    print("Archives processed:", total_archives)
    print("Required files:", total_files)
    print(f"Required size: {total_bytes / 1024**3:.2f} GB")
    print("Errors:", errors)
    print("Report:", REPORT_DIR / (
        "selective_extraction_plan.csv" if DRY_RUN
        else "selective_extraction_report.csv"
    ))

if __name__ == "__main__":
    main()
