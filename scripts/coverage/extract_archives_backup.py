
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv
import re
import shutil

# ==================================================
# CONFIGURATION
# ==================================================

BASE = Path(__file__).resolve().parents[2]

SOURCE = BASE / "downloaders" / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData"

HEL1OS_SOURCE = SOURCE / "hel1os" / "level1" / "2026"
SOLEXS_SOURCE = SOURCE / "solexs" / "level1" / "2026"

HEL1OS_OUTPUT = BASE / "raw_hel1os"
SOLEXS_OUTPUT = BASE / "raw_solexs"

REPORT_DIR = BASE / "output"
REPORT_DIR.mkdir(exist_ok=True)

START_DATE = "20260101"
END_DATE = "20260731"

# First run is a dry run. Change to False only after
# checking the extraction plan.
DRY_RUN = True

# ==================================================
# FILE SELECTION
# ==================================================

def get_archive_date(filename):
    match = re.search(r"2026\d{4}", filename)
    return match.group(0) if match else None


def required_file(instrument, filename):
    name = filename.lower()
    base = name.split("/")[-1]

    if instrument == "HEL1OS":
        is_lightcurve = (
            base.startswith("lightcurve_")
            and base.endswith(".fits")
        )

        is_gti = (
            "/aux/" in "/" + name
            and base.startswith("gti")
            and base.endswith(".fits")
        )

        return is_lightcurve or is_gti

    if instrument == "SoLEXS":
        return (
            "/sdd2/" in "/" + name
            and base.endswith((".lc.gz", ".gti.gz"))
        )

    return False


# ==================================================
# REPORTING
# ==================================================

def save_report(rows, filename):
    path = REPORT_DIR / filename

    fields = [
        "instrument",
        "date",
        "archive",
        "member",
        "output_path",
        "size_bytes",
        "status",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ==================================================
# EXTRACTION
# ==================================================

def main():

    sources = [
        ("HEL1OS", HEL1OS_SOURCE, HEL1OS_OUTPUT),
        ("SoLEXS", SOLEXS_SOURCE, SOLEXS_OUTPUT),
    ]

    report_rows = []

    total_archives = 0
    total_files = 0
    total_bytes = 0
    errors = 0

    for instrument, source, destination in sources:

        print(f"\nProcessing {instrument}")

        archives = sorted(source.rglob("*.zip"))

        selected_archives = []

        for archive in archives:
            date = get_archive_date(archive.name)

            if date and START_DATE <= date <= END_DATE:
                selected_archives.append((archive, date))

        print("Selected archives:", len(selected_archives))

        for archive_number, (archive, date) in enumerate(
            selected_archives, start=1
        ):

            total_archives += 1

            with ZipFile(archive) as z:

                members = [
                    info for info in z.infolist()
                    if not info.is_dir()
                    and required_file(instrument, info.filename)
                ]

                for info in members:

                    member_path = PurePosixPath(info.filename)

                    # Protect against unsafe archive paths.
                    if member_path.is_absolute() or ".." in member_path.parts:
                        errors += 1
                        continue

                    if instrument == "HEL1OS":
                        relative_path = Path(*member_path.parts)
                    else:
                        relative_path = (
                            Path("2026")
                            / date[4:6]
                            / date[6:8]
                            / Path(*member_path.parts)
                        )

                    target = destination / relative_path

                    total_files += 1
                    total_bytes += info.file_size

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

                    if DRY_RUN:
                        row["status"] = "planned"

                    else:
                        try:
                            target.parent.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            if target.exists():
                                if target.stat().st_size == info.file_size:
                                    row["status"] = "already_exists"
                                else:
                                    raise RuntimeError(
                                        "Existing file has different size"
                                    )

                            else:
                                temporary = target.with_name(
                                    target.name + ".part"
                                )

                                with z.open(info) as source_file:
                                    with temporary.open("wb") as output_file:
                                        shutil.copyfileobj(
                                            source_file,
                                            output_file,
                                        )

                                if temporary.stat().st_size != info.file_size:
                                    temporary.unlink(missing_ok=True)
                                    raise RuntimeError(
                                        "Extracted size mismatch"
                                    )

                                temporary.replace(target)
                                row["status"] = "extracted"

                        except Exception as exc:
                            errors += 1
                            row["status"] = "error"
                            row["error"] = str(exc)

                    report_rows.append(row)

            if archive_number % 25 == 0:
                print(
                    f"Processed {archive_number}/"
                    f"{len(selected_archives)} archives"
                )

            # Save progress so the report survives interruption.
            report_name = (
                "selective_extraction_plan.csv"
                if DRY_RUN
                else "selective_extraction_report.csv"
            )
            save_report(report_rows, report_name)

    print("\n========== SUMMARY ==========")
    print("Mode:", "DRY RUN" if DRY_RUN else "EXTRACTION")
    print("Archives processed:", total_archives)
    print("Required files:", total_files)
    print(f"Required size: {total_bytes / 1024**3:.2f} GB")
    print("Errors:", errors)

    report_name = (
        "selective_extraction_plan.csv"
        if DRY_RUN
        else "selective_extraction_report.csv"
    )

    print("Report:", REPORT_DIR / report_name)


if __name__ == "__main__":
    main()
