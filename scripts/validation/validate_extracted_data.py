
from pathlib import Path
from collections import defaultdict
from datetime import date
import re

import numpy as np
import pandas as pd
from astropy.io import fits


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[2]

HEL1OS_DIR = BASE / "raw_hel1os"
SOLEXS_DIR = BASE / "raw_solexs"
OUTPUT_DIR = BASE / "output"

# First test only the previously validated sample date.
# Set to None after the test to validate every extracted date.
DATE_FILTER = None

DETECTORS = ["cdte1", "cdte2", "czt1", "czt2"]


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_date(path):

    parts = path.parts

    for i in range(len(parts) - 2):
        try:
            year = parts[i]
            month = parts[i + 1]
            day = parts[i + 2]

            if (
                re.fullmatch(r"20\d{2}", year)
                and re.fullmatch(r"\d{2}", month)
                and re.fullmatch(r"\d{2}", day)
            ):
                return date(
                    int(year), int(month), int(day)
                ).isoformat()

        except ValueError:
            continue

    match = re.search(r"(20\d{2})(\d{2})(\d{2})", path.name)

    if match:
        try:
            return date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3))
            ).isoformat()
        except ValueError:
            pass

    return None


# ============================================================
# FITS VALIDATION
# ============================================================

def validate_fits(path):

    result = {
        "status": "PASS",
        "error": "",
        "hdu_count": 0,
        "data_hdu_count": 0,
        "table_columns": "",
        "row_count": 0,
        "gti_intervals": 0,
        "invalid_gti_intervals": 0,
        "overlapping_gti_intervals": 0,
    }

    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=False
        ) as hdul:

            hdul.verify("exception")

            result["hdu_count"] = len(hdul)

            for hdu in hdul:

                data = hdu.data

                if data is None:
                    continue

                result["data_hdu_count"] += 1

                if getattr(data, "dtype", None) is not None:
                    if data.dtype.names:
                        columns = list(data.dtype.names)

                        result["table_columns"] = ",".join(
                            columns
                        )

                        result["row_count"] += len(data)

                        column_map = {
                            col.upper(): col
                            for col in columns
                        }

                        if (
                            "START" in column_map
                            and "STOP" in column_map
                        ):

                            starts = np.asarray(
                                data[column_map["START"]],
                                dtype=float
                            )

                            stops = np.asarray(
                                data[column_map["STOP"]],
                                dtype=float
                            )

                            result["gti_intervals"] += len(starts)

                            invalid = (
                                ~np.isfinite(starts)
                                | ~np.isfinite(stops)
                                | (stops < starts)
                            )

                            result["invalid_gti_intervals"] += int(
                                invalid.sum()
                            )

                            valid = ~invalid

                            if valid.sum() > 1:

                                order = np.argsort(starts[valid])

                                sorted_starts = starts[valid][order]
                                sorted_stops = stops[valid][order]

                                overlaps = (
                                    sorted_starts[1:]
                                    < sorted_stops[:-1]
                                )

                                result[
                                    "overlapping_gti_intervals"
                                ] += int(overlaps.sum())

            if result["data_hdu_count"] == 0:
                result["status"] = "WARNING"
                result["error"] = "No data-bearing HDU found"

    except Exception as e:

        result["status"] = "FAIL"
        result["error"] = str(e)

    return result


# ============================================================
# INVENTORY AND FILE VALIDATION
# ============================================================

inventory = []

helios_groups = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(
            lambda: {"LIGHTCURVE": [], "GTI": []}
        )
    )
)

solexs_groups = defaultdict(
    lambda: {"LIGHTCURVE": [], "GTI": []}
)


def add_file(path, instrument, category, group_key):

    file_date = extract_date(path)

    if DATE_FILTER and file_date != DATE_FILTER:
        return

    if file_date is None:
        print("WARNING: Could not identify date:", path)
        return

    validation = validate_fits(path)

    row = {
        "instrument": instrument,
        "date": file_date,
        "category": category,
        "filename": path.name,
        "path": str(path.relative_to(BASE)),
        "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
        **validation
    }

    inventory.append(row)

    row["_valid"] = validation["status"] == "PASS"

    if instrument == "HEL1OS":
        obs_dir, detector = group_key
        helios_groups[file_date][obs_dir][detector][category].append(row)

    elif instrument == "SOLEXS":
        solexs_groups[group_key][category].append(row)


# HEL1OS scan

print("\nScanning HEL1OS...")

for path in HEL1OS_DIR.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in [".fits", ".fit"]:
        continue

    detector_match = re.search(
        r"(cdte1|cdte2|czt1|czt2)",
        path.name.lower()
    )

    if not detector_match:
        category = "OTHER"
        detector = "unknown"
    else:
        detector = detector_match.group(1)

        if path.parent.name.lower() == "aux":
            category = "GTI"
        else:
            category = "LIGHTCURVE"

    obs_dir = str(path.parent.parent.relative_to(BASE))

    add_file(
        path,
        "HEL1OS",
        category,
        (obs_dir, detector)
    )


# SoLEXS scan

print("\nScanning SoLEXS...")

for path in SOLEXS_DIR.rglob("*.gz"):

    filename = path.name.lower()

    if filename.endswith(".lc.gz"):
        category = "LIGHTCURVE"

    elif filename.endswith(".gti.gz"):
        category = "GTI"

    else:
        continue

    key = (extract_date(path), str(path.parent.relative_to(BASE)))

    if DATE_FILTER and key[0] != DATE_FILTER:
        continue

    add_file(
        path,
        "SOLEXS",
        category,
        key
    )


# ============================================================
# HEL1OS MATCHING
# ============================================================

helios_pair_rows = []
helios_date_stats = defaultdict(
    lambda: {
        "segments": 0,
        "matched_pairs": 0,
        "missing_pairs": 0,
        "invalid_pairs": 0
    }
)

for file_date, segments in helios_groups.items():

    for obs_dir, detectors in segments.items():

        helios_date_stats[file_date]["segments"] += 1

        for detector in DETECTORS:

            files = detectors[detector]

            lightcurves = files["LIGHTCURVE"]
            gtis = files["GTI"]

            matched = (
                len(lightcurves) == 1
                and len(gtis) == 1
                and lightcurves[0]["_valid"]
                and gtis[0]["_valid"]
            )

            if matched:
                helios_date_stats[file_date]["matched_pairs"] += 1
            else:
                helios_date_stats[file_date]["missing_pairs"] += 1

                if (
                    any(not x["_valid"] for x in lightcurves)
                    or any(not x["_valid"] for x in gtis)
                ):
                    helios_date_stats[file_date]["invalid_pairs"] += 1

            helios_pair_rows.append({
                "date": file_date,
                "observation": obs_dir,
                "detector": detector,
                "lightcurve_count": len(lightcurves),
                "gti_count": len(gtis),
                "pair_status": "PASS" if matched else "FAIL"
            })


# ============================================================
# SOLEXS MATCHING
# ============================================================

solexs_pair_rows = []
solexs_date_stats = defaultdict(
    lambda: {
        "lightcurves": 0,
        "gtis": 0,
        "valid_pairs": 0
    }
)

for (file_date, directory), files in solexs_groups.items():

    lightcurves = files["LIGHTCURVE"]
    gtis = files["GTI"]

    matched = (
        len(lightcurves) == 1
        and len(gtis) == 1
        and lightcurves[0]["_valid"]
        and gtis[0]["_valid"]
    )

    solexs_date_stats[file_date]["lightcurves"] += len(lightcurves)
    solexs_date_stats[file_date]["gtis"] += len(gtis)

    if matched:
        solexs_date_stats[file_date]["valid_pairs"] += 1

    solexs_pair_rows.append({
        "date": file_date,
        "directory": directory,
        "lightcurve_count": len(lightcurves),
        "gti_count": len(gtis),
        "pair_status": "PASS" if matched else "FAIL"
    })


# ============================================================
# DATE-WISE QUALITY REPORT
# ============================================================

all_dates = sorted(
    set(helios_date_stats) | set(solexs_date_stats)
)

date_rows = []
eligible_dates = []

for file_date in all_dates:

    h = helios_date_stats[file_date]
    s = solexs_date_stats[file_date]

    expected_helios_pairs = h["segments"] * len(DETECTORS)

    helios_complete = (
        expected_helios_pairs > 0
        and h["matched_pairs"] == expected_helios_pairs
    )

    solexs_complete = (
        s["valid_pairs"] > 0
        and s["lightcurves"] == s["gtis"]
    )

    candidate = helios_complete and solexs_complete

    if candidate:
        eligible_dates.append({
            "date": file_date,
            "helios_segments": h["segments"],
            "helios_matched_pairs": h["matched_pairs"],
            "solexs_valid_pairs": s["valid_pairs"],
            "status": "CANDIDATE"
        })

    date_rows.append({
        "date": file_date,
        "helios_segments": h["segments"],
        "helios_expected_pairs": expected_helios_pairs,
        "helios_matched_pairs": h["matched_pairs"],
        "helios_missing_pairs": h["missing_pairs"],
        "helios_invalid_pairs": h["invalid_pairs"],
        "solexs_lightcurves": s["lightcurves"],
        "solexs_gtis": s["gtis"],
        "solexs_valid_pairs": s["valid_pairs"],
        "helios_complete": helios_complete,
        "solexs_complete": solexs_complete,
        "candidate_eligible": candidate
    })


# ============================================================
# SAVE REPORTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inventory_df = pd.DataFrame(inventory)

if not inventory_df.empty:
    inventory_df.drop(
        columns=["_valid"],
        errors="ignore"
    ).to_csv(
        OUTPUT_DIR / "extraction_inventory.csv",
        index=False
    )

pd.DataFrame(date_rows).to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False
)

pd.DataFrame(eligible_dates).to_csv(
    OUTPUT_DIR / "eligible_dates.csv",
    index=False
)

pd.DataFrame(helios_pair_rows).to_csv(
    OUTPUT_DIR / "helios_matching_report.csv",
    index=False
)

pd.DataFrame(solexs_pair_rows).to_csv(
    OUTPUT_DIR / "solexs_matching_report.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("EXTRACTION VALIDATION SUMMARY")
print("=" * 55)

print("Date filter:", DATE_FILTER or "ALL DATES")
print("Total files inspected:", len(inventory))

if not inventory_df.empty:
    print("\nInstrument and category counts:")
    print(
        inventory_df.groupby(
            ["instrument", "category"]
        ).size()
    )

    print("\nFile validation statuses:")
    print(inventory_df["status"].value_counts())

print("\nDates inspected:", len(all_dates))
print("Candidate eligible dates:", len(eligible_dates))

print("\nReports saved in:", OUTPUT_DIR)
print("=" * 55)
