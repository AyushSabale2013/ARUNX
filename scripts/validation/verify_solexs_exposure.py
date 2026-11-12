
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "raw_solexs"
OUTPUT_DIR = ROOT / "output" / "scientific_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_gti_exposure(starts, stops):
    """Calculate union exposure for inclusive, integer-second GTIs."""

    valid = (
        np.isfinite(starts)
        & np.isfinite(stops)
        & (stops >= starts)
    )

    starts = starts[valid]
    stops = stops[valid]

    if not np.allclose(starts, np.round(starts), atol=1e-6):
        raise ValueError("GTI START values are not integer aligned")

    if not np.allclose(stops, np.round(stops), atol=1e-6):
        raise ValueError("GTI STOP values are not integer aligned")

    intervals = sorted(zip(starts, stops))

    merged = []

    for start, stop in intervals:
        start = int(round(start))
        stop = int(round(stop))

        if not merged or start > merged[-1][1] + 1:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)

    union_exposure = sum(
        stop - start + 1
        for start, stop in merged
    )

    raw_exposure = sum(
        int(round(stop)) - int(round(start)) + 1
        for start, stop in zip(starts, stops)
    )

    return valid, intervals, raw_exposure, union_exposure


def main():
    gti_files = sorted(DATA_DIR.rglob("*.gti.gz"))

    results = []

    print(f"Found GTI files: {len(gti_files)}")

    for index, gti_path in enumerate(gti_files, start=1):

        lc_path = gti_path.with_name(
            gti_path.name.replace(".gti.gz", ".lc.gz")
        )

        date = gti_path.name.split("_")[2]

        row = {
            "date": date,
            "gti_file": str(gti_path),
            "lightcurve_exists": lc_path.exists(),
        }

        try:
            with fits.open(gti_path) as hdul:
                gti = hdul["GTI"]
                starts = np.asarray(gti.data["START"], dtype=float)
                stops = np.asarray(gti.data["STOP"], dtype=float)
                header_exposure = float(
                    gti.header["EXPOSURE"]
                )

            valid, intervals, raw_exp, union_exp = (
                calculate_gti_exposure(starts, stops)
            )

            overlaps = 0

            for i in range(1, len(intervals)):
                if intervals[i][0] <= intervals[i - 1][1]:
                    overlaps += 1

            # Count actual samples inside the GTIs.
            with fits.open(lc_path) as hdul:
                lc = hdul["RATE"]
                times = np.asarray(lc.data["TIME"], dtype=float)

            inside = np.zeros(len(times), dtype=bool)

            for start, stop in intervals:
                inside |= (
                    np.isfinite(times)
                    & (times >= start - 1e-9)
                    & (times <= stop + 1e-9)
                )

            sample_count = int(inside.sum())

            metadata_match = np.isclose(
                header_exposure,
                union_exp,
                rtol=0,
                atol=1e-3,
            )

            samples_match = sample_count == union_exp

            status = (
                "PASS"
                if (
                    valid.all()
                    and metadata_match
                    and samples_match
                )
                else "REVIEW"
            )

            row.update({
                "gti_intervals": len(starts),
                "invalid_intervals": int((~valid).sum()),
                "zero_duration_intervals": int(
                    np.sum(stops[valid] == starts[valid])
                ),
                "overlapping_intervals": overlaps,
                "header_exposure_sec": header_exposure,
                "raw_inclusive_exposure_sec": raw_exp,
                "union_exposure_sec": union_exp,
                "lightcurve_samples_inside_gti": sample_count,
                "metadata_match": metadata_match,
                "samples_match": samples_match,
                "status": status,
            })

        except Exception as exc:
            row.update({
                "status": "ERROR",
                "error": str(exc),
            })

        results.append(row)

        if index % 25 == 0:
            print(f"Checked {index}/{len(gti_files)}")

    df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR / "solexs_exposure_consistency.csv"
    )

    df.to_csv(output_file, index=False)

    print("\n========== AUDIT SUMMARY ==========")
    print(f"GTI files: {len(gti_files)}")
    print(f"PASS: {(df['status'] == 'PASS').sum()}")
    print(f"REVIEW: {(df['status'] == 'REVIEW').sum()}")
    print(f"ERROR: {(df['status'] == 'ERROR').sum()}")

    if "metadata_match" in df.columns:
        print(
            "Metadata matches:",
            int(df["metadata_match"].fillna(False).sum())
        )

    if "samples_match" in df.columns:
        print(
            "Sample-count matches:",
            int(df["samples_match"].fillna(False).sum())
        )

    print(f"\nReport saved to:\n{output_file}")


if __name__ == "__main__":
    main()
