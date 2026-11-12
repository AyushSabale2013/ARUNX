
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "raw_hel1os"
OUTPUT_DIR = ROOT / "output" / "scientific_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_column(data, name):
    """Find a FITS table column without case sensitivity."""
    for column in data.columns.names:
        if column.lower() == name.lower():
            return np.asarray(data[column], dtype=float)

    raise KeyError(f"Column not found: {name}")


def process_gti(starts, stops):
    """Calculate GTI duration using continuous MJD endpoints."""

    valid = (
        np.isfinite(starts)
        & np.isfinite(stops)
        & (stops > starts)
    )

    intervals = sorted(zip(starts[valid], stops[valid]))

    merged = []
    overlaps = 0

    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            overlaps += 1
            merged[-1][1] = max(merged[-1][1], stop)

    duration_sec = sum(
        (stop - start) * 86400.0
        for start, stop in merged
    )

    return valid, merged, duration_sec, overlaps


def main():

    gti_files = sorted(DATA_DIR.rglob("gti*.fits"))
    results = []

    print(f"Found HEL1OS GTI files: {len(gti_files)}")

    for index, gti_path in enumerate(gti_files, start=1):

        detector = gti_path.stem.replace("gti", "").lower()

        subfolder = (
            "cdte" if detector.startswith("cdte") else "czt"
        )

        lc_path = (
            gti_path.parent.parent
            / subfolder
            / f"lightcurve_{detector}.fits"
        )

        # Extract date from the segment directory.
        segment_name = gti_path.parent.parent.name
        date = segment_name.split("_")[1] if "_" in segment_name else "UNKNOWN"

        try:
            with fits.open(gti_path) as hdul:

                gti_hdu = next(
                    hdu for hdu in hdul
                    if hdu.data is not None
                    and hdu.columns is not None
                    and {"tstart", "tstop"}.issubset(
                        {c.lower() for c in hdu.columns.names}
                    )
                )

                starts = get_column(gti_hdu.data, "tstart")
                stops = get_column(gti_hdu.data, "tstop")

                header_exposure = gti_hdu.header.get("EXPOSURE")

            valid, intervals, duration_sec, overlaps = process_gti(
                starts, stops
            )

            if not lc_path.exists():
                raise FileNotFoundError(
                    f"Matching lightcurve missing: {lc_path}"
                )

            with fits.open(lc_path) as hdul:

                band_hdus = [
                    hdu for hdu in hdul[1:]
                    if hdu.data is not None
                    and hdu.columns is not None
                    and "mjd" in [
                        c.lower() for c in hdu.columns.names
                    ]
                ]

                if len(band_hdus) != 5:
                    raise ValueError(
                        f"Expected 5 energy bands, found {len(band_hdus)}"
                    )

                for band_hdu in band_hdus:

                    times = get_column(band_hdu.data, "MJD")

                    timebin = float(
                        band_hdu.header.get("TIMEBIN", 1.0)
                    )

                    finite = np.isfinite(times)
                    finite_times = times[finite]

                    inside = np.zeros(len(times), dtype=bool)

                    for start, stop in intervals:
                        inside |= (
                            finite
                            & (times >= start - 1e-12)
                            & (times <= stop + 1e-12)
                        )

                    inside_count = int(inside.sum())
                    outside_count = int(
                        (finite & ~inside).sum()
                    )

                    differences = (
                        np.diff(finite_times) * 86400.0
                    )

                    duplicate_count = int(
                        np.sum(
                            np.isclose(
                                differences,
                                0,
                                rtol=0,
                                atol=1e-9
                            )
                        )
                    )

                    nonmonotonic_count = int(
                        np.sum(differences < -1e-9)
                    )

                    median_cadence = (
                        float(np.median(differences))
                        if len(differences) else np.nan
                    )

                    sample_coverage_sec = (
                        inside_count * timebin
                    )

                    coverage_fraction = (
                        sample_coverage_sec / duration_sec
                        if duration_sec > 0 else np.nan
                    )

                    outside_gti_fraction = (
                        outside_count / int(finite.sum())
                        if finite.sum() > 0 else np.nan
                    )

                    # Classify structural/data-quality problems separately
                    # from samples excluded by the GTI mask.
                    if (
                        not valid.all()
                        or len(intervals) == 0
                        or not finite.all()
                        or duplicate_count > 0
                        or nonmonotonic_count > 0
                        or timebin <= 0
                    ):
                        status = "ERROR"

                    elif outside_count > 0:
                        status = "REVIEW"

                    else:
                        status = "PASS"

                    results.append({
                        "date": date,
                        "segment": segment_name,
                        "detector": detector.upper(),
                        "energy_band": band_hdu.name,
                        "gti_intervals": len(starts),
                        "invalid_gti_intervals": int(
                            (~valid).sum()
                        ),
                        "overlapping_gti_intervals": overlaps,
                        "gti_duration_sec": round(
                            duration_sec, 3
                        ),
                        "header_exposure_sec": header_exposure,
                        "lightcurve_samples": len(times),
                        "finite_timestamps": int(finite.sum()),
                        "samples_inside_gti": inside_count,
                        "samples_outside_gti": outside_count,
                        "outside_gti_fraction": round(
                            outside_gti_fraction, 6
                        ) if np.isfinite(outside_gti_fraction) else np.nan,
                        "sample_coverage_sec": round(
                            sample_coverage_sec, 3
                        ),
                        "coverage_fraction": round(
                            coverage_fraction, 6
                        ) if np.isfinite(coverage_fraction) else np.nan,
                        "median_cadence_sec": round(
                            median_cadence, 6
                        ) if np.isfinite(median_cadence) else np.nan,
                        "duplicate_timestamps": duplicate_count,
                        "nonmonotonic_timestamps": nonmonotonic_count,
                        "status": status,
                    })

        except Exception as exc:

            results.append({
                "date": date,
                "segment": segment_name,
                "detector": detector.upper(),
                "energy_band": "ALL",
                "status": "ERROR",
                "error": str(exc),
            })

        if index % 100 == 0:
            print(
                f"Checked {index}/{len(gti_files)} GTI files"
            )

    df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR / "hel1os_exposure_consistency.csv"
    )

    df.to_csv(output_file, index=False)

    print("\n========== HEL1OS AUDIT SUMMARY ==========")
    print(f"GTI files: {len(gti_files)}")
    print(f"Energy-band records: {len(df)}")
    print(f"PASS: {(df['status'] == 'PASS').sum()}")
    print(f"REVIEW: {(df['status'] == 'REVIEW').sum()}")
    print(f"ERROR: {(df['status'] == 'ERROR').sum()}")

    print("\nDetector-wise results:")
    print(
        df.groupby(["detector", "status"])
        .size()
        .to_string()
    )

    print(f"\nReport saved to:\n{output_file}")


if __name__ == "__main__":
    main()
