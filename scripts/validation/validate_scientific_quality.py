
from pathlib import Path
from collections import defaultdict
import csv
import re

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[2]
HEL = ROOT / "raw_hel1os"
SLX = ROOT / "raw_solexs"
OUT = ROOT / "output" / "scientific_validation"

DATE_FILTER = None  # Set to None for the full dataset

CADENCE = 1.0
CADENCE_TOL = 0.15


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(dict.fromkeys(k for r in rows for k in r))

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def col(data, name):
    names = {n.lower(): n for n in data.names}

    if name.lower() not in names:
        raise ValueError(f"Missing {name}; found {data.names}")

    return np.asarray(data[names[name.lower()]])


def date_of(path, root):
    parts = path.relative_to(root).parts

    if len(parts) >= 3 and all(
        re.fullmatch(pattern, value)
        for pattern, value in [
            (r"20\d{2}", parts[0]),
            (r"\d{2}", parts[1]),
            (r"\d{2}", parts[2]),
        ]
    ):
        return f"{parts[0]}-{parts[1]}-{parts[2]}"

    match = re.search(r"(20\d{2})(\d{2})(\d{2})", path.name)

    return "-".join(match.groups()) if match else ""


def time_stats(t, seconds_per_unit):
    t = np.asarray(t, dtype=float)

    finite = t[np.isfinite(t)]

    # Preserve original order when checking monotonicity.
    ordered_diffs = np.diff(finite) * seconds_per_unit

    # Sort timestamps when estimating missing time bins.
    sorted_t = np.sort(finite)
    sorted_diffs = np.diff(sorted_t) * seconds_per_unit

    positive = sorted_diffs[sorted_diffs > 1e-8]

    median_cadence = (
        float(np.median(positive))
        if len(positive)
        else np.nan
    )

    missing_bins = 0

    if len(positive):
        estimated_bins = np.rint(positive / CADENCE).astype(int) - 1
        missing_bins = int(np.maximum(0, estimated_bins).sum())

    return {
        "rows": len(t),
        "nan_timestamps": int(np.isnan(t).sum()),
        "inf_timestamps": int(np.isinf(t).sum()),
        "duplicate_timestamps": int(
            len(finite) - len(np.unique(finite))
        ),
        "nonmonotonic_steps": int(
            np.sum(ordered_diffs <= 0)
        ),
        "median_cadence_sec": (
            round(median_cadence, 4)
            if np.isfinite(median_cadence)
            else ""
        ),
        "missing_1sec_bins_est": missing_bins,
    }


def gti_stats(
    t, starts, stops, seconds_per_unit,
    inclusive_endpoints=False
):
    t = np.asarray(t, dtype=float)
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)

    if inclusive_endpoints:
        valid_gti = (
            np.isfinite(starts)
            & np.isfinite(stops)
            & (stops >= starts)
        )
    else:
        valid_gti = (
            np.isfinite(starts)
            & np.isfinite(stops)
            & (stops > starts)
        )

    invalid_count = int((~valid_gti).sum())

    starts = starts[valid_gti]
    stops = stops[valid_gti]

    inside = np.zeros(len(t), dtype=bool)

    for start, stop in zip(starts, stops):
        inside |= (
            np.isfinite(t)
            & (t >= start - 1e-12)
            & (t <= stop + 1e-12)
        )

    finite_timestamps = np.isfinite(t)
    finite_count = int(finite_timestamps.sum())

    # Merge overlapping or adjacent intervals.
    intervals = sorted(zip(starts, stops))
    merged = []

    for start, stop in intervals:
        if inclusive_endpoints:
            separated = (
                not merged
                or start > merged[-1][1] + 1
            )
        else:
            separated = (
                not merged
                or start > merged[-1][1]
            )

        if separated:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(
                merged[-1][1], stop
            )

    if inclusive_endpoints:
        duration = sum(
            stop - start + 1
            for start, stop in merged
        )
    else:
        duration = sum(
            stop - start
            for start, stop in merged
        )

    inside_count = int(inside.sum())

    return {
        "gti_intervals": len(starts),
        "gti_duration_sec": round(
            float(duration * seconds_per_unit), 3
        ),
        "samples_inside_gti": inside_count,
        "samples_outside_gti": (
            finite_count - inside_count
        ),
        "sample_gti_fraction": round(
            inside_count / finite_count, 6
        ) if finite_count else 0,
        "invalid_gti_intervals": invalid_count,
    }, inside

def signal_stats(signal, inside):
    signal = np.asarray(signal, dtype=float)

    usable = inside & np.isfinite(signal)
    nan_mask = np.isnan(signal)
    inf_mask = np.isinf(signal)
    negative_mask = np.isfinite(signal) & (signal < 0)

    inside_count = int(inside.sum())
    valid_count = int(usable.sum())

    return {
        "nan_signal": int(nan_mask.sum()),
        "inf_signal": int(inf_mask.sum()),
        "negative_signal": int(negative_mask.sum()),

        "nan_signal_inside_gti": int(
            np.sum(nan_mask & inside)
        ),
        "inf_signal_inside_gti": int(
            np.sum(inf_mask & inside)
        ),
        "negative_signal_inside_gti": int(
            np.sum(negative_mask & inside)
        ),

        "nan_signal_outside_gti": int(
            np.sum(nan_mask & ~inside)
        ),
        "inf_signal_outside_gti": int(
            np.sum(inf_mask & ~inside)
        ),

        "valid_signal_samples_inside_gti": valid_count,

        "signal_completeness_inside_gti": round(
            valid_count / inside_count, 6
        ) if inside_count else 0,
    }


def error_stats(error, inside):
    error = np.asarray(error, dtype=float)

    nan_mask = np.isnan(error)
    inf_mask = np.isinf(error)
    negative_mask = np.isfinite(error) & (error < 0)

    return {
        "nan_stat_err": int(nan_mask.sum()),
        "inf_stat_err": int(inf_mask.sum()),
        "negative_stat_err": int(negative_mask.sum()),

        "nan_stat_err_inside_gti": int(
            np.sum(nan_mask & inside)
        ),
        "inf_stat_err_inside_gti": int(
            np.sum(inf_mask & inside)
        ),
        "negative_stat_err_inside_gti": int(
            np.sum(negative_mask & inside)
        ),
    }


def cadence_check(median):
    try:
        return (
            np.isfinite(float(median))
            and abs(float(median) - CADENCE) <= CADENCE_TOL
        )
    except (TypeError, ValueError):
        return False


def determine_status(ts, gx, signal, cadence_ok, error=None):
    problems = (
        ts["nan_timestamps"]
        or ts["inf_timestamps"]
        or ts["duplicate_timestamps"]
        or ts["nonmonotonic_steps"]
        or gx["invalid_gti_intervals"]
        or gx["gti_intervals"] == 0
        or gx["samples_inside_gti"] == 0
        or signal["nan_signal_inside_gti"]
        or signal["inf_signal_inside_gti"]
        or signal["negative_signal_inside_gti"]
        or not cadence_ok
    )

    if error is not None:
        problems = problems or (
            error["nan_stat_err_inside_gti"]
            or error["inf_stat_err_inside_gti"]
            or error["negative_stat_err_inside_gti"]
        )

    return "REVIEW" if problems else "PASS"


def validate_hel(lc, gt, date):
    match = re.search(
        r"lightcurve_(cdte[12]|czt[12])\.fits$",
        lc.name.lower()
    )

    detector = match.group(1)
    results = []

    try:
        with fits.open(gt, memmap=False) as h:
            g = next(
                x for x in h
                if x.data is not None
                and getattr(x.data, "names", None)
            )

            gti_start = col(g.data, "tstart")
            gti_stop = col(g.data, "tstop")

        with fits.open(lc, memmap=False) as h:
            for i, x in enumerate(h):

                if x.data is None or not getattr(
                    x.data, "names", None
                ):
                    continue

                names = {n.lower() for n in x.data.names}

                if not {"mjd", "ctr", "stat_err"}.issubset(names):
                    continue

                t = col(x.data, "MJD").astype(float)
                rate = col(x.data, "CTR").astype(float)
                err = col(x.data, "STAT_ERR").astype(float)

                if not (len(t) == len(rate) == len(err)):
                    raise ValueError(
                        "Timestamp, signal and error lengths differ"
                    )

                ts = time_stats(t, 86400)

                gx, inside = gti_stats(
                    t, gti_start, gti_stop, 86400
                )

                signal = signal_stats(rate, inside)
                errors = error_stats(err, inside)

                cadence_ok = cadence_check(
                    ts["median_cadence_sec"]
                )

                status = determine_status(
                    ts, gx, signal, cadence_ok, errors
                )

                results.append({
                    "date": date,
                    "instrument": "HEL1OS",
                    "detector": detector,
                    "band": x.header.get(
                        "EXTNAME", f"HDU_{i}"
                    ),
                    "file": str(lc.relative_to(ROOT)),
                    "gti_file": str(gt.relative_to(ROOT)),

                    **ts,
                    **gx,
                    **signal,
                    **errors,

                    "cadence_near_1sec": bool(cadence_ok),
                    "status": status,
                    "error": "",
                })

        if not results:
            raise ValueError(
                "No MJD/CTR/STAT_ERR table found"
            )

    except Exception as e:
        results.append({
            "date": date,
            "instrument": "HEL1OS",
            "detector": detector,
            "file": str(lc.relative_to(ROOT)),
            "gti_file": str(gt.relative_to(ROOT)),
            "status": "ERROR",
            "error": repr(e),
        })

    return results


def validate_slx(lc, gt, date):
    try:
        with fits.open(gt, memmap=False) as h:
            g = next(
                x for x in h
                if x.data is not None
                and getattr(x.data, "names", None)
            )

            gti_start = col(g.data, "START")
            gti_stop = col(g.data, "STOP")

        with fits.open(lc, memmap=False) as h:
            x = next(
                x for x in h
                if x.data is not None
                and getattr(x.data, "names", None)
                and {"time", "counts"}.issubset(
                    {n.lower() for n in x.data.names}
                )
            )

            t = col(x.data, "TIME").astype(float)
            counts = col(x.data, "COUNTS").astype(float)

        if len(t) != len(counts):
            raise ValueError(
                "Timestamp and signal lengths differ"
            )

        # SoLEXS TIME uses MJDREF=40587 (Unix epoch).
        # GTI START/STOP are Unix seconds.
        ts = time_stats(t, 1)

        gx, inside = gti_stats(
            t,
            gti_start,
            gti_stop,
            1,
            inclusive_endpoints=True
        )

        signal = signal_stats(counts, inside)

        cadence_ok = cadence_check(
            ts["median_cadence_sec"]
        )

        status = determine_status(
            ts, gx, signal, cadence_ok
        )

        return [{
            "date": date,
            "instrument": "SoLEXS",
            "detector": "SDD2",
            "band": "RATE",
            "file": str(lc.relative_to(ROOT)),
            "gti_file": str(gt.relative_to(ROOT)),

            **ts,
            **gx,
            **signal,

            "cadence_near_1sec": bool(cadence_ok),
            "status": status,
            "error": "",
        }]

    except Exception as e:
        return [{
            "date": date,
            "instrument": "SoLEXS",
            "detector": "SDD2",
            "file": str(lc.relative_to(ROOT)),
            "gti_file": str(gt.relative_to(ROOT)),
            "status": "ERROR",
            "error": repr(e),
        }]


def main():
    details = []
    pairs = []

    # HEL1OS
    for lc in HEL.rglob("lightcurve_*.fits"):
        date = date_of(lc, HEL)

        if DATE_FILTER and date != DATE_FILTER:
            continue

        match = re.search(
            r"lightcurve_(cdte[12]|czt[12])\.fits$",
            lc.name.lower()
        )

        if not match:
            continue

        detector = match.group(1)

        gt = (
            lc.parent.parent
            / "aux"
            / f"gti{detector}.fits"
        )

        pairs.append({
            "date": date,
            "instrument": "HEL1OS",
            "detector": detector,
            "lightcurve": str(lc.relative_to(ROOT)),
            "gti": (
                str(gt.relative_to(ROOT))
                if gt.exists()
                else ""
            ),
            "pair_status": (
                "PAIRED" if gt.exists() else "MISSING_GTI"
            ),
        })

        if gt.exists():
            details.extend(validate_hel(lc, gt, date))

    # SoLEXS
    for lc in SLX.rglob("*.lc.gz"):
        date = date_of(lc, SLX)

        if DATE_FILTER and date != DATE_FILTER:
            continue

        gt = lc.with_name(
            lc.name.replace(".lc.gz", ".gti.gz")
        )

        pairs.append({
            "date": date,
            "instrument": "SoLEXS",
            "detector": "SDD2",
            "lightcurve": str(lc.relative_to(ROOT)),
            "gti": (
                str(gt.relative_to(ROOT))
                if gt.exists()
                else ""
            ),
            "pair_status": (
                "PAIRED" if gt.exists() else "MISSING_GTI"
            ),
        })

        if gt.exists():
            details.extend(validate_slx(lc, gt, date))

    # Daily summary
    by_date = defaultdict(list)

    for row in details:
        by_date[row.get("date", "")].append(row)

    daily = []

    for date, rows in sorted(by_date.items()):
        hel = [
            r for r in rows
            if r.get("instrument") == "HEL1OS"
        ]

        slx = [
            r for r in rows
            if r.get("instrument") == "SoLEXS"
        ]

        daily.append({
            "date": date,

            "hel1os_detectors": ";".join(
                sorted({
                    r["detector"]
                    for r in hel
                    if r.get("detector")
                })
            ),

            "hel1os_band_tables": len(hel),
            "solexs_rate_tables": len(slx),

            "pass_rows": sum(
                r.get("status") == "PASS"
                for r in rows
            ),

            "review_rows": sum(
                r.get("status") == "REVIEW"
                for r in rows
            ),

            "error_rows": sum(
                r.get("status") == "ERROR"
                for r in rows
            ),

            "note": (
                "Scientific quality checks; "
                "not a final model-readiness decision."
            ),
        })

    # Save reports
    OUT.mkdir(parents=True, exist_ok=True)

    save_csv(
        OUT / "lightcurve_quality_detail.csv",
        details
    )

    save_csv(
        OUT / "lightcurve_gti_pairing.csv",
        pairs
    )

    save_csv(
        OUT / "daily_scientific_quality.csv",
        daily
    )

    print("Date filter:", DATE_FILTER or "FULL DATASET")

    print(
        "Detail rows:", len(details),
        "| PASS:", sum(
            r.get("status") == "PASS" for r in details
        ),
        "| REVIEW:", sum(
            r.get("status") == "REVIEW" for r in details
        ),
        "| ERROR:", sum(
            r.get("status") == "ERROR" for r in details
        )
    )

    print(
        "Pair records:", len(pairs),
        "| Daily records:", len(daily)
    )

    print("Reports:", OUT)


if __name__ == "__main__":
    main()
