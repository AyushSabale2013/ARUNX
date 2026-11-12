"""
ADITYA-L1 HEL1OS + SoLEXS multi-day merging pipeline (minute-end v2).

The signal aggregation and GTI rules are preserved from the validated
optimized pipeline. The important timestamp change is:

    Input samples in [10:00:00, 10:01:00)
    are assigned to output timestamp 10:01:00.

The original checkpoints and outputs are NOT overwritten. This version writes
to a separate checkpoint directory and versioned output files.

Usage:
    python scripts\\pipeline\\merge_lightcurves.py --dates 20260101 --workers 1
    python scripts\\pipeline\\merge_lightcurves.py --workers 4
    python scripts\\pipeline\\merge_lightcurves.py --limit 3
    python scripts\\pipeline\\merge_lightcurves.py --force --workers 4

NOTE:
    --force reprocesses dates in the NEW v2 checkpoint directory only.
"""

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[2]

HEL1OS_DIR = BASE / "raw_hel1os"
SOLEXS_DIR = BASE / "raw_solexs"

OUTPUT_DIR = BASE / "output" / "merged"

# Separate v2 paths: never reuse old left-labelled checkpoints.
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_minute_end_v2"

OUTPUT_PATH = OUTPUT_DIR / "merged_timeseries_minute_end_v2.parquet"
SUMMARY_PATH = OUTPUT_DIR / "merge_summary_minute_end_v2.csv"
QUALITY_PATH = OUTPUT_DIR / "merge_quality_report_minute_end_v2.csv"

DETECTORS = ["cdte1", "cdte2", "czt1", "czt2"]
EXPECTED_BANDS = 5
EXPECTED_CHANNELS = len(DETECTORS) * EXPECTED_BANDS

MINUTES_PER_DAY = 1440
US_PER_MINUTE = 60_000_000
MJD_UNIX_OFFSET = 40587.0
SECONDS_PER_DAY = 86400.0

HEL1OS_GTI_TOL = 1e-12  # MJD units, as in the original pipeline


# ============================================================
# COMMON HELPERS
# ============================================================

def extract_date(path):
    """Extract YYYYMMDD from a file or directory path."""
    match = re.search(r"20\d{6}", str(path))
    return match.group(0) if match else None


def get_column(data, name):
    """Read a FITS column (case-insensitive) as native float64."""
    for column in data.columns.names:
        if column.lower() == name.lower():
            return np.asarray(data[column], dtype=np.float64)
    raise KeyError(f"FITS column not found: {name}")


def find_gti_hdu(hdul, start_name, stop_name):
    """Find a GTI table using case-insensitive column names."""
    required = {start_name.lower(), stop_name.lower()}

    for hdu in hdul:
        if not isinstance(hdu, fits.BinTableHDU):
            continue

        names = {column.lower() for column in hdu.columns.names}
        if required.issubset(names):
            return hdu

    raise ValueError(
        f"GTI table not found: {start_name}, {stop_name}"
    )


def checkpoint_path(date):
    return CHECKPOINT_DIR / f"{date}.parquet"


def checkpoint_is_valid(path):
    """
    Validate both checkpoint length and its exact minute-end timestamp grid.

    A checkpoint named YYYYMMDD contains 1440 bins:
        YYYY-MM-DD 00:00–00:01 -> timestamp 00:01
        ...
        YYYY-MM-DD 23:59–24:00 -> timestamp next day 00:00
    """
    if not path.exists():
        return False

    try:
        frame = pd.read_parquet(path, columns=["timestamp"])

        if len(frame) != MINUTES_PER_DAY:
            return False

        actual = pd.to_datetime(frame["timestamp"], utc=True)
        if actual.isna().any() or actual.duplicated().any():
            return False

        date = path.stem
        day_start = pd.Timestamp(
            year=int(date[0:4]),
            month=int(date[4:6]),
            day=int(date[6:8]),
            tz="UTC",
        )

        expected = pd.date_range(
            start=day_start + pd.Timedelta(minutes=1),
            periods=MINUTES_PER_DAY,
            freq="1min",
        )

        actual_index = pd.DatetimeIndex(actual)
        return actual_index.equals(expected)

    except Exception:
        return False


# ============================================================
# GTI MASKING
# ============================================================

def merge_intervals(starts, stops):
    """Merge overlapping/touching intervals into sorted disjoint arrays."""
    if len(starts) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    order = np.argsort(starts, kind="stable")
    s = starts[order]
    e = stops[order]

    run_max = np.maximum.accumulate(e)

    new_group = np.empty(s.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = s[1:] > run_max[:-1]

    first = np.flatnonzero(new_group)
    last = np.concatenate([first[1:] - 1, [s.size - 1]])

    return s[first], run_max[last]


def gti_mask(times, starts, stops, tol=0.0):
    """True where start - tol <= time <= stop + tol for any GTI."""
    mask = np.zeros(times.shape, dtype=bool)

    if times.size == 0 or starts.size == 0:
        return mask

    gs, ge = merge_intervals(starts, stops)

    idx = np.searchsorted(gs - tol, times, side="right") - 1
    ok = idx >= 0
    mask[ok] = times[ok] <= ge[idx[ok]] + tol

    return mask


# ============================================================
# HEL1OS READERS
# ============================================================

def read_hel1os_gti(gti_path):
    with fits.open(gti_path) as hdul:
        hdu = find_gti_hdu(hdul, "tstart", "tstop")
        starts = get_column(hdu.data, "tstart")
        stops = get_column(hdu.data, "tstop")

    # Preserve original HEL1OS rule: zero-duration intervals are invalid.
    valid = (
        np.isfinite(starts)
        & np.isfinite(stops)
        & (stops > starts)
    )

    if not valid.all() or not valid.any():
        raise ValueError(f"Invalid HEL1OS GTI: {gti_path}")

    return starts[valid], stops[valid]


def mjd_to_unix_us(mjd):
    """MJD (UTC) -> integer Unix microseconds."""
    seconds = (mjd - MJD_UNIX_OFFSET) * SECONDS_PER_DAY
    return np.rint(seconds * 1e6).astype(np.int64)


def _is_band_hdu(hdu):
    if not isinstance(hdu, fits.BinTableHDU):
        return False

    names = {column.lower() for column in hdu.columns.names}
    return {"mjd", "ctr", "stat_err"}.issubset(names)


def read_hel1os_segment(lc_path):
    """
    Return {channel: (t_us, counts, stat_err, gti_valid)} for one segment.

    Samples outside GTIs are retained in the time series but their counts
    and errors are masked to NaN.
    """
    detector = lc_path.stem.replace("lightcurve_", "").lower()

    if detector not in DETECTORS:
        raise ValueError(f"Unknown HEL1OS detector: {detector}")

    gti_path = lc_path.parent.parent / "aux" / f"gti{detector}.fits"

    if not gti_path.exists():
        raise FileNotFoundError(gti_path)

    g_starts, g_stops = read_hel1os_gti(gti_path)

    channels = {}

    with fits.open(lc_path, memmap=True) as hdul:
        band_hdus = [hdu for hdu in hdul[1:] if _is_band_hdu(hdu)]

        if len(band_hdus) != EXPECTED_BANDS:
            raise ValueError(
                f"{lc_path}: expected {EXPECTED_BANDS} bands, "
                f"found {len(band_hdus)}"
            )

        for band_number, hdu in enumerate(band_hdus, start=1):
            mjd = get_column(hdu.data, "MJD")
            counts = get_column(hdu.data, "CTR")
            errors = get_column(hdu.data, "STAT_ERR")

            if not (len(mjd) == len(counts) == len(errors)):
                raise ValueError(f"Column length mismatch: {lc_path}")

            finite = np.isfinite(mjd)
            mjd = mjd[finite]
            counts = counts[finite]
            errors = errors[finite]

            gti_valid = gti_mask(
                mjd, g_starts, g_stops, tol=HEL1OS_GTI_TOL
            )

            counts = np.where(gti_valid, counts, np.nan)
            errors = np.where(gti_valid, errors, np.nan)

            channel = f"hel1os_{detector}_band{band_number}"

            channels[channel] = (
                mjd_to_unix_us(mjd),
                counts,
                errors,
                gti_valid,
            )

    return channels


# ============================================================
# SOLEXS READERS
# ============================================================

def read_solexs_gti(gti_path):
    with fits.open(gti_path) as hdul:
        hdu = find_gti_hdu(hdul, "start", "stop")
        starts = get_column(hdu.data, "START")
        stops = get_column(hdu.data, "STOP")

    # Inclusive endpoints; zero-duration GTIs are valid one-second intervals.
    valid = (
        np.isfinite(starts)
        & np.isfinite(stops)
        & (stops >= starts)
    )

    if not valid.all() or not valid.any():
        raise ValueError(f"Invalid SoLEXS GTI: {gti_path}")

    return starts[valid], stops[valid]


def read_solexs_day(lc_path):
    """Return (t_us, counts, gti_valid) for one SoLEXS day."""
    gti_path = lc_path.with_name(
        lc_path.name.replace(".lc.gz", ".gti.gz")
    )

    if not gti_path.exists():
        raise FileNotFoundError(gti_path)

    g_starts, g_stops = read_solexs_gti(gti_path)

    with fits.open(lc_path, memmap=False) as hdul:
        hdu = next(
            (
                item for item in hdul
                if isinstance(item, fits.BinTableHDU)
                and {"time", "counts"}.issubset(
                    {column.lower() for column in item.columns.names}
                )
            ),
            None,
        )

        if hdu is None:
            raise ValueError(f"RATE table not found in {lc_path}")

        times = get_column(hdu.data, "TIME")
        counts = get_column(hdu.data, "COUNTS")

    if len(times) != len(counts):
        raise ValueError(f"SoLEXS column length mismatch: {lc_path}")

    finite = np.isfinite(times)
    times = times[finite]
    counts = counts[finite]

    gti_valid = gti_mask(times, g_starts, g_stops, tol=0.0)
    counts = np.where(gti_valid, counts, np.nan)

    # SoLEXS TIME is seconds since MJDREF=40587 (Unix epoch).
    t_us = np.rint(times * 1e6).astype(np.int64)

    return t_us, counts, gti_valid


# ============================================================
# 1-MINUTE BINNING (VECTORIZED)
# ============================================================

def _sort_and_check(label, t_us, *arrays):
    """Sort by time and refuse duplicate timestamps."""
    order = np.argsort(t_us, kind="stable")
    t_us = t_us[order]
    arrays = [array[order] for array in arrays]

    if t_us.size > 1:
        n_dup = int(np.count_nonzero(t_us[1:] == t_us[:-1]))
        if n_dup:
            raise ValueError(f"{label}: {n_dup} duplicate timestamps.")

    return (t_us, *arrays)


def _minute_index(t_us, day_start_us):
    """
    Assign samples to bins using the original day-start boundary.

    This returns bin indices 0..1439. The output labels are shifted to
    right-edge timestamps separately; the sample binning itself is unchanged.
    """
    minute = (t_us - day_start_us) // US_PER_MINUTE
    inside = (minute >= 0) & (minute < MINUTES_PER_DAY)
    return minute[inside].astype(np.intp), inside


def _mean_by_minute(minute, values):
    """Mean of non-NaN values per minute; NaN where no valid value exists."""
    valid = ~np.isnan(values)
    m = minute[valid]

    n = np.bincount(m, minlength=MINUTES_PER_DAY)
    total = np.bincount(
        m, weights=values[valid], minlength=MINUTES_PER_DAY
    )

    out = np.full(MINUTES_PER_DAY, np.nan)
    nonzero = n > 0
    out[nonzero] = total[nonzero] / n[nonzero]

    return out


def _error_by_minute(minute, errors):
    """sqrt(sum(err^2)) / n over non-NaN errors per minute."""
    valid = ~np.isnan(errors)
    m = minute[valid]
    e = errors[valid]

    n = np.bincount(m, minlength=MINUTES_PER_DAY)
    sum_squares = np.bincount(
        m, weights=e * e, minlength=MINUTES_PER_DAY
    )

    out = np.full(MINUTES_PER_DAY, np.nan)
    nonzero = n > 0
    out[nonzero] = np.sqrt(sum_squares[nonzero]) / n[nonzero]

    return out


def bin_hel1os_channel(channel, parts, day_start_us):
    """Combine segments of one HEL1OS channel and bin to the 1-minute grid."""
    t_us = np.concatenate([part[0] for part in parts])
    counts = np.concatenate([part[1] for part in parts])
    errors = np.concatenate([part[2] for part in parts])
    gti = np.concatenate([part[3] for part in parts])

    t_us, counts, errors, gti = _sort_and_check(
        channel, t_us, counts, errors, gti
    )

    minute, inside = _minute_index(t_us, day_start_us)
    counts = counts[inside]
    errors = errors[inside]
    gti = gti[inside]

    gti_any = np.bincount(
        minute,
        weights=gti.astype(np.float64),
        minlength=MINUTES_PER_DAY,
    ) > 0

    signal = _mean_by_minute(minute, counts)
    error = _error_by_minute(minute, errors)

    signal[~gti_any] = np.nan
    error[~gti_any] = np.nan

    error_column = f"{channel}_stat_err"

    return {
        channel: signal,
        f"{channel}_gti": gti_any,
        error_column: error,
        f"{channel}_missing": np.isnan(signal),
        f"{error_column}_missing": np.isnan(error),
    }


def bin_solexs(t_us, counts, gti, day_start_us):
    channel = "solexs_sdd2_counts"

    t_us, counts, gti = _sort_and_check(
        channel, t_us, counts, gti
    )

    minute, inside = _minute_index(t_us, day_start_us)
    counts = counts[inside]
    gti = gti[inside]

    gti_any = np.bincount(
        minute,
        weights=gti.astype(np.float64),
        minlength=MINUTES_PER_DAY,
    ) > 0

    signal = _mean_by_minute(minute, counts)
    signal[~gti_any] = np.nan

    return {
        channel: signal,
        "solexs_sdd2_gti": gti_any,
        "solexs_sdd2_missing": np.isnan(signal),
    }


# ============================================================
# MERGE ONE DATE
# ============================================================

def merge_one_date(date, hel1os_files, solexs_path):
    day_start = pd.Timestamp(
        year=int(date[0:4]),
        month=int(date[4:6]),
        day=int(date[6:8]),
        tz="UTC",
    )
    day_start_us = int(day_start.value // 1000)

    # Samples remain binned relative to day_start_us.
    # Only the labels move to the RIGHT edge of each completed minute.
    grid = pd.date_range(
        start=day_start + pd.Timedelta(minutes=1),
        periods=MINUTES_PER_DAY,
        freq="1min",
    )

    channel_parts = {}
    detectors_found = set()

    for lc_path in hel1os_files:
        detector = lc_path.stem.replace("lightcurve_", "").lower()
        detectors_found.add(detector)

        for channel, part in read_hel1os_segment(lc_path).items():
            channel_parts.setdefault(channel, []).append(part)

    if detectors_found != set(DETECTORS):
        raise ValueError(
            f"{date}: expected detectors {DETECTORS}, "
            f"found {sorted(detectors_found)}"
        )

    if len(channel_parts) != EXPECTED_CHANNELS:
        raise ValueError(
            f"{date}: expected {EXPECTED_CHANNELS} HEL1OS channels, "
            f"found {len(channel_parts)}"
        )

    columns = {
        "date": np.full(
            MINUTES_PER_DAY,
            day_start.strftime("%Y-%m-%d"),
            dtype=object,
        ),
        "timestamp": grid,
    }

    for channel in sorted(channel_parts):
        columns.update(
            bin_hel1os_channel(
                channel, channel_parts[channel], day_start_us
            )
        )

    t_us, counts, gti = read_solexs_day(solexs_path)
    columns.update(bin_solexs(t_us, counts, gti, day_start_us))

    merged = pd.DataFrame(columns)

    if len(merged) != MINUTES_PER_DAY:
        raise ValueError(
            f"{date}: expected 1440 rows, found {len(merged)}"
        )

    if not merged["timestamp"].is_unique:
        raise ValueError(f"{date}: duplicate timestamps after merging")

    expected = pd.date_range(
        start=day_start + pd.Timedelta(minutes=1),
        periods=MINUTES_PER_DAY,
        freq="1min",
    )

    if not pd.DatetimeIndex(merged["timestamp"]).equals(expected):
        raise ValueError(f"{date}: incorrect minute-end timestamp grid")

    return merged


# ============================================================
# WORKER (ONE DATE -> ONE CHECKPOINT)
# ============================================================

def process_date(date, hel1os_files, solexs_path):
    """Merge one date and write its v2 checkpoint atomically."""
    started = time.perf_counter()

    try:
        merged = merge_one_date(date, hel1os_files, solexs_path)

        final_path = checkpoint_path(date)
        temp_path = final_path.with_name(final_path.name + ".tmp")

        merged.to_parquet(temp_path, index=False)
        os.replace(temp_path, final_path)

        return date, "MERGED", None, time.perf_counter() - started

    except Exception as exc:
        return date, "FAILED", str(exc), time.perf_counter() - started


# ============================================================
# QUALITY REPORT
# ============================================================

def quality_rows(date, merged):
    rows = []

    for column in merged.columns:
        if not column.endswith("_gti"):
            continue

        channel = column[:-4]
        signal_column = channel if channel in merged.columns else None
        missing_column = f"{channel}_missing"

        rows.append({
            "date": date,
            "channel": channel,
            "gti_valid_minutes": int(
                merged[column].fillna(False).astype(bool).sum()
            ),
            "valid_signal_minutes": (
                int(merged[signal_column].notna().sum())
                if signal_column else 0
            ),
            "missing_minutes": (
                int(
                    merged[missing_column]
                    .fillna(True)
                    .astype(bool)
                    .sum()
                )
                if missing_column in merged.columns
                else None
            ),
        })

    return rows


# ============================================================
# DISCOVERY
# ============================================================

def discover():
    print("\nDiscovering HEL1OS lightcurves...")
    hel1os_files = sorted(
        HEL1OS_DIR.rglob("lightcurve_*.fits")
    )

    # Select the highest processing version for each observation/detector.
    selected = {}

    for path in hel1os_files:
        date = extract_date(path)
        if not date:
            continue

        observation_name = path.parent.parent.name

        if "_V" in observation_name:
            base_name, version_text = observation_name.rsplit("_V", 1)
            version = (
                int(version_text)
                if version_text.isdigit()
                else -1
            )
        else:
            base_name = observation_name
            version = -1

        key = (date, base_name, path.name)

        if key not in selected or version > selected[key][0]:
            selected[key] = (version, path)

    hel1os_by_date = {}

    for (date, _, _), (_, path) in selected.items():
        hel1os_by_date.setdefault(date, []).append(path)

    for date in hel1os_by_date:
        hel1os_by_date[date].sort()

    print(f"HEL1OS original files: {len(hel1os_files)}")
    print(f"HEL1OS selected files: {len(selected)}")
    print(f"HEL1OS dates: {len(hel1os_by_date)}")

    print("\nDiscovering SoLEXS lightcurves...")
    solexs_files = sorted(SOLEXS_DIR.rglob("*.lc.gz"))

    solexs_by_date = {}

    for path in solexs_files:
        date = extract_date(path)
        if not date:
            continue

        if date in solexs_by_date:
            raise ValueError(
                f"Multiple SoLEXS lightcurves found for {date}"
            )

        solexs_by_date[date] = path

    print(f"SoLEXS lightcurve files: {len(solexs_files)}")
    print(f"SoLEXS dates: {len(solexs_by_date)}")

    return hel1os_by_date, solexs_by_date


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="parallel processes (default: min(8, cpus-1))",
    )

    parser.add_argument(
        "--dates",
        nargs="+",
        metavar="YYYYMMDD",
        help="process only these dates (test output gets a _test suffix)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N eligible dates (test mode)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore existing v2 checkpoints and reprocess selected dates",
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    print("=" * 60)
    print("ADITYA-L1 MULTI-DAY MERGING PIPELINE (MINUTE-END V2)")
    print("=" * 60)

    if not HEL1OS_DIR.exists():
        raise FileNotFoundError(HEL1OS_DIR)

    if not SOLEXS_DIR.exists():
        raise FileNotFoundError(SOLEXS_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    hel1os_by_date, solexs_by_date = discover()

    eligible_dates = sorted(
        set(hel1os_by_date) & set(solexs_by_date)
    )

    print(
        f"\nDates available in both instruments: {len(eligible_dates)}"
    )

    if not eligible_dates:
        raise RuntimeError(
            "No common HEL1OS and SoLEXS dates found."
        )

    test_mode = bool(args.dates) or args.limit is not None

    out_path = OUTPUT_PATH
    summary_path = SUMMARY_PATH
    quality_path = QUALITY_PATH

    if args.dates:
        requested_dates = sorted(set(args.dates))
        invalid_format = [
            date for date in requested_dates
            if not re.fullmatch(r"20\d{6}", date)
        ]

        if invalid_format:
            raise ValueError(
                f"Invalid date format (expected YYYYMMDD): {invalid_format}"
            )

        missing = [
            date for date in requested_dates
            if date not in eligible_dates
        ]

        if missing:
            raise ValueError(
                f"Dates not available in both instruments: {missing}"
            )

        eligible_dates = requested_dates

    if args.limit is not None:
        eligible_dates = eligible_dates[:args.limit]

    if test_mode:
        out_path = OUTPUT_PATH.with_name(
            "merged_timeseries_minute_end_v2_test.parquet"
        )
        summary_path = SUMMARY_PATH.with_name(
            "merge_summary_minute_end_v2_test.csv"
        )
        quality_path = QUALITY_PATH.with_name(
            "merge_quality_report_minute_end_v2_test.csv"
        )

        print(
            f"\nTEST MODE: {len(eligible_dates)} date(s): "
            f"{eligible_dates}"
        )

    # --------------------------------------------------------
    # Resume from v2 checkpoints only
    # --------------------------------------------------------

    if args.force:
        todo = list(eligible_dates)
    else:
        todo = [
            date for date in eligible_dates
            if not checkpoint_is_valid(checkpoint_path(date))
        ]

    skipped = len(eligible_dates) - len(todo)

    print(f"V2 checkpoints skipped: {skipped}")
    print(f"To process: {len(todo)} (workers: {args.workers})")

    # --------------------------------------------------------
    # Process remaining dates
    # --------------------------------------------------------

    results = {}
    started_all = time.perf_counter()

    def report(done, date, status, error, elapsed):
        total_elapsed = time.perf_counter() - started_all
        remaining = len(todo) - done
        eta = (
            total_elapsed / done * remaining
            if done > 0 else 0.0
        )

        line = (
            f"[{done}/{len(todo)}] {date} {status} "
            f"({elapsed:.1f}s, ETA {eta / 60:.1f} min)"
        )

        if error:
            line += f"\n    Reason: {error}"

        print(line, flush=True)

    if todo:
        if args.workers == 1:
            for done, date in enumerate(todo, start=1):
                result_date, status, error, elapsed = process_date(
                    date,
                    hel1os_by_date[date],
                    solexs_by_date[date],
                )

                results[result_date] = (status, error)
                report(done, result_date, status, error, elapsed)

        else:
            with ProcessPoolExecutor(
                max_workers=args.workers
            ) as pool:
                futures = [
                    pool.submit(
                        process_date,
                        date,
                        hel1os_by_date[date],
                        solexs_by_date[date],
                    )
                    for date in todo
                ]

                for done, future in enumerate(
                    as_completed(futures), start=1
                ):
                    result_date, status, error, elapsed = future.result()
                    results[result_date] = (status, error)
                    report(
                        done,
                        result_date,
                        status,
                        error,
                        elapsed,
                    )

    # --------------------------------------------------------
    # Assemble outputs from v2 checkpoints
    # --------------------------------------------------------

    summaries = []
    quality_reports = []
    frames = []

    for date in eligible_dates:
        base_row = {
            "date": date,
            "hel1os_segments": len(hel1os_by_date[date]),
            "solexs_file": solexs_by_date[date].name,
        }

        path = checkpoint_path(date)
        worker_failed = (
            results.get(date, (None, None))[0] == "FAILED"
        )

        if not worker_failed and checkpoint_is_valid(path):
            merged = pd.read_parquet(path)

            # Validate the complete checkpoint grid before including it.
            timestamp = pd.to_datetime(
                merged["timestamp"], utc=True
            )
            if timestamp.isna().any() or timestamp.duplicated().any():
                raise RuntimeError(
                    f"{date}: invalid timestamps in checkpoint {path}"
                )

            frames.append(merged)
            quality_reports.extend(quality_rows(date, merged))

            summaries.append({
                **base_row,
                "rows": len(merged),
                "columns": len(merged.columns),
                "status": "MERGED",
                "error": None,
            })

        else:
            summaries.append({
                **base_row,
                "rows": 0,
                "columns": 0,
                "status": "FAILED",
                "error": results.get(
                    date, (None, "missing or invalid checkpoint")
                )[1],
            })

    pd.DataFrame(summaries).to_csv(
        summary_path, index=False
    )

    pd.DataFrame(quality_reports).to_csv(
        quality_path, index=False
    )

    if not frames:
        raise RuntimeError(
            "No dates were successfully merged."
        )

    final_dataset = pd.concat(
        frames, ignore_index=True
    )

    final_dataset["timestamp"] = pd.to_datetime(
        final_dataset["timestamp"], utc=True
    )

    final_dataset = final_dataset.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if final_dataset["timestamp"].duplicated().any():
        duplicated = final_dataset.loc[
            final_dataset["timestamp"].duplicated(keep=False),
            "timestamp",
        ].head(10)

        raise RuntimeError(
            "Duplicate timestamps found across daily checkpoints:\n"
            + duplicated.to_string(index=False)
        )

    # Atomic master output write.
    temp_output = out_path.with_name(out_path.name + ".tmp")
    final_dataset.to_parquet(temp_output, index=False)
    os.replace(temp_output, out_path)

    success_count = sum(
        row["status"] == "MERGED" for row in summaries
    )
    failed_count = sum(
        row["status"] == "FAILED" for row in summaries
    )

    print("\n" + "=" * 60)
    print("MERGING PIPELINE FINISHED")
    print("=" * 60)
    print(f"Eligible dates: {len(eligible_dates)}")
    print(f"Successfully merged: {success_count}")
    print(f"Failed dates: {failed_count}")
    print(f"Total rows: {len(final_dataset):,}")
    print(f"Total columns: {len(final_dataset.columns)}")
    print(
        "Wall time this run: "
        f"{(time.perf_counter() - started_all) / 60:.1f} min"
    )

    print("\nOutput files:")
    print(out_path)
    print(summary_path)
    print(quality_path)
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print("=" * 60)

    if failed_count:
        raise RuntimeError(
            f"{failed_count} date(s) failed. "
            "Review the summary CSV before continuing."
        )


if __name__ == "__main__":
    main()
