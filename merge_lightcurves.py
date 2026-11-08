
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time


# ==================================================
# CONFIGURATION
# ==================================================

BASE = Path(__file__).resolve().parent

HEL1OS_DIR = BASE / "raw_helios"
SOLEXS_DIR = BASE / "raw_solexs"
OUTPUT_DIR = BASE / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

DATE = "2026-10-06"

START = pd.Timestamp(f"{DATE} 00:00:00", tz="UTC")
END = pd.Timestamp(f"{DATE} 23:59:00", tz="UTC")

GRID = pd.date_range(
    start=START,
    end=END,
    freq="1min"
)

OUTPUT_PATH = OUTPUT_DIR / f"merged_{DATE.replace('-', '')}.parquet"


# ==================================================
# HEL1OS GTI READER
# ==================================================

def read_hel1os_gti(gti_path):

    intervals = []

    with fits.open(gti_path) as hdul:

        for start, stop in zip(
            hdul[1].data["tstart"],
            hdul[1].data["tstop"]
        ):

            start_time = Time(
                start,
                format="mjd"
            ).to_datetime(timezone=timezone.utc)

            stop_time = Time(
                stop,
                format="mjd"
            ).to_datetime(timezone=timezone.utc)

            intervals.append((
                pd.Timestamp(start_time),
                pd.Timestamp(stop_time)
            ))

    return intervals


# ==================================================
# HEL1OS LIGHT CURVE READER
# ==================================================

def read_hel1os():

    channel_frames = {}

    lightcurve_files = sorted(
        path
        for path in HEL1OS_DIR.rglob("lightcurve_*.fits")
        if DATE.replace("-", "/") in str(path)
    )

    if not lightcurve_files:
        raise RuntimeError("No HEL1OS lightcurve files found.")

    print("HEL1OS files found:", len(lightcurve_files))

    for lc_path in lightcurve_files:

        detector = lc_path.stem.replace("lightcurve_", "")

        gti_path = (
            lc_path.parent.parent
            / "aux"
            / f"gti{detector}.fits"
        )

        if not gti_path.exists():
            raise FileNotFoundError(
                f"Missing GTI file: {gti_path}"
            )

        gti_intervals = read_hel1os_gti(gti_path)

        with fits.open(lc_path) as hdul:

            for hdu in hdul[1:]:

                if hdu.data is None:
                    continue

                if not {"MJD", "CTR"}.issubset(
                    set(hdu.columns.names)
                ):
                    continue

                data = hdu.data

                times = pd.to_datetime(
                    Time(
                        np.asarray(data["MJD"], dtype=float),
                        format="mjd"
                    ).to_datetime(timezone=timezone.utc),
                    utc=True
                )

                signal = np.asarray(
                    data["CTR"],
                    dtype=float
                )

                finite = np.isfinite(signal)

                # Check whether each sample belongs to a valid GTI.
                gti_valid = np.zeros(
                    len(times),
                    dtype=bool
                )

                for start, stop in gti_intervals:

                    gti_valid |= (
                        (times >= start)
                        & (times <= stop)
                    )

                channel = hdu.name.lower()

                column = (
                    f"hel1os_{detector}_{channel}"
                )

                frame = pd.DataFrame({

                    "timestamp": times,

                    column: np.where(
                        finite & gti_valid,
                        signal,
                        np.nan
                    ),

                    f"{column}_missing": ~finite,

                    f"{column}_gti": gti_valid

                })

                channel_frames.setdefault(
                    column,
                    []
                ).append(frame)

    if not channel_frames:
        raise RuntimeError(
            "No valid HEL1OS channels found."
        )

    aligned_frames = []

    for column, parts in channel_frames.items():

        # Preserve all original samples from both files.
        channel_frame = pd.concat(
            parts,
            ignore_index=True
        )

        channel_frame = channel_frame.sort_values(
            "timestamp"
        )

        # Exact duplicate timestamps are not expected.
        if channel_frame["timestamp"].duplicated().any():

            duplicates = channel_frame[
                channel_frame["timestamp"].duplicated(
                    keep=False
                )
            ]

            raise RuntimeError(
                f"Exact duplicate timestamps in {column}:\n"
                f"{duplicates.head()}"
            )

        channel_frame = channel_frame.set_index(
            "timestamp"
        )

        aligned_frames.append(channel_frame)

    result = pd.concat(
        aligned_frames,
        axis=1
    ).reset_index()

    return result


# ==================================================
# SOLEXS LIGHT CURVE READER
# ==================================================

def read_solexs():

    lc_files = list(
        SOLEXS_DIR.rglob("*.lc.gz")
    )

    gti_files = list(
        SOLEXS_DIR.rglob("SDD2/*.gti.gz")
    )

    if len(lc_files) != 1 or len(gti_files) != 1:

        raise RuntimeError(
            f"Expected one SDD2 light curve and GTI. "
            f"Found {len(lc_files)} light curves and "
            f"{len(gti_files)} GTI files."
        )

    with fits.open(lc_files[0]) as hdul:

        data = hdul["RATE"].data

        times = np.asarray(
            data["TIME"],
            dtype=float
        )

        signal = np.asarray(
            data["COUNTS"],
            dtype=float
        )

    # SoLEXS TIME is Unix seconds.
    timestamps = pd.to_datetime(
        times,
        unit="s",
        origin="unix",
        utc=True
    )

    with fits.open(gti_files[0]) as hdul:

        gti = hdul["GTI"].data

        gti_valid = np.zeros(
            len(times),
            dtype=bool
        )

        for start, stop in zip(
            gti["START"],
            gti["STOP"]
        ):

            gti_valid |= (
                (times >= start)
                & (times <= stop)
            )

    finite = np.isfinite(signal)

    valid = finite & gti_valid

    return pd.DataFrame({

        "timestamp": timestamps,

        "solexs_sdd2_counts": np.where(
            valid,
            signal,
            np.nan
        ),

        "solexs_sdd2_missing": ~finite,

        "solexs_sdd2_counts_gti": gti_valid

    })


# ==================================================
# RESAMPLE TO ONE-MINUTE UTC GRID
# ==================================================

def resample_to_minutes(frame):

    frame = frame.sort_values(
        "timestamp"
    ).copy()

    numeric_columns = [
        col
        for col in frame.columns
        if col != "timestamp"
        and not col.endswith("_missing")
        and not col.endswith("_gti")
    ]

    gti_columns = [
        col
        for col in frame.columns
        if col.endswith("_gti")
    ]

    frame = frame.set_index(
        "timestamp"
    )

    # Mean of available valid measurements in each minute.
    signals = (
        frame[numeric_columns]
        .resample("1min")
        .mean()
    )

    # A minute is GTI-valid if at least one sample is valid.
    gti_flags = (
        frame[gti_columns]
        .resample("1min")
        .max()
        .fillna(0)
        .astype(bool)
    )

    # Missing means no finite valid signal sample in that minute.
    missing_flags = (
        signals
        .isna()
    )

    missing_flags.columns = [
        f"{col}_missing"
        for col in numeric_columns
    ]

    result = (
        signals
        .join(gti_flags)
        .join(missing_flags)
    )

    # Reindex to exactly 1440 UTC minutes.
    result = result.reindex(GRID)

    # Fill missing quality flags.
    for col in numeric_columns:

        missing_col = f"{col}_missing"
        gti_col = f"{col}_gti"

        if missing_col not in result.columns:
            result[missing_col] = True

        if gti_col not in result.columns:
            result[gti_col] = False

        result[missing_col] = (
            result[missing_col]
            .fillna(True)
            .astype(bool)
        )

        result[gti_col] = (
            result[gti_col]
            .fillna(False)
            .astype(bool)
        )

        # Do not retain measurements outside GTI.
        result.loc[
            ~result[gti_col],
            col
        ] = np.nan

        # A signal is missing if no valid measurement exists.
        result[missing_col] = (
            result[col].isna()
        )

    result.index.name = "timestamp"

    return result.reset_index()


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 55)
    print("ADITYA-L1 DATA MERGING PIPELINE")
    print("Date:", DATE)
    print("=" * 55)

    print("\nReading HEL1OS...")

    hel1os = read_hel1os()

    print(
        "HEL1OS shape:",
        hel1os.shape
    )

    print("\nReading SoLEXS...")

    solexs = read_solexs()

    print(
        "SoLEXS shape:",
        solexs.shape
    )

    print("\nCombining instruments...")

    combined = hel1os.merge(
        solexs,
        on="timestamp",
        how="outer"
    )

    print(
        "Combined shape:",
        combined.shape
    )

    print("\nResampling to one-minute UTC grid...")

    merged = resample_to_minutes(
        combined
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    print("\nRunning validation...")

    assert len(merged) == 1440, (
        f"Expected 1440 rows, found {len(merged)}"
    )

    assert merged["timestamp"].is_unique, (
        "Duplicate timestamps found."
    )

    assert len(merged.columns) == len(
        set(merged.columns)
    ), "Duplicate columns found."

    signal_columns = [
        col
        for col in merged.columns
        if not col.endswith("_missing")
        and not col.endswith("_gti")
        and col != "timestamp"
    ]

    assert len(signal_columns) == 21, (
        f"Expected 21 signal columns, found {len(signal_columns)}"
    )

    for col in signal_columns:

        missing_col = f"{col}_missing"
        gti_col = f"{col}_gti"

        assert missing_col in merged.columns
        assert gti_col in merged.columns

        assert not (
            merged[missing_col].eq(False)
            & merged[col].isna()
        ).any(), f"Missing flag inconsistency: {col}"

        assert not (
            (~merged[gti_col])
            & merged[col].notna()
        ).any(), f"Signal outside GTI: {col}"

    print("All validation checks passed.")

    # ==================================================
    # SAVE
    # ==================================================

    merged.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 55)
    print("MERGE COMPLETE")
    print("=" * 55)

    print("Output:", OUTPUT_PATH)
    print("Rows:", len(merged))
    print("Columns:", len(merged.columns))
    print("Signals:", len(signal_columns))

    print(
        "First timestamp:",
        merged["timestamp"].iloc[0]
    )

    print(
        "Last timestamp:",
        merged["timestamp"].iloc[-1]
    )

    print("\nMissing signal counts:")

    for col in merged.columns:

        if col.endswith("_missing"):

            print(
                col,
                ":",
                int(merged[col].sum())
            )

    print("\nGTI-valid minute counts:")

    for col in merged.columns:

        if col.endswith("_gti"):

            print(
                col,
                ":",
                int(merged[col].sum())
            )


if __name__ == "__main__":
    main()
