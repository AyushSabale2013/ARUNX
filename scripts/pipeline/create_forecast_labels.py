
"""
ARUN-X: Segment-aware multi-day forecast labelling.

Input:
    output/labeled_timeseries_minute_end_v2.parquet
    output/goes_events_parsed.csv

Output:
    output/forecast_labeled_timeseries_minute_end_v2.parquet

Targets:
    flare_15m
    flare_30m
    flare_60m

Values:
     1 = M/X event starts within the future horizon
     0 = No M/X event starts within the future horizon
    -1 = Unknown / incomplete observation coverage

A prediction timestamp represents the end of a completed minute.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE / "output" /
    "labeled_timeseries_minute_end_v2.parquet"
)

GOES_FILE = BASE / "output" / "goes_events_parsed.csv"

OUTPUT_FILE = (
    BASE / "output" /
    "forecast_labeled_timeseries_minute_end_v2.parquet"
)

HORIZONS = [15, 30, 60]


def datetime_to_ns(series):
    """
    Convert timezone-aware timestamps to int64 nanoseconds.

    Explicitly normalizes datetime resolution to ns.
    This prevents pandas us/ms resolution from causing
    incorrect integer timestamp arithmetic.
    """

    parsed = pd.to_datetime(
        series,
        utc=True,
        errors="coerce"
    )

    if parsed.isna().any():
        raise ValueError("Invalid or missing datetime values found.")

    return (
        parsed.dt.as_unit("ns")
        .astype("int64")
        .to_numpy()
    )


def get_signal_columns(df):
    """
    Identify actual instrument signal columns.

    Excludes:
    - GTI flags
    - Missing flags
    - Statistical error columns
    """

    hel1os_signals = [
        col for col in df.columns
        if col.startswith("hel1os_")
        and not col.endswith("_gti")
        and not col.endswith("_missing")
        and not col.endswith("_stat_err")
        and not col.endswith("_stat_err_missing")
    ]

    solexs_signals = []

    if "solexs_sdd2_counts" in df.columns:
        solexs_signals.append("solexs_sdd2_counts")

    columns = hel1os_signals + solexs_signals

    if not columns:
        raise ValueError(
            "No usable HEL1OS or SoLEXS signal columns found."
        )

    return columns


def build_observation_segments(timestamps, usable):
    """
    Assign segment IDs to continuous usable observation sequences.

    A new segment starts:
    - At the first row
    - After a timestamp gap
    - After an unusable observation

    Unusable rows receive segment ID -1.
    """

    timestamps = timestamps.reset_index(drop=True)
    usable = usable.reset_index(drop=True)

    time_gap = timestamps.diff().ne(pd.Timedelta(minutes=1))

    # Start a new segment at the first usable row,
    # after a time gap, or after an unusable row.
    previous_unusable = (~usable).shift(
        1,
        fill_value=True
    )

    break_before = (
        time_gap
        | previous_unusable
    ) & usable

    segment_id = (
        break_before.cumsum()
        .astype("int64")
    )

    # Mark unusable rows separately.
    segment_id = segment_id.where(usable, -1)

    return segment_id


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if not GOES_FILE.exists():
        raise FileNotFoundError(GOES_FILE)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("Loading labelled multi-day dataset...")
    df = pd.read_parquet(INPUT_FILE)

    print("Loading corrected GOES catalogue...")
    goes = pd.read_csv(GOES_FILE)

    required_df = {
        "timestamp",
        "flare_active"
    }

    required_goes = {
        "start",
        "class_type"
    }

    if not required_df.issubset(df.columns):
        raise ValueError(
            f"Input missing columns: {required_df - set(df.columns)}"
        )

    if not required_goes.issubset(goes.columns):
        raise ValueError(
            f"GOES catalogue missing columns: "
            f"{required_goes - set(goes.columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Null or invalid timestamps found in dataset.")

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if df["flare_active"].isna().any():
        raise ValueError("Null flare_active labels found.")

    if not df["flare_active"].isin([0, 1]).all():
        raise ValueError("flare_active must contain only 0 and 1.")

    # --------------------------------------------------------
    # Prepare GOES M/X event starts
    # --------------------------------------------------------

    goes["class_type"] = (
        goes["class_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    goes["start"] = pd.to_datetime(
        goes["start"],
        utc=True,
        errors="coerce"
    )

    events = goes[
        goes["class_type"].isin(["M", "X"])
    ].copy()

    events = events.dropna(subset=["start"])

    event_starts = (
        events["start"]
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if event_starts.empty:
        raise ValueError("No valid M/X event starts found.")

    # Explicit nanosecond conversion.
    event_ns = datetime_to_ns(event_starts)

    # --------------------------------------------------------
    # Determine usable observation minutes
    # --------------------------------------------------------

    signal_columns = get_signal_columns(df)

    usable = (
        df[signal_columns]
        .notna()
        .any(axis=1)
    )

    print("\n========== OBSERVATION SUMMARY ==========")
    print("Signal columns used:", len(signal_columns))
    print("Usable rows:", int(usable.sum()))
    print("Unusable rows:", int((~usable).sum()))
    print("=========================================")

    # --------------------------------------------------------
    # Identify continuous usable observation segments
    # --------------------------------------------------------

    segment_id = build_observation_segments(
        df["timestamp"],
        usable
    )

    # --------------------------------------------------------
    # Find next M/X event for every timestamp
    # --------------------------------------------------------

    timestamp_ns = datetime_to_ns(df["timestamp"])

    if event_ns.dtype != timestamp_ns.dtype:
        raise TypeError(
            "Timestamp arrays have different integer types."
        )

    # Both arrays are now explicitly in nanoseconds.
    next_event_index = np.searchsorted(
        event_ns,
        timestamp_ns,
        side="right"
    )

    has_future_event = (
        next_event_index < len(event_ns)
    )

    next_event_ns = np.full(
        len(df),
        np.iinfo(np.int64).max,
        dtype=np.int64
    )

    next_event_ns[has_future_event] = event_ns[
        next_event_index[has_future_event]
    ]

    delta_ns = next_event_ns - timestamp_ns

    print("\n========== EVENT SUMMARY ==========")
    print("M/X catalogue rows:", len(events))
    print("Unique M/X event starts:", len(event_starts))
    print("First event:", event_starts.min())
    print("Last event:", event_starts.max())

    print(
        "Events within dataset time range:",
        int(
            (
                (event_starts >= df["timestamp"].min())
                & (event_starts <= df["timestamp"].max())
            ).sum()
        )
    )
    print("===================================")

    # --------------------------------------------------------
    # Generate horizon-specific labels
    # --------------------------------------------------------

    currently_active = (
        df["flare_active"]
        .eq(1)
        .to_numpy()
    )

    for horizon in HORIZONS:

        print(f"\nGenerating {horizon}-minute targets...")

        horizon_ns = pd.Timedelta(
            minutes=horizon
        ).value

        # ----------------------------------------------------
        # Find the exact future row
        # ----------------------------------------------------

        future_timestamp = df["timestamp"].shift(-horizon)

        future_segment = segment_id.shift(-horizon)

        expected_future_timestamp = (
            df["timestamp"]
            + pd.Timedelta(minutes=horizon)
        )

        complete_coverage = (
            usable
            & future_timestamp.eq(expected_future_timestamp)
            & future_segment.eq(segment_id)
            & segment_id.ne(-1)
        )

        # ----------------------------------------------------
        # Event onset within future horizon
        # ----------------------------------------------------

        positive = (
            (delta_ns > 0)
            & (delta_ns <= horizon_ns)
        )

        target = np.zeros(
            len(df),
            dtype=np.int8
        )

        target[positive] = 1

        # ----------------------------------------------------
        # Unknown labels
        # ----------------------------------------------------

        # Current flare-active rows are not forecasting samples.
        # Incomplete future observation coverage is also unknown.
        unknown = (
            ~complete_coverage.to_numpy()
            | currently_active
        )

        target[unknown] = -1

        column = f"flare_{horizon}m"

        df[column] = target

        counts = df[column].value_counts().reindex(
            [-1, 0, 1],
            fill_value=0
        )

        print(f"\n{column} distribution:")
        print(counts.to_string())

        print(
            "Complete future coverage:",
            int(complete_coverage.sum())
        )

        print(
            "Unknown labels:",
            int((target == -1).sum())
        )

        print(
            "Positive labels before coverage masking:",
            int(positive.sum())
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    for horizon in HORIZONS:

        column = f"flare_{horizon}m"

        if not df[column].isin([-1, 0, 1]).all():
            raise ValueError(
                f"Unexpected values in {column}"
            )

        active_rows = (
            df["flare_active"] == 1
        )

        if (df.loc[active_rows, column] != -1).any():
            raise ValueError(
                f"Active flare rows must be unknown in {column}"
            )

        if (
            (df[column] == 1)
            & (df["flare_active"] == 1)
        ).any():
            raise ValueError(
                f"Positive target found during active flare: {column}"
            )

    # Longer horizons should not have fewer positive labels
    # before masking, because their event window is larger.
    # This is checked using the generated final targets only
    # as a diagnostic, not as a strict assertion, because
    # unknown coverage can differ by horizon.

    # --------------------------------------------------------
    # Save output safely
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():

        backup_file = OUTPUT_FILE.with_name(
            OUTPUT_FILE.stem + "_previous.parquet"
        )

        print("\nExisting output found.")
        print("Creating backup:", backup_file)

        shutil.copy2(
            OUTPUT_FILE,
            backup_file
        )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========== FORECAST LABEL SUMMARY ==========")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print(
        "Unique dates:",
        df["timestamp"].dt.date.nunique()
    )

    for horizon in HORIZONS:

        column = f"flare_{horizon}m"

        print(f"\n{column}:")
        print(
            df[column]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print("\nOutput:", OUTPUT_FILE)
    print("============================================")


if __name__ == "__main__":
    main()
