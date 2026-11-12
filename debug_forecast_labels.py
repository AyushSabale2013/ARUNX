
"""
ARUN-X: Forecast-label timestamp and event diagnostics.

Checks:
1. Dataset and GOES timestamp ranges.
2. Timestamp precision (nanoseconds).
3. Number of unique M/X event starts.
4. Time until the next M/X event.
5. Positive counts before coverage masking.
6. Existing forecast-label distributions, if available.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE / "output" /
    "labeled_timeseries_minute_end_v2.parquet"
)

GOES_FILE = (
    BASE / "output" /
    "goes_events_parsed.csv"
)

FORECAST_FILE = (
    BASE / "output" /
    "forecast_labeled_timeseries_minute_end_v2.parquet"
)


def datetime_to_ns(series):
    """
    Explicitly convert timestamps to int64 nanoseconds.
    """

    parsed = pd.to_datetime(
        series,
        utc=True,
        errors="coerce"
    )

    if parsed.isna().any():
        raise ValueError("Invalid datetime values found.")

    return (
        parsed.dt.as_unit("ns")
        .astype("int64")
        .to_numpy()
    )


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if not GOES_FILE.exists():
        raise FileNotFoundError(GOES_FILE)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_FILE)

    goes = pd.read_csv(GOES_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    goes["start"] = pd.to_datetime(
        goes["start"],
        utc=True,
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Invalid dataset timestamps found.")

    # --------------------------------------------------------
    # Prepare events
    # --------------------------------------------------------

    goes["class_type"] = (
        goes["class_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    events = goes[
        goes["class_type"].isin(["M", "X"])
    ].copy()

    event_starts = (
        events["start"]
        .dropna()
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if event_starts.empty:
        raise ValueError("No valid M/X events found.")

    # --------------------------------------------------------
    # Explicit timestamp conversion
    # --------------------------------------------------------

    timestamp_ns = datetime_to_ns(
        df["timestamp"]
    )

    event_ns = datetime_to_ns(
        event_starts
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print("\n========== EVENT DIAGNOSTICS ==========")

    print("Dataset rows:", len(df))
    print("M/X catalogue rows:", len(events))
    print("Unique M/X event starts:", len(event_starts))

    print("\nDataset timestamp range:")
    print(
        df["timestamp"].min(),
        "to",
        df["timestamp"].max()
    )

    print("\nGOES event start range:")
    print(
        event_starts.min(),
        "to",
        event_starts.max()
    )

    print("\nDataset datetime dtype:")
    print(df["timestamp"].dtype)

    print("\nGOES datetime dtype:")
    print(event_starts.dtype)

    print("\nFirst 15 M/X event starts:")
    print(
        event_starts.head(15)
        .to_string(index=False)
    )

    print("\nLast 10 M/X event starts:")
    print(
        event_starts.tail(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Timestamp unit sanity checks
    # --------------------------------------------------------

    print("\n========== TIMESTAMP UNIT CHECK ==========")

    print("Dataset integer dtype:", timestamp_ns.dtype)
    print("GOES integer dtype:", event_ns.dtype)

    print("Dataset first timestamp (ns):", timestamp_ns[0])
    print("GOES first event (ns):", event_ns[0])

    # Nanosecond Unix timestamps should be around 10^18
    if not (
        10**18 < abs(int(timestamp_ns[0])) < 10**19
    ):
        raise ValueError(
            "Dataset timestamps are not represented in ns."
        )

    if not (
        10**18 < abs(int(event_ns[0])) < 10**19
    ):
        raise ValueError(
            "GOES timestamps are not represented in ns."
        )

    if timestamp_ns.dtype != event_ns.dtype:
        raise ValueError(
            "Dataset and GOES timestamp integer types differ."
        )

    print("Timestamp unit check: PASS")

    # --------------------------------------------------------
    # Find next future event
    # --------------------------------------------------------

    next_event_index = np.searchsorted(
        event_ns,
        timestamp_ns,
        side="right"
    )

    valid = (
        next_event_index < len(event_ns)
    )

    next_event_ns = np.full(
        len(df),
        np.iinfo(np.int64).max,
        dtype=np.int64
    )

    next_event_ns[valid] = event_ns[
        next_event_index[valid]
    ]

    delta_ns = next_event_ns - timestamp_ns

    delta_minutes = (
        delta_ns.astype(np.float64)
        / (60 * 1e9)
    )

    delta_minutes[~valid] = np.nan

    # --------------------------------------------------------
    # Delta distribution
    # --------------------------------------------------------

    print("\nNext-event delta distribution (minutes):")

    delta_series = pd.Series(
        delta_minutes
    ).dropna()

    print(
        delta_series.describe(
            percentiles=[
                0.01,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.99
            ]
        )
    )

    # --------------------------------------------------------
    # Positive counts before coverage masking
    # --------------------------------------------------------

    print("\nPositive counts BEFORE coverage masking:")

    for horizon in [15, 30, 60]:

        horizon_ns = pd.Timedelta(
            minutes=horizon
        ).value

        positive = (
            (delta_ns > 0)
            & (delta_ns <= horizon_ns)
        )

        print(
            f"{horizon} minutes: {int(positive.sum())}"
        )

    # --------------------------------------------------------
    # Next-event examples
    # --------------------------------------------------------

    print("\nFirst 20 timestamps and next event:")

    next_event = pd.Series(
        pd.to_datetime(
            next_event_ns,
            unit="ns",
            utc=True
        )
    )

    next_event.loc[~valid] = pd.NaT

    debug = pd.DataFrame({
        "timestamp": df["timestamp"],
        "next_event": next_event,
        "delta_minutes": delta_minutes
    })

    print(
        debug.head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Existing forecast labels
    # --------------------------------------------------------

    if FORECAST_FILE.exists():

        print("\n========== EXISTING TARGET DISTRIBUTIONS ==========")

        forecast = pd.read_parquet(
            FORECAST_FILE,
            columns=[
                "flare_15m",
                "flare_30m",
                "flare_60m"
            ]
        )

        for horizon in [15, 30, 60]:

            column = f"flare_{horizon}m"

            print(f"\n{column}:")

            print(
                forecast[column]
                .value_counts()
                .sort_index()
                .to_string()
            )

    else:
        print(
            "\nNo existing forecast output found."
        )

    print("\n========== DIAGNOSTICS COMPLETE ==========")


if __name__ == "__main__":
    main()
