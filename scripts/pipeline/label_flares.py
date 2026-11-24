
"""
ARUN-X: Multi-day GOES flare labelling.

Input:
    output/merged/merged_timeseries_minute_end_v2.parquet
    output/goes_events_parsed.csv

Output:
    output/labeled_timeseries_minute_end_v2.parquet

Timestamp convention:
    A row timestamp is the RIGHT EDGE of its completed 1-minute bin.
    Example: 10:01 represents measurements from 10:00:00 to 10:00:59.
"""

from pathlib import Path
import argparse
import os

import pandas as pd


BASE = Path(__file__).resolve().parents[2]

INPUT_FILE = Path(os.environ.get("ARUNX_MERGED_INPUT", str(BASE / "output" / "merged" / "merged_timeseries_minute_end_v2.parquet")))

GOES_FILE = Path(os.environ.get("ARUNX_GOES_FILE", str(BASE / "output" / "goes_events_parsed.csv")))

OUTPUT_FILE = Path(os.environ.get("ARUNX_LABELED_OUTPUT", str(BASE / "output" / "labeled_timeseries_minute_end_v2.parquet")))
EVENT_REPORT = Path(os.environ["ARUNX_EVENT_REPORT"]) if os.environ.get("ARUNX_EVENT_REPORT") else None


def main():

    global INPUT_FILE, GOES_FILE, OUTPUT_FILE, EVENT_REPORT
    parser = argparse.ArgumentParser(description="Create active/class/peak GOES labels.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--goes", type=Path, default=GOES_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--event-report", type=Path, default=EVENT_REPORT)
    args = parser.parse_args()
    INPUT_FILE, GOES_FILE, OUTPUT_FILE, EVENT_REPORT = args.input, args.goes, args.output, args.event_report

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if not GOES_FILE.exists():
        raise FileNotFoundError(GOES_FILE)

    print("Loading multi-day merged dataset...")
    df = pd.read_parquet(INPUT_FILE)

    print("Loading corrected GOES catalogue...")
    goes = pd.read_csv(GOES_FILE)

    required_columns = {"timestamp"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Merged dataset is missing timestamp.")

    required_goes = {
        "start", "peak", "end", "class", "class_type"
    }
    if not required_goes.issubset(goes.columns):
        raise ValueError(
            f"GOES dataset missing columns: "
            f"{required_goes - set(goes.columns)}"
        )

    # --------------------------------------------------------
    # Validate timestamps
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].isna().any():
        raise ValueError("Null timestamps found.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    minute_diffs = df["timestamp"].diff().dropna()

    if not minute_diffs.eq(pd.Timedelta(minutes=1)).any():
        raise ValueError(
            "No consecutive minute timestamps found. "
            "Check the merged dataset."
        )

    # --------------------------------------------------------
    # Validate GOES timestamps
    # --------------------------------------------------------

    for col in ["start", "peak", "end"]:
        goes[col] = pd.to_datetime(goes[col], utc=True)

    invalid = (
        (goes["start"] > goes["peak"])
        | (goes["peak"] > goes["end"])
    )

    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} invalid GOES events found."
        )

    # Only M/X events are used for these labels.
    events = goes[
        goes["class_type"].isin(["M", "X"])
    ].copy()

    events = events.sort_values("start").reset_index(drop=True)
    if events.duplicated(subset=["start", "peak", "end", "class_type"]).any():
        raise ValueError("Duplicate M/X GOES events found; resolve catalogue duplicates before labeling.")

    # --------------------------------------------------------
    # Initialize labels
    # --------------------------------------------------------

    df["flare_active"] = 0
    df["flare_class"] = 0
    df["flare_peak"] = 0

    bin_end = df["timestamp"]
    bin_start = bin_end - pd.Timedelta(minutes=1)
    event_rows = []

    # --------------------------------------------------------
    # Apply GOES event labels
    # --------------------------------------------------------

    for _, event in events.iterrows():

        event_start = event["start"]
        event_end = event["end"]

        # A completed bin is labelled active if the event
        # overlaps its interval (bin_start, bin_end].
        active = (
            (event_start < bin_end)
            & (event_end > bin_start)
        )

        label = 1 if event["class_type"] == "M" else 2

        df.loc[active, "flare_active"] = 1

        # Preserve the highest class in overlapping intervals.
        df.loc[active, "flare_class"] = (
            df.loc[active, "flare_class"]
            .clip(lower=label)
        )

        # Map peak to the right edge of the minute containing it.
        # Exact boundary peaks belong to the next completed bin.
        peak_minute_end = (
            event["peak"].floor("min")
            + pd.Timedelta(minutes=1)
        )

        peak_mask = bin_end == peak_minute_end

        df.loc[peak_mask, "flare_peak"] = 1
        event_rows.append({
            "event_index": int(len(event_rows) + 1),
            "start": event_start,
            "peak": event["peak"],
            "end": event_end,
            "class": event["class"],
            "class_type": event["class_type"],
            "active_minutes_labeled": int(active.sum()),
            "peak_minute_end_present": bool(peak_mask.any()),
        })

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not df["flare_class"].isin([0, 1, 2]).all():
        raise ValueError("Unexpected flare_class values.")

    if not df["flare_active"].isin([0, 1]).all():
        raise ValueError("Unexpected flare_active values.")

    if not df["flare_peak"].isin([0, 1]).all():
        raise ValueError("Unexpected flare_peak values.")

    # Every active row must have a non-zero class.
    if (
        (df["flare_active"] == 1)
        & (df["flare_class"] == 0)
    ).any():
        raise ValueError(
            "Active flare rows found without a flare class."
        )

    # --------------------------------------------------------
    # Save to a NEW output file
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    report_path = EVENT_REPORT or (OUTPUT_FILE.parent / "mx_event_label_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(event_rows).to_csv(report_path, index=False)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n========== MULTI-DAY LABEL SUMMARY ==========")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Unique dates:", df["timestamp"].dt.date.nunique())

    print("\nTimestamp range:")
    print("First:", df["timestamp"].min())
    print("Last: ", df["timestamp"].max())

    print("\nActive flare labels:")
    print(df["flare_active"].value_counts().sort_index())

    print("\nFlare class distribution:")
    print(df["flare_class"].value_counts().sort_index())

    print("\nPeak labels:")
    print(df["flare_peak"].value_counts().sort_index())

    print("\nM/X GOES events used:", len(events))
    print(
        "M events:",
        int((events["class_type"] == "M").sum())
    )
    print(
        "X events:",
        int((events["class_type"] == "X").sum())
    )

    print("\nSaved:", OUTPUT_FILE)
    print("===============================================")


if __name__ == "__main__":
    main()
