
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "output/merged/merged_timeseries.parquet"
GOES_FILE = "output/goes_events_parsed.csv"

OUTPUT_FILE = "output/labeled_timeseries.parquet"
EVENT_REPORT = "output/mx_event_label_report.csv"

HORIZONS = [15, 30, 60]

MINUTE_NS = 60 * 1_000_000_000


# ============================================================
# LOAD DATA
# ============================================================

print("Loading merged observations...")

df = pd.read_parquet(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True
)

df = df.sort_values("timestamp").reset_index(drop=True)

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps found in merged dataset.")

if not df["timestamp"].dt.second.eq(0).all():
    raise ValueError("Expected timestamps aligned to minute boundaries.")

print("Merged shape:", df.shape)

print("\nLoading GOES catalogue...")

goes = pd.read_csv(GOES_FILE)

for col in ["start", "peak", "end"]:
    goes[col] = pd.to_datetime(goes[col], utc=True)

events = goes[
    goes["class_type"].isin(["M", "X"])
].copy()

events = events.sort_values("start").reset_index(drop=True)

print("Total GOES events:", len(goes))
print("M/X events:", len(events))


# ============================================================
# CLASSIFICATION AND NOWCASTING LABELS
# ============================================================

print("\nGenerating classification and nowcasting labels...")

timestamps = (
    df["timestamp"].dt.as_unit("ns").astype("int64").to_numpy()
)

flare_active = np.zeros(len(df), dtype=np.int8)
flare_class = np.zeros(len(df), dtype=np.int8)
flare_peak = np.zeros(len(df), dtype=np.int8)

event_report = []

for _, event in events.iterrows():

    start_ns = event["start"].value
    end_ns = event["end"].value
    peak_ns = event["peak"].value

    label = 1 if event["class_type"] == "M" else 2

    left = np.searchsorted(timestamps, start_ns, side="left")
    right = np.searchsorted(timestamps, end_ns, side="right")

    if right > left:
        flare_active[left:right] = 1
        flare_class[left:right] = np.maximum(
            flare_class[left:right], label
        )

    peak_idx = np.searchsorted(
        timestamps, peak_ns, side="left"
    )

    peak_found = (
        peak_idx < len(timestamps)
        and timestamps[peak_idx] == peak_ns
    )

    if peak_found:
        flare_peak[peak_idx] = 1

    event_report.append({
        "start": event["start"],
        "peak": event["peak"],
        "end": event["end"],
        "class": event["class"],
        "class_type": event["class_type"],
        "active_minutes_in_dataset": right - left,
        "start_timestamp_present": (
            start_ns in timestamps
        ),
        "peak_timestamp_present": peak_found
    })


df["flare_active"] = flare_active
df["flare_class"] = flare_class
df["flare_peak"] = flare_peak


# ============================================================
# FORECASTING LABELS
# ============================================================

print("\nGenerating forecasting labels...")

event_starts = np.sort(
    events["start"].dt.as_unit("ns").astype("int64").to_numpy()
)

# Find the next M/X event start strictly after each timestamp.
next_event_idx = np.searchsorted(
    event_starts,
    timestamps,
    side="right"
)

next_event_exists = next_event_idx < len(event_starts)

next_event_ns = np.zeros(len(df), dtype=np.int64)

next_event_ns[next_event_exists] = event_starts[
    next_event_idx[next_event_exists]
]

delta_ns = next_event_ns - timestamps

# A complete future window requires every intervening
# minute to exist in the merged dataset.
complete_window = {}

for horizon in HORIZONS:

    steps = horizon

    valid = np.zeros(len(df), dtype=bool)

    if len(df) > steps:
        valid[:-steps] = (
            timestamps[steps:] - timestamps[:-steps]
            == steps * MINUTE_NS
        )

    complete_window[horizon] = valid

    target = np.zeros(len(df), dtype=np.int8)

    positive = (
        next_event_exists
        & (delta_ns > 0)
        & (delta_ns <= horizon * MINUTE_NS)
    )

    target[positive] = 1

    # Exclude currently active flare periods.
    target[flare_active == 1] = -1

    # Exclude incomplete future windows and dataset gaps.
    target[~valid] = -1

    df[f"flare_{horizon}m"] = target


# ============================================================
# SAVE OUTPUTS
# ============================================================

Path("output").mkdir(exist_ok=True)

df.to_parquet(OUTPUT_FILE, index=False)

report = pd.DataFrame(event_report)
report.to_csv(EVENT_REPORT, index=False)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 55)
print("PHASE 6 LABEL GENERATION COMPLETE")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("Unique dates:", df["date"].nunique())

print("\nClassification distribution:")
print(df["flare_class"].value_counts().sort_index())

print("\nNowcasting distribution:")
print(df["flare_active"].value_counts().sort_index())

print("\nPeak labels:")
print(df["flare_peak"].value_counts().sort_index())

for horizon in HORIZONS:
    col = f"flare_{horizon}m"

    print(f"\n{horizon}-minute forecasting labels:")
    print(df[col].value_counts().sort_index())

print("\nGOES event association:")
print("Total M/X events:", len(events))
print("Events with start timestamp present:",
      report["start_timestamp_present"].sum())
print("Events with peak timestamp present:",
      report["peak_timestamp_present"].sum())

print("\nForecast horizon nesting check:")

for short, long in [(15, 30), (30, 60)]:
    violation = (
        (df[f"flare_{short}m"] == 1)
        & (df[f"flare_{long}m"] != 1)
    ).sum()

    print(f"{short}m positives missing from {long}m:", violation)

print("\nSaved files:")
print(OUTPUT_FILE)
print(EVENT_REPORT)

print("\nNOTE: Label generation does not establish scientific "
      "validity. Review event associations and quality filters.")
