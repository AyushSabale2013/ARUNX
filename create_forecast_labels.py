import pandas as pd
import numpy as np

INPUT_FILE = "output/labeled_20261006.parquet"
OUTPUT_FILE = "output/forecast_labeled_20261006.parquet"

df = pd.read_parquet(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Load M/X event start times from parsed GOES catalogue
goes = pd.read_csv("output/goes_events_parsed.csv")
goes["start"] = pd.to_datetime(goes["start"], utc=True)

events = goes[goes["class_type"].isin(["M", "X"])]

event_starts = events["start"].sort_values().tolist()

# Generate forecasting targets
for horizon in [15, 30, 60]:

    target = np.zeros(len(df), dtype=np.int8)

    for event_start in event_starts:
        delta_minutes = (
            event_start - df["timestamp"]
        ).dt.total_seconds() / 60

        target[
            (delta_minutes > 0) &
            (delta_minutes <= horizon)
        ] = 1

    # Current M/X flare intervals are not forecasting samples
    target[df["flare_active"] == 1] = -1

    # Incomplete future windows: unknown
    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=horizon)

    target[df["timestamp"] > cutoff] = -1

    df[f"flare_{horizon}m"] = target

df.to_parquet(OUTPUT_FILE, index=False)

# Validation summary
print("Output:", OUTPUT_FILE)
print("Shape:", df.shape)

for horizon in [15, 30, 60]:
    print(f"\n{horizon}-minute forecast labels:")
    print(df[f"flare_{horizon}m"].value_counts().sort_index())

print("\nForecast-positive timestamps:")
for col in ["flare_15m", "flare_30m", "flare_60m"]:
    print(f"\n{col}")
    print(
        df.loc[df[col] == 1, ["timestamp", col]]
        .to_string(index=False)
    )
