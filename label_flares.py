import pandas as pd

INPUT_FILE = "output/merged_20261006.parquet"
GOES_FILE = "output/goes_events_parsed.csv"
OUTPUT_FILE = "output/labeled_20261006.parquet"

# Load datasets
df = pd.read_parquet(INPUT_FILE)
goes = pd.read_csv(GOES_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

for col in ["start", "peak", "end"]:
    goes[col] = pd.to_datetime(goes[col], utc=True)

# Keep only M and X class events
events = goes[goes["class_type"].isin(["M", "X"])].copy()

# Initialize labels
df["flare_active"] = 0
df["flare_class"] = 0
df["flare_peak"] = 0

# Apply GOES event labels
for _, event in events.iterrows():

    active = (
        (df["timestamp"] >= event["start"]) &
        (df["timestamp"] <= event["end"])
    )

    # Class: M = 1, X = 2
    label = 1 if event["class_type"] == "M" else 2

    df.loc[active, "flare_active"] = 1

    # Keep highest class if events overlap
    df.loc[active, "flare_class"] = df.loc[
        active, "flare_class"
    ].clip(lower=label)

    # Mark GOES peak minute
    peak_minute = event["peak"].floor("min")
    peak_mask = df["timestamp"] == peak_minute

    df.loc[peak_mask, "flare_peak"] = 1

# Save
df.to_parquet(OUTPUT_FILE, index=False)

# Validation
print("Output shape:", df.shape)
print("\nActive flare minutes:")
print(df["flare_active"].value_counts())

print("\nClassification distribution:")
print(df["flare_class"].value_counts().sort_index())

print("\nPeak labels:")
print(df["flare_peak"].value_counts())

print("\nM/X events:")
print(events[["start", "peak", "end", "class"]].to_string(index=False))

print("\nSaved:", OUTPUT_FILE)

