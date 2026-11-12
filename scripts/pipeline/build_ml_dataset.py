
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path(
    "output/forecast_labeled_timeseries_minute_end_v2.parquet"
)

OUTPUT = Path(
    "output/ml_ready_minute_end_v2"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "classification": ("flare_class", None),
    "nowcasting": ("flare_active", None),
    "peak_detection": ("flare_peak", None),
    "forecast_15m": ("flare_15m", 15),
    "forecast_30m": ("flare_30m", 30),
    "forecast_60m": ("flare_60m", 60),
}

ROLLING_WINDOWS = [5, 15, 30, 60]
DIFF_LAGS = [1, 5, 15]

TRAIN_END = pd.Timestamp("2026-06-01", tz="UTC")
VAL_END = pd.Timestamp("2026-07-01", tz="UTC")

EXPECTED_FEATURE_COUNT = 424

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PHASE 7: ML DATASET GENERATION")
print("=" * 60)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {SOURCE}"
    )

print("\nLoading master dataset...")

df = pd.read_parquet(SOURCE)

required_columns = [
    "timestamp",
    "solexs_sdd2_counts",
    "solexs_sdd2_missing",
    "solexs_sdd2_gti",
    *[target for target, _ in TARGETS.values()],
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

assert df["timestamp"].is_unique, (
    "Duplicate timestamps found"
)

assert df["timestamp"].is_monotonic_increasing, (
    "Timestamps are not sorted"
)

assert str(df["timestamp"].dt.tz) == "UTC"

print("Dataset shape:", df.shape)
print("Timestamp dtype:", df["timestamp"].dtype)
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())

# ============================================================
# IDENTIFY SIGNAL CHANNELS
# ============================================================

hel_signals = [
    col for col in df.columns
    if col.startswith("hel1os_")
    and not col.endswith((
        "_gti",
        "_missing",
        "_stat_err",
        "_stat_err_missing",
    ))
]

solexs_signals = [
    "solexs_sdd2_counts"
]

signal_cols = hel_signals + solexs_signals

if not hel_signals:
    raise ValueError("No HEL1OS signal columns found.")

for col in signal_cols:
    if col not in df.columns:
        raise ValueError(f"Missing signal column: {col}")

print("\nSignal channels:")
print("HEL1OS:", len(hel_signals))
print("SoLEXS:", len(solexs_signals))
print("Total:", len(signal_cols))

# ============================================================
# VALIDATE SIGNAL METADATA
# ============================================================

for col in signal_cols:

    if col == "solexs_sdd2_counts":
        missing_col = "solexs_sdd2_missing"
        gti_col = "solexs_sdd2_gti"
    else:
        missing_col = f"{col}_missing"
        gti_col = f"{col}_gti"

    required = [missing_col, gti_col]

    if col != "solexs_sdd2_counts":
        err_col = f"{col}_stat_err"
        err_missing_col = f"{col}_stat_err_missing"

        if err_col in df.columns:
            required.append(err_missing_col)

    for metadata_col in required:
        if metadata_col not in df.columns:
            raise ValueError(
                f"Missing metadata column: {metadata_col}"
            )

# ============================================================
# SENSOR AVAILABILITY
# ============================================================

print("\nChecking sensor availability...")

has_hel = df[hel_signals].notna().any(axis=1)

has_solexs = df[
    "solexs_sdd2_counts"
].notna()

usable = has_hel | has_solexs

print(
    "Rows with usable sensor data:",
    int(usable.sum())
)

print(
    "Rows without sensor data:",
    int((~usable).sum())
)

# ============================================================
# BUILD FEATURE FRAME
# ============================================================

print("\nGenerating basic signal features...")

X = pd.DataFrame(index=df.index)

# Sensor availability
X["hel1os_available"] = has_hel.astype("int8")

X["solexs_available"] = has_solexs.astype("int8")

# Signal transformations and metadata
for col in signal_cols:

    values = df[col]

    # Log transformation
    X[f"{col}_log1p"] = np.log1p(values)

    # Missingness and GTI indicators
    if col == "solexs_sdd2_counts":
        missing_col = "solexs_sdd2_missing"
        gti_col = "solexs_sdd2_gti"
    else:
        missing_col = f"{col}_missing"
        gti_col = f"{col}_gti"

    X[f"{col}_missing"] = (
        df[missing_col].astype("int8")
    )

    X[f"{col}_gti"] = (
        df[gti_col].astype("int8")
    )

    # HEL1OS statistical errors
    err_col = f"{col}_stat_err"

    if err_col in df.columns:

        X[f"{col}_stat_err_log1p"] = np.log1p(
            df[err_col]
        )

        err_missing_col = (
            f"{col}_stat_err_missing"
        )

        X[f"{col}_stat_err_missing"] = (
            df[err_missing_col].astype("int8")
        )

# ============================================================
# TEMPORAL SEGMENTS
# Reset rolling windows across:
# 1. Timestamp gaps
# 2. Unusable observation intervals
# ============================================================

print("\nIdentifying continuous observation segments...")

time_gap = df["timestamp"].diff().ne(
    pd.Timedelta(minutes=1)
)

previous_unusable = (~usable).shift(
    1,
    fill_value=True
)

break_before = (
    time_gap | previous_unusable
) & usable

segment_id = (
    break_before.cumsum().astype("int64")
)

# Unusable rows do not belong to an observation segment.
segment_id = segment_id.where(
    usable,
    -1
)

print(
    "Continuous segments:",
    int(segment_id[segment_id >= 0].nunique())
)

# ============================================================
# ROLLING STATISTICS AND TEMPORAL DIFFERENCES
# All features use current and past observations only.
# ============================================================

print("\nGenerating rolling features...")

for i, col in enumerate(signal_cols, start=1):

    print(
        f"[{i}/{len(signal_cols)}] Processing {col}"
    )

    values = np.log1p(df[col])

    # Prevent unusable observations from entering windows.
    values = values.where(usable)

    grouped = values.groupby(
        segment_id,
        sort=False
    )

    # Rolling statistics
    for window in ROLLING_WINDOWS:

        X[f"{col}_roll{window}_mean"] = (
            grouped.transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

        X[f"{col}_roll{window}_std"] = (
            grouped.transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=2
                ).std()
            )
        )

        X[f"{col}_roll{window}_count"] = (
            grouped.transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=1
                ).count()
            )
        )

    # Past-only differences
    for lag in DIFF_LAGS:

        X[f"{col}_diff_{lag}m"] = (
            grouped.diff(lag)
        )

# ============================================================
# UTC TIME FEATURES
# ============================================================

print("\nGenerating UTC time features...")

hour = (
    df["timestamp"].dt.hour
    + df["timestamp"].dt.minute / 60
)

day_of_year = df["timestamp"].dt.dayofyear

X["time_sin"] = np.sin(
    2 * np.pi * hour / 24
)

X["time_cos"] = np.cos(
    2 * np.pi * hour / 24
)

X["day_year_sin"] = np.sin(
    2 * np.pi * day_of_year / 365.25
)

X["day_year_cos"] = np.cos(
    2 * np.pi * day_of_year / 365.25
)

# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\nValidating feature matrix...")

assert X.shape[0] == len(df)

assert X.columns.is_unique, (
    "Duplicate feature names found"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_columns = X.columns.tolist()

print("Feature matrix:", X.shape)
print("Feature count:", len(feature_columns))

if len(feature_columns) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"but generated {len(feature_columns)}. "
        "Check the signal channel names and metadata."
    )

# ============================================================
# CHRONOLOGICAL SPLITS
# ============================================================

print("\nCreating chronological splits...")

timestamps = df["timestamp"]

train_mask = timestamps < TRAIN_END

val_mask = (
    (timestamps >= TRAIN_END)
    & (timestamps < VAL_END)
)

test_mask = timestamps >= VAL_END

splits = {
    "train": train_mask,
    "validation": val_mask,
    "test": test_mask,
}

for split_name, mask in splits.items():

    if mask.any():
        print(
            f"{split_name}: "
            f"{int(mask.sum()):,} rows | "
            f"{timestamps[mask].min()} -> "
            f"{timestamps[mask].max()}"
        )
    else:
        print(f"{split_name}: 0 rows")

# ============================================================
# CREATE TASK-SPECIFIC DATASETS
# ============================================================

print("\nCreating task-specific datasets...")

summary = []

for task_name, (
    target_col,
    horizon
) in TARGETS.items():

    print("\n" + "=" * 50)
    print("Processing:", task_name)
    print("=" * 50)

    target = df[target_col]

    # Exclude rows without usable sensor observations
    # and rows with missing targets.
    valid_mask = usable & target.notna()

    # -1 means unknown forecast target.
    if horizon is not None:
        valid_mask &= target != -1

    for split_name, split_mask in splits.items():

        final_mask = valid_mask & split_mask

        # Purge forecast samples whose prediction horizon
        # reaches or crosses the next split boundary.
        if horizon is not None and split_name != "test":

            boundary = (
                TRAIN_END
                if split_name == "train"
                else VAL_END
            )

            final_mask &= (
                timestamps
                < boundary - pd.Timedelta(
                    minutes=horizon
                )
            )

        indices = df.index[final_mask]

        dataset = X.loc[indices].copy()

        # Insert timezone-aware timestamps.
        dataset.insert(
            0,
            "timestamp",
            df.loc[indices, "timestamp"].array
        )

        # Insert target.
        dataset.insert(
            1,
            "target",
            target.loc[indices].astype(
                "int8"
            ).to_numpy()
        )

        # Verify timestamp preservation.
        assert str(
            dataset["timestamp"].dt.tz
        ) == "UTC", (
            f"Timezone lost in {task_name}_{split_name}"
        )

        assert len(dataset) == len(indices)

        # Save dataset.
        output_file = (
            OUTPUT
            / f"{task_name}_{split_name}.parquet"
        )

        dataset.to_parquet(
            output_file,
            index=False
        )

        # Target distribution.
        counts = (
            dataset["target"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        summary.append({
            "task": task_name,
            "split": split_name,
            "rows": len(dataset),
            "target_distribution": str(counts),
            "file": str(output_file),
        })

        print(
            f"{split_name}: {len(dataset):,} rows"
        )

        print(
            "Target distribution:",
            counts
        )

# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = OUTPUT / "dataset_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

# ============================================================
# SAVE FEATURE SCHEMA
# ============================================================

feature_schema = {
    "features": feature_columns,
    "feature_count": len(feature_columns),
    "signal_channels": signal_cols,
    "hel1os_channels": hel_signals,
    "solexs_channels": solexs_signals,
    "rolling_windows_minutes": ROLLING_WINDOWS,
    "difference_lags_minutes": DIFF_LAGS,
    "split_boundaries_utc": {
        "train_end": str(TRAIN_END),
        "validation_end": str(VAL_END),
    },
    "source": str(SOURCE),
    "output_directory": str(OUTPUT),
    "timestamp_convention": (
        "Minute-end UTC timestamps. "
        "Each row represents the preceding one-minute bin."
    ),
}

schema_file = OUTPUT / "feature_schema.json"

with open(
    schema_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_schema,
        f,
        indent=4
    )

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("PHASE 7 DATASET BUILD COMPLETE")
print("=" * 60)

print("Source:", SOURCE)
print("Output:", OUTPUT)
print("Total rows:", len(df))
print("Total features:", len(feature_columns))

print("\nGenerated files:")
print("1. Task-specific train/validation/test Parquet files")
print("2. dataset_summary.csv")
print("3. feature_schema.json")

print("\nDATASET SUMMARY:")
print(summary_df.to_string(index=False))

print("\nAll datasets generated successfully.")
