
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "output" / "ml_ready"
MODEL_DIR = ROOT / "output" / "models"

MODEL_PATH = MODEL_DIR / "xgboost_flare_detection.pkl"
IMPUTER_PATH = MODEL_DIR / "flare_detection_imputer.pkl"

VALIDATION_PATH = DATA_DIR / "classification_validation.parquet"
TEST_PATH = DATA_DIR / "classification_test.parquet"

OUTPUT_DIR = MODEL_DIR / "event_level_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.05, 0.10, 0.20, 0.50]

# Set this manually if automatic discovery does not find your GOES file.
GOES_EVENTS_PATH = None

# Expected date ranges
DATASETS = {
    "June": VALIDATION_PATH,
    "July": TEST_PATH
}


# ============================================================
# GOES CATALOGUE DISCOVERY
# ============================================================

def find_goes_file():

    if GOES_EVENTS_PATH is not None:
        path = Path(GOES_EVENTS_PATH)

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    search_dirs = [
        ROOT / "output",
        ROOT / "data",
    ]

    candidates = []

    for directory in search_dirs:

        if not directory.exists():
            continue

        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in [".csv", ".parquet"]:
                continue

            name = path.name.lower()

            if "goes" in name and any(
                word in name for word in ["event", "flare", "catalog"]
            ):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically find the GOES event catalogue.\n"
            "Set GOES_EVENTS_PATH manually at the top of this script."
        )

    if len(candidates) > 1:
        print("\nMultiple possible GOES files found:")

        for i, path in enumerate(candidates):
            print(f"{i + 1}. {path}")

        raise RuntimeError(
            "Multiple GOES files found. Set GOES_EVENTS_PATH manually."
        )

    return candidates[0]


# ============================================================
# LOAD GOES CATALOGUE
# ============================================================

def load_goes_catalogue(path):

    print("\nLoading GOES catalogue:")
    print(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("GOES catalogue must be CSV or Parquet.")

    print("\nGOES columns:")
    print(df.columns.tolist())

    return df


# ============================================================
# DETECT GOES COLUMN NAMES
# ============================================================

def find_column(df, aliases, description):

    normalized = {
        str(col).lower().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    for alias in aliases:

        key = alias.lower().replace(" ", "").replace("_", "")

        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        f"Expected one of: {aliases}"
    )


def prepare_events(df):

    start_col = find_column(
        df,
        [
            "event_starttime",
            "start_time",
            "starttime",
            "start",
            "begin_time"
        ],
        "event start"
    )

    peak_col = find_column(
        df,
        [
            "event_peaktime",
            "peak_time",
            "peaktime",
            "peak"
        ],
        "event peak"
    )

    end_col = find_column(
        df,
        [
            "event_endtime",
            "end_time",
            "endtime",
            "end"
        ],
        "event end"
    )

    class_col = find_column(
        df,
        [
            "goes_class",
            "class",
            "classification",
            "flare_class"
        ],
        "GOES flare class"
    )

    events = pd.DataFrame({
        "start_time": pd.to_datetime(
            df[start_col], utc=True, errors="coerce"
        ),
        "peak_time": pd.to_datetime(
            df[peak_col], utc=True, errors="coerce"
        ),
        "end_time": pd.to_datetime(
            df[end_col], utc=True, errors="coerce"
        ),
        "goes_class": df[class_col].astype(str).str.strip()
    })

    events = events.dropna(
        subset=["start_time", "peak_time", "end_time"]
    ).copy()

    # Keep only M-class and X-class events.
    events["flare_type"] = events["goes_class"].str.upper().str[0]

    events = events[
        events["flare_type"].isin(["M", "X"])
    ].copy()

    # Exclude malformed events.
    events = events[
        (events["start_time"] <= events["peak_time"]) &
        (events["peak_time"] <= events["end_time"])
    ].copy()

    events = events.sort_values("start_time").reset_index(drop=True)

    events.insert(
        0,
        "event_id",
        [f"GOES_{i + 1:04d}" for i in range(len(events))]
    )

    print(f"\nValid M/X GOES events: {len(events)}")
    print(events["flare_type"].value_counts())

    return events


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model and imputer...")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print("Model loaded successfully.")


# ============================================================
# GENERATE ROW-LEVEL PREDICTIONS
# ============================================================

def generate_predictions(path, dataset_name):

    print(f"\nProcessing {dataset_name}...")

    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing.")

    if "target" not in df.columns:
        raise ValueError("target column missing.")

    timestamps = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    X = df.drop(
        columns=["timestamp", "target"],
        errors="ignore"
    )

    probabilities = model.predict_proba(
        imputer.transform(X)
    )[:, 1]

    predictions = pd.DataFrame({
        "timestamp": timestamps,
        "true_class": df["target"].to_numpy(),
        "flare_probability": probabilities
    })

    predictions = predictions.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp")

    print(f"{dataset_name} observations: {len(predictions)}")

    return predictions


# ============================================================
# MATCH GOES EVENTS WITH MODEL PREDICTIONS
# ============================================================

def evaluate_events(events, predictions, dataset_name):

    print("\n" + "=" * 75)
    print(f"{dataset_name.upper()} EVENT-LEVEL EVALUATION")
    print("=" * 75)

    dataset_start = predictions["timestamp"].min()
    dataset_end = predictions["timestamp"].max()

    # Select events overlapping the observation period.
    relevant_events = events[
        (events["end_time"] >= dataset_start) &
        (events["start_time"] <= dataset_end)
    ].copy()

    print(f"GOES events overlapping dataset: {len(relevant_events)}")

    results = []

    for _, event in relevant_events.iterrows():

        event_start = event["start_time"]
        event_peak = event["peak_time"]
        event_end = event["end_time"]

        # Only evaluate the observed part of the event.
        event_rows = predictions[
            (predictions["timestamp"] >= event_start) &
            (predictions["timestamp"] <= event_end)
        ]

        if event_rows.empty:
            continue

        probabilities = event_rows["flare_probability"]

        row = {
            "dataset": dataset_name,
            "event_id": event["event_id"],
            "goes_class": event["goes_class"],
            "flare_type": event["flare_type"],
            "start_time": event_start,
            "peak_time": event_peak,
            "end_time": event_end,
            "duration_minutes": (
                event_end - event_start
            ).total_seconds() / 60,
            "observed_rows": len(event_rows),
            "max_probability": probabilities.max(),
            "mean_probability": probabilities.mean(),
            "median_probability": probabilities.median(),
            "peak_time_probability": np.nan
        }

        # Find the observation nearest to the GOES peak time.
        peak_idx = (
            event_rows["timestamp"] - event_peak
        ).abs().idxmin()

        row["peak_time_probability"] = event_rows.loc[
            peak_idx, "flare_probability"
        ]

        for threshold in THRESHOLDS:

            detected = bool(
                probabilities.max() >= threshold
            )

            row[f"detected_{threshold:.2f}"] = int(detected)

        results.append(row)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("No events could be matched to observations.")
        return result_df

    return result_df


# ============================================================
# PRINT EVENT-LEVEL SUMMARY
# ============================================================

def print_summary(results, dataset_name):

    print("\n" + "-" * 75)
    print(f"{dataset_name.upper()} EVENT-LEVEL SUMMARY")
    print("-" * 75)

    for flare_type in ["M", "X"]:

        subset = results[
            results["flare_type"] == flare_type
        ]

        print(f"\n{flare_type}-CLASS FLARES")
        print(f"Matched events: {len(subset)}")

        if subset.empty:
            continue

        print(
            f"Mean maximum probability: "
            f"{subset['max_probability'].mean():.4f}"
        )

        print(
            f"Median maximum probability: "
            f"{subset['max_probability'].median():.4f}"
        )

        print(
            f"Mean peak-time probability: "
            f"{subset['peak_time_probability'].mean():.4f}"
        )

        for threshold in THRESHOLDS:

            column = f"detected_{threshold:.2f}"

            detected = int(subset[column].sum())
            total = len(subset)

            recall = detected / total

            print(
                f"Threshold {threshold:.2f} | "
                f"Detected {detected}/{total} | "
                f"Event detection rate: {recall:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    goes_path = find_goes_file()

    raw_goes = load_goes_catalogue(goes_path)

    events = prepare_events(raw_goes)

    all_event_results = []

    for dataset_name, dataset_path in DATASETS.items():

        predictions = generate_predictions(
            dataset_path,
            dataset_name
        )

        results = evaluate_events(
            events,
            predictions,
            dataset_name
        )

        if results.empty:
            continue

        print_summary(results, dataset_name)

        output_file = (
            OUTPUT_DIR /
            f"{dataset_name.lower()}_event_results.csv"
        )

        results.to_csv(output_file, index=False)

        print(f"\nSaved: {output_file}")

        all_event_results.append(results)

    if not all_event_results:
        raise RuntimeError(
            "No event-level results generated. "
            "Check GOES timestamps and observation coverage."
        )

    combined = pd.concat(
        all_event_results,
        ignore_index=True
    )

    combined_path = OUTPUT_DIR / "all_event_results.csv"

    combined.to_csv(combined_path, index=False)

    print("\n" + "=" * 75)
    print("EVENT-LEVEL ANALYSIS COMPLETE")
    print("=" * 75)

    print(f"Combined results: {combined_path}")
    print(f"Total matched event records: {len(combined)}")


if __name__ == "__main__":
    main()
