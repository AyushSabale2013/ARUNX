
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


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

OUTPUT_PATH = MODEL_DIR / "class_specific_probability_analysis.csv"

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


# ============================================================
# LOAD MODEL AND IMPUTER
# ============================================================

print("\nLoading model and imputer...")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path, dataset_name):

    print(f"\nLoading {dataset_name}...")

    df = pd.read_parquet(path)

    if "target" not in df.columns:
        raise ValueError(f"'target' column missing in {path}")

    class_counts = df["target"].value_counts().sort_index()

    print(f"{dataset_name} shape: {df.shape}")
    print("Original flare-class distribution:")
    print(class_counts)

    unexpected = set(df["target"].dropna().unique()) - {0, 1, 2}

    if unexpected:
        raise ValueError(f"Unexpected target values: {unexpected}")

    return df


validation_df = load_dataset(VALIDATION_PATH, "June Validation")
test_df = load_dataset(TEST_PATH, "July Test")


# ============================================================
# PREPARE FEATURES AND PREDICT PROBABILITIES
# ============================================================

def predict_probabilities(df, dataset_name):

    X = df.drop(columns=["timestamp", "target"], errors="ignore")

    X = imputer.transform(X)

    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "true_class": df["target"].to_numpy(),
        "flare_probability": probabilities
    })

    print(f"{dataset_name}: predictions generated.")

    return result


validation = predict_probabilities(validation_df, "June")
test = predict_probabilities(test_df, "July")


# ============================================================
# CLASS-SPECIFIC PROBABILITY DISTRIBUTION
# ============================================================

def analyze_distribution(df, dataset_name):

    print("\n" + "=" * 75)
    print(f"{dataset_name.upper()} - CLASS-SPECIFIC PROBABILITY DISTRIBUTION")
    print("=" * 75)

    rows = []

    classes = {
        0: "No flare",
        1: "M-class",
        2: "X-class"
    }

    for class_id, class_name in classes.items():

        subset = df.loc[
            df["true_class"] == class_id,
            "flare_probability"
        ]

        if len(subset) == 0:
            print(f"\n{class_name}: No samples available.")
            continue

        print(f"\n{class_name}")
        print(f"Samples : {len(subset)}")
        print(f"Mean    : {subset.mean():.6f}")
        print(f"Median  : {subset.median():.6f}")
        print(f"25%     : {subset.quantile(0.25):.6f}")
        print(f"75%     : {subset.quantile(0.75):.6f}")
        print(f"90%     : {subset.quantile(0.90):.6f}")
        print(f"95%     : {subset.quantile(0.95):.6f}")

        rows.append({
            "dataset": dataset_name,
            "class": class_name,
            "samples": len(subset),
            "mean_probability": subset.mean(),
            "median_probability": subset.median(),
            "q25": subset.quantile(0.25),
            "q75": subset.quantile(0.75),
            "q90": subset.quantile(0.90),
            "q95": subset.quantile(0.95)
        })

    return rows


all_results = []

all_results.extend(analyze_distribution(validation, "June"))
all_results.extend(analyze_distribution(test, "July"))


# ============================================================
# CLASS-SPECIFIC RECALL AT DIFFERENT THRESHOLDS
# ============================================================

def analyze_thresholds(df, dataset_name):

    print("\n" + "=" * 75)
    print(f"{dataset_name.upper()} - CLASS-SPECIFIC THRESHOLD ANALYSIS")
    print("=" * 75)

    for threshold in THRESHOLDS:

        predictions = (
            df["flare_probability"] >= threshold
        ).astype(int)

        print(f"\nThreshold: {threshold:.2f}")

        for class_id, class_name in [(1, "M-class"), (2, "X-class")]:

            actual = df["true_class"] == class_id

            total = int(actual.sum())
            detected = int((predictions[actual] == 1).sum())

            recall = detected / total if total else np.nan

            print(
                f"{class_name:10s} | "
                f"Samples: {total:5d} | "
                f"Detected: {detected:5d} | "
                f"Recall: {recall:.4f}" if total else
                f"{class_name:10s} | Samples: 0 | Recall: N/A"
            )

        actual_binary = (df["true_class"] > 0).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            actual_binary,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        print(
            f"Combined M/X | "
            f"TP: {tp} FP: {fp} FN: {fn} TN: {tn} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )


analyze_thresholds(validation, "June")
analyze_thresholds(test, "July")


# ============================================================
# SAVE DISTRIBUTION RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(f"\nResults saved to:\n{OUTPUT_PATH}")

print("\nSaved distribution summary:")
print(results_df.to_string(index=False))

print("\nIMPORTANT:")
print("July contains very few X-class samples.")
print("Interpret X-class recall and probability statistics cautiously.")
