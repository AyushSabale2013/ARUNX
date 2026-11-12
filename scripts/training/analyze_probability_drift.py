
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
import joblib


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE = Path("output")
MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "ml_ready"

MODEL_PATH = MODEL_DIR / "xgboost_flare_detection.pkl"
IMPUTER_PATH = MODEL_DIR / "flare_detection_imputer.pkl"

SPLITS = {
    "June": DATA_DIR / "classification_validation.parquet",
    "July": DATA_DIR / "classification_test.parquet"
}


# --------------------------------------------------
# LOAD MODEL AND IMPUTER
# --------------------------------------------------

print("Loading model and imputer...")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)


# --------------------------------------------------
# LOAD DATA AND PREDICT
# --------------------------------------------------

results = {}

for split_name, path in SPLITS.items():

    print(f"\nProcessing {split_name}...")

    df = pd.read_parquet(path)

    X = df.drop(columns=["timestamp", "target"])
    y = (df["target"] > 0).astype(int)

    X = imputer.transform(X)

    probabilities = model.predict_proba(X)[:, 1]

    results[split_name] = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"], utc=True),
        "actual": y.to_numpy(),
        "probability": probabilities
    })

    print("Rows:", len(df))
    print("Actual flare count:", int(y.sum()))
    print("Mean predicted probability:", round(probabilities.mean(), 6))


# --------------------------------------------------
# PROBABILITY DISTRIBUTION SUMMARY
# --------------------------------------------------

print("\n\nPROBABILITY DISTRIBUTION")

for name, df in results.items():

    print(f"\n{name}")

    for label, group in df.groupby("actual"):

        print(
            "Flare" if label == 1 else "No flare",
            "| Count:", len(group),
            "| Mean:", round(group["probability"].mean(), 6),
            "| Median:", round(group["probability"].median(), 6),
            "| 75%:", round(group["probability"].quantile(0.75), 6),
            "| 90%:", round(group["probability"].quantile(0.90), 6),
            "| 95%:", round(group["probability"].quantile(0.95), 6)
        )


# --------------------------------------------------
# METRICS AT DIFFERENT THRESHOLDS
# --------------------------------------------------

print("\n\nTHRESHOLD COMPARISON")

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:

    print(f"\nThreshold: {threshold}")

    for name, df in results.items():

        y_true = df["actual"]
        y_pred = (df["probability"] >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        print(
            f"{name}: "
            f"Precision={precision_score(y_true, y_pred, zero_division=0):.4f}, "
            f"Recall={recall_score(y_true, y_pred, zero_division=0):.4f}, "
            f"F1={f1_score(y_true, y_pred, zero_division=0):.4f}, "
            f"Balanced Acc={balanced_accuracy_score(y_true, y_pred):.4f}, "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )


# --------------------------------------------------
# DAILY PROBABILITY AND FLARE ANALYSIS
# --------------------------------------------------

print("\n\nDAILY ANALYSIS")

daily_results = []

for name, df in results.items():

    temp = df.copy()
    temp["date"] = temp["timestamp"].dt.date

    daily = temp.groupby("date").agg(
        total_rows=("actual", "size"),
        actual_flares=("actual", "sum"),
        mean_probability=("probability", "mean"),
        max_probability=("probability", "max")
    )

    daily["predicted_flares_020"] = (
        temp.assign(predicted=(temp["probability"] >= 0.20).astype(int))
        .groupby("date")["predicted"]
        .sum()
    )

    daily["split"] = name
    daily_results.append(daily.reset_index())

    print(f"\n{name}")
    print(daily.to_string())


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output = MODEL_DIR / "probability_drift_daily.csv"

pd.concat(daily_results, ignore_index=True).to_csv(output, index=False)

print("\nSaved daily analysis:", output)


# --------------------------------------------------
# PLOT PROBABILITY DISTRIBUTIONS
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, name in zip(axes, ["June", "July"]):

    df = results[name]

    ax.hist(
        df.loc[df["actual"] == 0, "probability"],
        bins=50,
        alpha=0.6,
        label="No flare",
        density=True
    )

    ax.hist(
        df.loc[df["actual"] == 1, "probability"],
        bins=50,
        alpha=0.6,
        label="M/X flare",
        density=True
    )

    ax.set_title(f"{name} Probability Distribution")
    ax.set_xlabel("Predicted flare probability")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()

plot_path = MODEL_DIR / "probability_drift.png"
plt.savefig(plot_path, dpi=150)
plt.show()

print("\nSaved plot:", plot_path)
