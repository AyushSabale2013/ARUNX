
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "output" / "ml_ready"
MODEL_DIR = ROOT / "output" / "models"

# Load trained model and imputer
with open(MODEL_DIR / "xgboost_flare_detection.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "flare_detection_imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

# Load validation data ONLY
df = pd.read_parquet(
    DATA_DIR / "classification_validation.parquet"
)

X = df.drop(
    columns=["timestamp", "target"],
    errors="ignore"
)

y = (df["target"] > 0).astype(int)

# Apply saved training imputer
X = imputer.transform(X)

# Get flare probabilities
probabilities = model.predict_proba(X)[:, 1]

# Evaluate different thresholds
results = []

for threshold in np.arange(0.05, 0.96, 0.05):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y, predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    })

results_df = pd.DataFrame(results)

print("\nVALIDATION THRESHOLD ANALYSIS\n")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save results
output_path = (
    ROOT / "output" / "models" /
    "flare_threshold_analysis.csv"
)

results_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
