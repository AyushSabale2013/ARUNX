
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("output/ml_ready_minute_end_v2")

MODEL_DIR = Path("models/forecasting")

RESULT_DIR = Path("output/model_results/forecasting")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    "15m": "forecast_15m",
    "30m": "forecast_30m",
    "60m": "forecast_60m",
}

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(task, split):

    path = DATA_DIR / f"{task}_{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_parquet(path)

    if "timestamp" not in data.columns:
        raise ValueError(f"Missing timestamp in {path}")

    if "target" not in data.columns:
        raise ValueError(f"Missing target in {path}")

    return data


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, validation, test):

    excluded = ["timestamp", "target"]

    feature_columns = [
        col for col in train.columns
        if col not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    if feature_columns != [
        col for col in validation.columns
        if col not in excluded
    ]:
        raise ValueError(
            "Train and validation feature columns differ."
        )

    if feature_columns != [
        col for col in test.columns
        if col not in excluded
    ]:
        raise ValueError(
            "Train and test feature columns differ."
        )

    X_train = train[feature_columns].copy()
    X_val = validation[feature_columns].copy()
    X_test = test[feature_columns].copy()

    # XGBoost supports NaN values natively.
    # Do not fill missing values with arbitrary constants.

    for name, X in [
        ("train", X_train),
        ("validation", X_val),
        ("test", X_test),
    ]:

        non_numeric = X.select_dtypes(
            exclude=[np.number, "bool"]
        ).columns.tolist()

        if non_numeric:
            raise ValueError(
                f"Non-numeric features in {name}: "
                f"{non_numeric}"
            )

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

    return (
        X_train,
        X_val,
        X_test,
        feature_columns,
    )


# ============================================================
# SELECT THRESHOLD USING VALIDATION DATA
# ============================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    if len(thresholds) == 0:
        return 0.5

    # Precision and recall have one more element
    # than thresholds. Ignore their final element.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype("int8")

    metrics = {
        "task": name,
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    return metrics, predictions


# ============================================================
# TRAIN ONE FORECASTING MODEL
# ============================================================

def train_task(horizon, task):

    print("\n" + "=" * 65)
    print(f"TRAINING {horizon} FORECAST MODEL")
    print("=" * 65)

    # Load chronological datasets
    train = load_dataset(task, "train")
    validation = load_dataset(task, "validation")
    test = load_dataset(task, "test")

    y_train = train["target"].astype("int8")
    y_val = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")

    # Binary target validation
    for name, target in [
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:

        if not set(target.unique()).issubset({0, 1}):
            raise ValueError(
                f"Unexpected target values in {name}"
            )

        if target.nunique() != 2:
            raise ValueError(
                f"{name} must contain both classes."
            )

    # Prepare features
    (
        X_train,
        X_val,
        X_test,
        feature_columns,
    ) = prepare_features(
        train,
        validation,
        test,
    )

    print("\nDataset shapes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    # ========================================================
    # CLASS IMBALANCE
    # ========================================================

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\nClass imbalance:")
    print("Negative:", negative_count)
    print("Positive:", positive_count)
    print("scale_pos_weight:", scale_pos_weight)

    # ========================================================
    # MODEL
    # ========================================================

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,

        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,

        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nEvaluating on validation data...")

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    print(
        f"\nSelected validation threshold: "
        f"{threshold:.6f}"
    )

    validation_metrics, _ = evaluate_model(
        f"{horizon}_validation",
        y_val,
        val_probabilities,
        threshold,
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\nEvaluating on test data...")

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics, test_predictions = evaluate_model(
        f"{horizon}_test",
        y_test,
        test_probabilities,
        threshold,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR / f"xgboost_forecast_{horizon}.json"
    )

    model.save_model(model_path)

    # Save feature order and selected threshold.
    metadata = {
        "horizon": horizon,
        "model": "XGBoost",
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }

    metadata_path = (
        MODEL_DIR / f"forecast_{horizon}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({
        "timestamp": test["timestamp"].reset_index(
            drop=True
        ),
        "actual": y_test.reset_index(drop=True),
        "probability": test_probabilities,
        "predicted": test_predictions,
    })

    predictions_path = (
        RESULT_DIR / f"forecast_{horizon}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\nSaved:")
    print("Model:", model_path)
    print("Metadata:", metadata_path)
    print("Predictions:", predictions_path)

    return {
        "horizon": horizon,
        "validation": validation_metrics,
        "test": test_metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    for horizon, task in TASKS.items():

        result = train_task(
            horizon,
            task,
        )

        results.append(result)

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    validation_results = pd.DataFrame([
        result["validation"]
        for result in results
    ])

    test_results = pd.DataFrame([
        result["test"]
        for result in results
    ])

    validation_results.to_csv(
        RESULT_DIR / "validation_metrics.csv",
        index=False,
    )

    test_results.to_csv(
        RESULT_DIR / "test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("FORECASTING TRAINING COMPLETE")
    print("=" * 65)

    print("\nVALIDATION RESULTS:")
    print(
        validation_results[
            [
                "task",
                "threshold",
                "precision",
                "recall",
                "f1",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    print("\nTEST RESULTS:")
    print(
        test_results[
            [
                "task",
                "threshold",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print("\nResults directory:", RESULT_DIR)
    print("Models directory:", MODEL_DIR)


if __name__ == "__main__":
    main()
