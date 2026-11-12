
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
)

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("output/ml_ready_minute_end_v2")

MODEL_DIR = Path("models/baselines")

RESULT_DIR = Path("output/model_results/baselines")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    "15m": "forecast_15m",
    "30m": "forecast_30m",
    "60m": "forecast_60m",
}

RANDOM_STATE = 42

# Models to compare
MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(task, split):

    path = DATA_DIR / f"{task}_{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_parquet(path)

    return data


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_datasets(train, validation, test):

    feature_columns = [
        col for col in train.columns
        if col not in ["timestamp", "target"]
    ]

    for name, dataset in [
        ("validation", validation),
        ("test", test),
    ]:

        current_features = [
            col for col in dataset.columns
            if col not in ["timestamp", "target"]
        ]

        if current_features != feature_columns:
            raise ValueError(
                f"Feature mismatch in {name}"
            )

    if not set(train["target"].unique()).issubset({0, 1}):
        raise ValueError("Training target is not binary.")

    if not set(validation["target"].unique()).issubset({0, 1}):
        raise ValueError("Validation target is not binary.")

    if not set(test["target"].unique()).issubset({0, 1}):
        raise ValueError("Test target is not binary.")

    return feature_columns


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, validation, test, features):

    X_train = train[features].astype("float32")
    X_val = validation[features].astype("float32")
    X_test = test[features].astype("float32")

    for X in [X_train, X_val, X_test]:

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

    y_train = train["target"].astype("int8")
    y_val = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(name):

    if name == "logistic_regression":

        return Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=300,
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ])

    if name == "random_forest":

        return Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])

    if name == "extra_trees":

        return Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])

    if name == "hist_gradient_boosting":

        return Pipeline([
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=200,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ])

    if name == "lightgbm":

        return LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unknown model: {name}")


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        return 0.5

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

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": name,
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            predictions,
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

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    return metrics, predictions


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one_model(
    horizon,
    task,
    model_name,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    train_timestamps,
    test_timestamps,
    feature_columns,
):

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name} | HORIZON: {horizon}")
    print("=" * 65)

    model = create_model(model_name)

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nValidation evaluation...")

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    print("Selected threshold:", threshold)

    validation_metrics, _ = evaluate_model(
        f"{model_name}_{horizon}_validation",
        y_val,
        val_probabilities,
        threshold,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nTest evaluation...")

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics, test_predictions = evaluate_model(
        f"{model_name}_{horizon}_test",
        y_test,
        test_probabilities,
        threshold,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name}_{horizon}.joblib"
    )

    joblib.dump(model, model_path)

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "model": model_name,
        "horizon": horizon,
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "random_state": RANDOM_STATE,
    }

    metadata_path = (
        MODEL_DIR
        / f"{model_name}_{horizon}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "timestamp": test_timestamps.reset_index(drop=True),
        "actual": y_test.reset_index(drop=True),
        "probability": test_probabilities,
        "predicted": test_predictions,
    })

    predictions_df.to_csv(
        RESULT_DIR
        / f"{model_name}_{horizon}_predictions.csv",
        index=False,
    )

    print("\nSaved model:", model_path)

    return validation_metrics, test_metrics


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    all_validation_results = []
    all_test_results = []

    for horizon, task in TASKS.items():

        print("\n" + "#" * 70)
        print(f"FORECAST HORIZON: {horizon}")
        print("#" * 70)

        # Load only the current horizon.
        train = load_dataset(task, "train")
        validation = load_dataset(task, "validation")
        test = load_dataset(task, "test")

        feature_columns = validate_datasets(
            train,
            validation,
            test,
        )

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = prepare_features(
            train,
            validation,
            test,
            feature_columns,
        )

        print("\nDataset shapes:")
        print("Train:", X_train.shape)
        print("Validation:", X_val.shape)
        print("Test:", X_test.shape)

        print("\nTraining distribution:")
        print(y_train.value_counts().sort_index())

        for model_name in MODEL_NAMES:

            validation_metrics, test_metrics = (
                train_one_model(
                    horizon,
                    task,
                    model_name,
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    train["timestamp"],
                    test["timestamp"],
                    feature_columns,
                )
            )

            all_validation_results.append({
                "horizon": horizon,
                **validation_metrics,
            })

            all_test_results.append({
                "horizon": horizon,
                **test_metrics,
            })

        # Release memory before loading the next horizon.
        del train, validation, test
        del X_train, X_val, X_test
        del y_train, y_val, y_test

    # ========================================================
    # SAVE COMPARISON TABLES
    # ========================================================

    validation_df = pd.DataFrame(
        all_validation_results
    )

    test_df = pd.DataFrame(
        all_test_results
    )

    validation_df.to_csv(
        RESULT_DIR / "all_validation_metrics.csv",
        index=False,
    )

    test_df.to_csv(
        RESULT_DIR / "all_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("ALL BASELINE EXPERIMENTS COMPLETED")
    print("=" * 70)

    print("\nVALIDATION COMPARISON:")
    print(
        validation_df[
            [
                "horizon",
                "model",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ].sort_values(
            ["horizon", "pr_auc"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\nTEST COMPARISON:")
    print(
        test_df[
            [
                "horizon",
                "model",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ].sort_values(
            ["horizon", "pr_auc"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\nResults:", RESULT_DIR)
    print("Models:", MODEL_DIR)


if __name__ == "__main__":
    main()
