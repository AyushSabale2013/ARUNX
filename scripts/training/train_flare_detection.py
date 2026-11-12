
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "output" / "ml_ready"
MODEL_DIR = ROOT / "output" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "classification_train.parquet"
VAL_FILE = DATA_DIR / "classification_validation.parquet"
TEST_FILE = DATA_DIR / "classification_test.parquet"

RANDOM_STATE = 42

TARGET = "target"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_parquet(TRAIN_FILE)
val_df = pd.read_parquet(VAL_FILE)
test_df = pd.read_parquet(TEST_FILE)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train_df.drop(
    columns=["timestamp", TARGET],
    errors="ignore"
)

X_val = val_df.drop(
    columns=["timestamp", TARGET],
    errors="ignore"
)

X_test = test_df.drop(
    columns=["timestamp", TARGET],
    errors="ignore"
)

# Preserve identical feature order.
X_val = X_val.reindex(columns=X_train.columns)
X_test = X_test.reindex(columns=X_train.columns)


# ============================================================
# CONVERT THREE CLASSES TO BINARY
# ============================================================

def convert_to_binary(y):
    """
    Original:
        0 = No M/X flare
        1 = M-class
        2 = X-class

    Binary:
        0 = No M/X flare
        1 = M/X flare
    """

    return (y > 0).astype(int)


y_train = convert_to_binary(train_df[TARGET])
y_val = convert_to_binary(val_df[TARGET])
y_test = convert_to_binary(test_df[TARGET])


print("\nBinary training distribution:")
print(y_train.value_counts().sort_index())

print("\nBinary validation distribution:")
print(y_val.value_counts().sort_index())

print("\nBinary test distribution:")
print(y_test.value_counts().sort_index())

print("\nNumber of features:", X_train.shape[1])


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nApplying median imputation...")

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

# Fit only on training data.
X_train = imputer.fit_transform(X_train)

X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

print("Imputation complete.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = y_train.value_counts().sort_index()

total_samples = len(y_train)
number_of_classes = len(class_counts)

class_weights = {
    int(cls): total_samples / (number_of_classes * count)
    for cls, count in class_counts.items()
}

print("\nClass weights:")

for cls, weight in class_weights.items():
    print(f"Class {cls}: {weight:.6f}")


sample_weights = y_train.map(class_weights).to_numpy()


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\nTraining binary XGBoost classifier...")

model = XGBClassifier(
    objective="binary:logistic",

    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,

    eval_metric="logloss",
    early_stopping_rounds=50,

    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,

    sample_weight=sample_weights,

    eval_set=[
        (X_val, y_val)
    ],

    verbose=50,
)

print("\nTraining complete.")

print("Best iteration:", model.best_iteration)
print("Best validation score:", model.best_score)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, dataset_name):

    probabilities = model.predict_proba(X)[:, 1]

    # Default threshold = 0.50
    predictions = (probabilities >= 0.50).astype(int)

    print("\n" + "=" * 60)
    print(f"{dataset_name} RESULTS")
    print("=" * 60)

    print("Accuracy:", accuracy_score(y, predictions))

    print(
        "Balanced accuracy:",
        balanced_accuracy_score(y, predictions)
    )

    print(
        "Macro F1:",
        f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    print(
        "Flare precision:",
        precision_score(y, predictions, zero_division=0)
    )

    print(
        "Flare recall:",
        recall_score(y, predictions, zero_division=0)
    )

    print("\nClassification report:")

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                "No M/X flare",
                "M/X flare"
            ],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    return predictions, probabilities


# ============================================================
# VALIDATION EVALUATION
# ============================================================

val_predictions, val_probabilities = evaluate_model(
    model,
    X_val,
    y_val,
    "VALIDATION"
)


# ============================================================
# TEST EVALUATION
# ============================================================

# Record results only. Do not use test results for tuning.
test_predictions, test_probabilities = evaluate_model(
    model,
    X_test,
    y_test,
    "TEST"
)


# ============================================================
# SAVE MODEL AND IMPUTER
# ============================================================

model_path = MODEL_DIR / "xgboost_flare_detection.pkl"
imputer_path = MODEL_DIR / "flare_detection_imputer.pkl"
metadata_path = MODEL_DIR / "flare_detection_metadata.json"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(imputer_path, "wb") as f:
    pickle.dump(imputer, f)


metadata = {
    "model": "XGBoost",
    "task": "Binary M/X flare detection",
    "target": TARGET,

    "class_mapping": {
        "0": "No M/X flare",
        "1": "M/X flare"
    },

    "number_of_features": int(X_train.shape[1]),

    "best_iteration": int(model.best_iteration),
    "best_validation_score": float(model.best_score),

    "class_weights": class_weights,

    "decision_threshold": 0.50,

    "early_stopping_rounds": 50,
    "random_state": RANDOM_STATE,
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)


print("\nSaved model:", model_path)
print("Saved imputer:", imputer_path)
print("Saved metadata:", metadata_path)

print("\nBINARY FLARE DETECTION TRAINING COMPLETE.")
