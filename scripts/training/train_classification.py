
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

TARGET = "target"

RANDOM_STATE = 42

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
# PREPARE FEATURES AND TARGETS
# ============================================================

DROP_COLUMNS = ["timestamp", TARGET]

X_train = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
y_train = train_df[TARGET].astype(int)

X_val = val_df.drop(columns=DROP_COLUMNS, errors="ignore")
y_val = val_df[TARGET].astype(int)

X_test = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
y_test = test_df[TARGET].astype(int)

# Ensure identical feature order across all splits.
X_val = X_val.reindex(columns=X_train.columns)
X_test = X_test.reindex(columns=X_train.columns)

print("\nNumber of features:", X_train.shape[1])

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nApplying median imputation...")

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

# Fit ONLY on training data.
X_train = imputer.fit_transform(X_train)

# Apply the fitted imputer to validation and test.
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

print("Class weights:")
for cls, weight in class_weights.items():
    print(f"Class {cls}: {weight:.6f}")

sample_weights = y_train.map(class_weights).to_numpy()


# ============================================================
# TRAIN XGBOOST WITH EARLY STOPPING
# ============================================================

print("\nTraining XGBoost classifier...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,

    eval_metric="mlogloss",
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

    predictions = model.predict(X)

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} RESULTS")
    print(f"{'=' * 60}")

    print(
        "Accuracy:",
        accuracy_score(y, predictions)
    )

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

    print("\nClassification report:")

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "No M/X flare",
                "M-class",
                "X-class"
            ],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1, 2]))

    return predictions


# ============================================================
# VALIDATION EVALUATION
# ============================================================

val_predictions = evaluate_model(
    model,
    X_val,
    y_val,
    "VALIDATION"
)


# ============================================================
# TEST EVALUATION
# ============================================================

# Record this result, but do not use it to select hyperparameters.
test_predictions = evaluate_model(
    model,
    X_test,
    y_test,
    "TEST"
)


# ============================================================
# SAVE MODEL AND IMPUTER
# ============================================================

model_path = MODEL_DIR / "xgboost_classification.pkl"
imputer_path = MODEL_DIR / "classification_imputer.pkl"
metadata_path = MODEL_DIR / "classification_metadata.json"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(imputer_path, "wb") as f:
    pickle.dump(imputer, f)

metadata = {
    "model": "XGBoost",
    "target": TARGET,
    "features": X_train.shape[1],
    "classes": {
        "0": "No M/X flare",
        "1": "M-class",
        "2": "X-class"
    },
    "best_iteration": int(model.best_iteration),
    "best_validation_score": float(model.best_score),
    "class_weights": class_weights,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_STATE
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nSaved model:", model_path)
print("Saved imputer:", imputer_path)
print("Saved metadata:", metadata_path)

print("\nCLASSIFICATION TRAINING COMPLETE.")
