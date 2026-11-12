
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

DATA_DIR = Path("output/ml_ready")
MODEL_DIR = Path("output/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train = pd.read_parquet(DATA_DIR / "classification_train.parquet")
val = pd.read_parquet(DATA_DIR / "classification_validation.parquet")
test = pd.read_parquet(DATA_DIR / "classification_test.parquet")

# Remove metadata and target
X_train = train.drop(columns=["timestamp", "target"])
X_val = val.drop(columns=["timestamp", "target"])
X_test = test.drop(columns=["timestamp", "target"])

y_train = train["target"]
y_val = val["target"]
y_test = test["target"]

# Remove all statistical-error-related features
staterr_cols = [
    col for col in X_train.columns
    if "_stat_err" in col
]

X_train = X_train.drop(columns=staterr_cols)
X_val = X_val.drop(columns=staterr_cols)
X_test = X_test.drop(columns=staterr_cols)

print("Removed statistical-error features:", len(staterr_cols))
print("Remaining features:", X_train.shape[1])

# Imputation
imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Class weights
class_counts = y_train.value_counts().sort_index()

class_weights = (
    len(y_train) /
    (len(class_counts) * class_counts)
)

sample_weights = y_train.map(class_weights).to_numpy()

# Model
model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# Evaluation function
def evaluate(name, X, y):

    predictions = model.predict(X)

    print(f"\n{name}")
    print("Balanced accuracy:",
          balanced_accuracy_score(y, predictions))

    print("Macro F1:",
          f1_score(y, predictions, average="macro"))

    print("\nClassification report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2]
    ))

evaluate("VALIDATION", X_val, y_val)
evaluate("TEST", X_test, y_test)

# Save separately
joblib.dump(
    model,
    MODEL_DIR / "xgboost_no_staterr.pkl"
)

joblib.dump(
    imputer,
    MODEL_DIR / "classification_no_staterr_imputer.pkl"
)

print("\nNO-STATERR EXPERIMENT COMPLETE")
