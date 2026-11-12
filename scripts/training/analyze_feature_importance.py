
import joblib
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "output/models/xgboost_classification.pkl"
SCHEMA_PATH = "output/ml_ready/feature_schema.json"

model = joblib.load(MODEL_PATH)

import json

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

features = schema["features"]

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTOP 30 FEATURES")
print(importance.head(30).to_string(index=False))

importance.to_csv(
    "output/models/feature_importance.csv",
    index=False
)

plt.figure(figsize=(12, 9))

top = importance.head(25).sort_values("importance")

plt.barh(top["feature"], top["importance"])

plt.xlabel("Feature Importance")
plt.title("XGBoost - Top 25 Feature Importances")

plt.tight_layout()

plt.savefig(
    "output/models/feature_importance.png",
    dpi=200
)

plt.show()
