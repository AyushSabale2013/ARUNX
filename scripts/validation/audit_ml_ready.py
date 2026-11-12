import pandas as pd
import numpy as np
from pathlib import Path

folder = Path("output/ml_ready")
files = sorted(folder.glob("*.parquet"))

print("FILES:", len(files))

schemas = {}

for file in files:
    df = pd.read_parquet(file)

    features = df.drop(columns=["timestamp", "target"])
    schemas[file.name] = list(features.columns)

    numeric = features.select_dtypes(include=np.number)

    print(f"\n{file.name}")
    print("Rows:", len(df))
    print("Features:", len(features.columns))
    print("Timestamp dtype:", df["timestamp"].dtype)
    print("Missing values:", features.isna().sum().sum())
    print("Infinite values:", np.isinf(numeric).sum().sum())

print("\nSCHEMA CONSISTENCY:",
      len({tuple(v) for v in schemas.values()}) == 1)
print("\nTIMESTAMP CHECK")

for name in [
    "classification_train.parquet",
    "classification_validation.parquet",
    "classification_test.parquet"
]:
    df = pd.read_parquet(folder / name)

    print(f"\n{name}")
    print("First:", df["timestamp"].iloc[0])
    print("Last:", df["timestamp"].iloc[-1])
    print("Timezone:", df["timestamp"].dt.tz)