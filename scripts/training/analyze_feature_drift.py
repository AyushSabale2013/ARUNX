import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import ks_2samp


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("output/ml_ready")
OUTPUT_DIR = Path("output/models")

FILES = {
    "Train": DATA_DIR / "classification_train.parquet",
    "June": DATA_DIR / "classification_validation.parquet",
    "July": DATA_DIR / "classification_test.parquet",
}

MAX_SAMPLE = 50000
RANDOM_STATE = 42


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 75)
print("LOADING DATASETS")
print("=" * 75)

datasets = {}

for name, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    X = df.drop(columns=["timestamp", "target"])
    y = (df["target"] > 0).astype(int)

    datasets[name] = {
        "X": X,
        "y": y
    }

    print(
        f"{name}: "
        f"rows={len(df)}, "
        f"flare={int(y.sum())}, "
        f"non_flare={int((y == 0).sum())}, "
        f"features={X.shape[1]}"
    )


# ============================================================
# KS DRIFT ANALYSIS FUNCTION
# ============================================================

def analyze_drift(
    reference,
    comparison,
    reference_name,
    comparison_name,
    population
):

    results = []

    for feature in reference.columns:

        a = pd.to_numeric(
            reference[feature],
            errors="coerce"
        ).dropna()

        b = pd.to_numeric(
            comparison[feature],
            errors="coerce"
        ).dropna()

        if len(a) == 0 or len(b) == 0:
            continue

        if len(a) > MAX_SAMPLE:
            a = a.sample(
                MAX_SAMPLE,
                random_state=RANDOM_STATE
            )

        if len(b) > MAX_SAMPLE:
            b = b.sample(
                MAX_SAMPLE,
                random_state=RANDOM_STATE
            )

        ks = ks_2samp(a, b)

        results.append({

            "reference": reference_name,
            "comparison": comparison_name,
            "population": population,
            "feature": feature,

            "ks_statistic": ks.statistic,
            "p_value": ks.pvalue,

            "reference_median": a.median(),
            "comparison_median": b.median(),

            "reference_mean": a.mean(),
            "comparison_mean": b.mean(),

            "reference_missing_rate":
                reference[feature].isna().mean(),

            "comparison_missing_rate":
                comparison[feature].isna().mean(),

            "reference_count": len(a),
            "comparison_count": len(b),

        })

    return pd.DataFrame(results)


# ============================================================
# RUN TRAIN VS JUNE AND TRAIN VS JULY
# ============================================================

train_X = datasets["Train"]["X"]
train_y = datasets["Train"]["y"]

all_results = []

comparisons = ["June", "July"]

for split_name in comparisons:

    print("\n" + "=" * 75)
    print(f"ANALYZING TRAIN VS {split_name.upper()}")
    print("=" * 75)

    comparison_X = datasets[split_name]["X"]
    comparison_y = datasets[split_name]["y"]

    # --------------------------------------------------------
    # 1. ALL ROWS
    # --------------------------------------------------------

    print("\nAnalyzing all rows...")

    overall = analyze_drift(
        train_X,
        comparison_X,
        "Train",
        split_name,
        "All rows"
    )

    all_results.append(overall)

    # --------------------------------------------------------
    # 2. FLARE ROWS ONLY
    # --------------------------------------------------------

    print("Analyzing flare rows...")

    train_flare_X = train_X.loc[train_y == 1]
    comparison_flare_X = comparison_X.loc[
        comparison_y == 1
    ]

    flare_only = analyze_drift(
        train_flare_X,
        comparison_flare_X,
        "Train flare",
        split_name,
        "Flare rows"
    )

    all_results.append(flare_only)

    # --------------------------------------------------------
    # 3. NON-FLARE ROWS ONLY
    # --------------------------------------------------------

    print("Analyzing non-flare rows...")

    train_normal_X = train_X.loc[train_y == 0]
    comparison_normal_X = comparison_X.loc[
        comparison_y == 0
    ]

    normal_only = analyze_drift(
        train_normal_X,
        comparison_normal_X,
        "Train non-flare",
        split_name,
        "Non-flare rows"
    )

    all_results.append(normal_only)


# ============================================================
# COMBINE RESULTS
# ============================================================

final = pd.concat(
    all_results,
    ignore_index=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = OUTPUT_DIR / "feature_drift_analysis.csv"

final.to_csv(
    output_path,
    index=False
)

print("\nSaved:", output_path)


# ============================================================
# DISPLAY TOP 20 DRIFTED FEATURES
# ============================================================

for split_name in ["June", "July"]:

    for population in [
        "All rows",
        "Flare rows",
        "Non-flare rows"
    ]:

        subset = final[
            (final["comparison"] == split_name)
            &
            (final["population"] == population)
        ].copy()

        subset = subset.sort_values(
            "ks_statistic",
            ascending=False
        )

        print("\n" + "=" * 75)
        print(f"TRAIN VS {split_name} | {population}")
        print("=" * 75)

        if subset.empty:
            print("No valid feature comparisons.")
            continue

        columns = [
            "feature",
            "ks_statistic",
            "reference_median",
            "comparison_median",
            "reference_missing_rate",
            "comparison_missing_rate",
        ]

        print(
            subset[columns]
            .head(20)
            .to_string(index=False)
        )


# ============================================================
# DRIFT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DRIFT SUMMARY")
print("=" * 75)

for split_name in ["June", "July"]:

    for population in [
        "All rows",
        "Flare rows",
        "Non-flare rows"
    ]:

        subset = final[
            (final["comparison"] == split_name)
            &
            (final["population"] == population)
        ]

        if subset.empty:
            continue

        print(
            f"{split_name:5} | "
            f"{population:15} | "
            f"Mean KS={subset['ks_statistic'].mean():.4f} | "
            f"KS > 0.20: "
            f"{(subset['ks_statistic'] > 0.20).sum():3d} | "
            f"KS > 0.30: "
            f"{(subset['ks_statistic'] > 0.30).sum():3d}"
        )


# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n" + "=" * 75)
print("KS STATISTIC INTERPRETATION")
print("=" * 75)

print("""
KS statistic ranges from 0 to 1.

0.00 - 0.10 : Small distribution difference
0.10 - 0.20 : Moderate distribution difference
0.20 - 0.30 : Noticeable distribution difference
Above 0.30  : Large distribution difference

These are practical investigation bands, not universal
statistical significance thresholds.

Important:
- KS measures distribution differences, not their causes.
- day_year_sin and day_year_cos naturally differ across
  chronological splits.
- High KS does not automatically mean a feature is useless.
- Missingness differences should also be investigated.
""")

print("\nFeature drift analysis complete.")
