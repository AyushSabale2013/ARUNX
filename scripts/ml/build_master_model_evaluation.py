import json
from pathlib import Path

import pandas as pd

ROOT = Path("output/model_results")
COMPARISON = ROOT / "comparisons"
EVENT_DIR = ROOT / "event_level"

validation = pd.read_csv(
    COMPARISON / "all_models_validation_metrics.csv"
)
test = pd.read_csv(
    COMPARISON / "all_models_test_metrics.csv"
)
events = pd.read_csv(
    EVENT_DIR / "event_level_summary.csv"
)

key = ["model", "horizon"]

if validation.duplicated(key).any() or test.duplicated(key).any():
    raise ValueError("Duplicate model/horizon rows in comparison metrics.")

if set(map(tuple, validation[key].values)) != set(map(tuple, test[key].values)):
    raise ValueError("Validation and test model/horizon combinations differ.")

event_columns = [
    "event_detection_rate",
    "alert_episode_precision",
    "detected_events",
    "missed_events",
    "alert_episodes",
    "episodes_associated_with_event",
    "false_alarm_episodes",
    "false_alarms_per_day",
    "mean_lead_time_minutes",
    "median_lead_time_minutes",
    "observed_coverage_days",
    "missing_timestamp_minutes",
]

# Only join event metrics whose model names identify the architecture.
known_events = events[events["model"] != "forecast"].copy()
if known_events.duplicated(key).any():
    raise ValueError("Duplicate identifiable event-level model/horizon rows.")

event_lookup = {
    (row["model"], row["horizon"]): row
    for _, row in known_events.iterrows()
}

rows = []

# Traditional ML and XGBoost
for _, val in validation.iterrows():
    model = val["model"]
    horizon = val["horizon"]

    test_row = test[
        (test["model"] == model) &
        (test["horizon"] == horizon)
    ].iloc[0]

    row = {
        "family": val["family"],
        "model": model,
        "architecture": model,
        "horizon": horizon,
        "evaluation_source": "comparison_metrics_csv",
    }

    excluded = {"family", "model", "horizon"}
    for column in validation.columns:
        if column not in excluded:
            row[f"val_{column}"] = val[column]

    for column in test.columns:
        if column not in excluded:
            row[f"test_{column}"] = test_row[column]

    event_row = event_lookup.get((model, horizon))
    row["event_level_mapping_status"] = (
        "matched" if event_row is not None else "not_found"
    )

    for column in event_columns:
        row[f"event_{column}"] = (
            event_row[column] if event_row is not None else None
        )

    rows.append(row)

# Deep-learning models
for model in ["cnn", "cnn_avgmax", "cnn_lstm"]:
    for minutes in [15, 30, 60]:
        folder = (
            ROOT / "deep_learning" / model /
            f"forecast_{minutes}m"
        )
        metrics_path = folder / "metrics.json"

        with metrics_path.open() as f:
            data = json.load(f)

        val = data["validation_metrics"]
        tst = data["test_metrics"]

        row = {
            "family": "deep_learning",
            "model": model,
            "architecture": model,
            "horizon": f"{minutes}m",
            "evaluation_source": str(metrics_path),
            "sequence_length": data.get("sequence_length"),
            "feature_count": data.get("feature_count"),
            "best_epoch": data.get("best_epoch"),
            "epochs_run": data.get("epochs_run"),
            "best_validation_pr_auc": data.get("best_validation_pr_auc"),
            "trainable_parameters": data.get("trainable_parameters"),
            "gpu_peak_memory_mb": data.get("gpu_peak_memory_mb"),
            "device": data.get("device"),
            "pos_weight_train": data.get("pos_weight_train"),
            "val_threshold": val.get("threshold"),
            "test_threshold": tst.get("threshold"),
            "event_level_mapping_status": "unmapped_generic_forecast_label",
        }

        for column, value in val.items():
            row[f"val_{column}"] = value

        for column, value in tst.items():
            row[f"test_{column}"] = value

        for column in event_columns:
            row[f"event_{column}"] = None

        rows.append(row)

master = pd.DataFrame(rows)

# Keep horizon order consistent.
master["_horizon_order"] = (
    master["horizon"].str.replace("m", "", regex=False).astype(int)
)
master = (
    master.sort_values(["_horizon_order", "family", "model"])
    .drop(columns="_horizon_order")
    .reset_index(drop=True)
)

output = ROOT / "ARUNX_MASTER_MODEL_EVALUATION.csv"
master.to_csv(output, index=False)

print(f"Created: {output}")
print(f"Total rows: {len(master)}")
print("\nRows by family:")
print(master.groupby("family").size().to_string())
print("\nRows by horizon:")
print(master.groupby("horizon").size().to_string())
print("\nDeep-learning event metrics are intentionally blank because")
print("the event evaluator labels those runs with the ambiguous name 'forecast'.")
print("\nEvent-level metrics are preliminary; prediction coverage is incomplete.")
