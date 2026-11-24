"""Train reproducible XGBoost baselines on one preprocessing run.

The test split is loaded only after early stopping and threshold selection are
finished on the validation split. Outputs are written to a new run-specific
directory; this script refuses to overwrite an existing directory or report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


BASE = Path(__file__).resolve().parents[2]
DEFAULT_DATA_RUN = "preproc_20261124_v1"
HORIZONS = (15, 30, 60)
DEFAULT_SEED = 42
DEFAULT_CONFIG: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 1500,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 4,
    "early_stopping_rounds": 75,
    "max_bin": 256,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_file_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"Dataset hash mismatch for {path}: {actual} != {expected}")
    return actual


def validate_feature_schema(run_dir: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    schema_path = run_dir / "ml_ready" / "feature_schema.json"
    run_manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "config.json"
    schema = load_json(schema_path)
    manifest = load_json(run_manifest_path)
    config = load_json(config_path)
    features = schema.get("features")

    if manifest.get("run_id") != run_dir.name:
        raise ValueError(f"Run ID mismatch: directory={run_dir.name}, manifest={manifest.get('run_id')}")
    if not isinstance(features, list) or len(features) != 592 or schema.get("feature_count") != 592:
        raise ValueError("Expected exactly 592 approved feature columns in feature_schema.json")
    if len(features) != len(set(features)):
        raise ValueError("Feature schema contains duplicate names")
    forbidden_tokens = ("flare_", "target", "future", "next_event", "class_type", "intensity", "event_", "start", "peak", "end")
    forbidden = [name for name in features if any(token in name.lower() for token in forbidden_tokens)]
    if forbidden:
        raise ValueError(f"Potential target/metadata leakage in feature schema: {forbidden[:10]}")
    if schema.get("row_policy") != "both_instruments":
        raise ValueError(f"Unexpected row policy: {schema.get('row_policy')}")
    output_records = {row["path"]: row for row in manifest.get("output_files", [])}
    schema_record = output_records.get("ml_ready/feature_schema.json")
    if schema_record is None or sha256_file(schema_path).lower() != schema_record["sha256"].lower():
        raise ValueError("Feature schema hash does not match the preprocessing run manifest")
    for horizon in HORIZONS:
        task = f"forecast_{horizon}m"
        for split in ("train", "validation", "test"):
            relative = f"ml_ready/{task}_{split}.parquet"
            if relative not in output_records:
                raise ValueError(f"Run manifest does not hash required dataset {relative}")
    return features, schema, {"manifest": manifest, "config": config, "output_records": output_records}


def load_split(
    path: Path,
    features: list[str],
    split_name: str,
    config: dict[str, Any],
) -> tuple[pd.Series, np.ndarray, np.ndarray, dict[str, Any]]:
    expected_columns = ["timestamp", "target", *features]
    frame = pd.read_parquet(path, columns=expected_columns)
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"Column ordering mismatch in {path}")
    if frame.empty:
        raise ValueError(f"Empty split: {path}")

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamp.isna().any() or timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError(f"Invalid/duplicate/unsorted timestamps in {path}")
    target = frame["target"].to_numpy(dtype=np.int8, copy=True)
    if not set(np.unique(target)).issubset({0, 1}) or len(np.unique(target)) != 2:
        raise ValueError(f"Expected both binary classes 0/1 in {path}")

    # The run policy requires both instruments in every released sample.
    if not np.all(frame["hel1os_available"].to_numpy() == 1):
        raise ValueError(f"HEL1OS row-policy violation in {path}")
    if not np.all(frame["solexs_available"].to_numpy() == 1):
        raise ValueError(f"SoLEXS row-policy violation in {path}")

    represented_date = timestamp - pd.Timedelta(minutes=1)
    train_end = pd.Timestamp(config["splits"]["train_end_exclusive_utc"])
    validation_end = pd.Timestamp(config["splits"]["validation_end_exclusive_utc"])
    if split_name == "train" and represented_date.max() >= train_end:
        raise ValueError(f"Training sample crosses configured split boundary: {path}")
    if split_name == "validation" and (
        represented_date.min() < train_end or represented_date.max() >= validation_end
    ):
        raise ValueError(f"Validation sample is outside configured date range: {path}")
    if split_name == "test" and represented_date.min() < validation_end:
        raise ValueError(f"Test sample starts before configured test period: {path}")

    feature_frame = frame[features]
    nonnumeric = feature_frame.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if nonnumeric:
        raise ValueError(f"Non-numeric feature columns in {path}: {nonnumeric[:10]}")
    x_values = feature_frame.to_numpy(dtype=np.float32, copy=True)
    if np.isinf(x_values).any():
        raise ValueError(f"Infinite feature value found in {path}")
    counts = {str(int(k)): int(v) for k, v in pd.Series(target).value_counts().sort_index().items()}
    info = {
        "path": str(path),
        "rows": int(len(frame)),
        "start_utc": timestamp.min().isoformat(),
        "end_utc": timestamp.max().isoformat(),
        "class_counts": counts,
        "feature_null_values": int(np.isnan(x_values).sum()),
    }
    del feature_frame, frame
    return timestamp, x_values, target, info


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, {"validation_f1": float(f1_score(y_true, probabilities >= 0.5, zero_division=0))}
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1_values))
    return float(thresholds[best]), {
        "validation_f1": float(f1_values[best]),
        "validation_precision": float(precision[best]),
        "validation_recall": float(recall[best]),
    }


def metrics_at_threshold(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[dict[str, Any], np.ndarray]:
    predicted = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    result = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": {"labels": [0, 1], "rows_true_cols_predicted": [[tn, fp], [fn, tp]]},
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "positive_samples": int(tp + fn),
        "negative_samples": int(tn + fp),
        "samples": int(len(y_true)),
    }
    return result, predicted


def majority_metrics(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(y_train, minlength=2)
    majority_class = int(np.argmax(counts))
    predicted = np.full(y_eval.shape, majority_class, dtype=np.int8)
    scores = predicted.astype(np.float32)
    matrix = confusion_matrix(y_eval, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "classifier": "constant majority class selected from training split",
        "majority_class": majority_class,
        "training_class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "precision": float(precision_score(y_eval, predicted, zero_division=0)),
        "recall": float(recall_score(y_eval, predicted, zero_division=0)),
        "f1": float(f1_score(y_eval, predicted, zero_division=0)),
        "roc_auc_constant_score": float(roc_auc_score(y_eval, scores)),
        "pr_auc_average_precision": float(average_precision_score(y_eval, scores)),
        "confusion_matrix": {"labels": [0, 1], "rows_true_cols_predicted": [[tn, fp], [fn, tp]]},
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "positive_samples": int(tp + fn),
        "negative_samples": int(tn + fp),
        "samples": int(len(y_eval)),
    }


def save_pr_curve(path: Path, y_true: np.ndarray, probabilities: np.ndarray) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    threshold_column = np.append(thresholds, np.nan)
    pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": threshold_column,
    }).to_csv(path, index=False)


def train_one_horizon(
    horizon: int,
    run_dir: Path,
    output_dir: Path,
    features: list[str],
    run_config: dict[str, Any],
    source_records: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    task = f"forecast_{horizon}m"
    paths = {split: run_dir / "ml_ready" / f"{task}_{split}.parquet" for split in ("train", "validation", "test")}
    dataset_hashes = {}
    for split in ("train", "validation"):
        relative = f"ml_ready/{task}_{split}.parquet"
        dataset_hashes[split] = {
            "path": relative,
            "sha256": verify_file_hash(paths[split], source_records[relative]["sha256"]),
            "size_bytes": paths[split].stat().st_size,
        }
    horizon_dir = output_dir / f"forecast_{horizon}m"
    horizon_dir.mkdir(parents=True, exist_ok=False)

    print(f"[{horizon}m] Loading training and validation splits", flush=True)
    train_ts, x_train, y_train, train_info = load_split(paths["train"], features, "train", run_config)
    val_ts, x_val, y_val, val_info = load_split(paths["validation"], features, "validation", run_config)
    train_negative = int(np.sum(y_train == 0))
    train_positive = int(np.sum(y_train == 1))
    if not train_positive or not train_negative:
        raise ValueError(f"Training split for {horizon}m must contain positive and negative examples")
    scale_pos_weight = train_negative / train_positive

    parameters = {
        **DEFAULT_CONFIG,
        "random_state": seed,
        "scale_pos_weight": scale_pos_weight,
    }
    model = XGBClassifier(**parameters)
    eval_history: dict[str, Any] = {}
    print(f"[{horizon}m] Fitting XGBoost (scale_pos_weight={scale_pos_weight:.6f})", flush=True)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    eval_history = model.evals_result()

    best_iteration = int(getattr(model, "best_iteration", parameters["n_estimators"] - 1))
    iteration_range = (0, best_iteration + 1)
    val_probability = model.predict_proba(x_val, iteration_range=iteration_range)[:, 1]
    threshold, threshold_info = choose_threshold(y_val, val_probability)
    validation_metrics, _ = metrics_at_threshold(y_val, val_probability, threshold)
    validation_metrics.update(threshold_info)
    validation_metrics["pr_auc_xgboost_early_stopping_metric"] = float(eval_history["validation_0"]["aucpr"][best_iteration])
    save_pr_curve(horizon_dir / "validation_precision_recall_curve.csv", y_val, val_probability)

    # Freeze and save the selected model before the test split is opened.
    model_path = horizon_dir / f"xgboost_forecast_{horizon}m.json"
    model.save_model(model_path)
    del model, x_train, x_val, val_probability

    print(f"[{horizon}m] Model selected at iteration {best_iteration}; verifying and loading test for final evaluation", flush=True)
    test_relative = f"ml_ready/{task}_test.parquet"
    dataset_hashes["test"] = {
        "path": test_relative,
        "sha256": verify_file_hash(paths["test"], source_records[test_relative]["sha256"]),
        "size_bytes": paths["test"].stat().st_size,
    }
    test_ts, x_test, y_test, test_info = load_split(paths["test"], features, "test", run_config)
    final_model = XGBClassifier()
    final_model.load_model(model_path)
    test_probability = final_model.predict_proba(x_test, iteration_range=iteration_range)[:, 1]
    test_metrics, test_prediction = metrics_at_threshold(y_test, test_probability, threshold)
    majority = majority_metrics(y_train, y_test)
    save_pr_curve(horizon_dir / "test_precision_recall_curve.csv", y_test, test_probability)

    predictions_path = horizon_dir / "test_predictions.csv"
    pd.DataFrame({
        "timestamp": test_ts.astype(str),
        "actual": y_test,
        "probability": test_probability,
        "predicted": test_prediction,
    }).to_csv(predictions_path, index=False)

    # Verify the serialized model is reloadable and predictions are reproducible.
    reloaded = XGBClassifier()
    reloaded.load_model(model_path)
    replay_probability = reloaded.predict_proba(x_test, iteration_range=iteration_range)[:, 1]
    max_abs_difference = float(np.max(np.abs(test_probability - replay_probability)))
    if not np.allclose(test_probability, replay_probability, rtol=0.0, atol=1e-7):
        raise RuntimeError(f"Reloaded prediction mismatch for {horizon}m: max abs diff={max_abs_difference}")

    booster = final_model.get_booster()
    importance_gain = booster.get_score(importance_type="gain")
    importance_weight = booster.get_score(importance_type="weight")
    importance_rows = []
    for index, feature in enumerate(features):
        key = f"f{index}"
        importance_rows.append({
            "rank_by_gain": 0,
            "feature": feature,
            "gain": float(importance_gain.get(key, 0.0)),
            "split_count": int(importance_weight.get(key, 0.0)),
        })
    importance_rows.sort(key=lambda row: (-row["gain"], row["feature"]))
    for rank, row in enumerate(importance_rows, start=1):
        row["rank_by_gain"] = rank
    importance_path = horizon_dir / "feature_importance.csv"
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)

    metrics = {
        "dataset_run_id": run_dir.name,
        "task": task,
        "horizon_minutes": horizon,
        "feature_count": len(features),
        "train": {"data": train_info},
        "selected_threshold_from_validation_f1": threshold,
        "best_iteration": best_iteration,
        "validation": {"data": val_info, "metrics_at_selected_threshold": validation_metrics},
        "test": {"data": test_info, "metrics_at_validation_selected_threshold": test_metrics},
        "majority_class_baseline_test": majority,
        "prediction_reload_verification": {
            "model_loaded_from_saved_json": True,
            "allclose_atol_1e-7": True,
            "max_absolute_probability_difference": max_abs_difference,
        },
        "dataset_hashes": dataset_hashes,
        "model_sha256": sha256_file(model_path),
        "class_weight_scale_pos_weight_train_only": scale_pos_weight,
        "training_class_counts": {"0": train_negative, "1": train_positive},
        "validation_class_counts": val_info["class_counts"],
        "test_class_counts": test_info["class_counts"],
        "feature_schema_path": "ml_ready/feature_schema.json",
        "feature_schema_sha256": sha256_file(run_dir / "ml_ready" / "feature_schema.json"),
        "model_parameters": parameters,
        "xgboost_eval_history": eval_history,
        "versions": {"python": sys.version, "platform": platform.platform(), "xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    write_json(horizon_dir / "metrics.json", metrics)
    write_json(horizon_dir / "model_config.json", parameters)
    write_json(horizon_dir / "feature_names.json", {"dataset_run_id": run_dir.name, "feature_count": len(features), "features": features})
    write_json(horizon_dir / "training_metadata.json", {
        "dataset_run_id": run_dir.name,
        "task": task,
        "horizon_minutes": horizon,
        "seed": seed,
        "train": train_info,
        "validation": val_info,
        "test": test_info,
        "dataset_hashes": dataset_hashes,
        "feature_schema_sha256": metrics["feature_schema_sha256"],
        "training_completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name,
        "test_touched_after_model_selection": True,
    })
    print(f"[{horizon}m] complete: PR-AUC={test_metrics['pr_auc_average_precision']:.6f}, ROC-AUC={test_metrics['roc_auc']:.6f}, F1={test_metrics['f1']:.6f}", flush=True)

    return {
        "horizon": horizon,
        "directory": horizon_dir,
        "metrics": metrics,
        "importance": importance_rows,
        "parameters": parameters,
        "dataset_hashes": dataset_hashes,
    }


def render_report(
    output_path: Path,
    run_dir: Path,
    run_config: dict[str, Any],
    schema: dict[str, Any],
    trained: list[dict[str, Any]],
) -> None:
    run_report_path = run_dir / "reports" / "preprocessing_report.json"
    run_report = load_json(run_report_path)
    lines = [
        "# Baseline Model Results",
        "",
        f"**Dataset run:** `{run_dir.name}`  ",
        f"**Generated UTC:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Model run directory:** `{trained[0]['directory'].parent.relative_to(BASE).as_posix()}`  ",
        "**Model:** XGBoost binary classifier, separately trained for each forecast horizon.",
        "",
        "## Dataset and safeguards",
        "",
        f"- Dataset has {schema['feature_count']} approved features and uses `{schema['row_policy']}` rows.",
        "- The six prepared task/target mappings are classification → `flare_class`, nowcasting → `flare_active`, peak detection → `flare_peak`, and forecasting → `flare_15m` / `flare_30m` / `flare_60m`. This training run uses the three forecast target Parquets only.",
        "- The source feature order is loaded from `feature_schema.json`; each split is checked against the exact ordered columns `timestamp`, `target`, then the approved feature list.",
        "- Model inputs exclude the `timestamp` and `target` columns and are exactly the 592 schema features. Learned global scaling is not used; XGBoost handles missing values natively.",
        "- `scale_pos_weight` is computed only from training class counts. Early stopping uses validation PR-AUC; the decision threshold maximizes validation F1. Test file hashes are verified as byte-integrity metadata before fitting, but test feature/target rows are loaded only after model selection and final evaluation; they are then used only to verify serialized-model prediction reproducibility.",
        "- Chronological boundaries are preserved from the preprocessing config: training before 2026-06-01, validation in June, test in July. Forecast horizon purging is already applied in the input files.",
        "- Preprocessing validation status: `" + str(run_report.get("status")) + "`.",
        "",
        "## Training configuration",
        "",
        "```json",
        json.dumps({"seed": trained[0]["parameters"]["random_state"], "shared_xgboost_parameters": {k:v for k,v in trained[0]["parameters"].items() if k != "scale_pos_weight"}, "class_weight": "negative_train_count / positive_train_count, recomputed per horizon"}, indent=2),
        "```",
        "",
        "## Dataset distributions and evaluation",
        "",
        "All performance rows below are from the untouched test split after validation-based model selection. Precision, recall, and F1 use the validation-selected threshold. ROC-AUC and PR-AUC use continuous probability scores; PR-AUC is average precision.",
        "",
        "| Horizon | Split | Rows | Negatives | Positives |", "|---|---|---:|---:|---:|",
    ]
    for result in trained:
        h = result["horizon"]
        for split in ("train", "validation", "test"):
            if split == "train":
                info = result["metrics"].get("train", {}).get("data")
                if info is None:
                    info = load_json(result["directory"] / "training_metadata.json")["train"]
            else:
                info = result["metrics"][split]["data"]
            lines.append(f"| {h} min | {split} | {info['rows']:,} | {int(info['class_counts'].get('0', 0)):,} | {int(info['class_counts'].get('1', 0)):,} |")
    lines += [
        "",
        "### Test metrics compared with majority-class baseline",
        "",
        "| Horizon | Model threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Majority PR-AUC | Majority ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in trained:
        h = result["horizon"]
        m = result["metrics"]["test"]["metrics_at_validation_selected_threshold"]
        b = result["metrics"]["majority_class_baseline_test"]
        lines.append(f"| {h} min | {m['threshold']:.6g} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc_average_precision']:.4f} | {m['false_positive_rate']:.4f} | {m['false_negative_rate']:.4f} | {b['pr_auc_average_precision']:.4f} | {b['roc_auc_constant_score']:.4f} |")
    lines += [
        "",
        "The majority classifier predicts the majority class in training (class 0) for every test row. Its precision/recall/F1, confusion matrix, FPR/FNR, ROC-AUC, and PR-AUC are saved in each horizon’s `metrics.json`. Because it assigns a constant score, ROC-AUC is 0.5; its average precision equals the test positive prevalence.",
        "",
        "### Confusion matrices",
        "",
        "Rows are true labels `[0, 1]`; columns are predicted labels `[0, 1]`.",
        "",
        "| Horizon | XGBoost `[[TN, FP], [FN, TP]]` | Majority baseline `[[TN, FP], [FN, TP]]` |",
        "|---|---|---|",
    ]
    for result in trained:
        h = result["horizon"]
        m = result["metrics"]["test"]["metrics_at_validation_selected_threshold"]["confusion_matrix"]["rows_true_cols_predicted"]
        b = result["metrics"]["majority_class_baseline_test"]["confusion_matrix"]["rows_true_cols_predicted"]
        lines.append(f"| {h} min | `{m}` | `{b}` |")
    lines += ["", "### Feature importance", "", "Gain-based importance is saved for all 592 features, including zero-importance features. Top ten features per horizon:", "", "| Horizon | Rank | Feature | Gain | Split count |", "|---|---:|---|---:|---:|"]
    for result in trained:
        for row in result["importance"][:10]:
            lines.append(f"| {result['horizon']} min | {row['rank_by_gain']} | `{row['feature']}` | {row['gain']:.6g} | {row['split_count']} |")
    lines += [
        "",
        "### Precision-recall curves and artifacts",
        "",
        "Test and validation precision-recall curve points are saved as CSV files per horizon. Test predictions are saved with UTC timestamps, actual labels, probabilities, and predicted classes. Each saved model was reloaded and produced probabilities matching the original in-memory model within absolute tolerance `1e-7`.",
        "",
        "## Limitations",
        "",
        "- The readiness report identifies five requested dates without both-instrument coverage and one malformed July 3 M2.0 GOES event that was excluded. The July test labels may therefore be incomplete near that event.",
        "- There is only one valid X-class event in the test period. These binary forecast models predict M/X onset together; results do not establish X-specific skill.",
        "- All three horizons share one July test period. This is one temporal holdout, not evidence of generalization across multiple solar cycles or independent test periods.",
        "- Minute-level positives are clustered around individual events; event-level recall, false alarms per day, and lead-time evaluation remain necessary before scientific claims.",
        "- XGBoost gains are descriptive, not causal. Feature importance should be checked for stability across additional chronological periods.",
        "- Existing artifacts under `output/model_results/` use older 424-feature datasets and are not included in these results.",
        "",
        "## Recommended next experiments",
        "",
        "1. Resolve the excluded GOES event from an authoritative catalogue and rebuild a new versioned dataset before final July test reporting.",
        "2. Verify archive availability for the five missing dates and preserve the download/source provenance.",
        "3. Add a second or rolling-origin temporal evaluation period while keeping the current July test results as a single holdout reference.",
        "4. Report event-level detection, false alarms per day, lead time, and bootstrap uncertainty by event; do not claim X-specific performance from one X event.",
        "5. Compare class weighting against unweighted XGBoost and a simple regularized logistic model using the same chronological splits and validation-only selection.",
        "",
        "## Reproducibility and artifact mapping",
        "",
        f"- Dataset run ID: `{run_dir.name}`.",
        f"- Feature schema SHA-256: `{sha256_file(run_dir / 'ml_ready' / 'feature_schema.json')}`.",
        "- Each horizon’s `training_metadata.json` contains train/validation/test file hashes, class counts, time ranges, and the dataset run ID.",
        "- Each horizon directory contains the XGBoost JSON model, model config, ordered feature names, metrics and confusion matrix, PR curves, test predictions, and complete feature importance.",
        "- `artifact_manifest.json` records SHA-256 hashes for every saved artifact and this report.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    run_dir = (BASE / "output" / "preprocessing_runs" / args.dataset_run).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Preprocessing run not found: {run_dir}")
    features, schema, context = validate_feature_schema(run_dir)
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (BASE / args.output_dir).resolve()
    report_path = args.report.resolve() if args.report.is_absolute() else (BASE / args.report).resolve()
    if output_dir.exists() and getattr(args, "finalize_existing", False):
        return finalize_existing(run_dir, output_dir, report_path, features, schema, context)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing model run directory: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results report: {report_path}")

    output_records = context["output_records"]
    for horizon in HORIZONS:
        for split in ("train", "validation", "test"):
            relative = f"ml_ready/forecast_{horizon}m_{split}.parquet"
            verify_file_hash(run_dir / relative, output_records[relative]["sha256"])
    output_dir.mkdir(parents=True, exist_ok=False)

    shared = {
        "dataset_run_id": run_dir.name,
        "dataset_run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "feature_schema_sha256": sha256_file(run_dir / "ml_ready" / "feature_schema.json"),
        "seed": args.seed,
        "horizons_minutes": list(HORIZONS),
        "feature_count": len(features),
        "feature_order": features,
        "row_policy": schema["row_policy"],
        "split_boundaries": context["config"]["splits"],
        "xgboost_version": xgb.__version__,
        "environment_packages": {
            dist.metadata.get("Name", "unknown"): dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        },
        "parameters_without_horizon_class_weight": {k:v for k,v in DEFAULT_CONFIG.items()},
    }
    write_json(output_dir / "training_run_config.json", shared)
    write_json(output_dir / "training_environment.json", shared["environment_packages"])

    trained = []
    for horizon in HORIZONS:
        trained.append(train_one_horizon(
            horizon=horizon,
            run_dir=run_dir,
            output_dir=output_dir,
            features=features,
            run_config=context["config"],
            source_records=output_records,
            seed=args.seed,
        ))

    summary_rows = []
    for result in trained:
        horizon = result["horizon"]
        metrics = result["metrics"]["test"]["metrics_at_validation_selected_threshold"]
        baseline = result["metrics"]["majority_class_baseline_test"]
        summary_rows.append({
            "horizon_minutes": horizon,
            "threshold_from_validation": metrics["threshold"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc_average_precision": metrics["pr_auc_average_precision"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
            "majority_pr_auc": baseline["pr_auc_average_precision"],
            "majority_roc_auc": baseline["roc_auc_constant_score"],
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "test_metrics_summary.csv", index=False)
    render_report(report_path, run_dir, context["config"], schema, trained)

    artifact_paths = [p for p in output_dir.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    artifact_paths.append(report_path)
    artifacts = []
    for path in sorted(artifact_paths):
        try:
            relative = path.relative_to(output_dir).as_posix()
        except ValueError:
            relative = path.relative_to(BASE).as_posix()
        artifacts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output_dir / "artifact_manifest.json", {
        "model_run_id": output_dir.name,
        "dataset_run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    })
    return output_dir


def finalize_existing(
    run_dir: Path,
    output_dir: Path,
    report_path: Path,
    features: list[str],
    schema: dict[str, Any],
    context: dict[str, Any],
) -> Path:
    """Finish reporting for an already completed set of horizon fits."""
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results report: {report_path}")
    if (output_dir / "artifact_manifest.json").exists():
        raise FileExistsError(f"Artifact manifest already exists: {output_dir / 'artifact_manifest.json'}")

    trained = []
    for horizon in HORIZONS:
        horizon_dir = output_dir / f"forecast_{horizon}m"
        metrics = load_json(horizon_dir / "metrics.json")
        metadata = load_json(horizon_dir / "training_metadata.json")
        if metrics.get("dataset_run_id") != run_dir.name or metadata.get("dataset_run_id") != run_dir.name:
            raise ValueError(f"Dataset run ID mismatch in completed artifacts for {horizon}m")
        metrics["train"] = {"data": metadata["train"]}
        write_json(horizon_dir / "metrics.json", metrics)
        model_path = horizon_dir / f"xgboost_forecast_{horizon}m.json"
        if sha256_file(model_path) != metrics.get("model_sha256"):
            raise ValueError(f"Saved model hash mismatch for {horizon}m")
        importance = pd.read_csv(horizon_dir / "feature_importance.csv").to_dict(orient="records")
        if len(importance) != len(features):
            raise ValueError(f"Feature importance row count mismatch for {horizon}m")
        trained.append({
            "horizon": horizon,
            "directory": horizon_dir,
            "metrics": metrics,
            "importance": importance,
            "parameters": load_json(horizon_dir / "model_config.json"),
            "dataset_hashes": metrics["dataset_hashes"],
        })

    summary_rows = []
    for result in trained:
        m = result["metrics"]["test"]["metrics_at_validation_selected_threshold"]
        b = result["metrics"]["majority_class_baseline_test"]
        summary_rows.append({
            "horizon_minutes": result["horizon"],
            "threshold_from_validation": m["threshold"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "pr_auc_average_precision": m["pr_auc_average_precision"],
            "false_positive_rate": m["false_positive_rate"], "false_negative_rate": m["false_negative_rate"],
            "majority_pr_auc": b["pr_auc_average_precision"], "majority_roc_auc": b["roc_auc_constant_score"],
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "test_metrics_summary.csv", index=False)
    render_report(report_path, run_dir, context["config"], schema, trained)

    artifact_paths = [p for p in output_dir.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    artifact_paths.append(report_path)
    artifacts = []
    for path in sorted(artifact_paths):
        try:
            relative = path.relative_to(output_dir).as_posix()
        except ValueError:
            relative = path.relative_to(BASE).as_posix()
        artifacts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output_dir / "artifact_manifest.json", {
        "model_run_id": output_dir.name,
        "dataset_run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    })
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", default=DEFAULT_DATA_RUN)
    parser.add_argument("--output-dir", type=Path, default=Path("output/model_results/xgboost_baselines_preproc_20261124_v1"))
    parser.add_argument("--report", type=Path, default=Path("BASELINE_MODEL_RESULTS.md"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--finalize-existing", action="store_true", help="Build report and artifact manifest from completed horizon artifacts; does not retrain")
    args = parser.parse_args()
    path = run(args)
    print(f"Training complete. Artifacts: {path}")
    print(f"Results report: {(BASE / args.report).resolve()}")


if __name__ == "__main__":
    main()
