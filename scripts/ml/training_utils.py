"""
Shared utilities for the ARUN-X training scripts.

Used by:
    train_baseline_models.py
    train_forecasting_models.py   (XGBoost)
    train_deep_learning.py        (CNN / CNN-LSTM)

IMPORTANT: this module must be imported BEFORE numpy / sklearn / torch in each
script, because it caps BLAS/OpenMP thread counts through environment variables
(these are only read when the libraries are first loaded).

Set ARUNX_THREADS to change the CPU thread cap (default 4 for an 8 GB laptop).

Everything that touches data goes through load_split(), so every model family
sees exactly the same rows, the same feature order and the same cleaning, and
every model is scored by exactly the same compute_metrics().
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("ARUNX_THREADS", "4")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, _THREADS)

THREADS = max(1, int(os.environ["OMP_NUM_THREADS"]))

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "output" / "ml_ready_minute_end_v2"
DEFAULT_MODELS_ROOT = ROOT / "models"
DEFAULT_RESULTS_ROOT = ROOT / "output" / "model_results"

HORIZONS = (15, 30, 60)
SEED = 42
NON_FEATURE_COLUMNS = ("timestamp", "target")
FAMILIES = ("baselines", "forecasting", "deep_learning")


def add_path_args(parser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)


def split_path(data_dir: Path, horizon: int, split: str) -> Path:
    path = Path(data_dir) / f"forecast_{horizon}m_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


# ============================================================
# REPRODUCIBILITY / MEMORY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ============================================================
# DATA LOADING (memory-lean, column-pruned)
# ============================================================

def get_feature_names(data_dir: Path, horizon: int = 15, expected: int | None = None):
    """Feature list in a fixed order. Prefers feature_schema.json, otherwise
    uses every Parquet column except timestamp/target (the old baseline rule)."""
    import pyarrow.parquet as pq

    columns = pq.read_schema(split_path(data_dir, horizon, "train")).names
    schema_path = Path(data_dir) / "feature_schema.json"

    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as f:
            features = list(json.load(f)["features"])
        missing = [c for c in features if c not in columns]
        if missing:
            raise ValueError(f"Schema features missing from Parquet: {missing[:10]}")
        known = set(features) | set(NON_FEATURE_COLUMNS)
        extras = [c for c in columns if c not in known]
        if extras:
            print(
                f"WARNING: {len(extras)} Parquet column(s) are not in "
                f"feature_schema.json and will NOT be used: {extras[:5]}..."
            )
    else:
        features = [c for c in columns if c not in NON_FEATURE_COLUMNS]

    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names.")
    if expected is not None and len(features) != expected:
        raise ValueError(f"Expected {expected} features, found {len(features)}")
    return features


def load_split(path: Path, features: list[str], batch_rows: int = 65_536):
    """
    Read only timestamp + target + the requested features and stream them into
    ONE preallocated float32 array (no pandas copy, no float64 intermediate).

    Returns (timestamps: pd.Series[UTC], X: float32 [n, F], y: int8 [n]).
    +/-inf become NaN. Rows with a null timestamp/target are dropped; if the
    split is not already strictly chronological it is sorted and exact duplicate
    timestamps are removed (keep last). Already-clean splits are never copied.
    The Parquet files themselves are never modified.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    X = np.empty((n_rows, len(features)), dtype=np.float32)
    ts_parts, y_parts = [], []
    row = 0

    for batch in pf.iter_batches(
        batch_size=batch_rows, columns=["timestamp", "target", *features]
    ):
        names = batch.schema.names
        index = {name: i for i, name in enumerate(names)}
        m = batch.num_rows
        ts_parts.append(batch.column(index["timestamp"]).to_pandas())
        y_parts.append(
            np.asarray(
                batch.column(index["target"]).to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
        )
        block = X[row : row + m]
        for j, name in enumerate(features):
            block[:, j] = batch.column(index[name]).to_numpy(zero_copy_only=False)
        block[~np.isfinite(block)] = np.nan  # inf -> NaN, in place
        row += m
        del batch

    ts = pd.to_datetime(
        pd.concat(ts_parts, ignore_index=True), utc=True, errors="coerce"
    )
    y_raw = np.concatenate(y_parts)
    del ts_parts, y_parts

    valid = ts.notna().to_numpy() & np.isfinite(y_raw)
    clean = bool(valid.all()) and ts.is_monotonic_increasing and ts.is_unique
    if not clean:
        keep = np.flatnonzero(valid)
        order = keep[np.argsort(ts.to_numpy()[keep], kind="stable")]
        dup = ts.iloc[order].duplicated(keep="last").to_numpy()
        order = order[~dup]
        print(
            f"  {Path(path).name}: cleaning {n_rows - len(order):,} of {n_rows:,} rows "
            f"(null/duplicate/unsorted timestamps or null target)"
        )
        X = X[order]
        ts = ts.iloc[order].reset_index(drop=True)
        y_raw = y_raw[order]

    if not np.isin(y_raw, (0.0, 1.0)).all():
        raise ValueError(f"{Path(path).name}: target is not binary (0/1).")
    return ts, X, y_raw.astype(np.int8)


def check_chronology(ts_train, ts_val, ts_test) -> None:
    """Leakage guard only; it never changes the splits."""
    if not (ts_train.max() < ts_val.min() and ts_val.max() < ts_test.min()):
        raise ValueError(
            "Splits are not strictly chronological "
            f"(train<={ts_train.max()}, val {ts_val.min()}..{ts_val.max()}, "
            f"test>={ts_test.min()})."
        )


def require_both_classes(name: str, y: np.ndarray, strict: bool = True) -> None:
    if len(np.unique(y)) < 2:
        msg = f"{name} split contains only one class; metrics/threshold are undefined."
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)


def describe_split(name: str, ts, X, y) -> None:
    print(
        f"  {name:<10} rows={len(y):>9,}  positives={int(y.sum()):>7,} "
        f"({y.mean():.3%})  {ts.min()} -> {ts.max()}"
    )


# ============================================================
# PREPROCESSING (fit on TRAIN only; applied in place)
# ============================================================

def fit_preprocessing(X_train: np.ndarray) -> dict:
    """Per-column median (ignoring NaN) and mean/std of the median-imputed
    column. Column-by-column so no full-size temporary copy is made."""
    n, n_features = X_train.shape
    medians = np.zeros(n_features, dtype=np.float32)
    means = np.zeros(n_features, dtype=np.float32)
    stds = np.ones(n_features, dtype=np.float32)

    for j in range(n_features):
        col = X_train[:, j]
        finite = col[np.isfinite(col)].astype(np.float64)
        median = float(np.median(finite)) if finite.size else 0.0
        n_missing = n - finite.size
        mean = (finite.sum() + n_missing * median) / n
        var = (((finite - mean) ** 2).sum() + n_missing * (median - mean) ** 2) / n
        std = float(np.sqrt(var))
        medians[j] = median
        means[j] = mean
        stds[j] = std if std >= 1e-6 else 1.0
    return {"medians": medians, "means": means, "stds": stds}


def impute_inplace(X: np.ndarray, medians: np.ndarray, chunk: int = 100_000) -> None:
    for start in range(0, len(X), chunk):
        block = X[start : start + chunk]
        rows, cols = np.nonzero(np.isnan(block))
        if rows.size:
            block[rows, cols] = medians[cols]


def standardize_inplace(
    X: np.ndarray, means, stds, clip: float = np.inf, chunk: int = 100_000
) -> None:
    """Expects an already-imputed array."""
    for start in range(0, len(X), chunk):
        block = X[start : start + chunk]
        block -= means
        block /= stds
        if np.isfinite(clip):
            np.clip(block, -clip, clip, out=block)


def save_preprocessing(path: Path, prep: dict, clip: float = np.inf) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        medians=prep["medians"],
        means=prep["means"],
        stds=prep["stds"],
        clip=np.float32(clip),
    )


# ============================================================
# THRESHOLD + METRICS (identical for every model)
# ============================================================

def select_threshold(y_val: np.ndarray, probabilities: np.ndarray) -> float:
    """F1-optimal threshold from precision_recall_curve on VALIDATION data only."""
    p = np.asarray(probabilities, dtype=np.float64)
    if int(np.sum(y_val)) == 0:
        print("WARNING: validation has no positives; using threshold 0.5")
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_val, p)
    if len(thresholds) == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true, probabilities, threshold: float):
    y = np.asarray(y_true).astype(np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite probabilities produced by the model.")
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "samples": int(len(y)),
        "positive_samples": int(y.sum()),
    }
    return metrics, pred


def print_metrics(title: str, m: dict) -> None:
    def f(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(
        f"  {title:<11} thr={m['threshold']:.4f} acc={f(m['accuracy'])} "
        f"prec={f(m['precision'])} rec={f(m['recall'])} f1={f(m['f1'])} "
        f"PR-AUC={f(m['pr_auc'])} ROC-AUC={f(m['roc_auc'])}"
    )
    print(
        f"  {'':<11} confusion [[TN FP] [FN TP]] = "
        f"[[{m['true_negative']} {m['false_positive']}] "
        f"[{m['false_negative']} {m['true_positive']}]]"
    )


# ============================================================
# OUTPUT HELPERS
# ============================================================

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_predictions(path: Path, timestamps, y_true, probabilities, predicted) -> None:
    """Columns match what scripts/ml/evaluate_event_level.py expects. The file
    name MUST be '<model>_<15m|30m|60m>_predictions.csv' for that script."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.Series(timestamps).reset_index(drop=True),
            "actual": np.asarray(y_true),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "predicted": np.asarray(predicted),
        }
    ).to_csv(path, index=False)


def metrics_row(family: str, model: str, horizon: int, metrics: dict) -> dict:
    return {"family": family, "model": model, "horizon": f"{horizon}m", **metrics}


def update_metrics_table(path: Path, rows: list[dict]) -> pd.DataFrame:
    """Add/replace rows keyed by (model, horizon) so partial re-runs
    (e.g. --horizons 15) never wipe earlier results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists() and len(new):
        old = pd.read_csv(path)
        replaced = set(zip(new["model"], new["horizon"]))
        keep = [(m, h) not in replaced for m, h in zip(old["model"], old["horizon"])]
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def build_comparisons(results_root: Path) -> None:
    """Merge every family's tables into results_root/comparisons/."""
    results_root = Path(results_root)
    out_dir = results_root / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = ["family", "model", "horizon", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    for split in ("validation", "test"):
        frames = [
            pd.read_csv(results_root / fam / f"all_{split}_metrics.csv")
            for fam in FAMILIES
            if (results_root / fam / f"all_{split}_metrics.csv").exists()
        ]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True).sort_values(
            ["horizon", "pr_auc"], ascending=[True, False]
        )
        path = out_dir / f"all_models_{split}_metrics.csv"
        df.to_csv(path, index=False)
        print(f"\n{split.upper()} COMPARISON (all model families) -> {path}")
        print(df[shown].to_string(index=False, float_format=lambda x: f"{x:.4f}"))