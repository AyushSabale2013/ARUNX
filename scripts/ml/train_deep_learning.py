"""
ARUN-X PyTorch deep-learning trainer for M/X flare forecasting (upgraded).

Reads the existing chronological split files (never modified):
  output/ml_ready_minute_end_v2/forecast_{15m,30m,60m}_{train,validation,test}.parquet

Each file contains timestamp, target, and the 424 features listed in
feature_schema.json. Only complete 60-row windows with exactly one-minute
timestamp spacing are used, and windows are built INSIDE each split, so a
sequence can never cross a train/validation/test boundary.

Scientific guarantees (unchanged from the original script):
  * Imputation/scaling statistics are fitted on TRAIN rows only.
  * Validation PR-AUC is the only early-stopping / checkpoint-selection metric.
  * The F1 threshold is selected on VALIDATION only.
  * The TEST split is evaluated exactly once, with the best validation checkpoint.
  * No random splitting; target definition and labels are untouched.

Experiments
  cnn         original CNN (global AVERAGE pooling)           [baseline]
  cnn_lstm    original CNN-LSTM                               [baseline]
  cnn_avgmax  CNN with concatenated global average+max pooling [new experiment]
  --model both = cnn + cnn_lstm ; --model all = cnn + cnn_avgmax + cnn_lstm

Examples:
  python scripts/ml/train_deep_learning.py --model cnn_lstm
  python scripts/ml/train_deep_learning.py --model both --horizons 15
  python scripts/ml/train_deep_learning.py --model all --deterministic --torch-threads 4
  python scripts/ml/train_deep_learning.py --model cnn --learning-rate 5e-4 --tag lr5e-4

Outputs go to output/model_results/deep_learning/<model>[_<tag>]/forecast_<H>m/.
Use --tag whenever hyper-parameters differ from the defaults so baseline
results are not overwritten.

Thread limits: --torch-threads / --interop-threads, or the ARUNX_THREADS
environment variable (also caps OpenMP/BLAS threads, read at import time).
"""
from __future__ import annotations

import os

# Cap BLAS/OpenMP threads BEFORE numpy/torch are imported (only if requested).
_THREADS_ENV = os.environ.get("ARUNX_THREADS")
if _THREADS_ENV:
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_var, _THREADS_ENV)

import argparse
import gc
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "output" / "ml_ready_minute_end_v2"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "model_results" / "deep_learning"

HORIZONS = (15, 30, 60)
EXPECTED_FEATURES = 424
ONE_MINUTE_NS = 60 * 1_000_000_000
MINUTES_PER_DAY = 1440
SPLITS = ("train", "validation", "test")

MODEL_CHOICES = ("cnn", "cnn_avgmax", "cnn_lstm", "both", "all")
MODEL_GROUPS = {
    "cnn": ["cnn"],
    "cnn_avgmax": ["cnn_avgmax"],
    "cnn_lstm": ["cnn_lstm"],
    "both": ["cnn", "cnn_lstm"],
    "all": ["cnn", "cnn_avgmax", "cnn_lstm"],
}


# ============================================================
# SMALL UTILITIES (reproducibility, atomic writes, hashing)
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=_json_default) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    atomic_write_text(path, df.to_csv(index=False))


def atomic_torch_save(obj, path: Path) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def plain(obj):
    """JSON round-trip so checkpoints hold only plain Python types
    (lets torch.load(weights_only=True) work)."""
    return json.loads(json.dumps(obj, default=_json_default))


# ============================================================
# RUNTIME CONFIGURATION (threads, determinism) + ENVIRONMENT METADATA
# ============================================================

def configure_runtime(args) -> dict:
    if args.deterministic:
        # Must be set before the first CUDA/cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.interop_threads is not None:
        try:
            torch.set_num_interop_threads(args.interop_threads)
        except RuntimeError as exc:  # raised if parallel work already started
            print(f"WARNING: could not set inter-op threads: {exc}")

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_strict": bool(args.deterministic),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "env": {
            k: os.environ.get(k)
            for k in (
                "ARUNX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def collect_environment(device: torch.device, runtime: dict) -> dict:
    cuda_ok = torch.cuda.is_available()
    try:
        cudnn_version = torch.backends.cudnn.version() if cuda_ok else None
    except Exception:
        cudnn_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda_build_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_available": cuda_ok,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_count": torch.cuda.device_count() if cuda_ok else 0,
        "git_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "runtime": runtime,
    }


# ============================================================
# DATA LOADING, VALIDATION, CLEANING
# ============================================================

def load_feature_names(data_dir: Path, expected: int) -> tuple[list[str], str]:
    schema_path = data_dir / "feature_schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    features = list(schema["features"])
    if len(features) != expected:
        raise ValueError(f"Expected {expected} features in schema, found {len(features)}")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names in feature_schema.json")
    return features, sha256_file(schema_path)


def validate_target(series: pd.Series, name: str, allow_null_drop: bool):
    """Validate the ORIGINAL target column before any int8 conversion.

    Rejects: non-numeric dtypes, +/-inf, and any value other than 0/1.
    Null targets are rejected unless allow_null_drop is set (then reported).
    Returns (float64 values, boolean null mask).
    """
    dtype = series.dtype
    numeric = (
        pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_float_dtype(dtype)
    )
    if not numeric:
        raise ValueError(f"{name}: target dtype '{dtype}' is not numeric/boolean.")

    values = series.to_numpy(dtype=np.float64, na_value=np.nan)

    inf_mask = np.isinf(values)
    if inf_mask.any():
        raise ValueError(f"{name}: target contains {int(inf_mask.sum())} infinite value(s).")

    null_mask = np.isnan(values)
    if null_mask.any() and not allow_null_drop:
        raise ValueError(
            f"{name}: target contains {int(null_mask.sum())} null/NaN value(s). "
            "Prepared split files must have known labels (use --drop-null-target-rows "
            "to drop and report them instead)."
        )

    non_binary = ~null_mask & ~((values == 0.0) | (values == 1.0))
    if non_binary.any():
        bad = sorted(set(values[non_binary].tolist()))[:10]
        raise ValueError(
            f"{name}: target is not binary; {int(non_binary.sum())} invalid value(s), e.g. {bad}"
        )
    return values, null_mask


def timestamps_ns(ts: pd.Series) -> np.ndarray:
    return ts.dt.as_unit("ns").astype("int64").to_numpy()


def _count_inf_and_nan_inplace(x: np.ndarray, chunk: int = 100_000) -> tuple[int, int]:
    """Replace +/-inf with NaN in place; return (inf_replaced, total_nan_after)."""
    inf_count = nan_count = 0
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        bad_inf = np.isinf(block)
        n_inf = int(bad_inf.sum())
        if n_inf:
            block[bad_inf] = np.nan
            inf_count += n_inf
        nan_count += int(np.isnan(block).sum())
    return inf_count, nan_count


def load_split(path: Path, feature_names: list[str], allow_null_target_drop: bool):
    """Load one split file. Does the same cleaning as the original script
    (drop null/invalid timestamps, sort, drop duplicate timestamps keep-last)
    but VALIDATES the target first and REPORTS every removal.

    Returns (timestamps UTC Series, float32 features [n, F], int8 labels, report).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path, columns=["timestamp", "target", *feature_names])
    n_initial = len(df)
    if n_initial == 0:
        raise ValueError(f"{path.name}: file is empty.")

    # 1) Validate ORIGINAL target values before any dtype conversion.
    target, null_target = validate_target(df["target"], path.name, allow_null_target_drop)

    # 2) Timestamp parsing + cleaning.
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = ts.isna().to_numpy()
    valid = ~bad_ts & ~null_target
    keep = np.flatnonzero(valid)
    if keep.size == 0:
        raise ValueError(f"{path.name}: no rows with a valid timestamp and target.")

    keep_ns = timestamps_ns(ts.iloc[keep])
    rows_out_of_order = int(np.sum(np.diff(keep_ns) < 0))
    sort_idx = np.argsort(keep_ns, kind="stable")  # stable: "keep last" is deterministic
    sorted_ns = keep_ns[sort_idx]
    last_of_run = np.ones(len(sorted_ns), dtype=bool)
    last_of_run[:-1] = sorted_ns[1:] != sorted_ns[:-1]
    order = keep[sort_idx[last_of_run]]
    duplicates_removed = int(keep.size - order.size)

    # 3) Extract features / labels.
    x = df[feature_names].to_numpy(dtype=np.float32, copy=True)
    ts_clean = ts.iloc[order].reset_index(drop=True)
    y = target[order].astype(np.int8)  # safe: values verified to be exactly 0/1
    del df, ts, target
    if not (order.size == n_initial and np.array_equal(order, np.arange(n_initial))):
        x = x[order]
    inf_replaced, missing_values = _count_inf_and_nan_inplace(x)

    report = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "mtime_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        "rows_initial": int(n_initial),
        "rows_final": int(len(y)),
        "rows_removed_total": int(n_initial - len(y)),
        "removed_invalid_or_null_timestamp": int(bad_ts.sum()),
        "removed_null_target": int(null_target.sum()),
        "removed_invalid_rows_union": int(n_initial - keep.size),
        "removed_duplicate_timestamps": duplicates_removed,
        "rows_out_of_order_before_sort": rows_out_of_order,
        "inf_feature_values_replaced_with_nan": int(inf_replaced),
        "missing_feature_values_before_imputation": int(missing_values),
        "first_timestamp": ts_clean.iloc[0].isoformat(),
        "last_timestamp": ts_clean.iloc[-1].isoformat(),
    }
    print(
        f"  {path.name}: rows {n_initial:,} -> {len(y):,} "
        f"(removed: invalid_ts={report['removed_invalid_or_null_timestamp']:,}, "
        f"null_target={report['removed_null_target']:,}, "
        f"duplicate_ts={duplicates_removed:,}; out_of_order_steps={rows_out_of_order:,})"
    )
    return ts_clean, x, y, report


def validate_splits(ts_ns: dict, horizon: int, strict_gap: bool) -> dict:
    """Chronological order + timestamp-overlap checks across the three splits.
    Only inspects timestamps; never changes any split."""
    for name in SPLITS:
        arr = ts_ns[name]
        if len(arr) == 0:
            raise ValueError(f"{name} split is empty after cleaning.")
        if not np.all(np.diff(arr) > 0):
            raise ValueError(f"{name} split is not strictly increasing after cleaning.")

    tr, va, te = (ts_ns[s] for s in SPLITS)
    if not (tr[-1] < va[0] and va[-1] < te[0]):
        raise ValueError(
            "Splits are not strictly chronological: "
            f"train<= {pd.Timestamp(tr[-1], tz='UTC')}, "
            f"validation {pd.Timestamp(va[0], tz='UTC')}..{pd.Timestamp(va[-1], tz='UTC')}, "
            f"test>= {pd.Timestamp(te[0], tz='UTC')}"
        )
    for a, b, label in ((tr, va, "train/validation"), (tr, te, "train/test"), (va, te, "validation/test")):
        n_overlap = int(np.intersect1d(a, b).size)
        if n_overlap:
            raise ValueError(f"{n_overlap} overlapping timestamps between {label}.")

    gaps = {
        "train_to_validation_minutes": float((va[0] - tr[-1]) / ONE_MINUTE_NS),
        "validation_to_test_minutes": float((te[0] - va[-1]) / ONE_MINUTE_NS),
    }
    for key, gap in gaps.items():
        if gap < horizon:
            msg = (
                f"{key}={gap:.0f} min is shorter than the {horizon}-min forecast horizon: "
                "labels at the end of the earlier split look ahead into the next split."
            )
            if strict_gap:
                raise ValueError(msg)
            print("  WARNING:", msg)
    return {"strictly_chronological": True, "timestamp_overlaps": 0, **gaps}


def valid_sequence_endpoints(ts_ns: np.ndarray, seq_len: int) -> np.ndarray:
    """Endpoint row indices whose full seq_len window is minute-contiguous."""
    if len(ts_ns) < seq_len:
        return np.empty(0, dtype=np.int64)
    consecutive = np.diff(ts_ns) == ONE_MINUTE_NS
    bad_prefix = np.concatenate(([0], np.cumsum(~consecutive, dtype=np.int64)))
    endpoints = np.arange(seq_len - 1, len(ts_ns), dtype=np.int64)
    starts = endpoints - seq_len + 1
    return endpoints[(bad_prefix[endpoints] - bad_prefix[starts]) == 0]


def verify_sequence_endpoints(ts_ns: np.ndarray, endpoints: np.ndarray, seq_len: int, name: str,
                              chunk: int = 50_000) -> None:
    """Independent brute-force check: every window is inside the split and has
    exactly one-minute steps, and NO valid window was missed."""
    n = len(ts_ns)
    if endpoints.size:
        if endpoints.min() < seq_len - 1 or endpoints.max() >= n:
            raise AssertionError(f"{name}: sequence window leaves the split boundaries.")
        if np.any(np.diff(endpoints) <= 0):
            raise AssertionError(f"{name}: endpoints are not strictly increasing.")
    if n < seq_len:
        if endpoints.size:
            raise AssertionError(f"{name}: endpoints exist although split is shorter than a window.")
        return
    diffs = np.diff(ts_ns)
    windows = np.lib.stride_tricks.sliding_window_view(diffs, seq_len - 1)  # one row per start
    brute = []
    for s in range(0, windows.shape[0], chunk):
        ok = (windows[s:s + chunk] == ONE_MINUTE_NS).all(axis=1)
        brute.append(np.flatnonzero(ok) + s + seq_len - 1)
    brute = np.concatenate(brute) if brute else np.empty(0, dtype=np.int64)
    if not np.array_equal(brute, endpoints):
        raise AssertionError(
            f"{name}: sequence endpoints disagree with brute-force contiguity check "
            f"({len(endpoints)} vs {len(brute)})."
        )


# ============================================================
# PREPROCESSING (fit on TRAIN only; transform in place)
# ============================================================

def fit_scaler(x_train: np.ndarray) -> dict:
    """Per-column median (ignoring NaN) and mean/std of the median-imputed
    column. Identical statistics to the original, computed column by column
    so no full-size temporary copy is created."""
    n, n_features = x_train.shape
    medians = np.zeros(n_features, dtype=np.float32)
    means = np.zeros(n_features, dtype=np.float32)
    stds = np.ones(n_features, dtype=np.float32)
    for j in range(n_features):
        col = x_train[:, j]
        finite = col[np.isfinite(col)].astype(np.float64)
        median = float(np.median(finite)) if finite.size else 0.0
        n_missing = n - finite.size
        mean = (finite.sum() + n_missing * median) / n
        var = (((finite - mean) ** 2).sum() + n_missing * (median - mean) ** 2) / n
        std = float(np.sqrt(var))
        medians[j], means[j] = median, mean
        stds[j] = std if std >= 1e-6 else 1.0
    return {"medians": medians, "means": means, "stds": stds}


def transform_inplace(x: np.ndarray, prep: dict, clip: float = 10.0, chunk: int = 100_000) -> None:
    """Median-impute, standardize and clip IN PLACE (same maths as the original).
    The raw (unscaled) features therefore never coexist with the scaled copy."""
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        rows, cols = np.nonzero(~np.isfinite(block))
        if rows.size:
            block[rows, cols] = prep["medians"][cols]
        block -= prep["means"]
        block /= prep["stds"]
        np.clip(block, -clip, clip, out=block)


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, endpoints: np.ndarray, seq_len: int):
        if len(x) != len(y):
            raise ValueError("Feature/label length mismatch.")
        if endpoints.size and (endpoints.min() < seq_len - 1 or endpoints.max() >= len(x)):
            raise ValueError("Endpoints would create windows crossing the split boundary.")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # shares memory
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.endpoints = endpoints
        self.seq_len = seq_len

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = int(self.endpoints[index])
        start = end - self.seq_len + 1
        # [time, features] = [seq_len, n_features]
        return self.x[start:end + 1], self.y[end], end


def make_loader(dataset, shuffle: bool, device, batch_size: int, seed: int):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)  # reproducible shuffling independent of other RNG use
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# ============================================================
# MODELS
# ============================================================

class CNNModel(nn.Module):
    """Original CNN (pooling='avg') or avg+max variant (pooling='avgmax').

    Note on the baseline: global AVERAGE pooling collapses the 60-minute axis
    into a single mean per channel. A short, sharp precursor (e.g. a brightening
    spike) is diluted by the many quiet minutes and temporal order is lost.
    Global MAX pooling keeps the strongest activation of each filter, so the
    'avgmax' variant concatenates both (2 * conv2_channels features) at the
    cost of a slightly larger first head layer. Both are kept as experiments.
    """

    def __init__(self, n_features: int, conv1: int, conv2: int, kernel_size: int,
                 dropout: float, head_hidden: int, head_dropout: float, pooling: str = "avg"):
        super().__init__()
        if pooling not in ("avg", "avgmax"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(n_features, conv1, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(conv1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv1, conv2, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(conv2),
            nn.GELU(),
        )
        pooled = conv2 * (2 if pooling == "avgmax" else 1)
        self.head = nn.Sequential(
            nn.Linear(pooled, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        z = self.features(x.transpose(1, 2))  # [B, C, T]
        pooled = z.mean(dim=2)  # identical to AdaptiveAvgPool1d(1)+Flatten
        if self.pooling == "avgmax":
            pooled = torch.cat([pooled, z.amax(dim=2)], dim=1)
        return self.head(pooled).squeeze(1)


class CNNLSTMModel(nn.Module):
    def __init__(self, n_features: int, conv1: int, conv2: int, kernel1: int, kernel2: int,
                 conv_dropout: float, lstm_hidden: int, lstm_layers: int, lstm_dropout: float,
                 head_hidden: int, head_dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv1, kernel_size=kernel1, padding=kernel1 // 2),
            nn.BatchNorm1d(conv1),
            nn.GELU(),
            nn.Dropout(conv_dropout),
            nn.Conv1d(conv1, conv2, kernel_size=kernel2, padding=kernel2 // 2),
            nn.BatchNorm1d(conv2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(z)
        return self.head(sequence[:, -1, :]).squeeze(1)


def model_config(args, model_name: str) -> dict:
    if model_name in ("cnn", "cnn_avgmax"):
        return {
            "conv1_channels": args.conv1_channels,
            "conv2_channels": args.conv2_channels,
            "kernel_size": args.cnn_kernel_size,
            "dropout": args.cnn_dropout,
            "head_hidden": args.head_hidden,
            "head_dropout": args.cnn_head_dropout,
            "pooling": "avgmax" if model_name == "cnn_avgmax" else "avg",
        }
    return {
        "conv1_channels": args.conv1_channels,
        "conv2_channels": args.conv2_channels,
        "kernel1": args.lstm_conv_kernel1,
        "kernel2": args.lstm_conv_kernel2,
        "conv_dropout": args.lstm_conv_dropout,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "lstm_dropout": args.lstm_dropout,
        "head_hidden": args.head_hidden,
        "head_dropout": args.lstm_head_dropout,
    }


def build_model(model_name: str, n_features: int, cfg: dict) -> nn.Module:
    if model_name in ("cnn", "cnn_avgmax"):
        return CNNModel(
            n_features, cfg["conv1_channels"], cfg["conv2_channels"], cfg["kernel_size"],
            cfg["dropout"], cfg["head_hidden"], cfg["head_dropout"], cfg["pooling"],
        )
    return CNNLSTMModel(
        n_features, cfg["conv1_channels"], cfg["conv2_channels"], cfg["kernel1"], cfg["kernel2"],
        cfg["conv_dropout"], cfg["lstm_hidden"], cfg["lstm_layers"], cfg["lstm_dropout"],
        cfg["head_hidden"], cfg["head_dropout"],
    )


# ============================================================
# PREDICTION, LOSS, THRESHOLD, METRICS
# ============================================================

@torch.no_grad()
def predict(model, loader, device):
    """Returns (labels int8, logits float64, endpoint row indices)."""
    model.eval()
    logits_all, y_all, end_all = [], [], []
    for xb, yb, eb in loader:
        logits = model(xb.to(device, non_blocking=True))
        logits_all.append(logits.detach().double().cpu().numpy())
        y_all.append(yb.numpy())
        end_all.append(eb.numpy())
    logits = np.concatenate(logits_all)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Model produced non-finite logits.")
    return (
        np.concatenate(y_all).astype(np.int8),
        logits,
        np.concatenate(end_all).astype(np.int64),
    )


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def unweighted_bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean UNWEIGHTED binary cross-entropy computed directly from logits
    (numerically stable). Comparable across epochs/models, unlike the
    pos_weight-weighted training loss."""
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))))


def select_threshold(y_val: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """F1-optimal threshold from precision_recall_curve on VALIDATION only.
    (Predict positive when probability >= threshold.)"""
    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float):
    y = np.asarray(y_true).astype(np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    n = int(len(y))
    negatives = tn + fp
    both_classes = len(np.unique(y)) == 2
    evaluated_days = n / MINUTES_PER_DAY  # one evaluated sample == one minute
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / n) if n else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / negatives) if negatives else None,
        "false_positive_rate": float(fp / negatives) if negatives else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        # Minute-level false positives per day of evaluated data (each evaluated
        # sample is one minute). This is NOT an event-level false-alarm rate:
        # use scripts/ml/evaluate_event_level.py on the predictions CSV for that.
        "false_alarms_per_day": float(fp / evaluated_days) if evaluated_days > 0 else None,
        "evaluated_days": float(evaluated_days),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
        "positive_samples": int(y.sum()),
        "samples": n,
    }, pred


def class_distribution(y: np.ndarray) -> dict:
    n = int(len(y))
    pos = int(np.sum(y))
    neg = n - pos
    return {
        "samples": n,
        "positives": pos,
        "negatives": neg,
        "positive_rate": (pos / n) if n else None,
        "pos_weight_neg_over_pos": (neg / pos) if pos else None,
    }


# ============================================================
# ONE EXPERIMENT
# ============================================================

def train_one(args, model_name: str, horizon: int, device: torch.device,
              feature_names: list[str], schema_sha256: str, environment: dict) -> dict:
    print("\n" + "=" * 76)
    print(f"TRAINING {model_name.upper()} | FORECAST {horizon} MINUTES")
    print("=" * 76)

    seq_len = args.sequence_length
    data_dir = Path(args.data_dir)
    run_label = model_name + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(args.output_dir) / run_label / f"forecast_{horizon}m"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"

    # Heavy objects are declared up front so `finally` can always release them.
    model = optimizer = scheduler = criterion = None
    x_train = x_val = x_test = None
    train_ds = val_ds = test_ds = None
    train_loader = val_loader = test_loader = None
    gpu_peak_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        # ---- load + validate + clean --------------------------------------
        print("Loading splits (target validation, timestamp cleaning, dedup):")
        ts, y_rows, split_reports = {}, {}, {}
        raw = {}
        for split in SPLITS:
            path = data_dir / f"forecast_{horizon}m_{split}.parquet"
            ts[split], raw[split], y_rows[split], split_reports[split] = load_split(
                path, feature_names, args.drop_null_target_rows
            )
        x_train, x_val, x_test = raw["train"], raw["validation"], raw["test"]
        del raw

        ts_ns = {s: timestamps_ns(ts[s]) for s in SPLITS}
        split_check = validate_splits(ts_ns, horizon, args.strict_split_gap)
        print(
            "Split validation OK | strictly chronological, no timestamp overlap | "
            f"gaps (min): train->val={split_check['train_to_validation_minutes']:.0f}, "
            f"val->test={split_check['validation_to_test_minutes']:.0f}"
        )

        # ---- train-only preprocessing, in place ---------------------------
        prep = fit_scaler(x_train)
        for arr in (x_train, x_val, x_test):
            transform_inplace(arr, prep)
        gc.collect()  # raw (unscaled) values no longer exist anywhere
        atomic_save_npz(
            run_dir / "preprocessing.npz",
            medians=prep["medians"], means=prep["means"], stds=prep["stds"],
            clip=np.float32(10.0),
        )
        write_json(run_dir / "feature_names.json", feature_names)

        # ---- contiguous 60-minute windows, built inside each split ---------
        endpoints = {
            "train": valid_sequence_endpoints(ts_ns["train"], seq_len),
            "validation": valid_sequence_endpoints(ts_ns["validation"], seq_len),
            "test": valid_sequence_endpoints(ts_ns["test"], seq_len),
        }
        for split in SPLITS:
            verify_sequence_endpoints(ts_ns[split], endpoints[split], seq_len, split)
        if min(len(e) for e in endpoints.values()) == 0:
            raise RuntimeError(
                f"At least one split has no valid contiguous {seq_len}-minute sequences."
            )
        print(f"Sequence verification OK | window={seq_len} min, 1-min steps, inside each split")

        y_seq = {s: y_rows[s][endpoints[s]] for s in SPLITS}
        distributions = {
            "rows_after_cleaning": {s: class_distribution(y_rows[s]) for s in SPLITS},
            "sequences": {s: class_distribution(y_seq[s]) for s in SPLITS},
        }
        print(f"Features: {len(feature_names)} | Sequence: {seq_len} minutes")
        for s in SPLITS:
            d = distributions["sequences"][s]
            pw = d["pos_weight_neg_over_pos"]
            print(
                f"  {s:<10} sequences={d['samples']:>9,} positives={d['positives']:>7,} "
                f"({(d['positive_rate'] or 0):.3%}) pos_weight={'n/a' if pw is None else f'{pw:.3f}'}"
            )
        if distributions["sequences"]["train"]["positives"] == 0:
            raise RuntimeError("Training split has no positive forecast examples.")
        if distributions["sequences"]["validation"]["positives"] == 0:
            raise RuntimeError(
                "Validation split has no positive sequences; PR-AUC selection and "
                "threshold selection are undefined."
            )
        if distributions["sequences"]["test"]["positives"] == 0:
            print("  WARNING: test split has no positive sequences; PR/ROC-AUC will be undefined.")

        train_ds = SequenceDataset(x_train, y_rows["train"], endpoints["train"], seq_len)
        val_ds = SequenceDataset(x_val, y_rows["validation"], endpoints["validation"], seq_len)
        test_ds = SequenceDataset(x_test, y_rows["test"], endpoints["test"], seq_len)
        train_loader = make_loader(train_ds, True, device, args.batch_size, args.seed)
        val_loader = make_loader(val_ds, False, device, args.batch_size, args.seed)
        test_loader = make_loader(test_ds, False, device, args.batch_size, args.seed)

        # ---- model / loss / optimiser / scheduler ---------------------------
        pos_weight_value = distributions["sequences"]["train"]["pos_weight_neg_over_pos"]
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        cfg = model_config(args, model_name)
        model = build_model(model_name, len(feature_names), cfg).to(device)
        n_params = int(sum(p.numel() for p in model.parameters()))
        # Weighted loss is for TRAINING only; validation loss below is unweighted.
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience
        )

        experiment_config = {
            "model_name": model_name,
            "run_label": run_label,
            "horizon_minutes": horizon,
            "sequence_length": seq_len,
            "seed": args.seed,
            "arguments": vars(args),
            "model_config": cfg,
            "model_architecture": str(model),
            "trainable_parameters": n_params,
            "loss": {
                "training": "BCEWithLogitsLoss(pos_weight=neg/pos of TRAIN sequences) - weighted",
                "validation": "binary cross-entropy from logits, UNWEIGHTED",
                "pos_weight": float(pos_weight_value),
            },
            "optimizer": {
                "class": type(optimizer).__name__,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": list(optimizer.defaults.get("betas", [])),
                "eps": optimizer.defaults.get("eps"),
                "gradient_clip_max_norm": args.grad_clip,
            },
            "scheduler": {
                "class": type(scheduler).__name__,
                "mode": "max",
                "monitored_metric": "validation_pr_auc",
                "factor": args.lr_factor,
                "patience": args.lr_patience,
            },
            "early_stopping": {
                "metric": "validation_pr_auc",
                "patience": args.patience,
                "min_delta": args.min_delta,
                "max_epochs": args.epochs,
            },
            "preprocessing": {
                "fit_on": "train rows only",
                "imputation": "median", "standardization": "z-score", "clip": [-10.0, 10.0],
            },
            "threshold_selection": "F1-optimal on validation (precision_recall_curve); never test",
            "data": {
                "data_dir": str(data_dir),
                "feature_count": len(feature_names),
                "feature_schema_sha256": schema_sha256,
                "split_files": split_reports,
                "split_validation": split_check,
                "class_distributions": distributions,
            },
            "environment": environment,
        }
        write_json(run_dir / "experiment_config.json", experiment_config)
        print(f"Model: {model_name} | trainable parameters: {n_params:,} | device: {device}")

        # ---- training loop (validation PR-AUC drives everything) -----------
        best_pr_auc, best_epoch, stale = -1.0, 0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss, seen = 0.0, 0
            for xb, yb, _ in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                seen += len(yb)

            val_actual, val_logits, _ = predict(model, val_loader, device)
            val_prob = sigmoid_np(val_logits)
            val_pr_auc = float(average_precision_score(val_actual, val_prob))
            val_loss = unweighted_bce_from_logits(val_logits, val_actual)
            scheduler.step(val_pr_auc)

            row = {
                "epoch": epoch,
                "train_loss_weighted": total_loss / max(seen, 1),
                "validation_loss_unweighted": val_loss,
                "validation_pr_auc": val_pr_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            }
            history.append(row)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss(weighted)={row['train_loss_weighted']:.5f} | "
                f"val_loss(unweighted)={val_loss:.5f} | "
                f"val_PR_AUC={val_pr_auc:.5f} | "
                f"lr={row['learning_rate']:.2e} | {row['seconds']:.1f}s"
            )

            if val_pr_auc > best_pr_auc + args.min_delta:
                best_pr_auc, best_epoch, stale = val_pr_auc, epoch, 0
                atomic_torch_save(
                    {
                        "model_name": model_name,
                        "horizon_minutes": int(horizon),
                        "sequence_length": int(seq_len),
                        "feature_names": list(feature_names),
                        "model_config": plain(cfg),
                        "model_state_dict": model.state_dict(),
                        "best_validation_pr_auc": float(best_pr_auc),
                        "best_epoch": int(best_epoch),
                        "pos_weight": float(pos_weight_value),
                        "seed": int(args.seed),
                        "torch_version": str(torch.__version__),
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no validation PR-AUC improvement for {args.patience} epochs.")
                break

        write_csv(run_dir / "training_history.csv", pd.DataFrame(history))

        # ---- final evaluation with the best VALIDATION checkpoint ----------
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        val_actual, val_logits, _ = predict(model, val_loader, device)
        val_prob = sigmoid_np(val_logits)
        threshold, val_best_f1 = select_threshold(val_actual, val_prob)  # validation only

        test_actual, test_logits, test_end = predict(model, test_loader, device)
        test_prob = sigmoid_np(test_logits)
        test_metrics, test_pred = binary_metrics(test_actual, test_prob, threshold)
        val_metrics, _ = binary_metrics(val_actual, val_prob, threshold)

        predictions = pd.DataFrame({
            "timestamp": ts["test"].iloc[test_end].reset_index(drop=True),
            "actual": test_actual,
            "probability": test_prob,
            "predicted": test_pred,
        })
        write_csv(run_dir / f"forecast_{horizon}m_predictions.csv", predictions)

        if device.type == "cuda":
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        summary = {
            "model": model_name,
            "run_label": run_label,
            "horizon_minutes": horizon,
            "sequence_length": seq_len,
            "feature_count": len(feature_names),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_validation_pr_auc": best_pr_auc,
            "validation_best_f1_threshold": threshold,
            "validation_f1_at_selected_threshold": val_best_f1,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "sequence_counts": {s: int(len(endpoints[s])) for s in SPLITS},
            "class_distributions": distributions,
            "pos_weight_train": float(pos_weight_value),
            "trainable_parameters": n_params,
            "gpu_peak_memory_mb": gpu_peak_mb,
            "device": str(device),
            "loss_note": (
                "train_loss_weighted uses pos_weight; validation_loss_unweighted is plain BCE "
                "from logits. They are not directly comparable."
            ),
            "false_alarm_note": (
                "false_alarms_per_day counts minute-level false-positive samples per evaluated "
                "day; use evaluate_event_level.py for event-level false alarms."
            ),
            "config_file": "experiment_config.json",
        }
        write_json(run_dir / "metrics.json", summary)

        print("\nBEST CHECKPOINT:", checkpoint_path)
        print("VALIDATION threshold:", f"{threshold:.6f}")
        print("TEST METRICS:")
        print(json.dumps(test_metrics, indent=2))
        print("Predictions:", run_dir / f"forecast_{horizon}m_predictions.csv")
        return summary

    finally:
        # Release data, loaders, model and GPU memory after every experiment.
        model = optimizer = scheduler = criterion = None
        train_loader = val_loader = test_loader = None
        train_ds = val_ds = test_ds = None
        x_train = x_val = x_test = None
        free_memory(device)


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_CHOICES, default="cnn_lstm",
                   help="cnn | cnn_avgmax | cnn_lstm | both (cnn+cnn_lstm) | all (default: cnn_lstm)")
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--sequence-length", type=int, default=60)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--learning-rate", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--patience", type=int, default=6, help="early-stopping patience (validation PR-AUC)")
    g.add_argument("--min-delta", type=float, default=1e-6, help="minimum PR-AUC gain to count as improvement")
    g.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; <=0 disables clipping")
    g.add_argument("--lr-factor", type=float, default=0.5)
    g.add_argument("--lr-patience", type=int, default=2)

    g = p.add_argument_group("architecture (defaults reproduce the original models)")
    g.add_argument("--conv1-channels", type=int, default=128)
    g.add_argument("--conv2-channels", type=int, default=64)
    g.add_argument("--head-hidden", type=int, default=32)
    g.add_argument("--cnn-kernel-size", type=int, default=5)
    g.add_argument("--cnn-dropout", type=float, default=0.20)
    g.add_argument("--cnn-head-dropout", type=float, default=0.20)
    g.add_argument("--lstm-conv-kernel1", type=int, default=5)
    g.add_argument("--lstm-conv-kernel2", type=int, default=3)
    g.add_argument("--lstm-conv-dropout", type=float, default=0.15)
    g.add_argument("--lstm-hidden", type=int, default=64)
    g.add_argument("--lstm-layers", type=int, default=1)
    g.add_argument("--lstm-dropout", type=float, default=0.0, help="between LSTM layers (needs --lstm-layers > 1)")
    g.add_argument("--lstm-head-dropout", type=float, default=0.25)

    g = p.add_argument_group("reproducibility / runtime")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--deterministic", action="store_true",
                   help="strict deterministic mode (torch.use_deterministic_algorithms; may raise "
                        "on non-deterministic ops and can be slower)")
    g.add_argument("--torch-threads", type=int, default=int(_THREADS_ENV) if _THREADS_ENV else None,
                   help="torch intra-op CPU threads (default: ARUNX_THREADS or torch default)")
    g.add_argument("--interop-threads", type=int, default=None, help="torch inter-op CPU threads")

    g = p.add_argument_group("data / output")
    g.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    g.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    g.add_argument("--tag", type=str, default="", help="suffix for the output folder of non-default runs")
    g.add_argument("--expected-features", type=int, default=EXPECTED_FEATURES)
    g.add_argument("--drop-null-target-rows", action="store_true",
                   help="drop (and report) rows with null target instead of rejecting the file")
    g.add_argument("--strict-split-gap", action="store_true",
                   help="error (instead of warn) if a split gap is shorter than the forecast horizon")

    args = p.parse_args(argv)

    if any(h not in HORIZONS for h in args.horizons):
        p.error(f"--horizons must be chosen from {HORIZONS}")
    if args.sequence_length < 2:
        p.error("--sequence-length must be >= 2")
    for name in ("cnn_kernel_size", "lstm_conv_kernel1", "lstm_conv_kernel2"):
        if getattr(args, name) % 2 != 1 or getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be a positive odd integer (same-length padding)")
    for name in ("cnn_dropout", "cnn_head_dropout", "lstm_conv_dropout", "lstm_dropout", "lstm_head_dropout"):
        if not 0.0 <= getattr(args, name) < 1.0:
            p.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    for name in ("epochs", "batch_size", "patience", "lstm_layers", "conv1_channels",
                 "conv2_channels", "head_hidden", "lstm_hidden", "lr_patience"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 < args.lr_factor < 1:
        p.error("invalid --learning-rate / --weight-decay / --lr-factor")
    for name in ("torch_threads", "interop_threads"):
        v = getattr(args, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)

    runtime = configure_runtime(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = collect_environment(device, runtime)

    print("ARUN-X PyTorch Forecast Training")
    print("Project root:", ROOT)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", environment["gpu_name"], "| CUDA build:", environment["torch_cuda_build_version"])
    print(
        f"torch {torch.__version__} | threads={runtime['torch_num_threads']} "
        f"interop={runtime['torch_num_interop_threads']} | deterministic={args.deterministic}"
    )

    feature_names, schema_sha = load_feature_names(Path(args.data_dir), args.expected_features)
    model_names = MODEL_GROUPS[args.model]

    results = []
    for model_name in model_names:
        for horizon in args.horizons:
            set_seed(args.seed)
            results.append(
                train_one(args, model_name, horizon, device, feature_names, schema_sha, environment)
            )

    print("\nAll requested deep-learning runs completed.")
    print("Output directory:", args.output_dir)
    if results:
        table = pd.DataFrame([
            {
                "model": r["run_label"],
                "horizon": f"{r['horizon_minutes']}m",
                "best_epoch": r["best_epoch"],
                "val_pr_auc": r["best_validation_pr_auc"],
                "test_pr_auc": r["test_metrics"]["pr_auc"],
                "test_f1": r["test_metrics"]["f1"],
                "test_recall": r["test_metrics"]["recall"],
                "test_fpr": r["test_metrics"]["false_positive_rate"],
                "false_alarms/day": r["test_metrics"]["false_alarms_per_day"],
            }
            for r in results
        ])
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()