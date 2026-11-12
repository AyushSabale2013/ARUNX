#!/usr/bin/env python3
"""
ARUN-X: XGBoost flare-onset forecasting at 15 / 30 / 60 minute horizons.

Location: scripts/ml/train_xgboost_forecasting.py (next to training_utils.py).

What this script does
---------------------
1. Audits the raw Parquet splits (target values, timestamps, feature order,
   split chronology and the split boundaries in feature_schema.json) BEFORE any
   model is trained. It only reads; it never modifies datasets or labels.
2. Trains one XGBoost classifier per horizon with validation-based early
   stopping on PR-AUC (xgboost "aucpr"). Only the validation split is used for
   early stopping and for the F1-optimal threshold.
3. Evaluates on validation and test with the shared training_utils metrics and
   adds specificity / FPR / false-alarm rate.
4. Writes every artifact atomically. All heavy work finishes first; files are
   staged next to their destinations and swapped in with os.replace only after
   the whole horizon succeeded. The metadata file (which holds checksums of the
   other artifacts) is committed last and acts as the completion marker.

Scientific notes
----------------
* Minute-end samples; labels and features are used exactly as stored.
* The July test split was already used in earlier experiments. Treat the test
  numbers as an exploratory benchmark, not a pristine final holdout.
* Early stopping and the threshold both use the validation split, so the
  validation metrics are optimistic by construction. Compare models on test.
* Event-level flare metrics are NOT computed here. The Parquet files carry no
  flare/event identifiers, so event metrics cannot be derived reliably from
  row predictions. The saved predictions CSV follows the naming/column
  contract of scripts/ml/evaluate_event_level.py, which owns that evaluation.
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_N_JOBS = 3


def _preparse_threads(argv) -> int:
    """training_utils caps BLAS/OpenMP threads from ARUNX_THREADS at import time,
    so --n-jobs has to be known before it is imported."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    namespace, _ = pre.parse_known_args(argv)
    return max(1, namespace.n_jobs)


os.environ.setdefault("ARUNX_THREADS", str(_preparse_threads(sys.argv[1:])))

import training_utils as tu  # noqa: E402  (must come before numpy/sklearn/xgboost)

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

SCRIPT_VERSION = "2.0"
FAMILY = "forecasting"
MODEL_NAME = "xgboost"          # key used in the shared metrics tables / prediction file names
MODEL_LABEL = "XGBoost"         # legacy label kept in the metadata file
SPLITS = ("train", "validation", "test")
MINUTES_PER_DAY = 1440.0
LEAKY_NAME_TOKENS = ("target", "label", "future", "forecast")
MIN_XGBOOST = (1, 6, 0)         # constructor-level early_stopping_rounds / eval_metric

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
}


class DataIntegrityError(ValueError):
    """Raised when the datasets violate an invariant this script relies on."""


# ============================================================
# ARGUMENTS
# ============================================================

def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train XGBoost flare-onset forecasters (ARUN-X).",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tu.add_path_args(parser)  # --data-dir --models-root --results-root --seed
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=tu.HORIZONS,
        default=list(tu.HORIZONS), help="Forecast horizons in minutes.",
    )
    parser.add_argument(
        "--n-jobs", type=_positive_int, default=DEFAULT_N_JOBS,
        help="CPU threads for XGBoost (also caps BLAS/OpenMP via ARUNX_THREADS).",
    )

    hp = parser.add_argument_group("XGBoost hyperparameters (CPU only)")
    hp.add_argument("--n-estimators", type=_positive_int, default=DEFAULT_PARAMS["n_estimators"],
                    help="Upper bound on boosting rounds; early stopping picks the final count.")
    hp.add_argument("--early-stopping-rounds", type=_positive_int,
                    default=DEFAULT_PARAMS["early_stopping_rounds"],
                    help="Stop after this many rounds without validation PR-AUC improvement.")
    hp.add_argument("--learning-rate", type=_positive_float, default=DEFAULT_PARAMS["learning_rate"])
    hp.add_argument("--max-depth", type=_positive_int, default=DEFAULT_PARAMS["max_depth"])
    hp.add_argument("--min-child-weight", type=_non_negative_float,
                    default=DEFAULT_PARAMS["min_child_weight"])
    hp.add_argument("--subsample", type=_fraction, default=DEFAULT_PARAMS["subsample"])
    hp.add_argument("--colsample-bytree", type=_fraction, default=DEFAULT_PARAMS["colsample_bytree"])
    hp.add_argument("--gamma", type=_non_negative_float, default=DEFAULT_PARAMS["gamma"])
    hp.add_argument("--reg-alpha", type=_non_negative_float, default=DEFAULT_PARAMS["reg_alpha"])
    hp.add_argument("--reg-lambda", type=_non_negative_float, default=DEFAULT_PARAMS["reg_lambda"])
    hp.add_argument("--tree-method", choices=("hist", "approx", "exact"),
                    default=DEFAULT_PARAMS["tree_method"],
                    help="GPU methods are deliberately not offered.")

    run = parser.add_argument_group("run control")
    run.add_argument("--skip-existing", action="store_true",
                     help="Skip a horizon only if ALL artifacts exist, are consistent "
                          "(checksums) and were produced with the same configuration and data.")
    run.add_argument("--validate-only", action="store_true",
                     help="Run the data-integrity audit and exit without training.")
    run.add_argument("--allow-timestamp-cleaning", action="store_true",
                     help="Let training_utils.load_split drop null/duplicate/unsorted "
                          "timestamp rows (reported in metadata) instead of aborting.")
    run.add_argument("--build-comparisons", action="store_true",
                     help="Rebuild results_root/comparisons after training.")
    return parser


def hyperparameters_from_args(args) -> dict:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "early_stopping_rounds": args.early_stopping_rounds,
        "tree_method": args.tree_method,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
    }


# ============================================================
# ENVIRONMENT
# ============================================================

def _version_tuple(text: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def collect_versions() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pyarrow": pa.__version__,
    }


def thread_info(n_jobs: int) -> dict:
    return {
        "xgboost_n_jobs": n_jobs,
        "training_utils_cap": tu.THREADS,
        "environment": {
            key: os.environ.get(key)
            for key in ("ARUNX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


# ============================================================
# DATA INTEGRITY AUDIT (read-only; timestamp + target columns only)
# ============================================================

def audit_split(path: Path, split: str, horizon: int, allow_cleaning: bool) -> dict:
    """Validate raw target and timestamp columns without touching the file.

    Targets must be non-null, numeric, integer-valued and in {0, 1}; any
    violation aborts. Timestamp problems (null/unparsable, duplicates, rows out
    of order) abort unless allow_cleaning is set, in which case they are
    counted here and later removed by training_utils.load_split.
    """
    path = Path(path)
    label = f"{horizon}m/{split} ({path.name})"
    schema = pq.read_schema(path)

    for column in tu.NON_FEATURE_COLUMNS:
        if column not in schema.names:
            raise DataIntegrityError(f"{label}: missing required column '{column}'.")

    target_type = schema.field("target").type
    if not (pa.types.is_integer(target_type) or pa.types.is_floating(target_type)
            or pa.types.is_boolean(target_type)):
        raise DataIntegrityError(f"{label}: target has non-numeric type {target_type}.")
    ts_type = schema.field("timestamp").type
    if not (pa.types.is_timestamp(ts_type) or pa.types.is_string(ts_type)
            or pa.types.is_large_string(ts_type)):
        raise DataIntegrityError(f"{label}: timestamp has unsupported type {ts_type}.")

    table = pq.read_table(path, columns=["timestamp", "target"])
    n_rows = int(table.num_rows)
    if n_rows == 0:
        raise DataIntegrityError(f"{label}: split is empty.")

    # ---- target, validated on the ORIGINAL values (before any int8 cast) ----
    target_column = table.column("target")
    if target_column.null_count:
        raise DataIntegrityError(f"{label}: {target_column.null_count:,} null target value(s).")
    target = np.asarray(target_column.to_numpy(), dtype=np.float64)
    n_nonfinite = int((~np.isfinite(target)).sum())
    if n_nonfinite:
        raise DataIntegrityError(f"{label}: {n_nonfinite:,} NaN/inf target value(s).")
    n_fractional = int((target != np.floor(target)).sum())
    if n_fractional:
        raise DataIntegrityError(f"{label}: {n_fractional:,} fractional target value(s).")
    non_binary = ~np.isin(target, (0.0, 1.0))
    if non_binary.any():
        examples = np.unique(target[non_binary])[:5].tolist()
        raise DataIntegrityError(
            f"{label}: {int(non_binary.sum()):,} non-binary target value(s), e.g. {examples}."
        )
    positives = int(target.sum())

    # ---- timestamps ----
    ts_raw = table.column("timestamp").to_pandas()
    tz_in_file = getattr(ts_type, "tz", None)
    ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    n_null = int(ts.isna().sum())
    valid = ts[ts.notna()].reset_index(drop=True)
    if valid.empty:
        raise DataIntegrityError(f"{label}: no valid timestamps.")

    one_minute = pd.Timedelta(minutes=1)
    steps = valid.diff().iloc[1:]
    n_decreasing = int((steps < pd.Timedelta(0)).sum())
    n_duplicate = int(valid.duplicated().sum())
    n_misaligned = int((valid != valid.dt.floor("min")).sum())
    unique_sorted = valid.drop_duplicates().sort_values()
    gaps = unique_sorted.diff().iloc[1:]
    n_gaps = int((gaps > one_minute).sum())
    max_gap_minutes = float(gaps.max() / one_minute) if len(gaps) else 0.0

    problems = []
    if n_null:
        problems.append(f"{n_null:,} null/unparsable timestamp(s)")
    if n_duplicate:
        problems.append(f"{n_duplicate:,} duplicate timestamp(s)")
    if n_decreasing:
        problems.append(f"{n_decreasing:,} out-of-order step(s)")
    if problems:
        message = f"{label}: " + "; ".join(problems) + "."
        if not allow_cleaning:
            raise DataIntegrityError(
                message + " Fix the dataset or pass --allow-timestamp-cleaning "
                "to let training_utils.load_split drop them (reported in metadata)."
            )
        print(f"  WARNING {message} They will be cleaned by training_utils.load_split.")
    if n_misaligned:
        print(f"  WARNING {label}: {n_misaligned:,} timestamp(s) are not on whole-minute "
              "boundaries (schema says minute-end bins).")
    if tz_in_file is None and pa.types.is_timestamp(ts_type):
        print(f"  NOTE {label}: timestamps are timezone-naive in the file; interpreted as UTC.")

    return {
        "split": split,
        "file": path.name,
        "file_size_bytes": path.stat().st_size,
        "rows": n_rows,
        "expected_rows": int(len(valid) - n_duplicate),
        "null_timestamps": n_null,
        "duplicate_timestamps": n_duplicate,
        "out_of_order_steps": n_decreasing,
        "misaligned_timestamps": n_misaligned,
        "gaps_over_1min": n_gaps,
        "max_gap_minutes": max_gap_minutes,
        "timestamp_tz_in_file": None if tz_in_file is None else str(tz_in_file),
        "ts_min": valid.min(),
        "ts_max": valid.max(),
        "positives": positives,
        "negatives": n_rows - positives,
    }


def load_feature_schema(data_dir: Path) -> dict:
    path = Path(data_dir) / "feature_schema.json"
    if not path.exists():
        raise DataIntegrityError(f"feature_schema.json not found in {data_dir}.")
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    features = schema.get("features")
    if not isinstance(features, list) or not features:
        raise DataIntegrityError("feature_schema.json has no 'features' list.")
    if len(set(features)) != len(features):
        raise DataIntegrityError("feature_schema.json contains duplicate feature names.")
    declared = schema.get("feature_count")
    if declared is not None and declared != len(features):
        raise DataIntegrityError(
            f"feature_schema.json feature_count={declared} but lists {len(features)} features."
        )
    leaky = [f for f in features if any(token in f.lower() for token in LEAKY_NAME_TOKENS)]
    if leaky:
        raise DataIntegrityError(f"Suspicious feature names (possible label leakage): {leaky[:5]}")
    return schema


def verify_feature_order(data_dir: Path, horizons, schema_features: list) -> None:
    """The Parquet columns (minus timestamp/target) must equal the schema list,
    in the same order, for every horizon/split, and all must be numeric."""
    for horizon in horizons:
        for split in SPLITS:
            path = tu.split_path(data_dir, horizon, split)
            parquet_schema = pq.read_schema(path)
            columns = [c for c in parquet_schema.names if c not in tu.NON_FEATURE_COLUMNS]
            if columns != schema_features:
                if len(columns) != len(schema_features):
                    detail = f"{len(columns)} Parquet feature columns vs {len(schema_features)} in schema"
                else:
                    index = next(i for i, (a, b) in enumerate(zip(columns, schema_features)) if a != b)
                    detail = (f"first difference at position {index}: Parquet "
                              f"'{columns[index]}' vs schema '{schema_features[index]}'")
                raise DataIntegrityError(
                    f"{horizon}m/{split}: feature columns do not match feature_schema.json ({detail})."
                )
            bad = []
            for name in schema_features:
                kind = parquet_schema.field(name).type
                if not (pa.types.is_integer(kind) or pa.types.is_floating(kind)
                        or pa.types.is_boolean(kind)):
                    bad.append(f"{name}:{kind}")
            if bad:
                raise DataIntegrityError(f"{horizon}m/{split}: non-numeric feature columns {bad[:5]}")


def check_split_order_and_boundaries(horizon: int, audits: dict, boundaries) -> None:
    """Strict chronology (via training_utils) plus consistency with the
    boundaries recorded in feature_schema.json. Boundary comparisons are
    inclusive on both sides because minute-end stamps make the exact boundary
    minute ambiguous; real overlap is already excluded by the strict check."""
    spans = {s: pd.Series([audits[s]["ts_min"], audits[s]["ts_max"]]) for s in SPLITS}
    try:
        tu.check_chronology(spans["train"], spans["validation"], spans["test"])
    except ValueError as error:
        raise DataIntegrityError(f"{horizon}m: {error}") from error

    if not boundaries:
        print(f"  WARNING {horizon}m: no split_boundaries_utc in schema; boundary check skipped.")
        return
    train_end = pd.to_datetime(boundaries["train_end"], utc=True)
    validation_end = pd.to_datetime(boundaries["validation_end"], utc=True)
    checks = [
        (audits["train"]["ts_max"] <= train_end, f"train ends {audits['train']['ts_max']} after train_end {train_end}"),
        (audits["validation"]["ts_min"] >= train_end, f"validation starts {audits['validation']['ts_min']} before train_end {train_end}"),
        (audits["validation"]["ts_max"] <= validation_end, f"validation ends {audits['validation']['ts_max']} after validation_end {validation_end}"),
        (audits["test"]["ts_min"] >= validation_end, f"test starts {audits['test']['ts_min']} before validation_end {validation_end}"),
    ]
    for ok, message in checks:
        if not ok:
            raise DataIntegrityError(f"{horizon}m: split boundary violation: {message}.")


def print_audit(horizon: int, audits: dict) -> None:
    print(f"\n[{horizon}m] data audit")
    for split in SPLITS:
        a = audits[split]
        print(f"  {split:<10} rows={a['rows']:>9,} positives={a['positives']:>7,} "
              f"({a['positives'] / a['rows']:.3%}) {a['ts_min']} -> {a['ts_max']} "
              f"gaps>1min={a['gaps_over_1min']:,} (max {a['max_gap_minutes']:.0f} min)")


# ============================================================
# LOADED-DATA RECONCILIATION
# ============================================================

def reconcile_loaded(split: str, audit: dict, timestamps, y: np.ndarray) -> None:
    """The arrays returned by training_utils.load_split must agree with the audit."""
    if len(y) != audit["expected_rows"]:
        raise DataIntegrityError(
            f"{split}: loaded {len(y):,} rows but the audit expected {audit['expected_rows']:,}."
        )
    if timestamps.min() != audit["ts_min"] or timestamps.max() != audit["ts_max"]:
        raise DataIntegrityError(f"{split}: loaded time range differs from the audited range.")
    if len(y) == audit["rows"] and int(y.sum()) != audit["positives"]:
        raise DataIntegrityError(f"{split}: positive count changed between audit and load.")


def split_summary(split: str, audit: dict, timestamps, y: np.ndarray) -> dict:
    rows = int(len(y))
    positives = int(y.sum())
    span = timestamps.max() - timestamps.min() + pd.Timedelta(minutes=1)
    return {
        "file": audit["file"],
        "file_size_bytes": audit["file_size_bytes"],
        "rows": rows,
        "rows_in_file": audit["rows"],
        "rows_removed_by_cleaning": audit["rows"] - rows,
        "null_timestamps": audit["null_timestamps"],
        "duplicate_timestamps": audit["duplicate_timestamps"],
        "out_of_order_steps": audit["out_of_order_steps"],
        "gaps_over_1min": audit["gaps_over_1min"],
        "max_gap_minutes": audit["max_gap_minutes"],
        "start_utc": timestamps.min().isoformat(),
        "end_utc": timestamps.max().isoformat(),
        "observed_days": observed_days(rows),
        "calendar_span_days": float(span / pd.Timedelta(days=1)),
        "negatives": rows - positives,
        "positives": positives,
        "prevalence": positives / rows if rows else None,
    }


def observed_days(rows: int) -> float:
    """Each row is one one-minute bin, so observed time = rows / 1440 days.
    Unlike the calendar span this excludes gaps with no data."""
    return rows / MINUTES_PER_DAY


# ============================================================
# METRICS (shared training_utils metrics + operational extras)
# ============================================================

def add_operational_metrics(metrics: dict, days: float) -> dict:
    """Adds specificity, FPR, prevalence and row-level false alarms per day.
    (false_negative / true_positive / etc. already come from compute_metrics.)
    false_alarms_per_day counts false-positive ROWS (minutes) per observed day;
    it overstates distinct alarms when predictions persist across minutes."""
    tn, fp = metrics["true_negative"], metrics["false_positive"]
    negatives = tn + fp
    metrics["specificity"] = tn / negatives if negatives else None
    metrics["false_positive_rate"] = fp / negatives if negatives else None
    metrics["false_alarms_per_day"] = fp / days if days > 0 else None
    metrics["observed_days"] = float(days)
    metrics["prevalence"] = metrics["positive_samples"] / metrics["samples"]
    return metrics


def print_operational(title: str, metrics: dict) -> None:
    def f(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"

    print(f"  {'':<11} specificity={f(metrics['specificity'])} FPR={f(metrics['false_positive_rate'])} "
          f"FN={metrics['false_negative']:,} false-alarm rows/day={f(metrics['false_alarms_per_day'], 2)} "
          f"(prevalence/PR-AUC baseline={f(metrics['prevalence'])})")


# ============================================================
# ATOMIC ARTIFACT STAGING
# ============================================================

def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_quietly(path: Path) -> None:
    try:
        with open(path, "r+b") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


class ArtifactStager:
    """Writes artifacts to hidden temp files beside their destinations (same
    filesystem, original extension preserved so XGBoost keeps the JSON format)
    and swaps them in with os.replace only when commit() is called."""

    def __init__(self) -> None:
        self._items: list[tuple[Path, Path]] = []

    def tmp_path(self, final: Path) -> Path:
        final = Path(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(f".tmp-{os.getpid()}-{final.name}")
        self._items.append((tmp, final))
        return tmp

    def commit(self) -> None:
        for tmp, _ in self._items:
            if not tmp.exists():
                raise FileNotFoundError(f"Staged artifact missing, nothing committed: {tmp}")
        for tmp, _ in self._items:
            _fsync_quietly(tmp)
        items, self._items = self._items, []
        for tmp, final in items:
            os.replace(tmp, final)

    def discard(self) -> None:
        for tmp, _ in self._items:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        self._items = []


def stage_metrics_row(stager: ArtifactStager, table_path: Path, row: dict) -> None:
    """Reuse tu.update_metrics_table on a temp COPY so a crash can never leave a
    half-written shared table; the copy replaces the real file only on commit."""
    tmp = stager.tmp_path(table_path)
    if Path(table_path).exists():
        shutil.copy2(table_path, tmp)
    tu.update_metrics_table(tmp, [row])


# ============================================================
# CONFIGURATION FINGERPRINT / SKIP-EXISTING
# ============================================================

def data_signature(data_dir: Path, horizon: int) -> dict:
    signature = {}
    for split in SPLITS:
        stat = tu.split_path(data_dir, horizon, split).stat()
        signature[split] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    schema_stat = (Path(data_dir) / "feature_schema.json").stat()
    signature["schema"] = {"size": schema_stat.st_size, "mtime_ns": schema_stat.st_mtime_ns}
    return signature


def make_config_hash(params: dict, seed: int, features: list, signature: dict, horizon: int) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "horizon": horizon,
        "params": params,
        "seed": seed,
        "features": features,
        "data": signature,
        "xgboost": xgboost.__version__,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def artifact_paths(args, horizon: int) -> dict:
    model_dir = Path(args.models_root) / FAMILY
    result_dir = Path(args.results_root) / FAMILY
    return {
        "model": model_dir / f"xgboost_forecast_{horizon}m.json",
        "metadata": model_dir / f"forecast_{horizon}m_metadata.json",
        "test_predictions": result_dir / f"{MODEL_NAME}_{horizon}m_predictions.csv",
        "results": result_dir / f"{MODEL_NAME}_{horizon}m_results.json",
        "validation_table": result_dir / "all_validation_metrics.csv",
        "test_table": result_dir / "all_test_metrics.csv",
    }


def existing_run_is_complete(horizon: int, paths: dict, config_hash: str):
    """True only if every required artifact exists, the metadata says complete,
    the configuration/data fingerprint matches, recorded checksums match the
    files on disk, and both shared tables hold this model/horizon row."""
    for role, path in paths.items():
        if not Path(path).exists() or Path(path).stat().st_size == 0:
            return False, f"missing or empty {role}: {path}"
    try:
        with Path(paths["metadata"]).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"unreadable metadata ({error})"
    if metadata.get("status") != "complete":
        return False, "metadata status is not 'complete'"
    if metadata.get("config_hash") != config_hash:
        return False, "configuration, features or data changed since the stored run"
    artifacts = metadata.get("artifacts", {})
    for role in ("model", "test_predictions", "results"):
        recorded = artifacts.get(role, {}).get("sha256")
        if recorded is None or recorded != file_sha256(paths[role]):
            return False, f"{role} checksum does not match metadata (partial or modified run)"
    for role in ("validation_table", "test_table"):
        try:
            table = pd.read_csv(paths[role])
            present = ((table["model"] == MODEL_NAME) & (table["horizon"] == f"{horizon}m")).any()
        except (OSError, KeyError, pd.errors.ParserError):
            return False, f"{role} unreadable"
        if not present:
            return False, f"{role} has no {MODEL_NAME}/{horizon}m row"
    return True, "all artifacts present and consistent"


# ============================================================
# MODEL
# ============================================================

def make_model(params: dict, scale_pos_weight: float, seed: int, n_jobs: int) -> XGBClassifier:
    """early_stopping_rounds and eval_metric go to the constructor (required API
    for xgboost >= 2.0, accepted since 1.6). 'aucpr' is the last (only) metric, so
    it drives early stopping on the single validation eval set. CPU only."""
    kwargs = dict(
        objective=params["objective"],
        eval_metric=params["eval_metric"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        tree_method=params["tree_method"],
        early_stopping_rounds=params["early_stopping_rounds"],
        random_state=seed,
        n_jobs=n_jobs,
    )
    if _version_tuple(xgboost.__version__) >= (2, 0, 0):
        kwargs["device"] = "cpu"  # never pick a GPU implicitly
    return XGBClassifier(**kwargs)


def predict_best(model: XGBClassifier, X: np.ndarray, best_iteration: int) -> np.ndarray:
    """Probabilities from trees 0..best_iteration only, stated explicitly so the
    result does not depend on version-specific defaults."""
    proba = model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
    return np.asarray(proba, dtype=np.float64)


def roundtrip_check(model_path: Path, X_val: np.ndarray, p_val: np.ndarray, best_iteration: int) -> None:
    """Reload the staged model file and confirm it reproduces validation
    probabilities on a sample, before anything is committed."""
    stride = max(1, len(X_val) // 2000)
    sample = np.ascontiguousarray(X_val[::stride])
    reloaded = XGBClassifier()
    reloaded.load_model(str(model_path))
    again = reloaded.predict_proba(sample, iteration_range=(0, best_iteration + 1))[:, 1]
    if not np.allclose(again, p_val[::stride], rtol=0.0, atol=1e-5):
        raise RuntimeError("Saved model does not reproduce validation probabilities; aborting commit.")


# ============================================================
# ONE HORIZON
# ============================================================

def run_horizon(horizon: int, args, features: list, audits: dict, paths: dict,
                params: dict, config_hash: str) -> dict:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    task = f"{horizon}m"
    stager = ArtifactStager()
    data_dir = Path(args.data_dir)
    split_files = {s: tu.split_path(data_dir, horizon, s) for s in SPLITS}

    print("\n" + "=" * 70)
    print(f"TRAINING {task} XGBoost FORECAST MODEL")
    print("=" * 70)

    try:
        # ---- load train + validation only (test stays untouched until the
        #      model, best iteration and threshold are all frozen) ----
        ts_train, X_train, y_train = tu.load_split(split_files["train"], features)
        ts_val, X_val, y_val = tu.load_split(split_files["validation"], features)
        reconcile_loaded("train", audits["train"], ts_train, y_train)
        reconcile_loaded("validation", audits["validation"], ts_val, y_val)
        tu.require_both_classes("train", y_train)
        tu.require_both_classes("validation", y_val)
        print("\nSplits:")
        tu.describe_split("train", ts_train, X_train, y_train)
        tu.describe_split("validation", ts_val, X_val, y_val)

        # ---- class imbalance from TRAINING labels only ----
        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())
        scale_pos_weight = negative_count / positive_count
        print(f"\nTrain negatives={negative_count:,} positives={positive_count:,} "
              f"scale_pos_weight={scale_pos_weight:.4f}")

        # ---- fit with validation-only early stopping on PR-AUC ----
        model = make_model(params, scale_pos_weight, args.seed, args.n_jobs)
        fit_started = time.perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        fit_seconds = time.perf_counter() - fit_started

        best_iteration = int(model.best_iteration)
        best_score = float(model.best_score)
        rounds_built = int(model.get_booster().num_boosted_rounds())
        stopped_early = rounds_built < params["n_estimators"]
        history = model.evals_result().get("validation_0", {}).get("aucpr", [])
        print(f"\nbest_iteration={best_iteration} (trees used={best_iteration + 1}) "
              f"best validation aucpr={best_score:.6f} rounds built={rounds_built} "
              f"early stopped={stopped_early} fit={fit_seconds:.1f}s")
        if not stopped_early:
            print("  WARNING: early stopping never triggered; validation PR-AUC may still be "
                  "improving. Consider a larger --n-estimators.")

        # ---- validation threshold (validation only) + validation metrics ----
        p_val = predict_best(model, X_val, best_iteration)
        threshold = tu.select_threshold(y_val, p_val)
        val_days = observed_days(len(y_val))
        val_metrics, _ = tu.compute_metrics(y_val, p_val, threshold)
        add_operational_metrics(val_metrics, val_days)
        print(f"\nSelected validation threshold: {threshold:.6f}")
        tu.print_metrics("validation", val_metrics)
        print_operational("validation", val_metrics)

        # ---- stage the model and verify it round-trips ----
        staged_model = stager.tmp_path(paths["model"])
        model.save_model(str(staged_model))
        roundtrip_check(staged_model, X_val, p_val, best_iteration)

        train_summary = split_summary("train", audits["train"], ts_train, y_train)
        val_summary = split_summary("validation", audits["validation"], ts_val, y_val)
        del X_train, X_val
        tu.free_memory()

        # ---- test: loaded only now; threshold and model are already fixed ----
        ts_test, X_test, y_test = tu.load_split(split_files["test"], features)
        reconcile_loaded("test", audits["test"], ts_test, y_test)
        tu.check_chronology(ts_train, ts_val, ts_test)
        tu.require_both_classes("test", y_test, strict=False)
        tu.describe_split("test", ts_test, X_test, y_test)

        p_test = predict_best(model, X_test, best_iteration)
        test_days = observed_days(len(y_test))
        test_metrics, test_pred = tu.compute_metrics(y_test, p_test, threshold)
        add_operational_metrics(test_metrics, test_days)
        print("\nTest (threshold fixed from validation; exploratory benchmark):")
        tu.print_metrics("test", test_metrics)
        print_operational("test", test_metrics)
        test_summary = split_summary("test", audits["test"], ts_test, y_test)

        # ---- stage remaining artifacts ----
        tu.save_predictions(stager.tmp_path(paths["test_predictions"]),
                            ts_test, y_test, p_test, test_pred)
        results_doc = {
            "model": MODEL_NAME, "horizon": task, "threshold": threshold,
            "best_iteration": best_iteration,
            "validation": val_metrics, "test": test_metrics,
        }
        staged_results = stager.tmp_path(paths["results"])
        tu.write_json(staged_results, results_doc)
        stage_metrics_row(stager, paths["validation_table"],
                          tu.metrics_row(FAMILY, MODEL_NAME, horizon, val_metrics))
        stage_metrics_row(stager, paths["test_table"],
                          tu.metrics_row(FAMILY, MODEL_NAME, horizon, test_metrics))

        # ---- metadata (committed last; carries checksums of the other artifacts) ----
        staged_predictions = stager._items[[t.name for t, _ in stager._items].index(
            f".tmp-{os.getpid()}-{paths['test_predictions'].name}")][0]
        total_seconds = time.perf_counter() - started
        metadata = {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_utc": started_utc.isoformat(timespec="seconds"),
            "config_hash": config_hash,
            # legacy keys kept for downstream compatibility
            "horizon": task,
            "model": MODEL_LABEL,
            "features": features,
            "feature_count": len(features),
            "threshold": threshold,
            "scale_pos_weight": scale_pos_weight,
            "random_state": args.seed,
            "train_rows": train_summary["rows"],
            "validation_rows": val_summary["rows"],
            "test_rows": test_summary["rows"],
            # new fields
            "horizon_minutes": horizon,
            "family": FAMILY,
            "hyperparameters": {**params, "scale_pos_weight": scale_pos_weight,
                                "random_state": args.seed, "n_jobs": args.n_jobs},
            "early_stopping": {
                "metric": "aucpr (xgboost, validation split only)",
                "rounds": params["early_stopping_rounds"],
                "best_iteration": best_iteration,
                "best_score": best_score,
                "trees_used_for_prediction": best_iteration + 1,
                "rounds_built": rounds_built,
                "n_estimators_requested": params["n_estimators"],
                "stopped_early": stopped_early,
                "validation_aucpr_history": [float(v) for v in history],
            },
            "prediction": {"iteration_range": [0, best_iteration + 1], "probability_threshold": threshold},
            "threshold_selection": "F1-optimal on validation precision-recall curve (training_utils.select_threshold); test data never used",
            "class_distribution": {
                s: {"negative": summary["negatives"], "positive": summary["positives"],
                    "prevalence": summary["prevalence"]}
                for s, summary in (("train", train_summary), ("validation", val_summary), ("test", test_summary))
            },
            "splits": {"train": train_summary, "validation": val_summary, "test": test_summary},
            "feature_schema": {"file": "feature_schema.json", "order_validated": True},
            "seed": args.seed,
            "threads": thread_info(args.n_jobs),
            "versions": collect_versions(),
            "runtime_seconds": {"fit": round(fit_seconds, 3), "total": round(total_seconds, 3)},
            "metrics": {"validation": val_metrics, "test": test_metrics},
            "metric_definitions": {
                "false_alarms_per_day": "false-positive ROWS (minute bins) per observed day; "
                                        "not distinct alarm episodes",
                "observed_days": "rows / 1440 (each row is one one-minute bin)",
                "pr_auc": "sklearn average_precision_score; xgboost aucpr used for early stopping differs slightly",
                "event_level": "not computed here; see scripts/ml/evaluate_event_level.py",
            },
            "evaluation_notes": [
                "Test split (July) was used in earlier experiments: exploratory benchmark, not a pristine holdout.",
                "Early stopping and threshold both use the validation split; validation metrics are optimistic.",
                "Labels of the last horizon-minutes of a split look into the next split; splits were not altered.",
            ],
            "artifacts": {
                "model": {"file": paths["model"].name, "sha256": file_sha256(staged_model),
                          "bytes": staged_model.stat().st_size},
                "test_predictions": {"file": paths["test_predictions"].name,
                                     "sha256": file_sha256(staged_predictions),
                                     "bytes": staged_predictions.stat().st_size},
                "results": {"file": paths["results"].name, "sha256": file_sha256(staged_results),
                            "bytes": staged_results.stat().st_size},
            },
        }
        tu.write_json(stager.tmp_path(paths["metadata"]), metadata)

        stager.commit()
        print(f"\nSaved ({task}): {paths['model']}\n  {paths['metadata']}\n  {paths['test_predictions']}\n"
              f"  {paths['results']}\n  tables: {paths['validation_table'].name}, {paths['test_table'].name}")
        return {"horizon": task, "validation": val_metrics, "test": test_metrics,
                "best_iteration": best_iteration, "seconds": total_seconds}
    finally:
        stager.discard()


# ============================================================
# MAIN
# ============================================================

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if _version_tuple(xgboost.__version__) < MIN_XGBOOST:
        raise RuntimeError(f"xgboost >= {'.'.join(map(str, MIN_XGBOOST))} required, "
                           f"found {xgboost.__version__}.")
    tu.set_seed(args.seed)
    horizons = list(dict.fromkeys(args.horizons))
    data_dir = Path(args.data_dir)
    params = hyperparameters_from_args(args)

    print(f"ARUN-X XGBoost forecasting v{SCRIPT_VERSION} | xgboost {xgboost.__version__} | "
          f"n_jobs={args.n_jobs} (training_utils cap {tu.THREADS}) | seed={args.seed}")
    if args.n_jobs > tu.THREADS:
        print(f"  NOTE: --n-jobs {args.n_jobs} exceeds ARUNX_THREADS cap {tu.THREADS} "
              "(the cap only limits BLAS/OpenMP libraries other than XGBoost).")

    # ---------- integrity audit: nothing is trained or written if this fails ----------
    try:
        schema = load_feature_schema(data_dir)
        verify_feature_order(data_dir, horizons, schema["features"])
        features = tu.get_feature_names(data_dir, horizons[0], expected=schema.get("feature_count"))
        if features != schema["features"]:
            raise DataIntegrityError("Feature list from training_utils differs from feature_schema.json.")
        audits = {}
        for horizon in horizons:
            audits[horizon] = {
                split: audit_split(tu.split_path(data_dir, horizon, split), split, horizon,
                                   args.allow_timestamp_cleaning)
                for split in SPLITS
            }
            check_split_order_and_boundaries(horizon, audits[horizon], schema.get("split_boundaries_utc"))
            print_audit(horizon, audits[horizon])
    except DataIntegrityError as error:
        print(f"\nDATA INTEGRITY ERROR: {error}", file=sys.stderr)
        return 2
    print(f"\nFeature order validated against feature_schema.json ({len(features)} features).")

    if args.validate_only:
        print("Validation only: no training performed.")
        return 0

    # ---------- train ----------
    completed, skipped, failed = [], [], []
    for horizon in horizons:
        paths = artifact_paths(args, horizon)
        config_hash = make_config_hash(params, args.seed, features,
                                       data_signature(data_dir, horizon), horizon)
        if args.skip_existing:
            complete, reason = existing_run_is_complete(horizon, paths, config_hash)
            if complete:
                print(f"\n[{horizon}m] SKIP: {reason}")
                skipped.append(f"{horizon}m")
                continue
            print(f"\n[{horizon}m] not skipping: {reason}")
        try:
            completed.append(run_horizon(horizon, args, features, audits[horizon],
                                         paths, params, config_hash))
        except Exception:
            print(f"\n[{horizon}m] FAILED; existing artifacts were left untouched.", file=sys.stderr)
            traceback.print_exc()
            failed.append(f"{horizon}m")
        finally:
            tu.free_memory()

    # ---------- summary ----------
    print("\n" + "=" * 70)
    print("FORECASTING RUN COMPLETE" if not failed else "FORECASTING RUN FINISHED WITH FAILURES")
    print("=" * 70)
    if completed:
        rows = [{"horizon": r["horizon"], "best_iter": r["best_iteration"],
                 "val_thr": r["validation"]["threshold"], "val_pr_auc": r["validation"]["pr_auc"],
                 "test_prec": r["test"]["precision"], "test_rec": r["test"]["recall"],
                 "test_f1": r["test"]["f1"], "test_pr_auc": r["test"]["pr_auc"],
                 "test_roc_auc": r["test"]["roc_auc"], "test_FPR": r["test"]["false_positive_rate"],
                 "test_FA_rows/day": r["test"]["false_alarms_per_day"]} for r in completed]
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nTest metrics are an exploratory benchmark (July split reused across experiments).")
    if skipped:
        print("Skipped (already complete):", ", ".join(skipped))
    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
    if args.build_comparisons and (completed or skipped):
        tu.build_comparisons(Path(args.results_root))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())