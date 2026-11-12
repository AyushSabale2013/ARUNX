"""
Baseline model training for ARUN-X (upgraded).

Models: logistic_regression, random_forest, extra_trees,
        hist_gradient_boosting, lightgbm

Design rules
------------
* All data loading, preprocessing, threshold selection, metrics and prediction
  saving come from training_utils.py, so every model family is scored
  identically.
* Features come from feature_schema.json (fixed order) via get_feature_names().
* Preprocessing statistics are fitted on TRAIN only. Validation is the only
  data used for tuning (early stopping, C selection, threshold). Test is only
  scored once per model with the validation-selected threshold.
* Datasets, labels and chronological splits are never modified.

Memory strategy (8 GB RAM)
--------------------------
Each split lives in ONE float32 array. Models run in "stages" so the arrays are
transformed in place at most twice and never copied:
    stage 0  raw (NaN kept)          -> lightgbm, hist_gradient_boosting
    stage 1  median-imputed          -> random_forest, extra_trees
    stage 2  imputed + standardized  -> logistic_regression
Preprocessing parameters are saved next to the models so inference can
reproduce them (preprocessing_<horizon>m.npz).

Usage
-----
    python scripts/ml/train_baseline_models.py
    python scripts/ml/train_baseline_models.py --horizons 15 --models lightgbm --threads 3
    python scripts/ml/train_baseline_models.py --skip-existing
"""
from __future__ import annotations

import os
import sys

# ------------------------------------------------------------------
# Thread cap. Must happen BEFORE training_utils / numpy / sklearn load,
# because BLAS/OpenMP read these variables only at import time.
# ------------------------------------------------------------------
def _early_threads(argv):
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--threads="):
            return arg.split("=", 1)[1]
    return None


_requested_threads = _early_threads(sys.argv[1:])
if _requested_threads is not None:
    try:
        _n = max(1, min(8, int(_requested_threads)))
    except ValueError:
        _n = 4
    os.environ["ARUNX_THREADS"] = str(_n)
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_var] = str(_n)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import training_utils as tu  # noqa: E402  (must precede numpy/sklearn)

import argparse  # noqa: E402
import platform  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

FAMILY = "baselines"
ALL_MODELS = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
)

# Which in-place preprocessing stage each model needs (see module docstring).
MODEL_STAGE = {
    "lightgbm": 0,
    "hist_gradient_boosting": 0,
    "random_forest": 1,
    "extra_trees": 1,
    "logistic_regression": 2,
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train ARUN-X baseline models.")
    tu.add_path_args(p)
    p.add_argument("--horizons", type=int, nargs="+", default=list(tu.HORIZONS),
                   choices=list(tu.HORIZONS))
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument("--threads", type=int, default=tu.THREADS,
                   help="CPU thread cap (3-4 recommended for 8 GB laptops).")
    p.add_argument("--expected-features", type=int, default=None,
                   help="Fail if the feature count differs (sanity check).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip only when model, metadata, predictions, and required preprocessing artifacts exist.")

    # Logistic regression
    p.add_argument("--lr-c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0],
                   help="C values; best is chosen by VALIDATION PR-AUC.")
    p.add_argument("--lr-max-iter", type=int, default=1000)
    p.add_argument("--lr-tol", type=float, default=1e-4)
    p.add_argument("--lr-clip", type=float, default=10.0,
                   help="Clip standardized features to +/-clip for LR (inf to disable).")

    # Random forest / extra trees
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=14)
    p.add_argument("--rf-min-leaf", type=int, default=10)
    p.add_argument("--rf-max-samples", type=float, default=0.5,
                   help="Bootstrap fraction per RF tree (cuts memory/time).")
    p.add_argument("--et-trees", type=int, default=200)
    p.add_argument("--et-max-depth", type=int, default=14)
    p.add_argument("--et-min-leaf", type=int, default=10)

    # HistGradientBoosting
    p.add_argument("--hgb-max-iter", type=int, default=400)
    p.add_argument("--hgb-eval-every", type=int, default=10)
    p.add_argument("--hgb-patience", type=int, default=10,
                   help="Stop scanning after this many evaluations without validation PR-AUC gain.")

    # LightGBM
    p.add_argument("--lgbm-max-estimators", type=int, default=2000)
    p.add_argument("--lgbm-early-stopping", type=int, default=100)
    p.add_argument("--lgbm-imbalance", choices=("scale_pos_weight", "none"),
                   default="scale_pos_weight",
                   help="Class imbalance handling (never combined with class_weight).")
    return p.parse_args()


def validate_args(args) -> None:
    """Reject invalid configurations before loading large datasets."""
    positive_ints = {
        "threads": args.threads,
        "lr_max_iter": args.lr_max_iter,
        "rf_trees": args.rf_trees,
        "rf_min_leaf": args.rf_min_leaf,
        "et_trees": args.et_trees,
        "et_min_leaf": args.et_min_leaf,
        "hgb_max_iter": args.hgb_max_iter,
        "hgb_eval_every": args.hgb_eval_every,
        "hgb_patience": args.hgb_patience,
        "lgbm_max_estimators": args.lgbm_max_estimators,
        "lgbm_early_stopping": args.lgbm_early_stopping,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.threads > 8:
        raise ValueError("--threads must be between 1 and 8 on this training setup.")
    if not args.lr_c_grid or any(not math.isfinite(c) or c <= 0 for c in args.lr_c_grid):
        raise ValueError("Every --lr-c-grid value must be finite and greater than zero.")
    if not math.isfinite(args.lr_tol) or args.lr_tol <= 0:
        raise ValueError("--lr-tol must be finite and greater than zero.")
    if not (math.isinf(args.lr_clip) or (math.isfinite(args.lr_clip) and args.lr_clip > 0)):
        raise ValueError("--lr-clip must be positive or inf.")
    if not (0.0 < args.rf_max_samples <= 1.0):
        raise ValueError("--rf-max-samples must be in the interval (0, 1].")
    if args.rf_max_depth <= 0 or args.et_max_depth <= 0:
        raise ValueError("RF/ET max depth must be positive; use a large value instead of None.")
    if args.expected_features is not None and args.expected_features <= 0:
        raise ValueError("--expected-features must be positive.")
    if len(set(args.horizons)) != len(args.horizons):
        raise ValueError("--horizons contains duplicates.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models contains duplicates.")


# ============================================================
# HELPERS
# ============================================================

def library_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "pyarrow", "scipy"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def class_counts(y: np.ndarray):
    n_pos = int(y.sum())
    return len(y) - n_pos, n_pos


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """sklearn 'balanced' weights from TRAIN labels: n / (2 * n_class)."""
    n_neg, n_pos = class_counts(y)
    n = len(y)
    w = np.where(y == 1, n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1)))
    return w.astype(np.float32)


def predict_proba_chunked(predict, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Return validated positive-class probabilities without a full-size copy of X."""
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"Expected a non-empty 2D feature matrix; got shape {X.shape}.")
    if chunk <= 0:
        raise ValueError("Prediction chunk size must be positive.")
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), chunk):
        proba = np.asarray(predict(X[start : start + chunk]))
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError(f"predict_proba must return shape (n, 2); got {proba.shape}.")
        positive = proba[:, 1]
        if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
            raise ValueError("Model produced non-finite or out-of-range probabilities.")
        out[start : start + len(positive)] = positive.astype(np.float32, copy=False)
    return out


def save_joblib_atomic(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(obj, tmp, compress=3)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_preprocessing_atomic(path: Path, prep: dict, clip: float) -> None:
    """Write preprocessing parameters atomically so interrupted runs cannot leave a corrupt NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        tu.save_preprocessing(tmp, prep, clip=clip)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_predictions_atomic(path: Path, ts, y, probabilities, predictions) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        tu.save_predictions(tmp, ts, y, probabilities, predictions)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def split_summary(ts, y) -> dict:
    n_neg, n_pos = class_counts(y)
    return {
        "rows": int(len(y)),
        "positives": n_pos,
        "negatives": n_neg,
        "positive_rate": float(y.mean()) if len(y) else None,
        "start": str(ts.min()),
        "end": str(ts.max()),
    }


class Data:
    """Holds the three float32 arrays and applies preprocessing IN PLACE,
    one stage at a time (raw -> imputed -> standardized)."""

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, ts_test, lr_clip):
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.ts_test = ts_test
        self.lr_clip = lr_clip
        self.stage = 0
        self.prep = None  # fitted on raw TRAIN only

    def advance_to(self, stage: int) -> None:
        if stage < self.stage:
            raise RuntimeError("Models must run in non-decreasing preprocessing-stage order.")
        if stage >= 1 and self.stage < 1:
            self.prep = tu.fit_preprocessing(self.X_train)  # train statistics only
            for X in (self.X_train, self.X_val, self.X_test):
                tu.impute_inplace(X, self.prep["medians"])
            self.stage = 1
        if stage >= 2 and self.stage < 2:
            for X in (self.X_train, self.X_val, self.X_test):
                tu.standardize_inplace(X, self.prep["means"], self.prep["stds"], clip=self.lr_clip)
            self.stage = 2


# ============================================================
# MODEL FITTERS  (each returns: estimator, info dict)
# All tuning uses VALIDATION data only.
# ============================================================

def fit_logistic_regression(args, d: Data):
    results, best = [], None

    for C in args.lr_c_grid:
        t0 = time.perf_counter()
        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=args.lr_max_iter,
            tol=args.lr_tol,
            random_state=args.seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(d.X_train, d.y_train)
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        n_iter = int(np.max(model.n_iter_))
        converged = (not warned) and n_iter < args.lr_max_iter

        val_p = predict_proba_chunked(model.predict_proba, d.X_val)
        ap = float(average_precision_score(d.y_val, val_p))
        results.append({"C": C, "val_pr_auc": ap, "n_iter": n_iter, "converged": converged,
                        "seconds": round(time.perf_counter() - t0, 2)})
        print(f"    C={C:<7} val PR-AUC={ap:.4f} n_iter={n_iter} converged={converged}")

        if best is None or ap > best[0]:
            best = (ap, model, C, converged, n_iter)
        else:
            del model

    _, model, C, converged, n_iter = best
    if not converged:
        print(f"  WARNING: logistic regression did not converge (n_iter={n_iter}); "
              f"consider raising --lr-max-iter.")
    return model, {
        "tuning_grid": results,
        "selected_C": C,
        "converged": converged,
        "n_iter": n_iter,
        "clip": args.lr_clip,
        "selection_metric": "validation PR-AUC",
    }


def fit_random_forest(args, d: Data):
    model = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_leaf,
        max_features="sqrt",
        bootstrap=True,
        max_samples=args.rf_max_samples,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=args.threads,
    )
    model.fit(d.X_train, d.y_train)
    return model, {
        "imbalance_strategy": "balanced_subsample",
        "training_rows": int(len(d.y_train)),
        "n_features_in": int(model.n_features_in_),
    }


def fit_extra_trees(args, d: Data):
    model = ExtraTreesClassifier(
        n_estimators=args.et_trees,
        max_depth=args.et_max_depth,
        min_samples_leaf=args.et_min_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.threads,
    )
    model.fit(d.X_train, d.y_train)
    return model, {
        "imbalance_strategy": "balanced class_weight",
        "training_rows": int(len(d.y_train)),
        "n_features_in": int(model.n_features_in_),
    }


def fit_hist_gradient_boosting(args, d: Data):
    """
    sklearn's built-in early_stopping=True carves a RANDOM validation_fraction
    out of the training rows, which leaks future information into past rows for
    time-series data. Instead:
      1. fit with early_stopping=False for hgb_max_iter iterations,
      2. score the CHRONOLOGICAL validation split at checkpoints via
         staged_predict_proba and keep the best iteration (PR-AUC),
      3. refit with max_iter = best iteration (identical seed/params).
    Balanced class weights are passed as sample_weight (works on every sklearn).
    """
    weights = balanced_sample_weight(d.y_train)

    def build(max_iter):
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=max_iter,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.seed,
        )

    model = build(args.hgb_max_iter)
    model.fit(d.X_train, d.y_train, sample_weight=weights)
    full_iters = int(model.n_iter_)

    best_ap, best_iter, bad, history = -1.0, full_iters, 0, []
    for i, proba in enumerate(model.staged_predict_proba(d.X_val), start=1):
        if i % args.hgb_eval_every and i != full_iters:
            continue
        ap = float(average_precision_score(d.y_val, proba[:, 1]))
        history.append((i, ap))
        if ap > best_ap + 1e-6:
            best_ap, best_iter, bad = ap, i, 0
        else:
            bad += 1
            if bad >= args.hgb_patience:
                break

    refit = best_iter < full_iters
    if refit:
        del model
        tu.free_memory()
        model = build(best_iter)
        model.fit(d.X_train, d.y_train, sample_weight=weights)
    del weights

    print(f"    best iteration={best_iter}/{full_iters} (val PR-AUC={best_ap:.4f}, refit={refit})")
    return model, {
        "early_stopping": "chronological validation PR-AUC via staged_predict_proba (no random split)",
        "max_iter": args.hgb_max_iter,
        "best_iteration": best_iter,
        "refit_with_best_iteration": refit,
        "best_val_pr_auc": best_ap,
        "imbalance": "balanced sample_weight (train labels)",
    }


def fit_lightgbm(args, d: Data):
    """
    Imbalance: scale_pos_weight = n_neg / n_pos from TRAIN labels, and NO
    class_weight (the two both reweight positives; stacking them would square the
    effect). Use --lgbm-imbalance none for an unweighted model.
    Early stopping monitors validation average precision (same family of metric
    as the PR-AUC used for model comparison).
    Early-stopping API: callbacks (lightgbm >= 3.3 / 4.x) with a fallback to
    early_stopping_rounds for older releases.
    """
    import lightgbm as lgb

    n_neg, n_pos = class_counts(d.y_train)
    spw = (n_neg / max(n_pos, 1)) if args.lgbm_imbalance == "scale_pos_weight" else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=args.lgbm_max_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        max_bin=255,
        force_col_wise=True,
        random_state=args.seed,
        n_jobs=args.threads,
        verbosity=-1,
    )

    if hasattr(lgb, "early_stopping") and hasattr(lgb, "log_evaluation"):
        fit_kwargs = {
            "callbacks": [
                lgb.early_stopping(args.lgbm_early_stopping, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        }
        api = "callbacks"
    else:  # very old lightgbm
        fit_kwargs = {"early_stopping_rounds": args.lgbm_early_stopping, "verbose": False}
        api = "early_stopping_rounds"

    model.fit(d.X_train, d.y_train, eval_set=[(d.X_val, d.y_val)], **fit_kwargs)

    best_iter = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
    print(f"    best iteration={best_iter}/{args.lgbm_max_estimators} scale_pos_weight={spw:.3f}")
    return model, {
        "imbalance_strategy": args.lgbm_imbalance,
        "scale_pos_weight": float(spw),
        "class_weight": None,
        "early_stopping_api": api,
        "early_stopping_metric": "average_precision (validation)",
        "early_stopping_rounds": args.lgbm_early_stopping,
        "best_iteration": best_iter,
        "lightgbm_version": lgb.__version__,
    }


FITTERS = {
    "logistic_regression": fit_logistic_regression,
    "random_forest": fit_random_forest,
    "extra_trees": fit_extra_trees,
    "hist_gradient_boosting": fit_hist_gradient_boosting,
    "lightgbm": fit_lightgbm,
}


# ============================================================
# ONE MODEL
# ============================================================

def train_one_model(name, horizon, args, d, features, split_info, model_dir, result_dir, versions):
    print("\n" + "=" * 65)
    print(f"MODEL: {name} | HORIZON: {horizon}m")
    print("=" * 65)

    d.advance_to(MODEL_STAGE[name])
    completion_marker = model_dir / f".{name}_{horizon}m.incomplete"
    completion_marker.write_text("Training started; artifacts are not yet complete.\n")
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    print("  Training...")
    model, info = FITTERS[name](args, d)
    train_seconds = time.perf_counter() - t0
    print(f"  Training time: {train_seconds:.1f}s")

    # ---- validation: threshold selection (validation only) ----
    val_p = predict_proba_chunked(model.predict_proba, d.X_val)
    threshold = tu.select_threshold(d.y_val, val_p)
    val_metrics, _ = tu.compute_metrics(d.y_val, val_p, threshold)
    tu.print_metrics("validation", val_metrics)

    # ---- test: scored once with the validation threshold ----
    test_p = predict_proba_chunked(model.predict_proba, d.X_test)
    test_metrics, test_pred = tu.compute_metrics(d.y_test, test_p, threshold)
    tu.print_metrics("test", test_metrics)

    # ---- artifacts ----
    model_path = model_dir / f"{name}_{horizon}m.joblib"
    prep_path = None
    if MODEL_STAGE[name] >= 1:
        prep_path = model_dir / f"preprocessing_{horizon}m_{name}.npz"
        save_preprocessing_atomic(
            prep_path, d.prep,
            clip=d.lr_clip if MODEL_STAGE[name] == 2 else np.inf,
        )
    save_joblib_atomic(model, model_path)

    prediction_path = result_dir / f"{name}_{horizon}m_predictions.csv"
    save_predictions_atomic(prediction_path, d.ts_test, d.y_test, test_p, test_pred)

    metadata = {
        "model": name,
        "family": FAMILY,
        "horizon": f"{horizon}m",
        "created_utc": started.isoformat(),
        "training_seconds": round(train_seconds, 2),
        "total_seconds": round(time.perf_counter() - t0, 2),
        "seed": args.seed,
        "threads": args.threads,
        "features": features,
        "feature_count": len(features),
        "feature_source": "feature_schema.json" if (Path(args.data_dir) / "feature_schema.json").exists()
        else "parquet columns (excluding timestamp/target)",
        "preprocessing": {
            "stage": ["raw (NaN-native)", "median-impute (train)", "median-impute + standardize (train)"][
                MODEL_STAGE[name]],
            "file": prep_path.name if prep_path else None,
            "fitted_on": "train split only",
        },
        "hyperparameters": model.get_params(deep=False),
        "model_details": info,
        "threshold": threshold,
        "threshold_selection": "F1-optimal on validation split only; validation is also used for model/hyperparameter selection",
        "splits": split_info,
        "training_class_distribution": {
            "negative": class_counts(d.y_train)[0],
            "positive": class_counts(d.y_train)[1],
            "positive_rate": float(np.mean(d.y_train)),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "library_versions": versions,
        "command_line": sys.argv,
    }
    # Metadata is written last and acts as the completion marker for --skip-existing.
    tu.write_json(model_dir / f"{name}_{horizon}m_metadata.json", metadata)

    # ---- partial-rerun-safe metric tables (keyed by model+horizon) ----
    extra = {"training_seconds": round(train_seconds, 2)}
    tu.update_metrics_table(result_dir / "all_validation_metrics.csv",
                            [tu.metrics_row(FAMILY, name, horizon, {**val_metrics, **extra})])
    tu.update_metrics_table(result_dir / "all_test_metrics.csv",
                            [tu.metrics_row(FAMILY, name, horizon, {**test_metrics, **extra})])

    # A stale metadata file from an earlier run cannot make a failed rerun look complete.
    completion_marker.unlink(missing_ok=True)
    print(f"  Saved: {model_path}")

    del model, val_p, test_p, test_pred
    tu.free_memory()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    validate_args(args)
    if not 3 <= args.threads <= 4:
        print(f"NOTE: using {args.threads} thread(s); 3-4 is recommended for an 8 GB machine.")
    tu.set_seed(args.seed)

    model_dir = Path(args.models_root) / FAMILY
    result_dir = Path(args.results_root) / FAMILY
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    versions = library_versions()
    # Run in preprocessing-stage order so arrays are transformed in place only once.
    models = sorted(dict.fromkeys(args.models), key=lambda m: MODEL_STAGE[m])
    print(f"Threads: {args.threads} | models: {models} | horizons: {args.horizons}")

    for horizon in args.horizons:
        print("\n" + "#" * 70)
        print(f"FORECAST HORIZON: {horizon}m")
        print("#" * 70)

        def artifacts_complete(model_name: str) -> bool:
            required = [
                model_dir / f"{model_name}_{horizon}m.joblib",
                model_dir / f"{model_name}_{horizon}m_metadata.json",
                result_dir / f"{model_name}_{horizon}m_predictions.csv",
            ]
            if MODEL_STAGE[model_name] >= 1:
                required.append(model_dir / f"preprocessing_{horizon}m_{model_name}.npz")
            marker = model_dir / f".{model_name}_{horizon}m.incomplete"
            return (not marker.exists()) and all(
                path.is_file() and path.stat().st_size > 0 for path in required
            )

        pending = [
            m for m in models
            if not (args.skip_existing and artifacts_complete(m))
        ]
        skipped = sorted(set(models) - set(pending))
        if skipped:
            print(f"Skipping existing: {skipped}")
        if not pending:
            continue

        features = tu.get_feature_names(args.data_dir, horizon, args.expected_features)
        print(f"Features: {len(features)}")

        ts_tr, X_tr, y_tr = tu.load_split(tu.split_path(args.data_dir, horizon, "train"), features)
        ts_va, X_va, y_va = tu.load_split(tu.split_path(args.data_dir, horizon, "validation"), features)
        ts_te, X_te, y_te = tu.load_split(tu.split_path(args.data_dir, horizon, "test"), features)

        tu.check_chronology(ts_tr, ts_va, ts_te)
        tu.require_both_classes("train", y_tr)
        tu.require_both_classes("validation", y_va)
        tu.require_both_classes("test", y_te, strict=False)

        print("\nSplits:")
        tu.describe_split("train", ts_tr, X_tr, y_tr)
        tu.describe_split("validation", ts_va, X_va, y_va)
        tu.describe_split("test", ts_te, X_te, y_te)

        split_info = {
            "train": split_summary(ts_tr, y_tr),
            "validation": split_summary(ts_va, y_va),
            "test": split_summary(ts_te, y_te),
        }

        d = Data(X_tr, X_va, X_te, y_tr, y_va, y_te, ts_te, args.lr_clip)
        del X_tr, X_va, X_te, ts_tr, ts_va

        for name in pending:
            try:
                train_one_model(name, horizon, args, d, features, split_info,
                                model_dir, result_dir, versions)
            except Exception as exc:  # preserve completed models and continue the batch
                print(f"\nERROR: {name} {horizon}m failed: {type(exc).__name__}: {exc}")
                print("  This model/horizon is incomplete; rerun without --skip-existing after fixing the cause.")
                tu.free_memory()

        del d
        tu.free_memory()

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS COMPLETED")
    print("=" * 70)
    tu.build_comparisons(Path(args.results_root))
    print("\nResults:", result_dir)
    print("Models: ", model_dir)


if __name__ == "__main__":
    main()