"""Build an immutable, full-row canonical feature/target table for a run.

This is an additive documentation/training convenience artifact. It never
changes files under the preprocessing run directory and refuses to overwrite.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "preproc_20261124_v1"
RUN = ROOT / "output" / "preprocessing_runs" / RUN_ID
OUT = ROOT / "output" / "master_datasets" / RUN_ID
SOURCE = RUN / "labels" / "forecast_labeled_timeseries_minute_end_v2.parquet"
SCHEMA = RUN / "ml_ready" / "feature_schema.json"
TARGETS = ["flare_active", "flare_class", "flare_peak", "flare_15m", "flare_30m", "flare_60m"]
ROLLING = [5, 15, 30, 60]
LAGS = [1, 5, 15]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite canonical master directory: {OUT}")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    approved = schema["features"]
    df = pd.read_parquet(SOURCE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if not df.timestamp.is_unique or not df.timestamp.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique and increasing")
    required = {"timestamp", "date", *TARGETS, "solexs_sdd2_counts"}
    if not required.issubset(df.columns):
        raise ValueError(f"Source missing required fields: {sorted(required-set(df.columns))}")

    hel = [c for c in df if c.startswith("hel1os_") and not c.endswith(("_gti", "_missing", "_stat_err", "_stat_err_missing"))]
    signals = hel + ["solexs_sdd2_counts"]
    usable = df[signals].notna().any(axis=1)
    features = {}
    features["hel1os_available"] = df[hel].notna().any(axis=1).astype("int8")
    features["solexs_available"] = df["solexs_sdd2_counts"].notna().astype("int8")

    for col in signals:
        features[f"{col}_log1p"] = np.log1p(df[col])
        base = "solexs_sdd2" if col == "solexs_sdd2_counts" else col
        features[f"{col}_missing"] = df[f"{base}_missing"].astype("int8")
        features[f"{col}_gti"] = df[f"{base}_gti"].astype("int8")
        err = f"{col}_stat_err"
        if err in df:
            features[f"{col}_stat_err_log1p"] = np.log1p(df[err])
            features[f"{col}_stat_err_missing"] = df[f"{err}_missing"].astype("int8")

    time_gap = df.timestamp.diff().ne(pd.Timedelta(minutes=1))
    prev_bad = (~usable).shift(1, fill_value=True)
    breaks = (time_gap | prev_bad) & usable
    segment = breaks.cumsum().astype("int64").where(usable, -1)
    for col in signals:
        values = np.log1p(df[col]).where(usable)
        grouped = values.groupby(segment, sort=False)
        for w in ROLLING:
            for stat, minp in [("mean", 1), ("std", 2), ("count", 1), ("min", 1), ("max", 1)]:
                features[f"{col}_roll{w}_{stat}"] = grouped.transform(lambda s, w=w, stat=stat, minp=minp: getattr(s.rolling(window=w, min_periods=minp), stat)())
        for lag in LAGS:
            features[f"{col}_diff_{lag}m"] = grouped.diff(lag)

    hour = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    doy = df.timestamp.dt.dayofyear
    features["time_sin"] = np.sin(2 * np.pi * hour / 24)
    features["time_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    features["day_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X = pd.DataFrame(features, index=df.index)
    if X.columns.tolist() != approved:
        missing = [x for x in approved if x not in X]
        extra = [x for x in X if x not in approved]
        first_order_mismatch = next((i for i,(a,b) in enumerate(zip(X.columns, approved)) if a != b), None)
        raise ValueError(f"Feature schema mismatch: got={X.shape[1]} expected={len(approved)}, missing={missing[:5]}, extra={extra[:5]}, order_mismatch_at={first_order_mismatch}")

    # Preserve raw quality columns alongside identically named model indicators.
    # The collision is resolved with a raw__ prefix (for example raw__..._gti).
    raw_renames = {c: f"raw__{c}" for c in df.columns if c in approved}
    raw = df.rename(columns=raw_renames)
    out = pd.concat([raw.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    out.insert(0, "observation_id", out.timestamp.dt.strftime("%Y%m%dT%H%M%SZ"))
    out.insert(1, "source_run_id", RUN_ID)
    if out.observation_id.duplicated().any() or out.shape[1] != 705 or len(out) != 298080:
        raise ValueError(f"Unexpected master shape or duplicate IDs: {out.shape}")

    if OUT.exists() and not any(OUT.iterdir()):
        OUT.rmdir()  # only remove the empty directory left by a failed first attempt
    OUT.mkdir(parents=True, exist_ok=False)
    master = OUT / "canonical_master_features_targets.parquet"
    out.to_parquet(master, index=False, compression="zstd")
    schema_out = OUT / "canonical_master_schema.json"
    schema_doc = {
        "dataset_run_id": RUN_ID,
        "rows": len(out),
        "columns": len(out.columns),
        "feature_count": len(approved),
        "features": approved,
        "target_columns": TARGETS,
        "metadata_columns": ["observation_id", "source_run_id", "date", "timestamp"],
        "source_columns": [c for c in df.columns],
        "raw_column_renames_for_feature_collisions": raw_renames,
        "timestamp_convention": "UTC minute-end; each row represents the preceding one-minute bin",
        "storage": "Parquet, Zstandard compression",
    }
    schema_out.write_text(json.dumps(schema_doc, indent=2), encoding="utf-8")
    manifest = {
        "dataset_run_id": RUN_ID,
        "source_run_manifest_sha256": sha256(RUN / "run_manifest.json"),
        "source_forecast_table_sha256": sha256(SOURCE),
        "feature_schema_sha256": sha256(SCHEMA),
        "master_file": master.name,
        "master_sha256": sha256(master),
        "schema_file": schema_out.name,
        "schema_sha256": sha256(schema_out),
        "rows": len(out), "columns": len(out.columns), "feature_count": len(approved),
        "first_timestamp_utc": str(out.timestamp.min()),
        "last_timestamp_utc": str(out.timestamp.max()),
        "represented_dates": int(out.date.nunique()),
        "note": "Additive artifact. Original preprocessing outputs remain unchanged.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
