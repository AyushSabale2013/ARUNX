"""Validate every artifact produced by run_preprocessing.py and write a report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_master(df: pd.DataFrame, label: str) -> dict[str, Any]:
    issues: list[str] = []
    if "timestamp" not in df:
        return {"name": label, "rows": len(df), "issues": ["timestamp column missing"]}
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any(): issues.append(f"invalid timestamps: {int(ts.isna().sum())}")
    if ts.duplicated().any(): issues.append(f"duplicate timestamps: {int(ts.duplicated().sum())}")
    if not ts.is_monotonic_increasing: issues.append("timestamps are not sorted")

    signal_cols = [
        col for col in df.columns
        if col.startswith(("hel1os_", "solexs_"))
        and not col.endswith(("_gti", "_missing", "_stat_err", "_stat_err_missing"))
    ]
    mismatches = outside_gti = 0
    for col in signal_cols:
        suffix = "solexs_sdd2" if col == "solexs_sdd2_counts" else col
        gti_col = f"{suffix}_gti"
        missing_col = f"{suffix}_missing"
        if gti_col not in df or missing_col not in df:
            issues.append(f"quality flags missing for {col}")
            continue
        signal = pd.to_numeric(df[col], errors="coerce")
        gti = df[gti_col].fillna(False).astype(bool)
        missing = df[missing_col].fillna(True).astype(bool)
        mismatches += int((signal.isna() != missing).sum())
        outside_gti += int((signal.notna() & ~gti).sum())
        if np.isinf(signal.dropna().to_numpy(dtype=float)).any():
            issues.append(f"infinite values in {col}")
    if mismatches: issues.append(f"missing-flag mismatches: {mismatches}")
    if outside_gti: issues.append(f"signal values outside GTI: {outside_gti}")

    target_cols = [c for c in df.columns if c.startswith("flare_")]
    target_counts = {c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).sort_index().items()} for c in target_cols}
    for col in target_cols:
        allowed = {-1, 0, 1} if col.endswith(("15m", "30m", "60m")) else {0, 1, 2}
        if not set(pd.to_numeric(df[col], errors="coerce").dropna().unique()).issubset(allowed):
            issues.append(f"unexpected target values in {col}")
    if {"flare_active", "flare_class"}.issubset(df.columns):
        if ((df["flare_active"] == 1) & (df["flare_class"] == 0)).any():
            issues.append("active flare rows have class 0")
    # Minute-end rows are named for the UTC day containing their represented
    # interval, so 00:00 belongs to the previous bin day.
    bin_dates = (ts - pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d")
    if "date" in df:
        stored_dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if (stored_dates != bin_dates).any():
            issues.append("stored date disagrees with minute-end bin ownership")
    dates = bin_dates
    daily_rows = dates.value_counts().to_dict()
    bad_days = {str(day): int(count) for day, count in daily_rows.items() if int(count) != 1440}
    if bad_days:
        issues.append(f"daily row counts differ from 1,440: {len(bad_days)} dates")
    return {
        "name": label,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start_utc": str(ts.min()),
        "end_utc": str(ts.max()),
        "unique_dates": int(dates.nunique()),
        "date_row_counts": {str(k): int(v) for k, v in sorted(daily_rows.items())},
        "dates_not_1440_rows": bad_days,
        "signal_channel_count": len(signal_cols),
        "target_distributions": target_counts,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--merge-suffix", choices=["", "_test"], default="")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    merged_path = run_dir / "merged" / f"merged_timeseries_minute_end_v2{args.merge_suffix}.parquet"
    labeled_path = run_dir / "labels" / "labeled_timeseries_minute_end_v2.parquet"
    forecast_path = run_dir / "labels" / "forecast_labeled_timeseries_minute_end_v2.parquet"
    required = [merged_path, labeled_path, forecast_path]
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Required pipeline outputs missing: " + ", ".join(missing_paths))

    checks = [
        validate_master(pd.read_parquet(merged_path), "merged"),
        validate_master(pd.read_parquet(labeled_path), "labeled"),
        validate_master(pd.read_parquet(forecast_path), "forecast_labeled"),
    ]

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ml_dir = run_dir / "ml_ready"
    schema_path = ml_dir / "feature_schema.json"
    summary_path = ml_dir / "dataset_summary.csv"
    if not schema_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Feature schema or dataset summary is missing")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_names = schema.get("features", [])
    feature_count = len(feature_names)
    if feature_count != schema.get("feature_count"):
        raise ValueError("Feature schema count does not match the feature list")
    forbidden = [name for name in feature_names if name.startswith("flare_") or name in {"target", "class_type", "intensity", "start", "peak", "end"}]
    if forbidden:
        raise ValueError(f"Label or event fields leaked into features: {forbidden[:10]}")
    if schema.get("row_policy") != config["features"].get("row_policy", "either_instrument"):
        raise ValueError("Feature schema row policy does not match run config")

    tasks = ["classification", "nowcasting", "peak_detection", *[f"forecast_{int(h)}m" for h in config.get("targets", {}).get("forecast_horizons_minutes", [15, 30, 60])]]
    splits = ["train", "validation", "test"]
    dataset_files = []
    ml_stats = []
    train_end = pd.Timestamp(config["splits"]["train_end_exclusive_utc"])
    val_end = pd.Timestamp(config["splits"]["validation_end_exclusive_utc"])
    for task in tasks:
        for split in splits:
            path = ml_dir / f"{task}_{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            if not {"timestamp", "target", *feature_names}.issubset(frame.columns):
                raise ValueError(f"Dataset schema mismatch: {path.name}")
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
                raise ValueError(f"Timestamp validation failed: {path.name}")
            sample_dates = ts - pd.Timedelta(minutes=1)
            if split == "train" and len(ts) and sample_dates.max() >= train_end:
                raise ValueError(f"Train rows cross train boundary: {path.name}")
            if split == "validation" and len(ts) and (sample_dates.min() < train_end or sample_dates.max() >= val_end):
                raise ValueError(f"Validation rows cross date boundaries: {path.name}")
            if split == "test" and len(ts) and sample_dates.min() < val_end:
                raise ValueError(f"Test rows precede validation boundary: {path.name}")
            target_values = set(pd.to_numeric(frame["target"], errors="coerce").dropna().unique())
            expected = {0, 1, 2} if task == "classification" else {0, 1}
            if not target_values.issubset(expected):
                raise ValueError(f"Invalid target values {target_values} in {path.name}")
            numeric = frame[feature_names].select_dtypes(include=np.number)
            if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Infinite feature values in {path.name}")
            dataset_files.append({"path": str(path.relative_to(run_dir)), "rows": int(len(frame)), "sha256": sha256_file(path)})
            ml_stats.append({
                "task": task,
                "split": split,
                "rows": int(len(frame)),
                "start_utc": str(ts.min()) if len(ts) else None,
                "end_utc": str(ts.max()) if len(ts) else None,
                "target_distribution": {str(k): int(v) for k, v in frame["target"].value_counts().sort_index().items()},
                "feature_missing_values": int(frame[feature_names].isna().sum().sum()),
            })

    extraction_path = reports / "extraction" / "selective_extraction_report.csv"
    extraction = pd.read_csv(extraction_path) if extraction_path.exists() else pd.DataFrame()
    acquisition = json.loads((reports / "acquisition_manifest.json").read_text(encoding="utf-8"))
    archive_counts: dict[str, int] = {}
    for row in acquisition.get("entries", []):
        status = row.get("status", "unknown")
        archive_counts[status] = archive_counts.get(status, 0) + 1

    report = {
        "run_id": run_dir.name,
        "generated_utc": datetime_now_utc(),
        "status": "PASS",
        "date_range_requested": config["date_range"],
        "archive_count_by_status": archive_counts,
        "extraction_file_count": int(len(extraction)),
        "extraction_count_by_status": extraction["status"].value_counts().to_dict() if not extraction.empty else {},
        "raw_sample_quality": {},
        "merged_rows": checks[0]["rows"],
        "merged_dates": checks[0]["unique_dates"],
        "labelled_rows": checks[1]["rows"],
        "forecast_labelled_rows": checks[2]["rows"],
        "feature_count": feature_count,
        "row_policy": schema.get("row_policy"),
        "task_split_datasets": ml_stats,
        "task_dataset_files": dataset_files,
        "master_table_checks": checks,
        "notes": [
            "NaN feature values are preserved and the training scripts fit imputation/scaling on the training split only.",
            "Hard-to-soft ratio is not computed from uncalibrated count/count-rate columns; response/units must be reconciled first.",
            "This run uses M/X targets and 15/30/60-minute horizons defined by the current code, not the alternative 4-state and 1-24-hour proposal in attached documents.",
        ],
    }
    quality_path = reports / "lightcurve_quality_detail.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        count_column = lambda name: int(pd.to_numeric(quality[name], errors="coerce").fillna(0).sum()) if name in quality else 0
        report["raw_sample_quality"] = {
            "files_audited": int(len(quality)),
            "passed_files": int(quality["status"].eq("PASS").sum()) if "status" in quality else None,
            "failed_or_review_files": int((~quality["status"].eq("PASS")).sum()) if "status" in quality else None,
            "raw_samples": count_column("rows"),
            "nonfinite_timestamps": count_column("nan_timestamps") + count_column("inf_timestamps"),
            "duplicate_timestamps": count_column("duplicate_timestamps"),
            "samples_outside_gti": count_column("samples_outside_gti"),
        }
    warnings: list[str] = []
    date_coverage_path = reports / "date_coverage.csv"
    if date_coverage_path.exists():
        date_coverage = pd.read_csv(date_coverage_path)
        missing_dates = date_coverage.loc[date_coverage["merge_status"] != "MERGED", "date"].astype(str).tolist()
        report["requested_dates_without_merged_rows"] = missing_dates
        if missing_dates:
            warnings.append(f"{len(missing_dates)} requested calendar dates have no merged table")

    manual_path = run_dir / "labels" / "goes_events_manual_review.csv"
    unmatched_path = run_dir / "labels" / "goes_unmatched_lines.csv"
    manual = pd.read_csv(manual_path) if manual_path.exists() else pd.DataFrame()
    unmatched = pd.read_csv(unmatched_path) if unmatched_path.exists() else pd.DataFrame()
    report["goes_catalogue_review"] = {
        "manual_review_rows": int(len(manual)),
        "unmatched_event_like_rows": int(len(unmatched)),
        "manual_review_file": str(manual_path.relative_to(run_dir)) if manual_path.exists() else None,
        "unmatched_file": str(unmatched_path.relative_to(run_dir)) if unmatched_path.exists() else None,
    }
    if len(manual):
        warnings.append(f"{len(manual)} GOES event rows require manual review and were excluded")
    if len(unmatched):
        warnings.append(f"{len(unmatched)} date-like GOES catalogue rows did not match the parser format")

    goes_path = run_dir / "labels" / "goes_events_parsed.csv"
    goes = pd.read_csv(goes_path) if goes_path.exists() else pd.DataFrame()
    split_event_counts = {}
    if not goes.empty and {"start", "class_type"}.issubset(goes.columns):
        goes["start"] = pd.to_datetime(goes["start"], utc=True, errors="coerce")
        goes["class_type"] = goes["class_type"].astype(str).str.upper()
        mx_events = goes[goes["class_type"].isin(["M", "X"]) & goes["start"].notna()]
        study_start = pd.Timestamp(config["date_range"]["start"], tz="UTC")
        train_end = pd.Timestamp(config["splits"]["train_end_exclusive_utc"])
        val_end = pd.Timestamp(config["splits"]["validation_end_exclusive_utc"])
        study_end = pd.Timestamp(config["date_range"]["end"], tz="UTC") + pd.Timedelta(days=1)
        event_intervals = {"train": (study_start, train_end), "validation": (train_end, val_end), "test": (val_end, study_end)}
        for split, (left, right) in event_intervals.items():
            part = mx_events[(mx_events["start"] >= left) & (mx_events["start"] < right)]
            split_event_counts[split] = {"M": int((part["class_type"] == "M").sum()), "X": int((part["class_type"] == "X").sum()), "M_or_X": int(len(part))}
    report["goes_mx_events_by_split"] = split_event_counts
    issues = [f"{c['name']}: {issue}" for c in checks for issue in c["issues"]]
    if any(status in {"local_invalid", "failed", "invalid_url", "existing_invalid", "existing_checksum_mismatch"} for status in archive_counts):
        issues.append("one or more source archives failed verification")
    if not extraction.empty and extraction["status"].eq("error").any():
        issues.append("one or more required archive members failed extraction")
    if issues:
        report["status"] = "FAIL"
        report["issues"] = issues
    elif warnings:
        report["status"] = "PASS_WITH_REVIEW"
    report["warnings"] = warnings
    (reports / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(ml_stats).to_csv(reports / "dataset_quality.csv", index=False)
    print(f"Validation {report['status']}: {reports / 'preprocessing_report.json'}")
    if issues:
        raise RuntimeError("Validation failed: " + "; ".join(issues[:12]))


def datetime_now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
