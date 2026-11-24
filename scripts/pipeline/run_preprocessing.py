"""Run the real-data preprocessing pipeline into a new, versioned run folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = BASE / "config" / "preprocessing.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_step(name: str, command: list[str], env: dict[str, str], log_dir: Path) -> None:
    result = subprocess.run(command, cwd=BASE, env=env, capture_output=True, text=True)
    (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"Pipeline stage {name} failed (exit {result.returncode}). See {log_dir}.\n{tail}")


def date_inventory(config: dict, run_dir: Path) -> None:
    start = date.fromisoformat(config["date_range"]["start"])
    end = date.fromisoformat(config["date_range"]["end"])
    paths = config["paths"]
    archive_root = resolve(BASE, paths["archive_root"])
    raw_hel = resolve(BASE, paths["hel1os_raw"])
    raw_slx = resolve(BASE, paths["solexs_raw"])
    merge_summary = run_dir / "merged" / "merge_summary_minute_end_v2.csv"
    if not merge_summary.exists():
        merge_summary = run_dir / "merged" / "merge_summary_minute_end_v2_test.csv"
    merged: dict[str, str] = {}
    if merge_summary.exists():
        for row in csv.DictReader(merge_summary.open(encoding="utf-8-sig", newline="")):
            merged[row["date"]] = row["status"]

    rows = []
    current = start
    while current <= end:
        day = current.strftime("%Y%m%d")
        hel_archives = list((archive_root / "hel1os" / "level1" / day[:4]).rglob(f"HLS_{day}_*.zip")) if (archive_root / "hel1os" / "level1" / day[:4]).exists() else []
        slx_archives = list((archive_root / "solexs" / "level1" / day[:4]).rglob(f"AL1_SLX_L1_{day}_*.zip")) if (archive_root / "solexs" / "level1" / day[:4]).exists() else []
        hel_files = list(raw_hel.rglob(f"*{day}*")) if raw_hel.exists() else []
        slx_files = list(raw_slx.rglob(f"*{day}*")) if raw_slx.exists() else []
        hel_present = any(p.is_file() and p.name.startswith("lightcurve_") for p in hel_files)
        slx_present = any(p.is_file() and p.name.endswith(".lc.gz") for p in slx_files)
        rows.append({
            "date": current.isoformat(),
            "hel1os_archives": len(hel_archives),
            "solexs_archives": len(slx_archives),
            "hel1os_extracted_lightcurves_present": hel_present,
            "solexs_extracted_lightcurve_present": slx_present,
            "merge_status": merged.get(day, "NOT_PROCESSED"),
        })
        current += timedelta(days=1)
    pd.DataFrame(rows).to_csv(run_dir / "reports" / "date_coverage.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="unique output directory name; defaults to a UTC timestamp")
    parser.add_argument("--workers", type=int, help="override configured merge worker count")
    parser.add_argument("--limit", type=int, help="process first N common dates for a quick real-data trial")
    parser.add_argument("--skip-extract", action="store_true", help="use existing extracted FITS files after inventory checks")
    args = parser.parse_args()
    config_path = resolve(BASE, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start = date.fromisoformat(config["date_range"]["start"])
    end = date.fromisoformat(config["date_range"]["end"])
    if start > end:
        raise ValueError("date_range.start must not be after date_range.end")
    if config.get("time_grid", {}).get("frequency") != "1min" or config.get("time_grid", {}).get("timestamp_label") != "minute_end_utc":
        raise ValueError("This release supports only 1min minute_end_utc grids; update the merge and horizon semantics before changing cadence.")
    horizons = config.get("targets", {}).get("forecast_horizons_minutes", [15, 30, 60])
    if not horizons or any(int(h) <= 0 for h in horizons):
        raise ValueError("Forecast horizons must be positive integer minutes")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    run_root = resolve(BASE, config["paths"]["run_root"])
    run_dir = run_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_dir}")
    for name in ("logs", "reports", "merged", "labels", "ml_ready"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    # Keep the exact config with the run for reproducibility.
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    python = sys.executable
    scripts = BASE / "scripts"
    paths = config["paths"]
    archive_root = resolve(BASE, paths["archive_root"])
    manifest_path = run_dir / "reports" / "acquisition_manifest.json"

    # Hash and ZIP-validate local archives without modifying the source archives.
    run_step("acquisition_inventory", [python, str(scripts / "pipeline" / "acquire_data.py"),
             "--initialize", "--manifest", str(manifest_path), "--archive-root", str(archive_root),
             "--start", start.isoformat(), "--end", end.isoformat()], os.environ.copy(), run_dir / "logs")

    if not args.skip_extract:
        run_step("extract", [python, str(scripts / "coverage" / "extract_archives.py"),
                 "--start-date", start.strftime("%Y%m%d"), "--end-date", end.strftime("%Y%m%d"),
                 "--report-dir", str(run_dir / "reports" / "extraction")], os.environ.copy(), run_dir / "logs")
    else:
        (run_dir / "reports" / "extraction_skipped.txt").write_text(
            "Extraction skipped by request; existing raw FITS were required and checked during merge.\n",
            encoding="utf-8")
        prior_extraction = BASE / "output" / "selective_extraction_report.csv"
        prior_science = BASE / "output" / "scientific_validation"
        extraction_dir = run_dir / "reports" / "extraction"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        if prior_extraction.exists():
            shutil.copy2(prior_extraction, extraction_dir / "selective_extraction_report.csv")
        for report_name in ("lightcurve_quality_detail.csv", "lightcurve_gti_pairing.csv", "daily_scientific_quality.csv"):
            source_report = prior_science / report_name
            if source_report.exists():
                shutil.copy2(source_report, run_dir / "reports" / report_name)

    goes_file = resolve(BASE, paths["goes_catalogue"])
    run_step("parse_goes", [python, str(scripts / "pipeline" / "parse_goes.py"),
             "--input", str(goes_file), "--output-dir", str(run_dir / "labels")], os.environ.copy(), run_dir / "logs")

    workers = args.workers or int(config["merge"]["workers"])
    merge_cmd = [python, str(scripts / "pipeline" / "merge_lightcurves.py"),
                 "--workers", str(workers), "--start-date", start.strftime("%Y%m%d"),
                 "--end-date", end.strftime("%Y%m%d"), "--output-dir", str(run_dir / "merged")]
    if args.limit:
        merge_cmd.extend(["--limit", str(args.limit)])
    run_step("merge", merge_cmd, os.environ.copy(), run_dir / "logs")

    merge_suffix = "_test" if args.limit else ""
    merged_file = run_dir / "merged" / f"merged_timeseries_minute_end_v2{merge_suffix}.parquet"
    goes_csv = run_dir / "labels" / "goes_events_parsed.csv"
    labeled_file = run_dir / "labels" / "labeled_timeseries_minute_end_v2.parquet"
    forecast_file = run_dir / "labels" / "forecast_labeled_timeseries_minute_end_v2.parquet"
    run_step("label", [python, str(scripts / "pipeline" / "label_flares.py"),
             "--input", str(merged_file), "--goes", str(goes_csv), "--output", str(labeled_file),
             "--event-report", str(run_dir / "reports" / "mx_event_label_report.csv")],
             os.environ.copy(), run_dir / "logs")
    target_env = os.environ.copy()
    target_env["ARUNX_HORIZONS_MINUTES"] = ",".join(map(str, horizons))
    run_step("forecast_targets", [python, str(scripts / "pipeline" / "create_forecast_labels.py"),
             "--input", str(labeled_file), "--goes", str(goes_csv), "--output", str(forecast_file)],
             target_env, run_dir / "logs")

    env = os.environ.copy()
    env.update({
        "ARUNX_FORECAST_INPUT": str(forecast_file),
        "ARUNX_ML_OUTPUT": str(run_dir / "ml_ready"),
        "ARUNX_TRAIN_END": config["splits"]["train_end_exclusive_utc"],
        "ARUNX_VAL_END": config["splits"]["validation_end_exclusive_utc"],
        "ARUNX_ROW_POLICY": config["features"].get("row_policy", "both_instruments"),
        "ARUNX_ROLLING_WINDOWS": ",".join(map(str, config["features"]["rolling_windows_minutes"])),
        "ARUNX_DIFF_LAGS": ",".join(map(str, config["features"]["difference_lags_minutes"])),
        "ARUNX_HORIZONS_MINUTES": ",".join(map(str, horizons)),
    })
    run_step("build_features_and_splits", [python, str(scripts / "pipeline" / "build_ml_dataset.py")], env, run_dir / "logs")

    date_inventory(config, run_dir)
    run_step("validate", [python, str(scripts / "validation" / "validate_preprocessing_run.py"), "--run-dir", str(run_dir), "--merge-suffix", merge_suffix], os.environ.copy(), run_dir / "logs")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(run_dir / "config.json"),
        "source_goes_sha256": sha256_file(goes_file),
        "run_directory": str(run_dir.relative_to(BASE)).replace("\\", "/"),
        "stages": ["local archive inventory and ZIP validation", "selective extraction", "GOES parsing", "multi-day UTC merge", "GOES interval labels", "segment-aware forecast targets", "past-only features", "chronological task splits", "output validation"],
        "validation_report": str((run_dir / "reports" / "preprocessing_report.json").relative_to(BASE)).replace("\\", "/"),
        "output_files": [],
    }
    output_candidates = [merged_file, labeled_file, forecast_file, run_dir / "ml_ready" / "feature_schema.json", run_dir / "ml_ready" / "dataset_summary.csv"]
    output_candidates.extend(sorted((run_dir / "ml_ready").glob("*.parquet")))
    for output_file in output_candidates:
        if output_file.exists():
            manifest["output_files"].append({
                "path": str(output_file.relative_to(run_dir)).replace("\\", "/"),
                "size_bytes": output_file.stat().st_size,
                "sha256": sha256_file(output_file),
            })
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Preprocessing run completed: {run_dir}")
    print(f"Final task datasets: {run_dir / 'ml_ready'}")
    print(f"Report: {run_dir / 'reports' / 'preprocessing_report.json'}")


if __name__ == "__main__":
    main()
