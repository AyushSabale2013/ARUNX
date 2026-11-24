# ARUN-X preprocessing

This repository prepares real Aditya-L1 HEL1OS and SoLEXS observations and GOES event labels for time-series model training. It contains FITS extraction, a multi-day minute-end UTC merge, GOES-based active/class/peak labels, segment-aware forecast targets, past-only features, chronological splits, and a validation report.

## Quick start

Use the repository virtual environment on Windows:

```powershell
cd ARUNX
.\venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python scripts\pipeline\run_preprocessing.py --config config\preprocessing.json --run-id my_run
```

For a short run on three real observation dates:

```powershell
python scripts\pipeline\run_preprocessing.py --config config\preprocessing_smoke.json --run-id smoke_run --skip-extract
```

The full command creates a new run folder and refuses to reuse an existing run ID. Run products are written to `output/preprocessing_runs/<run-id>/`; previous Parquet datasets are not overwritten.

## Downloading additional data

PRADAN requires an authenticated account and provides session-specific archive selections. Export the portal-generated HEL1OS and SoLEXS Python selector files after selecting a date range, then inventory any current local archives, import only the selector URL lists, and download through the manifest:

```powershell
python scripts\pipeline\acquire_data.py --initialize --start 2026-01-01 --end 2026-07-31
python scripts\pipeline\acquire_data.py --portal-hel1os-script <HEL1OS-selector.py> --portal-solexs-script <SoLEXS-selector.py> --start 2026-01-01 --end 2026-07-31
python scripts\pipeline\acquire_data.py --download
```

Set `PRADAN_COOKIE` in the process environment through an approved local secret mechanism before downloading. Do not paste credentials into the manifest, documentation, shell history, or source control. The importer parses literal URL lists from portal selectors without executing them or importing cookie values. Missing URLs are reported as blocked; the downloader does not invent portal endpoints.

See [ACQUISITION.md](ACQUISITION.md) for the manifest format and failure handling.

## Configuration and outputs

- `config/preprocessing.json` selects dates, 1-minute minute-end UTC cadence, feature windows, row policy, target horizons, and chronological split boundaries.
- `output/preprocessing_runs/<run-id>/run_manifest.json` records configuration and report locations.
- `output/preprocessing_runs/<run-id>/merged/` contains the synchronized Parquet table, per-date checkpoints, and merge/quality reports.
- `output/preprocessing_runs/<run-id>/labels/` contains parsed GOES events and labeled Parquet tables.
- `output/preprocessing_runs/<run-id>/ml_ready/` contains task-specific train/validation/test Parquet files and `feature_schema.json`.
- `output/preprocessing_runs/<run-id>/reports/preprocessing_report.json` summarizes archive/extraction audit evidence, rows, feature count, target distributions, split ranges, event counts, and validation status. `PASS_WITH_REVIEW` means structural checks passed while a documented date gap or GOES catalogue row remains unresolved.

Train scripts accept `--data-dir output/preprocessing_runs/<run-id>/ml_ready`, so they can consume a selected run directly. Example:

```powershell
python scripts\ml\train_baseline_models.py --data-dir output\preprocessing_runs\my_run\ml_ready
python scripts\ml\train_deep_learning.py --data-dir output\preprocessing_runs\my_run\ml_ready
```

## Current task definitions and limitations

- Active/class/peak labels use GOES M/X events; classification values are 0=no active M/X event, 1=M, and 2=X. Forecast targets are M/X onset within 15, 30, or 60 minutes. Unknown forecast rows use `-1` and are excluded from supervised split files.
- Minute-end row timestamps label the completed minute immediately before that time. The split builder assigns each row to its represented UTC date and purges forecast examples near split boundaries.
- The final feature table defaults to rows with at least one valid signal from each instrument. Per-channel missing and GTI indicators remain features. NaNs are retained for training-time imputation fitted on training data only.
- The current 592 features include per-channel log values, quality flags, HEL1OS error values, past-only rolling mean/std/count/min/max, lagged differences (including the one-minute rate of change), and cyclical UTC time features.
- A hard-to-soft ratio is intentionally not computed from raw `CTR` and `COUNTS` columns: their units and response calibration are not established as comparable. A four-state quiet/pre-flare/flare/decay target and 1-24-hour forecast horizons in the proposal PDFs are also not part of this current dataset contract; choose and validate those definitions before adding them.
- This repository stops at validated model-ready data. It does not implement the web application, live alert delivery, or measured model evaluation. The attached architecture PDFs describe a separate synthetic-data application snapshot and should not be read as a status report for this workspace.

See [architecture.md](architecture.md), [data_pipeline.md](data_pipeline.md), and [DATA_PREPROCESSING_STATUS.md](DATA_PREPROCESSING_STATUS.md) for more detail.
# Technical data documentation

- [Complete end-to-end preprocessing documentation (PDF)](docs/ADITYA_L1_COMPLETE_DATA_PREPROCESSING_DOCUMENTATION.pdf)
- [Canonical 592-feature master dataset and manifest](output/master_datasets/preproc_20261124_v1/manifest.json)
- Master Parquet: `output/master_datasets/preproc_20261124_v1/canonical_master_features_targets.parquet`
- Rebuild the additive master from the run's frozen source/schema: `venv\Scripts\python.exe scripts\pipeline\build_canonical_master.py`

The master is versioned separately from `output/preprocessing_runs/preproc_20261124_v1`; it has 298,080 rows, 705 columns, 592 approved features, and six targets. The preprocessing outputs are not overwritten.

