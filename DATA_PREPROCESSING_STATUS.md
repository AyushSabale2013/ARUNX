# ARUN-X data preprocessing: current workflow and next work

## Scope and current state

This document follows the implemented path from downloaded archives to the task-specific ML tables. It describes the current multi-day **minute-end v2** pipeline, which supersedes the older one-day example and older minute-start outputs. Paths below are relative to `ARUNX/`.

The repository contains HEL1OS and SoLEXS archives, archive and extraction reports, parsed GOES events, a 207-date merged v2 table, labeled and forecast-labeled tables, and task-specific train/validation/test Parquet files with **592 engineered features**. The current reproducible run is `preproc_20261124_v1` under `output/preprocessing_runs/`. Structural checks passed, but its status is **`PASS_WITH_REVIEW`**: five requested dates lack merged coverage and one malformed GOES event remains excluded for manual review. The split files are generated artifacts; they do not establish model performance or scientific validity by themselves.

## Verified run snapshot (24 November 2026)

- The configured range is 2026-01-01 through 2026-07-31. Existing archives were inventoried and checked; the pipeline was run with `--skip-extract`, reusing prior extraction and raw-science audit evidence. No remote downloads were performed during this run.
- The run produced 298,080 minute-end rows across 207 dates, used the configured `both_instruments` policy, and generated six task families: classification, nowcasting, peak detection, and 15/30/60-minute forecast targets. Split files and SHA-256 hashes are recorded in the run manifest.
- Missing dates requiring resolution: 2026-02-17, 2026-04-16, 2026-05-22, 2026-06-04, and 2026-06-09.
- One July 3 M2.0 catalogue row has inconsistent chronology (start 23:50, peak 12:03, end 00:05). It is marked for review and excluded; its times were not inferred or repaired.
- The configured test period has only one X-class event, so X-class evaluation will be highly uncertain. No model training or performance evaluation is claimed.
- The downloader is implemented, but its authenticated remote path has not been exercised. Training CLI execution also remains unverified in this environment because the project virtual environment lacks `torch` and `joblib`.

## Implemented workflow, from source data to ML tables

### 1. Acquire and inventory source data

- The source observations are HEL1OS Level 1 ZIP archives and SoLEXS Level 1 ZIP archives staged under `downloaders/pradan1.issdc.gov.in/.../downloadData/`.
- Downloader scripts under `downloaders/` hold the portal-generated archive selections. `scripts/coverage/inventory_archives.py` inventories archives and `scripts/coverage/verify_archives.py` checks ZIP integrity. Reports are in `output/archive_inventory_report.csv` and `output/archive_integrity_report.csv`.
- The repository includes `output/acquisition_manifest.json`, but it currently has no entries. The run manifest records SHA-256 hashes for derived outputs, while the source archive manifest still lacks the original portal query, retrieval provenance, remote checksums, and a reproducible download record. Local archive integrity and hashes are recorded in the run reports.

### 2. Select files and extract only needed FITS products

- `scripts/coverage/extract_archives.py` selects the requested 2026 interval and extracts HEL1OS light-curve FITS files plus detector GTIs, and SoLEXS SDD2 light curves plus matching GTIs. It validates archive paths, file sizes, and CRC values and writes extraction evidence into the selected run’s `reports/` directory when run by the orchestrator.
- Extracted files are stored under `raw_hel1os/` and `raw_solexs/`. `output/extraction_inventory.csv`, instrument matching reports, and `output/data_quality_report.csv` help find missing or invalid expected products.
- `output/date_selection.csv`, `output/eligible_dates.csv`, `output/selected_dates_jan_jul_2026.csv`, and `output/final_date_report.csv` capture different screening stages. They do not all mean the same thing: archive duration/filename checks are candidate screens, while the merge reads the actual FITS and GTIs. Preserve these distinctions when describing which dates are usable.

### 3. Parse the GOES event catalogue

- `scripts/pipeline/parse_goes.py` reads `goes/goes_xray_event_list_2026.txt`, parses event start, peak, end, class, and intensity, handles midnight rollover, and writes `output/goes_events_parsed.csv` plus `output/goes_events_manual_review.csv`.
- `scripts/pipeline/audit_goes_catalogue.py` and the chronology anomaly/manual-review reports provide catalogue checks. GOES times are interpreted as UTC by the parser; the time convention and minute-rounding behavior still need to be independently confirmed against the source catalogue and mission products.
- GOES is a label source only. GOES-derived event fields and labels must never be included in model features.

### 4. Read FITS, apply GTIs, and aggregate to UTC minutes

- `scripts/pipeline/merge_lightcurves.py` discovers files by date, selects the highest HEL1OS processing version per observation/detector, checks that all four detectors and five bands are present, and locates the SoLEXS SDD2 light curve with its GTI.
- HEL1OS samples use MJD timestamps, count rates (`CTR`), and statistical errors; SoLEXS samples use Unix-epoch seconds and `COUNTS`. The script converts both to integer UTC microseconds. It masks measurements outside the corresponding GTI and non-finite timestamps/measurements.
- Each channel is averaged within one-minute bins. HEL1OS statistical errors are aggregated as `sqrt(sum(error^2))/n`. The output grid contains 1,440 **minute-end** timestamps per date: the first row is 00:01 UTC and the last is 00:00 UTC of the following date, representing the preceding minute.
- GTI, missing, signal, and HEL1OS statistical-error columns are retained. The orchestrator runs per-date workers, writes checkpoints and merge/quality reports under the selected run directory, and combines them into `merged/merged_timeseries_minute_end_v2.parquet`.
- Date range, paths, split boundaries, feature windows, row policy, and target horizons are now supplied through JSON configuration. The run report records requested dates without merged rows; a fuller per-date inclusion/exclusion decision table tied to a formally approved coverage policy remains to implement.

### 5. Add active-flare, class, and peak labels

- `scripts/pipeline/create_multiday_labels.py` adds `flare_active`, `flare_class`, and `flare_peak` to the merged observations using M/X GOES event intervals.
- Active labels cover event start through end on the minute-end grid; class is 0 for no active M/X event, 1 for M, and 2 for X, with the larger class retained where intervals overlap. Peak labels mark an exact matching timestamp.
- The labeled master table is written under the run directory at `labels/labeled_timeseries_minute_end_v2.parquet`; event association and GOES review details are retained in that run’s `labels/` reports.

### 6. Add future-start forecast targets

- `scripts/pipeline/create_forecast_labels.py` adds `flare_15m`, `flare_30m`, and `flare_60m` to the labeled master table and writes `labels/forecast_labeled_timeseries_minute_end_v2.parquet` under the selected run directory.
- A target is 1 when an M/X event starts strictly after the row timestamp and within the horizon, 0 when no start falls in the horizon, and -1 when the target is unknown. A row is unknown during an active M/X flare or when a continuous usable observation segment does not cover the entire future window.
- Segment boundaries are created by timestamp gaps and rows with no finite signal from either instrument. This is more conservative than merely checking the end of the overall file.

### 7. Build features and chronological task splits

- `scripts/pipeline/build_ml_dataset.py` reads the forecast-labeled v2 file, verifies sorted unique UTC timestamps and expected quality fields, and identifies the 20 HEL1OS channels plus SoLEXS SDD2.
- It creates 592 features: sensor availability; `log1p` signal and HEL1OS statistical-error values; missing and GTI flags; past-only rolling mean, standard deviation, count, minimum, and maximum over 5/15/30/60 minutes; lagged differences over 1/5/15 minutes; and cyclical UTC hour/day-of-year features. Rolling features reset across time gaps and unusable observation intervals.
- It applies chronological boundaries of 2026-06-01 UTC (train end) and 2026-07-01 UTC (validation end). Forecast unknowns (`-1`) and rows not meeting the configured `both_instruments` validity policy are excluded. Forecast samples near train/validation boundaries are purged by the horizon length.
- It writes six tasks (`classification`, `nowcasting`, `peak_detection`, and forecast horizons 15/30/60 minutes), each split into train/validation/test Parquet files under the selected run directory’s `ml_ready/`, plus `dataset_summary.csv` and `feature_schema.json`.

## Last implemented step and current stopping point

The last materialized preprocessing step is **feature engineering plus task-specific chronological split files** in `output/preprocessing_runs/preproc_20261124_v1/ml_ready/`. An orchestrator, configurable run file, stage logs, validation report, and output-hash manifest now link the stages. The production-quality build is still incomplete: review items above remain unresolved, authenticated remote acquisition was not exercised, and there is no approved date-eligibility policy reconciling all coverage reports.

The current configuration requires valid signal from both instruments. The generated splits are not yet automatically restricted to an approved eligible-date list or a formally agreed per-date coverage threshold. In addition, `log1p` is applied but no train-fitted scaler/normalizer is saved or applied. The audit is reproducible and the outputs are structurally validated, but `PASS_WITH_REVIEW` means these splits should not be treated as final until the listed dates and catalogue row are resolved.

## Work still to implement, in priority order

### P0 — Lock down data identity, time semantics, and date eligibility

1. **Partial:** local archives have sizes and hashes in inventory/run evidence, and resumable authenticated download handling is implemented. Populate the source acquisition manifest with the exact portal query, retrieval time, source URLs, and remote checksums when available; remote acquisition still needs an authenticated end-to-end verification.
2. **Remaining:** create an authoritative per-date status table joining archive integrity, required-file matching, extraction, actual GTI coverage, merge status, GOES events, and the final inclusion/exclusion reason. The current configuration requires both instruments, but the minimum per-date coverage threshold still needs approval.
3. **Partial:** UTC conversion, inclusive GTI endpoints, minute-end binning, and day-boundary ownership are implemented and unit-tested. Independently verify instrument time references and GOES alignment, and document onset/peak rounding for non-minute events.
4. **Remaining:** reconcile the date reports, approve inclusion policy, resolve the five missing days, and report actual partial-day exposure. A complete 1,440-row grid represents bins, not proof of uninterrupted observation.

### P1 — Make preprocessing reproducible and fail-safe

5. **Implemented with gaps:** the orchestrator runs inventory/validation through split generation and records configuration and output hashes. Add code revision identity and strengthen source-input provenance in the manifest.
6. **Mostly implemented:** configuration, unique run directories, stage logs, and resumable merge checkpoints are in place. Finish validating all config fields and review atomicity/idempotency across every stage.
7. **Remaining:** expand quality enforcement to duplicate samples, physically invalid values, cadence anomalies, outliers, long gaps, and minimum coverage. Current checks cover archive integrity, required channel matching, GTIs, finite values, missingness, and structural consistency.
8. **Partial:** overlap priority, interval inclusion, peak matching, forecast onset semantics, and unknown windows are encoded. Independently confirm catalogue time convention and rounding; document events outside coverage and validate a curated set of event examples.

### P2 — Finalize ML data preparation and evaluation safety

9. **Configured:** all generated tasks use `both_instruments`; publish per-split instrument coverage/missingness and confirm this policy is scientifically intended.
10. Fit any imputation, scaling, clipping, or other learned transform on training data only, save its parameters and schema, and reuse them unchanged for validation, test, and inference. Keep the feature code strictly past-only and audit features for label leakage.
11. Review split design and event counts by split. Keep dates chronological; consider an embargo/purge policy that accounts for both target horizon and feature lookback, and keep the final test dates untouched during tuning.
12. **Partial:** a machine-readable run audit checks timestamp/date structure, quality flags, target domains, feature schema, split readability/chronology, and output hashes. Extend it to per-date coverage, duplicates/gaps, all feature NaN/infinity cases, and explicit cross-split leakage checks.
13. Once the dataset passes the audit, document the exact dataset version and train baselines first (climatology/no-skill and logistic regression), then a class-weighted CatBoost or LightGBM model. Temporal CNN/CNN-LSTM models are later experiments once the independent event count supports them. Tune thresholds on validation dates only; leave test dates untouched and retain natural class prevalence in validation and test.
14. Report PR-AUC, threshold-specific precision/recall, event-level recall, false alarms per day, lead time, event counts, and date context. Accuracy alone is not informative for these rare targets. Any X-class claim must reflect the small X-event sample. These are proposed evaluation practices, not results measured by this preprocessing document.
15. For application integration, package the saved feature schema and any training-fitted preprocessing parameters with the validated model. Ensure inference uses the same past-only feature definitions and timestamp convention as training.

## Planning context from the progress report

The attached progress report proposed February-July 2026 as the broad study window and recommended finalizing dates from verified shared coverage and event counts. That is a planning proposal, not the boundary implemented in `build_ml_dataset.py`: the current script uses timestamps before 2026-06-01 for train, June for validation, and July onward for test. Reconcile this date policy with the intended study window and actual inclusion list before treating the generated splits as final. The report's CatBoost/LightGBM, baseline, metric, and integration recommendations are captured above as downstream work, not as completed model results.

## Key implementation references

- Acquisition/extraction: `downloaders/`, `scripts/coverage/`, `output/acquisition_manifest.json`, `output/selective_extraction_report.csv`
- GOES parsing/audit: `scripts/pipeline/parse_goes.py`, `scripts/pipeline/audit_goes_catalogue.py`, `output/goes_events_parsed.csv`
- Multi-day merge: `scripts/pipeline/merge_lightcurves.py`, `output/preprocessing_runs/preproc_20261124_v1/merged/merge_summary_minute_end_v2.csv`, `output/preprocessing_runs/preproc_20261124_v1/merged/merge_quality_report_minute_end_v2.csv`
- Labels: `scripts/pipeline/create_multiday_labels.py`, `scripts/pipeline/create_forecast_labels.py`
- Features/splits: `scripts/pipeline/build_ml_dataset.py`, `output/preprocessing_runs/preproc_20261124_v1/ml_ready/dataset_summary.csv`, `output/preprocessing_runs/preproc_20261124_v1/ml_ready/feature_schema.json`
- Historical one-day walkthrough: `PREPROCESSING_AND_MODEL_GUIDE.md` (its descriptions of paths and timestamp behavior predate the multi-day minute-end v2 pipeline).
